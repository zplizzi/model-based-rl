from lib.resnet import ResidualBlock, ResidualBlockPre
from torch import nn
import torch
import math
from lib.utils.utils import select_from_axis

hidden_channels = 32
hidden_dim = 32


# takes 64x64 input
class RepresentationNetwork(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        # 64x64
        self.c1 = nn.Conv2d(args.image_channels,
                            hidden_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        self.stack = nn.Sequential(*[
            ResidualBlockPre(
                hidden_channels, normalization="group_norm", residual_scale=.1)
            for _ in range(3)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.args.image_channels, 64, 64)
        x = self.c1(x)
        x = nn.functional.leaky_relu(x)
        x = self.stack(x)
        assert x.shape == (batch_size, hidden_channels, hidden_dim, hidden_dim)
        return x


class DecoderNetwork(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.stack = nn.Sequential(*[
            ResidualBlockPre(
                hidden_channels, normalization="group_norm", residual_scale=.1)
            for _ in range(3)
        ])
        # self.c1 = nn.Conv2d(hidden_channels, args.color_channels, kernel_size=3, stride=1, padding=1)

        # One layer to upsample in resolution
        self.c1 = nn.ConvTranspose2d(hidden_channels,
                                     hidden_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0)
        # And another to project down in channels
        self.c2 = nn.Conv2d(hidden_channels,
                            args.color_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, hidden_channels, hidden_dim, hidden_dim)
        x = self.stack(x)
        x = nn.functional.leaky_relu(x)
        x = self.c1(x)
        x = nn.functional.leaky_relu(x)
        x = self.c2(x)
        x = x[:, :, :64, :64]
        assert x.shape == (batch_size, self.args.color_channels, 64, 64)
        return x


class DynamicsNetwork(nn.Module):
    """
    The trick here is how to get the actions available for computation, while
    keeping the network residual - meaning that it performs the identity
    transform by default at initialization.

    The approach used here is to just maintain the action channels all the way
    through, and slice them off at the end. That way by default the actions
    stay in their channel and just get ignored, and the hidden is passed thru.
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.groups = 4
        self.num_action_layers = math.ceil(
            self.args.num_actions / self.groups) * self.groups
        self.in_channels = hidden_channels + self.num_action_layers

        self.stack = nn.Sequential(*[
            ResidualBlockPre(self.in_channels,
                             normalization="group_norm",
                             residual_scale=.1) for _ in range(4)
        ])

    def forward(self, x):
        # TODO: scale input hidden to (0, 1) as in paper
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.in_channels, hidden_dim,
                           hidden_dim)
        x = self.stack(x)
        # Trim the channel dim back down to 256
        # We trim instead of project through a learned layer to maintain the
        # residual propery
        x = x[:, :32]
        assert x.shape == (batch_size, hidden_channels, hidden_dim, hidden_dim)
        return x


class VectorHead(nn.Module):
    def __init__(self, args, n_outputs: int):
        super().__init__()
        self.in_channels = 32
        self.n_outputs = n_outputs

        self.stack = nn.Sequential(*[
            ResidualBlock(self.in_channels, normalization="group_norm")
            for _ in range(2)
        ])

        # TODO: fix + improve this
        self.c1 = nn.Conv2d(self.in_channels,
                            4,
                            kernel_size=3,
                            stride=4,
                            padding=1)
        self.fc1 = nn.Linear(16 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, self.n_outputs)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.in_channels, 6, 6)

        x = self.stack(x)
        x = nn.functional.leaky_relu(x)

        x = self.c1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc1(x.reshape((batch_size, -1)))
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        return x


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.representation_net = RepresentationNetwork(args)
        self.dynamics_net = DynamicsNetwork(args)
        self.i = 0

    def representation(self, image):
        """Take an image, convert to hidden."""
        return self.representation_net(image)

    def reward(self, hidden, action):
        raw_reward = self.reward_net(hidden)
        reward = select_from_axis(raw_reward, action, axis=1)
        return reward

    def dynamics(self, hidden, action):
        # reward = self.reward(hidden, action)
        reward = None

        batch_size = hidden.shape[0]
        assert hidden.shape == (batch_size, hidden_channels, hidden_dim, hidden_dim)
        assert action.shape == (batch_size, )
        # Convert the action into a format suitable for a convnet
        # Note: the paper didn't really explain how this should be done
        # So I just expanded each dim of a one-hot representation of the action
        # into a constant plane
        action = nn.functional.one_hot(
            action, num_classes=self.dynamics_net.num_action_layers)
        # assert action.shape == (batch_size, self.args.num_actions)
        action_planes = action.reshape(
            (batch_size, self.dynamics_net.num_action_layers, 1, 1))
        action_planes = action_planes.expand(
            (batch_size, self.dynamics_net.num_action_layers, hidden_dim,
             hidden_dim))
        action_planes = action_planes.float()

        dynamics_input = torch.cat((hidden, action_planes), dim=1)
        hidden = self.dynamics_net(dynamics_input)

        return hidden, reward
