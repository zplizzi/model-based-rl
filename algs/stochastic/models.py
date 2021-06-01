from lib.resnet import ResidualBlock
from torch import nn
import torch
import math
from lib.utils.utils import select_from_axis

NUM_CODE_LAYERS = 64

# takes 64x64 input
class RepresentationNetwork(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        # 64x64
        self.c1 = nn.Conv2d(
            args.image_channels, 32, kernel_size=3, stride=2, padding=1)
        # 32
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # 16
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # 8
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        # 6
        self.stack = nn.Sequential(*[
            ResidualBlock(256, normalization="group_norm") for _ in range(2)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        # assert x.shape == (batch_size, self.args.image_channels, 64, 64)
        x = self.c1(x)
        x = nn.functional.leaky_relu(x)
        x = self.c2(x)
        x = nn.functional.leaky_relu(x)
        x = self.c3(x)
        x = nn.functional.leaky_relu(x)
        x = self.c4(x)
        x = nn.functional.leaky_relu(x)
        x = self.stack(x)
        # assert x.shape == (batch_size, 256, 6, 6)
        return x


class CodeNetwork(nn.Module):
    """
    The code network takes 2 frames as input (prediction and target) and
    outputs a code representing the stochastic noise.
    """
    def __init__(self, args):
        self.args = args
        super().__init__()
        # 64x64
        self.c1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)
        # 32
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # 16
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # 8
        self.c4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        # 6
        self.fc1 = nn.Linear(32 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, NUM_CODE_LAYERS)

    def forward(self, x):
        batch_size = x.shape[0]
        # assert x.shape == (batch_size, self.args.image_channels, 64, 64)
        x = self.c1(x)
        x = nn.functional.leaky_relu(x)
        x = self.c2(x)
        x = nn.functional.leaky_relu(x)
        x = self.c3(x)
        x = nn.functional.leaky_relu(x)
        x = self.c4(x)
        x = nn.functional.leaky_relu(x)
        x = x.reshape((batch_size, 32 * 6 * 6))
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        assert x.shape == (batch_size, NUM_CODE_LAYERS)
        return x

class DecoderNetwork(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.stack = nn.Sequential(*[
            ResidualBlock(256, normalization="group_norm") for _ in range(2)
        ])

        # 6x6x256
        self.c1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=2)
        # 8x8x128
        self.c2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                     padding=0)
        # 16x16x64
        self.c3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                     padding=1)
        # 32x32x32
        self.c4 = nn.ConvTranspose2d(32, args.color_channels, kernel_size=3,
                                     stride=2, padding=1)

    def forward(self, x):
        # TODO: normalization?
        batch_size = x.shape[0]

        x = self.stack(x)
        x = nn.functional.leaky_relu(x)

        # assert x.shape == (batch_size, 256, 6, 6)
        x = self.c1(x)
        x = nn.functional.leaky_relu(x)
        # assert x.shape == (batch_size, 128, 8, 8)
        x = self.c2(x)
        x = nn.functional.leaky_relu(x)
        # assert x.shape == (batch_size, 64, 16, 16)
        x = self.c3(x)
        # TODO: Should I omit this second-to-last relu?
        x = nn.functional.leaky_relu(x)
        # assert x.shape == (batch_size, 32, 32, 32)
        x = self.c4(x)
        # The weird padding settings and slicing here are to avoid using
        # the output_padding option, which has a huge perf penalty
        x = x[:, :, :64, :64]
        assert x.shape == (batch_size, self.args.color_channels, 64, 64), x.shape

        # For now let's simplify things since we're only using greyscale
        assert self.args.color_channels == 1
        x = x.reshape((batch_size, 64, 64))

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
        # self.in_channels = 256 + self.args.num_actions
        num_action_layers = math.ceil(self.args.num_actions / 16) * 16
        self.in_channels = 256 + num_action_layers + NUM_CODE_LAYERS
        self.stack = nn.Sequential(*[
            # ResidualBlock(256, normalization="group_norm") for _ in range(5)
            ResidualBlock(self.in_channels, normalization="group_norm") for _ in range(8)
        ])

    def forward(self, x):
        # TODO: scale input hidden to (0, 1) as in paper
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.in_channels, 6, 6)
        x = self.stack(x)
        # Trim the channel dim back down to 256
        # We trim instead of project through a learned layer to maintain the
        # residual propery
        x = x[:, :256]
        assert x.shape == (batch_size, 256, 6, 6)
        return x


class VectorHead(nn.Module):
    def __init__(self, args, n_outputs: int):
        super().__init__()
        self.in_channels = 256
        self.n_outputs = n_outputs

        self.stack = nn.Sequential(*[
            ResidualBlock(256, normalization="group_norm") for _ in range(2)
        ])

        self.c1 = nn.Conv2d(self.in_channels, 16, kernel_size=1)
        self.fc1 = nn.Linear(16 * 6 * 6, 256)
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
        self.decoder_net = DecoderNetwork(args)
        self.code_net = CodeNetwork(args)

        self.i = 0

    def prediction(self, hidden):
        """Predict value, reward, and policy given a hidden state."""
        batch_size = hidden.shape[0]
        assert hidden.shape == (batch_size, 256, 6, 6)

        value_logits = self.value_net(hidden).reshape((batch_size, ))
        policy_logits = self.policy_net(hidden)

        return value_logits, policy_logits

    def representation(self, image):
        """Take an image, convert to hidden."""
        return self.representation_net(image)

    def reward(self, hidden, action):
        raw_reward = self.reward_net(hidden)
        reward = select_from_axis(raw_reward, action, axis=1)
        return reward

    def dynamics(self, hidden, action, code):
        batch_size = hidden.shape[0]
        assert hidden.shape == (batch_size, 256, 6, 6)
        assert action.shape == (batch_size, )

        assert code.shape == (batch_size, NUM_CODE_LAYERS)
        code = code.reshape((batch_size, NUM_CODE_LAYERS, 1, 1))
        code = code.expand((batch_size, NUM_CODE_LAYERS, 6, 6))
        assert code.shape == (batch_size, NUM_CODE_LAYERS, 6, 6)

        # Convert the action into a format suitable for a convnet
        # Note: the paper didn't really explain how this should be done
        # So I just expanded each dim of a one-hot representation of the action
        # into a constant plane
        num_action_layers = math.ceil(self.args.num_actions / 16) * 16
        action = nn.functional.one_hot(
            action, num_classes=num_action_layers)
        # assert action.shape == (batch_size, self.args.num_actions)
        action_planes = action.reshape((batch_size, num_action_layers, 1,
                                        1))
        action_planes = action_planes.expand((batch_size,
                                              num_action_layers, 6, 6))
        action_planes = action_planes.float()

        dynamics_input = torch.cat((hidden, action_planes, code), dim=1)
        hidden = self.dynamics_net(dynamics_input)
        return hidden
