import torch
from torch import nn

import ray
import ray.tune
import sys
import time

from lib import tracker_global as t
from papers.muzero.lib.utils import explained_variance
from papers.muzero.trainable import BaseTrainer
from papers.muzero.player import RandomPlayer, SAPlayer
# from papers.muzero.network import DQNNetwork
from papers.muzero.parse_args import parse_args

import plotly.graph_objects as go
import numpy as np
from lib.utils.utils import load_state_dict_debug, select_from_axis
from papers.muzero.lib.distributional import distributional_loss, plot_distribution

N = 200




class PolicyNetwork(nn.Module):
    """The baseline architecture described in Rainbow. Note this is approx
    2x as large as the one described in the original DQN paper, but
    is similar to the one in their Github repo."""
    def __init__(self, args):
        self.args = args
        super().__init__()
        # 84x84x4 input
        self.c1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        # 21x21x16
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        # 21x21x32
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # 11x11x32
        self.fc1 = nn.Linear(64 * 11 * 11, 512)
        self.fc2 = nn.Linear(512, args.num_actions * N)

    def forward(self, x):
        batch_size = x.shape[0]
        # assert x.shape == (batch_size, 4, 84, 84)
        x = self.c1(x)
        x = nn.functional.relu(x)
        # assert x.shape == (batch_size, 32, 21, 21), x.shape
        x = self.c2(x)
        x = nn.functional.relu(x)
        # assert x.shape == (batch_size, 64, 11, 11), x.shape
        x = self.c3(x)
        x = nn.functional.relu(x)
        # assert x.shape == (batch_size, 64, 11, 11), x.shape
        x = x.reshape((batch_size, 64 * 11 * 11))
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = x.reshape((batch_size, self.args.num_actions, N))
        # assert x.shape == (batch_size, self.args.num_actions)
        return x


class DQNNetwork(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.policy_net = PolicyNetwork(args)
        self.i = 0

    def representation(self, x):
        # We don't have a separate representation net
        return x

    def prediction(self, x):
        return self.policy_net(x)


class DQNPlayer(SAPlayer):
    network_cls = DQNNetwork

    def get_values(self, hiddens):
        supports = self.network.prediction(hiddens)
        # Compute the expected value of the distribution
        values = supports.mean(dim=2)
        # assert values.shape == (len(self.games), self.args.num_actions)
        return values


class Trainer(BaseTrainer):
    # network_cls = DQNNetwork

    def _setup(self, config):
        self.args = config["args"]
        self.network = DQNNetwork(self.args)
        self.network = self.network.to(args.device)

        self.target_network = DQNNetwork(self.args)
        self.target_network = self.target_network.to(args.device)

        path = t.download_model("dd04403859464f08a42ba18a51b28eaa", "muzero")
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

        super()._setup(config, player_cls=DQNPlayer, args=self.args)
        # super()._setup(config, player_cls=RandomPlayer, args=self.args)

    def _train_step(self, batch, logging=False):
        """
        For training examples, we have pairs of
            (state, action, reward, next_state)
        """
        self.network.train()
        self.optimizer.zero_grad()

        # batch = [x.cuda() for x in batch]
        images = batch["images"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        # images, actions, rewards, _, _, weights, keys = batch
        initial_image = images[:, 0]
        final_image = images[:, 1]

        reward = rewards[:, 0]

        # Get the network prediction of the value given the initial state, action
        # pair
        initial_action = actions[:, 0]
        hidden = self.network.representation(initial_image)
        supports = self.network.prediction(hidden)

        # Note the use of the target network
        with torch.no_grad():
            hidden = self.target_network.representation(final_image)
            next_state_supports = self.target_network.prediction(hidden)

        batch_size = self.args.batch_size
        assert supports.shape == (batch_size, self.args.num_actions, N)

        # Get the supports for the chosen actions - the values of z(s, a)
        supports = select_from_axis(supports, initial_action, axis=1)
        assert supports.shape == (batch_size, N)

        best_next_action = next_state_supports.mean(dim=2).argmax(dim=1)
        assert best_next_action.shape == (batch_size, )
        next_supports = select_from_axis(next_state_supports, best_next_action,
                                         axis=1)
        assert next_supports.shape == (batch_size, N)

        # Contract next-state distribution by gamma and shift by rewards to get the target dist
        assert reward.shape == (batch_size, )
        # We want rewards to pair up on the batch axis
        target_supports = reward.unsqueeze(1) + self.args.discount * next_supports
        assert target_supports.shape == (batch_size, N)

        loss = distributional_loss(supports, target_supports)
        loss = loss.mean(dim=0)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), .5)
        self.optimizer.step()

        freq = 300

        if logging and t.i % freq == 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_image_grid("input_final",
                             final_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_histogram("reward", reward, freq=freq)
            t.add_histogram("initial action", initial_action, freq=freq)
            t.add_chart("charts/target_dist",
                        plot_distribution(target_supports[0].cpu().detach()))
            t.add_chart("charts/pred_dist",
                        plot_distribution(supports[0].cpu().detach()))

            t.add_histogram("charts/pred_mean", supports.mean(dim=1))
            t.add_histogram("charts/target_mean", target_supports.mean(dim=1))
            t.add_histogram("charts/pred_var", supports.std(dim=1))
            t.add_histogram("charts/target_var", target_supports.std(dim=1))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # ray.init(ignore_reinit_error=True, local_mode=True)
    ray.init(args.ray_address,
             ignore_reinit_error=True,
             object_store_memory=5 * 1024**3)

    config = {
        "group": "asdf",
        "args": args,
    }
    trainer = Trainer(config)

    while True:
        trainer.train()
