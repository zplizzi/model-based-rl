import torch
from torch import nn

import ray
import ray.tune
import sys
import time
import numpy as np
import scipy.stats
import random

from lib import tracker_global as t
from papers.muzero.lib.utils import explained_variance
from papers.muzero.trainable import BaseTrainer
from papers.muzero.player import RandomPlayer, SAPlayer
from papers.muzero.network import DQNNetwork, VectorHead
# from papers.muzero import network_full as network
from papers.muzero import network
from papers.muzero.parse_args import parse_args

import collections
from collections import defaultdict

from lib.resnet import ResidualBlock

from lib.utils.utils import load_state_dict_debug, debug_grad_norm, select_from_axis
from papers.muzero.lib.distributional import plot_distribution, distributional_loss

N = 100


class LearnedDynamicsNetwork(network.Network):
    def __init__(self, args):
        super().__init__(args)
        self.decoder_net = network.DecoderNetwork(args)
        self.previous_reward_net = VectorHead(args, n_outputs=1)

        self.policy_net = VectorHead(args, n_outputs=self.args.num_actions * N)
        self.reward_net = VectorHead(args, n_outputs=self.args.num_actions * N)

    def reward(self, hidden, action):
        raw_reward = self.reward_net(hidden)
        raw_reward = raw_reward.reshape((self.args.batch_size, self.args.num_actions, N))
        reward = select_from_axis(raw_reward, action, axis=1)
        return reward


class LearnedDynamicsPlayer(SAPlayer):
    network_cls = LearnedDynamicsNetwork

    def get_values(self, hiddens):
        batch_size = len(hiddens)
        # Use a DQN-style network where we predict all values of Q(S, A)
        # in the same forward pass without a dynamics step
        supports = self.network.policy_net(hiddens)
        supports = supports.reshape((batch_size, self.args.num_actions, N))
        # Compute the expected value of the distribution
        values = supports.mean(dim=2)

        # Values are the mean of the bottom 10 percent
        # values = supports[:, :, :10].mean(dim=2)

        assert values.shape == (len(self.games), self.args.num_actions)
        return values

    def push_to_buffer(self):
        super().push_to_buffer()
        # TODO: remove !!!!
        # time.sleep(.03)


class Trainer(BaseTrainer):
    def _setup(self, config):
        self.args = config["args"]
        self.network = LearnedDynamicsNetwork(self.args)
        self.network = self.network.to(args.device)

        self.target_network = LearnedDynamicsNetwork(self.args)
        self.target_network = self.target_network.to(args.device)

        path = t.download_model("1becab1c2d9a4bf7af2571ce7e3a73f5", "muzero")
        load_state_dict_debug(self.network, torch.load(path))
        load_state_dict_debug(self.target_network, torch.load(path))
        # self.network.load_state_dict(torch.load(path))
        # self.target_network.load_state_dict(torch.load(path))

        # super()._setup(config, player_cls=RandomPlayer, args=self.args)
        super()._setup(config,
                       player_cls=LearnedDynamicsPlayer,
                       args=self.args)

    def do_per_eval_step(self):
        super().do_per_eval_step()

    def _train_step(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        initial_image = images[:, 0]
        final_image = images[:, -1]

        hidden = self.network.representation(initial_image)
        hidden_dynamics = hidden

        losses = defaultdict(list)
        for i in range(0, self.args.num_unroll_steps):
            # for a given i we have (state, action, etc). So we want the action
            # taken on step i, doing dynamics then decoder will give state
            # i+1
            action = actions[:, i]
            # The reward is the one resulting immediately from the action
            reward = rewards[:, i]
            # Hiddens at t-1
            old_hiddens = hidden_dynamics
            hidden_dynamics, reward_pred = self.network.dynamics(
                hidden_dynamics, action)
            assert reward_pred.shape == (self.args.batch_size, N)
            image = images[:, i + 1]
            hidden_repr = self.network.representation(image)
            # Get the last frame from the framestack
            # TODO: this will only work for greyscale images
            target_frame = images[:, i + 1, -1]

            # Unrolled autoencoding loss term
            decoded_dynamics = self.network.decoder_net(hidden_dynamics)
            assert decoded_dynamics.shape == (self.args.batch_size,
                                              self.args.color_channels,
                                              self.args.dim, self.args.dim)
            decoded_dynamics = decoded_dynamics.reshape(
                (self.args.batch_size, self.args.dim, self.args.dim))
            l = nn.functional.mse_loss(decoded_dynamics,
                                       target_frame,
                                       reduction="none")
            l = l.clamp(0, .0025)
            l = l.mean(-1).mean(-1)
            assert l.shape == (self.args.batch_size, )
            l = l * 5
            # For riverraid
            l = l * 10
            losses["unrolled_autoencoder"].append(l)

            # TODO: remove
            # hidden_dynamics, reward_pred = self.network.dynamics(
            #     hidden_dynamics, action)

            # Autoencoding loss without dynamics network
            # TODO: we could also do this at timestep i=0
            decoded_repr = self.network.decoder_net(hidden_repr)
            assert decoded_repr.shape == (self.args.batch_size,
                                          self.args.color_channels,
                                          self.args.dim, self.args.dim)
            decoded_repr = decoded_repr.reshape(
                (self.args.batch_size, self.args.dim, self.args.dim))
            l = nn.functional.mse_loss(decoded_repr,
                                       target_frame,
                                       reduction="none")
            l = l.clamp(0, .0025)
            l = l.mean(-1).mean(-1)
            l = l * 5
            # For riverraid
            l = l * 100
            # l = l * 10
            assert l.shape == (self.args.batch_size, )
            losses["direct_autoencoder"].append(l)

            # Reward loss
            assert reward.shape == (self.args.batch_size, )
            # Expand so all supports have the same value
            reward_target = reward.unsqueeze(1).expand((self.args.batch_size, N))
            assert reward_pred.shape == reward_target.shape
            l = distributional_loss(reward_pred, reward_target)
            l = l * .5
            assert l.shape == (self.args.batch_size, )
            losses["reward"].append(l)

            # # One-step Reward loss
            # try:
            #     one_step_rewards = self.network.reward(hidden_repr, actions[:, i+1])
            #     # Expand so all supports have the same value
            #     one_step_reward_target = rewards[:, i + 1].unsqueeze(1).expand(
            #         (self.args.batch_size, N))
            #     assert one_step_rewards.shape == one_step_reward_target.shape
            #     l = distributional_loss(one_step_rewards, one_step_reward_target)
            #     l = l * .5
            #     assert l.shape == (self.args.batch_size, )
            #     losses["one_step_reward"].append(l)
            # except IndexError:
            #     # This won't work on the last step
            #     pass

            # Value Loss
            supports = self.network.policy_net(old_hiddens)
            supports = supports.reshape((self.args.batch_size, self.args.num_actions, N))
            assert supports.shape == (self.args.batch_size,
                                                 self.args.num_actions, N)
            supports = select_from_axis(supports, action, axis=1)

            # TODO: hidden_repr is not from the target network. So this is
            # kinda weird.
            # TODO: put back the target network??
            # next_state_supports = self.target_network.policy_net(hidden_repr)
            next_state_supports = self.network.policy_net(hidden_repr)
            next_state_supports = next_state_supports.reshape((self.args.batch_size, self.args.num_actions, N))
            best_next_action = next_state_supports.mean(dim=2).argmax(dim=1)
            assert best_next_action.shape == (self.args.batch_size, )
            next_supports = select_from_axis(next_state_supports,
                                             best_next_action,
                                             axis=1)
            assert next_supports.shape == (self.args.batch_size, N)

            # Contract next-state distribution by gamma and shift by rewards to get the target dist
            # TODO: train with predicted rewards??
            assert reward.shape == (self.args.batch_size, )
            # We want rewards to pair up on the batch axis
            target_supports = reward.unsqueeze(
                1) + self.args.discount * next_supports
            assert target_supports.shape == (self.args.batch_size, N)

            l = distributional_loss(supports, target_supports.detach())

            assert l.shape == (self.args.batch_size, )
            l = l * .2
            l = l * .1
            losses["value"].append(l)

            # # One-step value loss (for training dynamics)
            # pred = self.network.policy_net(hidden)
            # pred = pred.reshape((self.args.batch_size, self.args.num_actions, N)).mean(dim=2)
            # # target = self.network.policy_net(hidden_repr).detach()
            # target = next_state_supports.mean(dim=2).detach()
            # # TODO: this is an arbitrary selection of loss (thus dist func)
            # # Most likely not stable under stochasic estimation.
            # assert pred.shape == target.shape
            # l = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
            # # Shape is (batch_size, actions, N)
            # # So we want to sum over N, mean over actions
            # # l = l.sum(dim=2).mean(dim=1)
            # l = l.mean(dim=1)
            # l = l * .002
            # assert l.shape == (self.args.batch_size, )
            # losses["value_dynamics"].append(l)

        # Get the loss for each type and each sample in the batch
        loss_dict = {
            k: torch.stack(values).sum(dim=0)
            for k, values in losses.items()
        }

        # Get the total loss for each item in the batch
        per_sample_loss = torch.stack(list(loss_dict.values())).sum(dim=0)
        assert per_sample_loss.shape == (self.args.batch_size, )

        loss = per_sample_loss.mean()

        loss.backward()
        # if logging and t.i % 100 == 0:
        #     grad_norm = debug_grad_norm(self.network.parameters())
        #     t.add_scalar("charts/grad_norm", grad_norm, freq=1)
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), .4)
        self.optimizer.step()

        freq = 1000
        if logging and t.i % freq == 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            for key, value in loss_dict.items():
                t.add_scalar(f"loss/{key}", value.mean(), freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_image_grid("input_final",
                             final_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_image("final_unrolled", decoded_dynamics[0], freq=freq)
            t.add_image("target frame", target_frame[0], freq=freq)
            t.add_image("final_direct", decoded_repr[0], freq=freq)
            t.add_scalar(f"charts/reward_expl_var",
                         explained_variance(reward_pred.mean(dim=1), reward),
                         freq=1)
            t.add_scalar(f"charts/q_expl_var",
                         explained_variance(supports.mean(dim=1),
                                            target_supports.mean(dim=1)),
                         freq=1)
            t.add_histogram("charts/reward_target", reward, freq=1)

            t.add_histogram("charts/reward_pred_mean",
                            reward_pred.mean(dim=1))
            t.add_histogram("charts/reward_pred_std", reward_pred.std(dim=1))

            t.add_histogram("charts/pred_value_mean", supports.mean(dim=1))
            t.add_histogram("charts/target_value_mean",
                            target_supports.mean(dim=1))
            t.add_histogram("charts/pred_value_var", supports.std(dim=1))
            t.add_histogram("charts/target_value_var",
                            target_supports.std(dim=1))

            t.add_chart("value_pred_dist",
                        plot_distribution(supports[0].cpu().detach()))
            t.add_chart("value_target_dist",
                        plot_distribution(target_supports[0].cpu().detach()))
            t.add_chart("reward_pred_dist",
                        plot_distribution(reward_pred[0].cpu().detach()))

            # supports have shape (batch_size, N)
            w1_val = (supports - target_supports).abs().sum(dim=1)
            t.add_histogram("charts/w1_value", w1_val, plot_mean=True)

            w1_reward = (reward_target - reward_pred).abs().sum(dim=1)
            t.add_histogram("charts/w1_reward", w1_reward, plot_mean=True)

            # w1_one_step_reward = (one_step_reward_target - one_step_rewards).abs().sum(dim=1)
            # t.add_histogram("charts/w1_one_step_reward", w1_one_step_reward, plot_mean=True)

            # # w1_value_dynamics = (target - pred).abs().sum(dim=2).mean(dim=1)
            # w1_value_dynamics = (target - pred).abs().mean(dim=1)
            # t.add_histogram("charts/w1_value_dynamics", w1_value_dynamics, plot_mean=True)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # ray.init(ignore_reinit_error=True, local_mode=True)
    ray.init(args.ray_address,
             ignore_reinit_error=True,
             object_store_memory=10 * 1024**3)

    config = {
        "group": "asdf",
        "args": args,
    }
    trainer = Trainer(config)

    while True:
        trainer.train()
