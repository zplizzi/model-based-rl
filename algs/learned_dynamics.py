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
from papers.muzero import network
from papers.muzero.parse_args import parse_args

import collections
from collections import defaultdict

from lib.resnet import ResidualBlock
class LargerVectorHead(nn.Module):
    def __init__(self, args, n_outputs: int):
        super().__init__()
        self.stack = nn.Sequential(*[
            ResidualBlock(256, normalization="group_norm") for _ in range(3)
        ])
        self.in_channels = 256
        self.n_outputs = n_outputs

        self.c0 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1)

        self.c1 = nn.Conv2d(self.in_channels, 2, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(2 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, self.n_outputs)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.in_channels, 6, 6)

        x = self.stack(x)
        x = nn.functional.leaky_relu(x)

        # TODO: remove this!
        x = self.c0(x)
        x = nn.functional.leaky_relu(x)

        x = self.c1(x)
        # x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        assert x.shape == (batch_size, 2, 6, 6)
        x = self.fc1(x.reshape((batch_size, -1)))
        # TODO: maybe get rid of this?
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        assert x.shape == (batch_size, self.n_outputs)
        return x


def debug_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    t.add_scalar("charts/grad_norm", total_norm, freq=1)


class LearnedDynamicsNetwork(network.Network):
    def __init__(self, args):
        super().__init__(args)
        self.decoder_net = network.DecoderNetwork(args)
        self.previous_reward_net = VectorHead(args, n_outputs=1)

class LargerLearnedDynamicsNetwork(LearnedDynamicsNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.policy_net = LargerVectorHead(args, args.num_actions)


class RandomSleepingPlayer(RandomPlayer):
    def push_to_buffer(self):
        super().push_to_buffer()
        # TODO: remove !!!!
        # time.sleep(.03)


class LearnedDynamicsPlayer(SAPlayer):
    # network_cls = LearnedDynamicsNetwork
    network_cls = LargerLearnedDynamicsNetwork

    def get_values(self, hiddens):
        # Use a DQN-style network where we predict all values of Q(S, A)
        # in the same forward pass without a dynamics step
        values = self.network.policy_net(hiddens)
        assert values.shape == (len(self.games), self.args.num_actions)
        return values

    def push_to_buffer(self):
        super().push_to_buffer()
        # TODO: remove !!!!
        # time.sleep(.03)


class Trainer(BaseTrainer):
    def _setup(self, config):
        self.args = config["args"]
        # self.network = LearnedDynamicsNetwork(self.args)
        self.network = LargerLearnedDynamicsNetwork(self.args)
        self.network = self.network.to(args.device)

        # self.target_network = LearnedDynamicsNetwork(self.args)
        self.target_network = LargerLearnedDynamicsNetwork(self.args)
        self.target_network = self.target_network.to(args.device)

        self.overfitting_samples = collections.deque(maxlen=100000)

        path = t.download_model("6a69b47be8c0454a96163490d75c5457", "muzero")
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

        # TODO: remove
        # self.network.policy_net = LargerVectorHead(self.args, self.args.num_actions).cuda()
        # self.target_network.policy_net = LargerVectorHead(self.args, self.args.num_actions).cuda()

        # super()._setup(config, player_cls=RandomSleepingPlayer, args=self.args)
        super()._setup(
            config, player_cls=LearnedDynamicsPlayer, args=self.args)

    def do_per_eval_step(self):
        super().do_per_eval_step()
        # Do something with the age samples
        # Ideally I'd plot a big scatter plot.. but i don't think wandb
        # can handle that
        if len(self.overfitting_samples) > 1000:
            # t.save_to_file("overfitting_samples", self.overfitting_samples, freq=1)
            data = np.array(self.overfitting_samples)
            x = data[:, 0]
            buffer_size = x.max()
            x = buffer_size - x
            y = data[:, 1]
            results = scipy.stats.linregress(x, y)
            delta = results.slope * buffer_size
            percent_delta = delta / results.intercept

            t.add_scalar("replay_buffer/size", buffer_size, freq=1)
            t.add_scalar("replay_buffer/percent_overfit",
                         percent_delta,
                         freq=1)
            t.add_scalar("replay_buffer/fit_confidence",
                         results.pvalue,
                         freq=1)

    def _train_step(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        initial_image = images[:, 0]
        final_image = images[:, -1]
        keys = batch["key"]
        weights = batch["weight"]

        # TODO: loss/gradient scaling?

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
            image = images[:, i + 1]
            hidden_repr = self.network.representation(image)
            # Get the last frame from the framestack
            # TODO: this will only work for greyscale images
            target_frame = images[:, i + 1, -1]

            # Loss term where we compare the hidden state computed by unrolling
            # the dynamics network to the hidden state from embedding the
            # input frames directly with the representation network.
            # TODO: should they be scaled by the same amount?
            # TODO: if we try this again, you need to detach the initial
            # representation that is used to seed the dynamics net. otherwise
            # grads will still flow in to repr
            # t1 = hidden_dynamics / hidden_dynamics.norm()
            # t2 = hidden_repr / hidden_repr.norm()
            # l = nn.functional.smooth_l1_loss(t1, t2.detach(), reduction="none")
            # # l = l * .01
            # l = l * 1e5
            # # l = l.clamp(-.1, .1)
            # hidden_repr_losses.append(l)

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
            l = l.mean(-1).mean(-1)
            assert l.shape == (self.args.batch_size, )
            # l = l * .01
            l = l * 5
            # For riverraid
            l = l * 10
            losses["unrolled_autoencoder"].append(l)

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
            l = l.mean(-1).mean(-1)
            l = l * 5
            # For riverraid
            l = l * 100
            assert l.shape == (self.args.batch_size, )
            losses["direct_autoencoder"].append(l)

            # Reward loss
            assert reward_pred.shape == reward.shape
            l = nn.functional.smooth_l1_loss(reward_pred,
                                             reward,
                                             reduction="none")
            l = l * .5
            assert l.shape == (self.args.batch_size, )
            losses["reward"].append(l)

            # One-step Previous Reward loss
            # The idea here is that we use hidden[i] to predict reward[i-1]
            prev_reward = self.network.previous_reward_net(hidden_repr)
            prev_reward = prev_reward.reshape((self.args.batch_size, ))
            assert prev_reward.shape == reward.shape
            l = nn.functional.smooth_l1_loss(prev_reward,
                                             reward,
                                             reduction="none")
            l = l * .5
            assert l.shape == (self.args.batch_size, )
            losses["previous_reward"].append(l)

            # One-step Reward loss
            try:
                one_step_rewards = self.network.reward_net(hidden_repr).gather(
                    1, actions[:, i + 1].unsqueeze(1))
                one_step_rewards = one_step_rewards.reshape(
                    (self.args.batch_size, ))
                assert one_step_rewards.shape == reward.shape
                l = nn.functional.smooth_l1_loss(one_step_rewards,
                                                 rewards[:, i + 1],
                                                 reduction="none")
                l = l * .5
                assert l.shape == (self.args.batch_size, )
                losses["one_step_reward"].append(l)
            except IndexError:
                # This won't work on the last step
                pass

            # Value Loss
            values_pred_initial = self.network.policy_net(old_hiddens)
            assert values_pred_initial.shape == (self.args.batch_size,
                                                 self.args.num_actions)
            q_pred = values_pred_initial.gather(1, action.unsqueeze(1))
            q_pred = q_pred.reshape((self.args.batch_size, ))

            values_pred_final = self.target_network.policy_net(hidden_repr)
            assert values_pred_final.shape == (self.args.batch_size,
                                               self.args.num_actions)
            best_next_q = torch.max(values_pred_final, dim=1).values
            assert best_next_q.shape == (self.args.batch_size, )
            # The updated estimate of the value of the selected action
            q_target = reward + self.args.discount * best_next_q
            assert q_target.shape == (self.args.batch_size, )
            assert q_pred.shape == q_target.shape
            l = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach(), reduction="none")
            assert l.shape == (self.args.batch_size, )
            l = l * .2
            losses["value"].append(l)

        # Get the loss for each type and each sample in the batch
        loss_dict = {
            k: torch.stack(values).sum(dim=0)
            for k, values in losses.items()
        }

        # Get the total loss for each item in the batch
        per_sample_loss = torch.stack(list(loss_dict.values())).sum(dim=0)
        assert per_sample_loss.shape == (self.args.batch_size, )

        # age = batch["age"].cpu().tolist()
        # self.overfitting_samples += list(zip(age, per_sample_loss.tolist()))

        weighted_losses = per_sample_loss * weights
        loss = weighted_losses.mean()
        # loss = per_sample_loss.mean()

        loss.backward()
        debug_grad_norm(self.network.parameters())
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), .05)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), .4)
        self.optimizer.step()

        freq = 100
        # TODO: implement importance scaling of loss!
        # Without it, weird shit will happen to the value function
        if self.args.prioritized_replay:
            # Don't allow loss greater than 1 to avoid an outlier throwing
            # everything off.
            # TODO: make a more general approach to this
            new_priorities = per_sample_loss.clamp(.01, 1).detach().cpu().numpy()
            keys = keys.cpu().numpy()
            try:
                self.replay_buffer.update_priorities.remote(keys, new_priorities)
            except AttributeError:
                self.replay_buffer.update_priorities(keys, new_priorities)
            if logging:
                old_priorities = batch["priority"].cpu().numpy()
                t.add_histogram(
                    "replay/old_priorities", old_priorities, freq=freq)
                t.add_histogram(
                    "replay/new_priorities", new_priorities, freq=freq)
                t.add_histogram(
                    "replay/diff_priorities",
                    new_priorities - old_priorities,
                    freq=freq)
                t.add_histogram("replay/weights", weights, freq=freq)

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
                         explained_variance(reward_pred, reward),
                         freq=1)
            # t.add_scalar(f"charts/one_step_reward_expl_var",
            #              explained_variance(one_step_rewards, rewards[:, -1]),
            #              freq=1)
            t.add_scalar(f"charts/previous_reward_expl_var",
                         explained_variance(prev_reward, rewards[:, -1]),
                         freq=1)
            t.add_scalar(
                f"charts/q_expl_var",
                explained_variance(q_pred, q_target),
                freq=1)
            t.add_histogram("charts/reward_target", reward, freq=1)
            t.add_histogram("charts/reward_pred", reward_pred, freq=1)
            t.add_histogram("charts/q_pred", q_pred, freq=1)
            t.add_histogram("charts/q_target", q_target, freq=1)

    def train_using_model(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        input_action = batch["actions"][:, 0]
        initial_image = images[:, 0]
        rewards = batch["rewards"]
        actions = batch["actions"]

        with torch.no_grad():
            hidden = self.network.representation(initial_image)
        # TODO: re-detach this
        # hidden = self.network.representation(initial_image)

        losses = []
        for i in range(1):
            # Hiddens at t-1
            old_hiddens = hidden

            # TODO: optimization for storing policy from last step
            values_pred_initial = self.network.policy_net(old_hiddens)
            assert values_pred_initial.shape == (self.args.batch_size,
                                                 self.args.num_actions)

            with torch.no_grad():
                # Take an epsilon-greedy action
                best_values, best_actions = values_pred_initial.max(dim=1)
                eps = .2
                should_sample = torch.rand_like(best_values) < eps
                samples = torch.randint_like(best_actions,
                                             self.args.num_actions)
                best_actions[should_sample] = samples[should_sample]
                action = best_actions

                # action = torch.randint_like(input_action, self.args.num_actions)
                assert action.shape == (self.args.batch_size, )
                hidden, reward = self.network.dynamics(hidden, action)
                # TODO: remove me!
                action = actions[:, i]
                reward = rewards[:, i]

            q_pred = values_pred_initial.gather(1, action.unsqueeze(1))
            q_pred = q_pred.reshape((self.args.batch_size, ))

            with torch.no_grad():
                if not self.args.double_q:
                    # Note: target network is used here
                    values_pred_final = self.target_network.policy_net(hidden)
                    assert values_pred_final.shape == (self.args.batch_size,
                                                       self.args.num_actions)
                    best_next_q = torch.max(values_pred_final, dim=1).values
                else:
                    # Get action with training network
                    values_pred_final = self.network.policy_net(hidden)
                    best_action = torch.argmax(values_pred_final, dim=1)
                    # Get value of that action with target network
                    values_pred_final = self.target_network.policy_net(hidden)
                    best_next_q = values_pred_final.gather(
                        1, best_action.unsqueeze(1))
                    best_next_q = best_next_q.reshape((self.args.batch_size, ))

                assert best_next_q.shape == (self.args.batch_size, )
                # The updated estimate of the value of the selected action
                q_target = reward + self.args.discount * best_next_q
                assert q_target.shape == (self.args.batch_size, )

            assert q_pred.shape == q_target.shape
            l = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach())
            losses.append(l)

        loss = sum(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), .05)
        self.optimizer.step()

        freq = 200
        if logging and t.i % freq == 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_scalar(f"charts/q_expl_var",
                         explained_variance(q_pred, q_target),
                         freq=1)

            # This is the advantage function: q(s, a) - v(s)
            max_val = values_pred_initial.max()
            t.add_scalar(f"charts/advantage_expl_var",
                         explained_variance(q_pred - max_val,
                                            q_target - max_val),
                         freq=1)

            t.add_histogram("charts/rewards", reward, freq=1)
            t.add_histogram("charts/q_pred", q_pred, freq=1)
            t.add_histogram("charts/q_target", q_target, freq=1)

    def train_just_dqn(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        initial_image = images[:, 0]
        next_image = images[:, 1]
        rewards = batch["rewards"]
        actions = batch["actions"]
        reward = rewards[:, 0]
        action = actions[:, 0]

        # TODO: re-detach this
        # hidden = self.network.representation(initial_image)
        with torch.no_grad():
            hidden = self.network.representation(initial_image)

        values_pred_initial = self.network.policy_net(hidden)
        assert values_pred_initial.shape == (self.args.batch_size,
                                             self.args.num_actions)

        q_pred = values_pred_initial.gather(1, action.unsqueeze(1))
        q_pred = q_pred.reshape((self.args.batch_size, ))

        with torch.no_grad():
            # Note: target network is used here for repr and policy
            hidden = self.target_network.representation(next_image)
            values_pred_final = self.target_network.policy_net(hidden)
            assert values_pred_final.shape == (self.args.batch_size,
                                               self.args.num_actions)
            best_next_q = torch.max(values_pred_final, dim=1).values

            assert best_next_q.shape == (self.args.batch_size, )
            # The updated estimate of the value of the selected action
            q_target = reward + self.args.discount * best_next_q
            assert q_target.shape == (self.args.batch_size, )

        assert q_pred.shape == q_target.shape
        loss = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), .05)
        self.optimizer.step()

        freq = 200
        if logging and t.i % freq == 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_scalar(f"charts/q_expl_var",
                         explained_variance(q_pred, q_target),
                         freq=1)

            # This is the advantage function: q(s, a) - v(s)
            max_val = values_pred_initial.max()
            t.add_scalar(f"charts/advantage_expl_var",
                         explained_variance(q_pred - max_val,
                                            q_target - max_val),
                         freq=1)

            t.add_histogram("charts/rewards", reward, freq=1)
            t.add_histogram("charts/q_pred", q_pred, freq=1)
            t.add_histogram("charts/q_target", q_target, freq=1)

    # This one was the first model-based dqn to work
    def train_just_dqn_2(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        initial_image = images[:, 0]
        next_image = images[:, 1]
        rewards = batch["rewards"]
        actions = batch["actions"]
        reward = rewards[:, 0]
        input_action = actions[:, 0]

        use_hiddens = np.random.rand() > .1

        # TODO: re-detach this
        # hidden = self.network.representation(initial_image)
        with torch.no_grad():
            hidden = self.network.representation(initial_image)
            # Step forward one step based on the real action
            if use_hiddens:
                hidden, _ = self.network.dynamics(hidden, input_action)

        values_pred_initial = self.network.policy_net(hidden)
        assert values_pred_initial.shape == (self.args.batch_size,
                                             self.args.num_actions)

        # Get the action we want to predict value for.
        # here, we sample randomly
        # action = torch.randint_like(input_action, self.args.num_actions)

        if use_hiddens:
            # Take an epsilon-greedy action
            best_values, best_actions = values_pred_initial.max(dim=1)
            eps = .2
            should_sample = torch.rand_like(best_values) < eps
            samples = torch.randint_like(best_actions,
                                         self.args.num_actions)
            best_actions[should_sample] = samples[should_sample]
            action = best_actions
        else:
            # TODO: we could actually sample an action here too.
            # But i'm not too worried about this leakage, it's not interesting
            action = input_action
        assert action.shape == (self.args.batch_size, )

        q_pred = values_pred_initial.gather(1, action.unsqueeze(1))
        q_pred = q_pred.reshape((self.args.batch_size, ))

        with torch.no_grad():
            # Note: target network is used here for repr and policy

            if use_hiddens:
                # The idea here is that we have s_t and s_t+1, and we are
                # backing up the value at a hypothetical s_t+2 based on
                # the sampled action. We use two initial states so that
                # both the prediction and target can be from a 1-step dynamics
                # hidden.
                hidden = self.target_network.representation(next_image)
                # Step forward one step using the sampled action
                hidden, reward = self.target_network.dynamics(hidden, action)
            else:
                # This works, but allows real rewards to leak in.
                hidden = self.target_network.representation(next_image)
                reward = rewards[:, 0]

                # We want to never allow the real rewards to be used
                # in order to verify we are learning just from the model.
                # So, we'll allow 0-step preds to be trained with 1-step
                # targets.
                # TODO: potentially revert back once we're done evaluating;
                # allowing real rewards may help maximize performance.
                # hidden, reward = self.target_network.dynamics(hidden, action)

            values_pred_final = self.target_network.policy_net(hidden)
            assert values_pred_final.shape == (self.args.batch_size,
                                               self.args.num_actions)
            best_next_q = torch.max(values_pred_final, dim=1).values

            assert best_next_q.shape == (self.args.batch_size, )
            # The updated estimate of the value of the selected action
            q_target = reward + self.args.discount * best_next_q
            assert q_target.shape == (self.args.batch_size, )

        assert q_pred.shape == q_target.shape
        per_sample_loss = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach(), reduction="none")
        assert per_sample_loss.shape == (self.args.batch_size, )
        loss = per_sample_loss.mean()

        age = batch["age"].cpu().tolist()
        self.overfitting_samples += list(zip(age, per_sample_loss.tolist()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), .05)
        self.optimizer.step()

        freq = 200
        if logging and t.i % freq == 0 and t.i != 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_scalar(f"charts/q_expl_var",
                         explained_variance(q_pred, q_target),
                         freq=1)

            # This is the advantage function: q(s, a) - v(s)
            # This isn't quite right, ex if the target is greater than
            # v(s), target_adv will be positive which should be impossible
            # it's impossible to get a measure of the true adv function
            # target without evaluating all actions
            # max_val = values_pred_initial.max(dim=1).values
            # mean will help stabilize this somewhat
            max_val = values_pred_initial.mean(dim=1)
            t.add_scalar(f"charts/advantage_expl_var",
                         explained_variance(q_pred - max_val,
                                            q_target - max_val),
                         freq=1)

            t.add_scalar(f"charts/reward_expl_var",
                         explained_variance(q_pred - best_next_q * self.args.discount,
                                            reward),
                         freq=1)

            t.add_scalar("charts/q_pred_mean", values_pred_initial.mean())
            t.add_scalar("charts/q_pred_std", values_pred_initial.std(dim=1).mean())

            t.add_histogram("charts/rewards", reward, freq=1)
            t.add_histogram("charts/implied_rewards", (q_pred - best_next_q * self.args.discount).clamp(-.05, .05), freq=1)
            t.add_histogram("charts/q_pred", q_pred, freq=1)
            t.add_histogram("charts/q_target", q_target, freq=1)


    def train_just_dqn_3(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        initial_image = images[:, 0]
        next_image = images[:, 1]
        # rewards = batch["rewards"]
        # actions = batch["actions"]
        # reward = rewards[:, 0]
        # input_action = actions[:, 0]

        with torch.no_grad():
            next_hidden = self.network.representation(initial_image)

        losses = []
        for i in range(3):
            hidden = next_hidden

            values_pred_initial = self.network.policy_net(hidden.detach())
            assert values_pred_initial.shape == (self.args.batch_size,
                                                 self.args.num_actions)

            # Take an epsilon-greedy action
            best_values, best_actions = values_pred_initial.max(dim=1)
            eps = .4
            should_sample = torch.rand_like(best_values) < eps
            samples = torch.randint_like(best_actions,
                                         self.args.num_actions)
            best_actions[should_sample] = samples[should_sample]
            action = best_actions
            assert action.shape == (self.args.batch_size, )

            q_pred = values_pred_initial.gather(1, action.unsqueeze(1))
            q_pred = q_pred.reshape((self.args.batch_size, ))

            with torch.no_grad():
                next_hidden, reward = self.network.dynamics(hidden, action)

                values_pred_final = self.target_network.policy_net(next_hidden)
                assert values_pred_final.shape == (self.args.batch_size,
                                                   self.args.num_actions)
                best_next_q = torch.max(values_pred_final, dim=1).values

                assert best_next_q.shape == (self.args.batch_size, )
                # The updated estimate of the value of the selected action
                q_target = reward + self.args.discount * best_next_q
                assert q_target.shape == (self.args.batch_size, )

            assert q_pred.shape == q_target.shape
            per_sample_loss = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach(), reduction="none")
            assert per_sample_loss.shape == (self.args.batch_size, )
            losses.append(per_sample_loss)

        per_sample_loss = torch.stack(losses).mean(dim=0)
        assert per_sample_loss.shape == (self.args.batch_size, )
        loss = per_sample_loss.mean()

        age = batch["age"].cpu().tolist()
        self.overfitting_samples += list(zip(age, per_sample_loss.tolist()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), .05)
        self.optimizer.step()

        freq = 200
        if logging and t.i % freq == 0 and t.i != 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_scalar(f"charts/q_expl_var",
                         explained_variance(q_pred, q_target),
                         freq=1)

            # This is the advantage function: q(s, a) - v(s)
            # This isn't quite right, ex if the target is greater than
            # v(s), target_adv will be positive which should be impossible
            # it's impossible to get a measure of the true adv function
            # target without evaluating all actions
            # max_val = values_pred_initial.max(dim=1).values
            # mean will help stabilize this somewhat
            max_val = values_pred_initial.mean(dim=1)
            t.add_scalar(f"charts/advantage_expl_var",
                         explained_variance(q_pred - max_val,
                                            q_target - max_val),
                         freq=1)

            t.add_scalar(f"charts/reward_expl_var",
                         explained_variance(q_pred - best_next_q * self.args.discount,
                                            reward),
                         freq=1)

            t.add_scalar("charts/q_pred_mean", values_pred_initial.mean())
            t.add_scalar("charts/q_pred_std", values_pred_initial.std(dim=1).mean())

            t.add_histogram("charts/rewards", reward, freq=1)
            t.add_histogram("charts/implied_rewards", (q_pred - best_next_q * self.args.discount), freq=1)
            t.add_histogram("charts/q_pred", q_pred, freq=1)
            t.add_histogram("charts/q_target", q_target, freq=1)


    def train_just_dqn_4(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        initial_image = images[:, 0]
        next_image = images[:, 1]
        rewards = batch["rewards"]
        actions = batch["actions"]
        input_reward = rewards[:, 0]
        input_action = actions[:, 0]

        with torch.no_grad():
            hidden_pred = self.network.representation(initial_image)
            hidden_target = self.network.representation(next_image)

        losses = []
        for i in range(3):
            # hidden = next_hidden

            values_pred_initial = self.network.policy_net(hidden_pred.detach())
            assert values_pred_initial.shape == (self.args.batch_size,
                                                 self.args.num_actions)

            if i == 0:
                action = input_action
            else:
                # Take an epsilon-greedy action
                best_values, best_actions = values_pred_initial.max(dim=1)
                eps = .4
                should_sample = torch.rand_like(best_values) < eps
                samples = torch.randint_like(best_actions,
                                             self.args.num_actions)
                best_actions[should_sample] = samples[should_sample]
                action = best_actions
                assert action.shape == (self.args.batch_size, )

            q_pred = values_pred_initial.gather(1, action.unsqueeze(1))
            q_pred = q_pred.reshape((self.args.batch_size, ))

            with torch.no_grad():
                # next_hidden, reward = self.network.dynamics(hidden, action)
                if i == 0:
                    reward = input_reward
                else:
                    # TODO:
                    pass

                values_pred_final = self.target_network.policy_net(hidden_target)
                assert values_pred_final.shape == (self.args.batch_size,
                                                   self.args.num_actions)
                best_next_q = torch.max(values_pred_final, dim=1).values

                assert best_next_q.shape == (self.args.batch_size, )
                # The updated estimate of the value of the selected action
                q_target = reward + self.args.discount * best_next_q
                assert q_target.shape == (self.args.batch_size, )

            assert q_pred.shape == q_target.shape
            per_sample_loss = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach(), reduction="none")
            assert per_sample_loss.shape == (self.args.batch_size, )
            losses.append(per_sample_loss)

            with torch.no_grad():
                hidden_pred, reward = self.network.dynamics(hidden_pred, action)

        per_sample_loss = torch.stack(losses).mean(dim=0)
        assert per_sample_loss.shape == (self.args.batch_size, )
        loss = per_sample_loss.mean()

        age = batch["age"].cpu().tolist()
        self.overfitting_samples += list(zip(age, per_sample_loss.tolist()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), .05)
        self.optimizer.step()

        freq = 200
        if logging and t.i % freq == 0 and t.i != 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_scalar(f"charts/q_expl_var",
                         explained_variance(q_pred, q_target),
                         freq=1)

            # This is the advantage function: q(s, a) - v(s)
            # This isn't quite right, ex if the target is greater than
            # v(s), target_adv will be positive which should be impossible
            # it's impossible to get a measure of the true adv function
            # target without evaluating all actions
            # max_val = values_pred_initial.max(dim=1).values
            # mean will help stabilize this somewhat
            max_val = values_pred_initial.mean(dim=1)
            t.add_scalar(f"charts/advantage_expl_var",
                         explained_variance(q_pred - max_val,
                                            q_target - max_val),
                         freq=1)

            t.add_scalar(f"charts/reward_expl_var",
                         explained_variance(q_pred - best_next_q * self.args.discount,
                                            reward),
                         freq=1)

            t.add_scalar("charts/q_pred_mean", values_pred_initial.mean())
            t.add_scalar("charts/q_pred_std", values_pred_initial.std(dim=1).mean())

            t.add_histogram("charts/rewards", reward, freq=1)
            t.add_histogram("charts/implied_rewards", (q_pred - best_next_q * self.args.discount), freq=1)
            t.add_histogram("charts/q_pred", q_pred, freq=1)
            t.add_histogram("charts/q_target", q_target, freq=1)

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
