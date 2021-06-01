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
        self.fc2 = nn.Linear(512, args.num_actions)

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
        values = self.network.prediction(hiddens)
        assert values.shape == (len(self.games), self.args.num_actions)
        return values


class Trainer(BaseTrainer):
    # network_cls = DQNNetwork

    def _setup(self, config):
        self.args = config["args"]
        self.network = DQNNetwork(self.args)
        self.network = self.network.to(args.device)

        self.target_network = DQNNetwork(self.args)
        self.target_network = self.target_network.to(args.device)

        super()._setup(config, player_cls=DQNPlayer, args=self.args)

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
        weights = batch["weight"]
        keys = batch["key"]
        # images, actions, rewards, _, _, weights, keys = batch
        initial_image = images[:, 0]
        final_image = images[:, 1]

        # Compute discounted rewards. If td_steps=1, this reduces to just
        # discounted_rewards = rewards[0]
        discounted_rewards = torch.zeros_like(rewards[:, 0])
        for i in range(self.args.td_steps):
            discounted_rewards += rewards[:, i] * self.args.discount**i

        # Get the network prediction of the value given the initial state, action
        # pair
        initial_action = actions[:, 0]
        hidden = self.network.representation(initial_image)
        values_pred_initial = self.network.prediction(hidden)
        q_pred = values_pred_initial.gather(1, initial_action.unsqueeze(1))
        q_pred = q_pred.reshape((self.args.batch_size, ))

        # Now we want to build an improved target to improve the prediction with.
        # It's based on n-step TD with re-computed final Q(s,a) value
        if not self.args.double_q:
            # Note: this has been tested and works nicely
            # This is the loss used in original DQN, with a target network
            # for stability. The target network is used to select the best final
            # action and predict its value.
            with torch.no_grad():
                hidden = self.target_network.representation(final_image)
                values_pred_final = self.target_network.prediction(hidden)
                best_next_q = torch.max(values_pred_final, dim=1).values
        else:
            # TODO: test
            # This is the "Double DQN" loss. Here, the current network is used
            # to select the final action, but the target network is used to get
            # its value.
            with torch.no_grad():
                # Get action with training network
                hidden = self.network.representation(final_image)
                values_pred_final = self.network.prediction(hidden)
                best_action = torch.argmax(values_pred_final)
                # Get value of that action with target network
                hidden = self.target_network.representation(final_image)
                values_pred_final = self.target_network.prediction(hidden)
                best_next_q = values_pred_final.gather(
                    1, best_action.unsqueeze(1))
                best_next_q = best_next_q.reshape((self.args.batch_size, ))

        # The updated estimate of the value of the selected action
        q_target = discounted_rewards + (self.args.discount**
                                         self.args.td_steps) * best_next_q

        losses = torch.nn.functional.smooth_l1_loss(q_pred,
                                                    q_target.detach(),
                                                    reduction="none")
        # Apply importance sampling weights
        weighted_losses = losses * weights
        loss = weighted_losses.mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), .5)
        self.optimizer.step()

        freq = 300
        # if self.args.prioritized_replay:
        #     td_error = (q_pred - q_target.detach()).abs()
        #     # Don't allow loss greater than 1 to avoid an outlier throwing
        #     # everything off.
        #     # TODO: make a more general approach to this
        #     new_priorities = td_error.clamp(.01, 1).detach().cpu().numpy()
        #     keys = keys.cpu().numpy()
        #     try:
        #         self.replay_buffer.update_priorities.remote(keys, new_priorities)
        #     except AttributeError:
        #         self.replay_buffer.update_priorities(keys, new_priorities)
        #     if logging:
        #         old_priorities = batch["priority"].cpu().numpy()
        #         t.add_histogram(
        #             "replay/old_priorities", old_priorities, freq=freq)
        #         t.add_histogram(
        #             "replay/new_priorities", new_priorities, freq=freq)
        #         t.add_histogram(
        #             "replay/diff_priorities",
        #             new_priorities - old_priorities,
        #             freq=freq)
        #         t.add_histogram("replay/weights", weights, freq=freq)

        # if logging and t.i % freq == 0:
        #     freq = 1
        #     t.add_scalar("loss/total", loss, freq=freq)
        #     t.add_histogram("loss/losses", losses, freq=freq)
        #     t.add_histogram("loss/weighted_losses", weighted_losses, freq=freq)
        #     t.add_image_grid(
        #         "input", initial_image[0], freq=freq, range=(0, 1))
        #     t.add_image_grid(
        #         "input_final", final_image[0], freq=freq, range=(0, 1))
        #     t.add_histogram("discounted reward", discounted_rewards, freq=freq)
        #     t.add_histogram("initial action", initial_action, freq=freq)
        #     t.add_histogram(
        #         "values_pred_initial", values_pred_initial, freq=freq)
        #     t.add_histogram("values_pred_final", values_pred_final, freq=freq)
        #     t.add_histogram("q_target", q_target, freq=freq)
        #     t.add_scalar("q_target_mean", q_target.mean(), freq=freq)
        #     t.add_histogram("q_pred", q_pred, freq=freq)

        #     t.add_scalar(
        #         f"ev/pred_target",
        #         explained_variance(q_pred, q_target),
        #         freq=freq)
        #     t.add_scalar(
        #         f"ev/reward",
        #         explained_variance(
        #             q_pred - best_next_q * (self.args.discount**self.args.td_steps),
        #             discounted_rewards),
        #         freq=freq)


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
