import numpy as np
from typing import Dict, List, Optional
import torch
import copy

from papers.muzero.network import Network, UniformNetwork, NetworkUniformPolicy, \
    DQNNetwork

from papers.muzero.types import Action, Image, Value, Reward, NormalizedPolicy
from papers.muzero.player import Player

import pyximport
pyximport.install()
from papers.muzero.player_cy import Node, MinMaxStats, select_child, \
    expand_node, backpropagate, add_exploration_noise, epsilon_greedy

class MCTS:
    def __init__(self, root: "Node", actions: List[Action]):
        self.root = root
        self.min_max_stats = MinMaxStats()
        self.actions = actions

class MCTSPlayer(Player):
    def __init__(self, args, model_storage, replay_buffer, stats):
        super().__init__(args, model_storage, replay_buffer, stats)

    def run(self):
        self.pull_latest_model()
        i = 0
        while True:
            if i % 256 == 0:
                frames = self.games[0].get_video_frames()
                self.stats.log_video_frames.remote(frames)
            if i % 5 == 0:
                self.pull_latest_model()
            self.step()
            i += 1

    def compute(self):
        # Do representation step
        images = []
        for game in self.games:
            root = Node(0)
            image = game.make_image(len(game.obs) - 1)
            images.append(image)

        images = torch.stack(images)
        images = images.to(self.args.device)
        hiddens = self.network.representation_eval(images)
        predictions = self.network.prediction_eval(hiddens)
        predictions = zip(*predictions)
        roots = []
        for hidden, prediction, game in zip(hiddens, predictions, self.games):
            root = Node(0)
            _, policy_logits = prediction

            # For now, just use a uniform policy
            # exp() of any uniform vector is safe
            policy_logits = torch.zeros(self.args.num_actions)

            # The reward for transitioning to the parent doesn't matter
            reward = 0
            expand_node(root, reward, policy_logits, hidden,
                        self.args.num_actions)
            # add_exploration_noise(self.args, root)
            roots.append(MCTS(root, game.actions))

        for _ in range(self.args.num_simulations):
            model_inputs = []
            for mcts in roots:
                history = copy.copy(mcts.actions)
                node = mcts.root
                search_path = [node]

                # print("prior, value, visit_count")
                # print([c.prior for c in node.children])
                # print([c.value() for c in node.children])
                # print([c.visit_count for c in node.children])

                while node.expanded():
                    action, node = select_child(node, mcts.min_max_stats)
                    history.append(action)
                    search_path.append(node)

                # Inside the search tree we use the dynamics function to obtain the
                # next hidden state given an action and the previous hidden state.
                parent = search_path[-2]
                model_inputs.append((parent.hidden_state, history[-1]))
                mcts.current_node = node
                mcts.current_search_path = search_path

            hiddens, actions = zip(*model_inputs)
            hiddens = torch.stack(hiddens)
            actions = torch.tensor(actions).to(self.args.device)
            hiddens, rewards = self.network.dynamics_eval(hiddens, actions)
            predictions = self.network.prediction_eval(hiddens)
            predictions = zip(*predictions)
            for hidden, reward, prediction, mcts in zip(hiddens, rewards, predictions, roots):
                value, policy_logits = prediction

                # For now, just use a uniform policy
                # exp() of any uniform vector is safe
                policy_logits = torch.zeros(self.args.num_actions)

                # TODO: it seems like a problem if we select actions in a manner
                # differently in MCTS than the actual game. If we do this, how
                # are the MCTS values going to represent the true values?
                expand_node(mcts.current_node, reward, policy_logits, hidden,
                            self.args.num_actions)

                backpropagate(mcts.current_search_path, value,
                              self.args.discount, mcts.min_max_stats)

        # We need the root node to use later
        root_nodes = [mcts.root for mcts in roots]
        # For debugging
        self.root_nodes = root_nodes

        def eps_decay(i):
            if i < 2000:
                eps = 1
            elif i < 4000:
                di = i - 2000
                eps = 1 - min(di / 4000, .8)
            else:
                di = i - 4000
                eps = .2 - .2 * min(di / 10000, .8)
            return eps

        epsilon = eps_decay(self.network.i)
        if np.random.rand() < .01:
            print(f"eps is {epsilon}")

        actions = []
        root_values = []
        policies = []
        # Select an action and step the environment
        for game, root in zip(self.games, root_nodes):
            # TODO: this still isn't quite right.. we want to select the action
            # that has the best expected value under our action selection
            # policy, not under the MCTS UCB policy. Something like "max
            # value discovered under this node" might be closer?
            values = [child.value() for child in root.children]
            values = torch.tensor(values)
            policy = values.exp() / values.exp().sum()
            # action = np.random.choice(range(len(policy)), p=policy)
            action = epsilon_greedy(policy, epsilon=epsilon)

            # value = root.value()
            # For now let's not boostrap
            value = 0

            actions.append(action)
            root_values.append(value)
            policies.append(policy)

        return actions, root_values, policies



def compute_value_loss(values, targets):
    # return CrossEntropyValueHead.loss(values, targets).mean()
    return torch.nn.functional.mse_loss(values, targets)


def compute_reward_loss(rewards, targets):
    assert rewards.shape == targets.shape
    # return CrossEntropyValueHead.loss(rewards, targets).mean()
    return torch.nn.functional.mse_loss(rewards, targets)


def compute_policy_loss(policy_logits, target_p):
    # print(policy_logits)
    policy_logits = nn.functional.log_softmax(policy_logits, dim=1)
    policy_loss = -1 * (target_p * policy_logits).sum(-1)
    return policy_loss.mean()

# TODO: add back their weird gradient scaling for losses and hidden
"""
Gradient scaling:
- scaling the losses such that the loss of the 0th step is of the same magnitude
as the other steps combined:
    - potentially important for ensuring that the representation network
    fully encodes the state into the hidden - otherwise, some of the encoding
    may happen in the "dynamics" model.
    - for example, if the representation network doesn't have enough capacity,
    it could overflow into the hidden the job of doing computations required
    for predicting the output. this violates the desire for the dynamics net
    to just be doing dynamics.
    - this is especially likely when the targets are relatively stationary -
    meaning there's less "dynamics" to simulate.

"""
def train_step(optimizer,
               network,
               batch: Batch,
               args,
               logging=False,
               eval=False):
    """Execute a single training step."""
    # if t.i // 50 % 2:
    #     print("eval mode!")
    #     eval = True
    if eval:
        network.eval()
    else:
        network.train()
        optimizer.zero_grad()

    print("doing train step")
    batch = [x.cuda() for x in batch]
    images, actions, rewards, values, policies, weights = batch
    batch_size = images.shape[0]
    # assert images.shape == (batch_size, args.n_framestack * 4, 96, 96)
    assert actions.shape == (batch_size, args.num_unroll_steps)

    # Initial step, from the real observation.
    hidden = network.representation(images)

    loss = []

    # Recurrent steps, from action and previous hidden state.
    for i in range(args.num_unroll_steps):
        value_pred, reward_pred, policy_pred = network.prediction(hidden)
        # For a given index, state[i] is used to predict value[i] and
        # reward_pred[i] and policy[i]. We then selected an action action[i]
        # and received true reward reward[i].

        value_loss = compute_value_loss(value_pred, values[:, i])
        reward_loss = compute_reward_loss(reward_pred, rewards[:, i])
        policy_loss = compute_policy_loss(policy_pred, policies[:, i])

        if i == 0:
            scale = 1
        else:
            scale = 1 / args.num_unroll_steps
        # loss += [value_loss*scale, reward_loss*scale, policy_loss*scale]
        loss += [value_loss]

        # print(reward_pred)
        # print(rewards[:, i])

        # Get the action taken after this state
        action = actions[:, i]
        # ..and step the dynamics model
        hidden = network.dynamics(hidden, action)

        # Log on the first and last unroll step
        if logging and (i == 0
                        or i == (args.num_unroll_steps - 1)) and not eval:
            t.add_scalar(f"loss/value{i}", value_loss, freq=5)
            t.add_scalar(f"loss/reward{i}", reward_loss, freq=5)
            t.add_scalar(f"loss/policy{i}", policy_loss, freq=5)

            # print(f"value pred {i}")
            # print(value_pred)
            # print(f"value target {i}")
            # print(values[:, i])
            t.add_scalar(f"training/value_pred{i}", value_pred, freq=5)
            t.add_histogram(f"training/value_target{i}", values[:, i], freq=10)
            t.add_scalar(
                f"training/value_target{i} mean", values[:, i].mean(), freq=5)
            t.add_scalar(
                f"training/value_expl_var{i}",
                explained_variance(value_pred, values[:, i]),
                freq=5)

            t.add_scalar(f"training/reward_pred{i}", reward_pred, freq=5)
            t.add_scalar(f"training/reward_target{i}", rewards[:, i], freq=5)
            t.add_scalar(
                f"training/reward_expl_var{i}",
                explained_variance(reward_pred, rewards[:, i]),
                freq=5)

            # The target is normalized, but the predicted isn't.
            policy_p = torch.nn.functional.softmax(policy_pred[0], dim=0)
            t.add_bar_chart(f"training/policy_pred{i}", policy_p, freq=10)
            t.add_bar_chart(
                f"training/policy_target{i}", policies[0, i], freq=10)

        if logging and (i == 0 or i == (args.num_unroll_steps - 1)) and eval:
            t.add_scalar(
                f"eval/value_expl_var{i}",
                explained_variance(value_pred, values[:, i]),
                freq=1)
            t.add_scalar(
                f"eval/reward_expl_var{i}",
                explained_variance(reward_pred, rewards[:, i]),
                freq=1)
        # print(policies)
        # print(rewards)
        # print(values)

    if not eval:
        loss = sum(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), .5)
        optimizer.step()

    if logging and not eval:
        t.add_scalar("loss/total", loss, freq=20)
        t.add_image_grid("input", images[0], freq=20, range=(0, 1))
        t.add_bar_chart("rewards", rewards[0], freq=20)
        t.add_bar_chart("values", values[0], freq=20)
        # t.add_histogram("training/input_hist", images[0], freq=10)

def train_step_simple(optimizer,
                      network,
                      batch: Batch,
                      args,
                      logging=False,
                      eval=False):
    if eval:
        network.eval()
    else:
        network.train()
        optimizer.zero_grad()

    # print("doing train step")
    batch = [x.cuda() for x in batch]
    images, actions, rewards, values, policies = batch
    batch_size = images.shape[0]
    assert actions.shape == (batch_size, args.num_unroll_steps)

    # Initial step, from the real observation.
    hidden = network.representation(images)
    value_pred, reward_pred, policy_pred = network.prediction(hidden)
    value_loss = compute_value_loss(value_pred, values[:, 0])
    loss = value_loss.mean()

    if logging and not eval:
        t.add_scalar(f"loss/value", value_loss, freq=30)

        t.add_scalar(f"training/value_pred", value_pred, freq=30)
        t.add_histogram(f"training/value_target", values[:, 0], freq=30)
        t.add_scalar(
            f"training/value_target mean", values[:, 0].mean(), freq=30)
        t.add_scalar(
            f"training/value_expl_var",
            explained_variance(value_pred, values[:, 0]),
            freq=30)

    if logging and eval:
        t.add_scalar(
            f"eval/value_expl_var",
            explained_variance(value_pred, values[:, 0]),
            freq=1)
        t.add_scalar(
            f"eval/reward_expl_var",
            explained_variance(reward_pred, rewards[:, 0]),
            freq=1)

    if not eval:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), .5)
        optimizer.step()

    if logging and not eval:
        t.add_image_grid("input", images[0], freq=100, range=(0, 1))

def train_step_sa(optimizer, network, batch: Batch, args, logging=False):
    network.train()
    optimizer.zero_grad()

    batch = [x.cuda() for x in batch]
    images, actions, rewards, values, policies = batch
    batch_size = images.shape[0]
    # assert images.shape == (batch_size, args.n_framestack * 4, 96, 96)
    assert actions.shape == (batch_size, args.num_unroll_steps)

    if logging:
        t.add_bar_chart(f"training/policy_target{0}", policies[0, 0], freq=30)

    # Initial step, from the real observation.
    hidden = network.representation(images)

    loss = []

    # Recurrent steps, from action and previous hidden state.
    for i in range(args.num_unroll_steps):
        # value_pred, reward_pred, policy_pred = network.prediction(hidden)
        value_pred, policy_pred = network.prediction(hidden)

        # Get the action taken after this state
        action = actions[:, i]
        # ..and step the dynamics model
        hidden, reward_pred = network.dynamics(hidden, action)

        value_loss = compute_value_loss(value_pred, values[:, i])
        # TODO: this is not quite right; rewards should be predicted dependent
        # on the action selected. Need to fix the rollout worker also.
        reward_loss = compute_reward_loss(reward_pred, rewards[:, i])

        if i == 0:
            scale = 1
        else:
            scale = 1 / args.num_unroll_steps
        # loss += [value_loss*scale, reward_loss*scale, policy_loss*scale]
        loss += [value_loss * scale, reward_loss * scale]
        # loss += [value_loss]

        # Log on the first and last unroll step
        if logging and (i == 0 or i == (args.num_unroll_steps - 1)):
            t.add_scalar(f"loss/value{i}", value_loss, freq=20)
            t.add_scalar(f"loss/reward{i}", reward_loss, freq=20)

            t.add_scalar(f"training/value_pred{i}", value_pred, freq=20)
            t.add_histogram(f"training/value_target{i}", values[:, i], freq=20)
            t.add_scalar(
                f"training/value_target{i} mean", values[:, i].mean(), freq=20)
            t.add_scalar(
                f"training/value_expl_var{i}",
                explained_variance(value_pred, values[:, i]),
                freq=20)

            t.add_scalar(f"training/reward_pred{i}", reward_pred, freq=20)
            t.add_scalar(f"training/reward_target{i}", rewards[:, i], freq=20)
            t.add_scalar(
                f"training/reward_expl_var{i}",
                explained_variance(reward_pred, rewards[:, i]),
                freq=20)

        if i == 0:
            initial_value_pred = value_pred

        if logging and i > 0:
            # We want a measure of how much of the variance is explained
            # beyond just using the zero-step prediction.
            # Seems like there's a handful of different ways we could slice this tho:
            # predictions:
            # - pred[i] - target[0]: absolute residual
            # - pred[i] - pred[0]: residual relative to predicted base
            # targets:
            # - target[i] - target[0]: absolute target
            # - target[i] - pred[0]: actual residual
            """
            So comparing abs pred to abs target doesn't make sense.
            values[:, 0] is the value of the same rollout that values[:, i]
            came from - meaning the only difference in the two are rewards in
            between or decay. In other words, values[:, 0] is conditional on
            a certain set of future actions.

            On the other hand, initial_value_pred is the expected value of the
            initial state under the policy, but not conditional on any future
            actions. So they're fundamentally different quantities.

            On one hand this is obvious - we train the value function on many
            such conditional samples, and if the conditional samples follow
            the distribution of the policy then we eventually converge properly.
            But in this case it doesn't work because we're trying to figure out
            how much using the actions gets us additional info and specificity
            over just the initial prediction without knowledge of the future actions - 
            so we have to have an initial value that isn't conditional on future actions.

            So that leaves us with the relative ones. What is rel/rel?
            I think this is what we care about. I'm not 100% certain it's
            mathematically different than the normal EV. but i think so.
            because initial_value_pred is a random variable across the batch
            that can add or remove variance. here we expect it will remove a
            lot of variance - making it harder to explain the remaining.
            Indeeed, rel_target has lower variance than just the main
            value_target.

            """
            # abs_pred = value_pred - values[:, 0]
            rel_pred = value_pred - initial_value_pred

            # abs_target = values[:, i] - values[:, 0]
            rel_target = values[:, i] - initial_value_pred

            # t.add_histogram(f"residuals/abs_pred{i}", abs_pred, freq=100)
            t.add_histogram(f"residuals/rel_pred{i}", rel_pred, freq=100)
            # t.add_histogram(f"residuals/abs_target{i}", abs_target, freq=100)
            t.add_histogram(f"residuals/rel_target{i}", rel_target, freq=100)

            # t.add_scalar(f"residuals/ev_abs_rel{i}", explained_variance(abs_pred, rel_target), freq=30)
            # t.add_scalar(f"residuals/ev_abs_abs{i}", explained_variance(abs_pred, abs_target), freq=30)
            t.add_scalar(
                f"residuals/ev_rel_rel{i}",
                explained_variance(rel_pred, rel_target),
                freq=30)
            # t.add_scalar(f"residuals/ev_rel_abs{i}", explained_variance(rel_pred, abs_target), freq=30)

    loss = sum(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), .5)
    optimizer.step()

    if logging:
        t.add_scalar("loss/total", loss, freq=50)
        t.add_image_grid("input", images[0], freq=30, range=(0, 1))
        t.add_bar_chart("rewards", rewards[0], freq=50)
        t.add_bar_chart("values", values[0], freq=50)
