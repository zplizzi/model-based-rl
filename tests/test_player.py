import time
import torch
from torch import nn
from papers.muzero import player
from papers.muzero.model_storage import ModelStorage
from papers.muzero.replay_buffer import ReplayBuffer
from papers.muzero.network import Network, DQNNetwork
from papers.muzero.parse_args import parse_args
from papers.muzero.tests.test_game import MockEnv, MockNetwork
from papers.muzero.player_statistics import PlayerStatistics

import pyximport
pyximport.install()
from papers.muzero.player_cy import Node, MinMaxStats, select_child, \
    expand_node, backpropagate, add_exploration_noise

import numpy as np
"""
Things I would like tested:
- test the env by itself (all other tests use mocked env)
- network doesn't really need too much testing (just use mocked)

- test action selection (one step simulation, just checking policy is used)
- thoroughly test the format pushed to buffer is correct
- test games finish/terminate properly
- somehow test core mcts + UCB

"""

def test_random_player():
    args = parse_args([])
    args.num_vec_env = 4
    replay_buffer = ReplayBuffer(args)
    p = player.RandomPlayer(args, None, replay_buffer)
    for i in range(100):
        p.step()

    assert len(replay_buffer.buffer) > 0




def test_mocked():
    """Test that the mock doesn't explode."""
    args = parse_args([])
    args.device = torch.device("cpu")
    args.num_vec_env = 4
    model_storage = ModelStorage()
    replay_buffer = ReplayBuffer(args)
    network = MockNetwork(args).to(args.device)
    player.Environment = MockEnv
    p = player.GamePlayer(args, model_storage, replay_buffer)
    p.network = network
    for i in range(10):
        p.step()


# The following is a mockup that "hides" a large reward behind a certain
# series of actions, so that we can test that the reward is discovered.


def encode_history(actions):
    return torch.tensor(actions)


def decode_history(x):
    return [int(a) for a in x.tolist()]


class MockNetworkMCTS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prediction_eval(self, hidden):
        policy_logits = torch.ones(
            (1, self.args.num_actions)) / self.args.num_actions

        value = 10

        # Hide a large value behind this particular action sequence
        if decode_history(hidden[0]) == [3, 2]:
            # value, reward, policy
            return torch.tensor([value]), torch.tensor([100]), policy_logits
        else:
            return torch.tensor([value]), torch.tensor([.1]), policy_logits

    def representation_eval(self, image: torch.Tensor) -> torch.Tensor:
        return encode_history([]).unsqueeze(0)

    def dynamics_eval(self, hidden: torch.Tensor, action: torch.Tensor):
        action = action[0]
        history = decode_history(hidden[0])
        history.append(action)
        return encode_history(history).unsqueeze(0)

    def training_steps(self) -> int:
        return 0


class MockEnvMCTS(MockEnv):
    def __init__(self, args):
        super().__init__(args)

    def step(self, action):
        obs = torch.rand((3, 96, 96))
        # reward = random.randint(-100, 100)
        reward = 1
        done = False
        return obs, reward, done

    def reset(self):
        return torch.rand((3, 96, 96))


def test_hidden_reward():
    args = parse_args([])
    args.num_simulations = 50
    args.discount = .9
    args.device = torch.device("cpu")
    # TODO: this mock shouldn't work for more than one env
    args.num_vec_env = 4
    model_storage = ModelStorage()
    replay_buffer = ReplayBuffer(args)
    stats = PlayerStatistics()
    network = MockNetworkMCTS(args).to(args.device)
    player.Environment = MockEnvMCTS
    p = player.MCTSPlayer(args, model_storage, replay_buffer, stats)
    p.network = network
    for i in range(100):
        p.step()

    actions = [x[1][0][0] for x in replay_buffer.buffer]
    # Confirm that the hidden action was chosen most often
    # assert len([x for x in actions if x == 3]) > 30

    rewards = [x[1][0][1] for x in replay_buffer.buffer]
    values = [x[1][0][2] for x in replay_buffer.buffer]
    print(actions)
    print(values)
    print(rewards)
    # for value in values:
    #     target = (100 * .9) * (1/6) **2 + .1
    #     # Value should be roughly close to the target
    #     assert value > target - 2
    #     assert value < target + 2
    # policies = [x[1][0][3] for x in replay_buffer.buffer]

    assert False


class MockNetworkQSA(MockNetworkMCTS):
    def prediction_eval(self, hidden):
        policy_logits = torch.ones(
            (1, self.args.num_actions)) / self.args.num_actions

        assert len(decode_history(hidden[0])) in (0, 1)

        # Hide a large value behind this particular action
        if decode_history(hidden[0]) == [3]:
            # value, reward, policy
            return torch.tensor([.8]), torch.tensor([0]), policy_logits
        else:
            return torch.tensor([.5]), torch.tensor([0]), policy_logits


def test_qsa_player():
    args = parse_args([])
    args.num_vec_env = 1
    args.device = torch.device("cpu")
    network = MockNetworkQSA(args)
    replay_buffer = ReplayBuffer(args)
    p = player.SAPlayer(args, None, replay_buffer)
    p.network = network
    for i in range(100):
        p.step()

    assert len(replay_buffer.buffer) > 0

    game = p.games[0]
    count = (np.array(game.actions) == 3).sum()
    # most of the actions should be "3"
    assert count > 100 / 2

def test_dqn_player():
    args = parse_args([])
    args.num_vec_env = 1
    # args.device = torch.device("cpu")
    network = DQNNetwork(args).to(args.device)
    stats = PlayerStatistics()
    replay_buffer = ReplayBuffer(args)
    p = player.SAPlayer(args, None, replay_buffer, stats)
    p.network = network
    for i in range(100):
        p.step()

    assert len(replay_buffer.buffer) > 0

# Tests for the functions in player_cy


def test_minmax():
    minmax = MinMaxStats()
    minmax.update(-5)
    minmax.update(10)
    assert np.isclose(minmax.normalize(7), (7 + 5) / 15)


def test_select_child_values():
    """All else equal, a node with higher value_sum should be chosen."""
    minmax = MinMaxStats()
    minmax.update(-10)
    minmax.update(10)

    # repeat test 10 times
    for i in range(10):
        root = Node(0)
        root.visit_count = 10 * 10
        values = np.random.randint(0, 10000, size=10)
        children = []
        # Add children to root
        for i in range(10):
            child = Node(prior=1 / 10)
            child.visit_count = 10
            child.value_sum = values[i]
            children.append(child)
            root.add_child(child)

        action, child = select_child(root, minmax)
        assert child == children[action]
        assert action == np.argmax(values)


def test_select_child_priors():
    """All else equal, a node with higher prior should be chosen."""
    minmax = MinMaxStats()
    minmax.update(-10)
    minmax.update(10)

    # repeat test 10 times
    for i in range(10):
        root = Node(0)
        root.visit_count = 10 * 10
        priors = np.random.randint(0, 10000, size=10)
        priors = priors / priors.sum()
        children = []
        # Add children to root
        for i in range(10):
            child = Node(prior=priors[i])
            child.visit_count = 10
            child.value_sum = 10
            children.append(child)
            root.add_child(child)

        action, child = select_child(root, minmax)
        assert child == children[action]
        assert action == np.argmax(priors)


def test_select_child_counts():
    """All else equal, a node with lower visit_count should be chosen
    (because this one has more uncertainty?)"""
    minmax = MinMaxStats()
    minmax.update(-10)
    minmax.update(10)

    # repeat test 10 times
    for i in range(10):
        root = Node(0)
        root.visit_count = 10 * 10
        counts = np.random.randint(0, 10000, size=10)
        root.visit_count = counts.sum()
        children = []
        # Add children to root
        for i in range(10):
            child = Node(prior=1 / 10)
            child.visit_count = counts[i]
            child.value_sum = 10
            children.append(child)
            root.add_child(child)

        action, child = select_child(root, minmax)
        assert child == children[action]
        assert action == np.argmax(-1 * counts)


# def test_select_action_counts():
#     """Confirm that an action with significantly higher visit count is chosen.

#     TODO: make this not slightly flaky.
#     """
#     args = parse_args([])
#     # repeat test 10 times
#     for i in range(10):
#         root = Node(0)
#         # Add children to root
#         for i in range(10):
#             child = Node(prior=1 / 10)
#             if i == 4:
#                 child.visit_count = 10000
#             else:
#                 child.visit_count = 1
#             root.add_child(child)

#         action = select_action(args, root, 0)
#         assert action == 4


def test_backpropagate():
    n1 = Node(.1)
    n1.reward = 0
    n2 = Node(.1)
    n2.reward = 0
    n3 = Node(.1)
    n3.reward = 10
    search_path = [n1, n2, n3]
    value = 0
    discount = .9
    minmax = MinMaxStats()
    minmax.update(-10)
    minmax.update(10)
    backpropagate(search_path, value, discount, minmax)
    assert n1.value_sum == 10 * .9**1
