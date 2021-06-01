from papers.muzero import player
from papers.muzero.parse_args import parse_args

import torch
import random
from torch import nn
import pytest

class MockALE:
    def step(self, action):
        return None, None, False, None

class MockEnv:
    def __init__(self, args):
        # self.obs_test = []
        # self.reward_test = []
        self.env = MockALE()

    def step(self, action):
        obs = torch.rand((3, 96, 96))
        reward = random.randint(-100, 100)
        # self.obs_test.append(obs)
        # self.reward_test.append(reward)
        done = False
        return obs, reward, done

    def reset(self):
        return torch.rand((3, 96, 96))


class MockNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.i = 0

    def prediction_eval(self, hidden):
        batch_size = hidden.shape[0]
        assert hidden.device == self.args.device
        assert hidden.shape == (batch_size, 256 + self.args.num_actions, 6, 6)
        value = torch.zeros(batch_size)
        reward = torch.zeros(batch_size)
        policy_logits = torch.ones(
            (batch_size, self.args.num_actions)) / self.args.num_actions
        return value, reward, policy_logits

    def representation_eval(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        assert image.device == self.args.device
        assert image.shape == (batch_size, self.args.n_framestack * 4, 96, 96)

        hidden = torch.zeros((batch_size, 256 + self.args.num_actions, 6, 6))
        hidden = hidden.to(self.args.device)
        return hidden

    def dynamics_eval(self, hidden: torch.Tensor, action: torch.Tensor):
        batch_size = hidden.shape[0]
        assert hidden.device == self.args.device
        assert hidden.shape == (batch_size, 256 + self.args.num_actions, 6, 6)
        assert action.device == self.args.device
        assert action.shape == (batch_size, )

        hidden = torch.zeros((batch_size, 256 + self.args.num_actions, 6, 6))
        hidden = hidden.to(self.args.device)
        return hidden

    def training_steps(self) -> int:
        return 0


def test_make_image():
    args = parse_args([])
    args.n_framestack = 3
    game = player.Game(args)
    env = MockEnv(args)
    actions = [1, 2, 3, 4, 5]
    game.obs.append(env.reset())
    for action in actions:
        obs, reward, done = env.step(action)
        game.obs.append(obs)
        game.actions.append(action)

    image = game.make_image(i=1)

    # For n_framestack = 3, and i=1, we will expect 1 padded observation
    # and two padded actions, because we only want the actions that were taken
    # before obs[i].

    frames = image[0:3 * 3]

    def allclose_to_scalar(a, b):
        b = torch.tensor(b).float()
        return torch.allclose(a, b)

    assert allclose_to_scalar(frames[0:3], 0)
    assert torch.allclose(frames[3:6], game.obs[0])
    assert torch.allclose(frames[6:9], game.obs[1])
    actions = image[3 * 3:]
    assert allclose_to_scalar(actions[0], 0)
    assert allclose_to_scalar(actions[1], 0)
    assert allclose_to_scalar(actions[2], game.actions[0] / args.num_actions)


def test_make_value_target():
    args = parse_args([])
    args.td_steps = 2
    args.discount = .9
    game = player.Game(args)
    values = [10, 20, 30, 40]
    rewards = [5, 4, 3, 2]
    game.values = values
    game.rewards = rewards
    game.policies = [None] * len(values)

    game.done = False
    value = game.make_value_target(0)

    # So we're at a state, which is an observation.
    # If we want the two-step reward, that's the reward following the next
    # two actions. If we take two steps, we will have 3 total hidden states
    # (start, state after step 1, state after step 2), so we should use the
    # value prediction after step 2 for boostrapping.
    assert value == 5 + 4 * .9 + 30 * .9**2

    value = game.make_value_target(1)
    assert value == 4 + 3 * .9 + 40 * .9**2

    # If the game isn't done, we can't make targets if we don't have enough
    # data.
    with pytest.raises(AssertionError):
        value = game.make_value_target(2)

    # But if the game is done, we set terminal values to 0
    game.done = True
    value = game.make_value_target(2)
    assert value == 3 + 2 * .9 + 0 * .9**2

    value = game.make_value_target(3)
    assert value == 2

    value = game.make_value_target(4)
    assert value == 0


def test_get_training_example():
    args = parse_args([])
    args.num_unroll_steps = 2
    args.td_steps = 2
    game = player.Game(args)
    game.obs = [torch.zeros((3, 96, 96))] * 6
    game.actions = [1, 2, 3, 4, 5]
    game.rewards = [4, 5, 6, 7, 8]
    game.values = [-1, -2, -3, -4, -5]
    game.policies = ["test1", "test2", "test3", "test4", "test5"]

    image, targets = game.get_training_example(1)
    assert torch.allclose(image, game.make_image(1))
    assert len(targets) == 2

    action, reward, value, policy = targets[0]
    assert action == 2
    assert reward == 5
    assert value == game.make_value_target(1)
    assert policy == "test2"

    action, reward, value, policy = targets[1]
    assert action == 3
    assert reward == 6
    assert value == game.make_value_target(2)
    assert policy == "test3"


def test_get_new_training_examples():
    args = parse_args([])
    args.num_unroll_steps = 2
    args.td_steps = 2
    game = player.Game(args)
    n = 10
    samples = []
    game.obs.append(torch.zeros((3, 96, 96)))
    for i in range(n):
        game.obs.append(torch.zeros((3, 96, 96)))
        game.actions.append(1)
        game.rewards.append(1)
        game.values.append(1)
        game.policies.append("blah")
        samples += game.get_new_training_examples()

    # Get the number of valid training examples, relying on the bounds
    # checking in the sub-methods.
    n_target = 0
    for i in range(20):
        try:
            game.get_training_example(i)
            n_target += 1
        except AssertionError:
            pass
    assert len(samples) == n_target
    # Also check against what I think it should be
    assert len(samples) == 10 - 1 - 2

    # Repeat but with the game marked as done
    game.done = True
    samples += game.get_new_training_examples()
    n_target = 0
    for i in range(20):
        try:
            game.get_training_example(i)
            n_target += 1
        except AssertionError:
            pass
    assert len(samples) == n_target
    assert len(samples) == 10 - 1
