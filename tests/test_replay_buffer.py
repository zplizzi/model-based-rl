from papers.muzero.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from papers.muzero.parse_args import parse_args

import torch
import random
from torch import nn
import pytest
import numpy as np

class PatchedBuffer(ReplayBuffer):
    # Override sample to linearly sample through the buffer
    def sample(self):
        try:
            self.last_sample += 1
        except AttributeError:
            self.last_sample = 0
        return self.buffer[self.last_sample]


def test_sample_batch():
    args = parse_args([])
    args.num_unroll_steps = 2
    args.td_steps = 2
    args.batch_size = 2
    args.num_actions = 3
    buf = PatchedBuffer(args)

    img1 = torch.ones((args.n_framestack * 4, 96, 96)) * 7
    policy1 = torch.tensor([1,2,3])
    s1 = (img1, ((1, 10, 100, policy1), (2, 20, 200, policy1*2)))
    s2 = (img1*2, ((3, 30, 300, policy1*3), (4, 40, 400, policy1*4)))

    buf.save_samples([s1, s2])

    batch = buf.sample_batch()
    image, actions, rewards, values, policies = batch
    assert image.shape == (2, args.n_framestack * 4, 96, 96)

    assert torch.allclose(image[0], img1)
    assert torch.allclose(image[1], img1 * 2)

    assert torch.allclose(actions[:, 0], torch.tensor([1, 3]))
    assert torch.allclose(rewards[:, 0], torch.tensor([10, 30]).float())
    assert torch.allclose(values[:, 0], torch.tensor([100, 300]).float())
    assert torch.allclose(policies[:, 0], torch.stack([policy1, policy1*3]))

    assert torch.allclose(actions[:, 1], torch.tensor([2, 4]))
    assert torch.allclose(rewards[:, 1], torch.tensor([20, 40]).float())
    assert torch.allclose(values[:, 1], torch.tensor([200, 400]).float())
    assert torch.allclose(policies[:, 1], torch.stack([policy1*2, policy1*4]))


def test_uniform():
    args = parse_args([])
    args.buffer_size = 500
    b = ReplayBuffer(args)
    for i in range(1000):
        # samples are (data, weight)
        b.save_samples([(i, 1)])

    assert len(b) == 500
    batch = b.sample(100)
    mean = np.array(batch).mean()
    assert mean > 700
    assert mean < 800


def test_prioritized_uniform():
    args = parse_args([])
    args.buffer_size = 500
    b = PrioritizedReplayBuffer(args)
    for i in range(1000):
        # samples are (data, weight)
        b.save_samples([(i, 1)])

    assert len(b) == 500
    batch = b.sample(100)
    values = np.array(batch)[:, 0]
    weights = np.array(batch)[:, 1]
    indexes = np.array(batch)[:, 2]
    assert values.mean() > 700
    assert values.mean() < 800
    assert weights.mean() == 1
    assert indexes.mean() > 700
    assert indexes.mean() < 800

def test_prioritized_nonuniform():
    args = parse_args([])
    args.buffer_size = 1000
    args.replay_buffer_alpha = 1
    b = PrioritizedReplayBuffer(args)
    for i in range(900):
        # samples are (data, weight)
        b.save_samples([(i, .01)])
    for i in range(900, 1000):
        # samples are (data, weight)
        b.save_samples([(i, 1)])

    assert len(b) == 1000
    batch = b.sample(1000)
    values = np.array(batch)[:, 0]
    weights = np.array(batch)[:, 1]
    indexes = np.array(batch)[:, 2]
    # 10% of the samples should be <900, the rest >900
    assert values.mean() > 900
    assert values.mean() < 915
    # The weight of the frequent samples should be .01, because they get
    # sampled 100x as frequently as average samples in the buffer
    # Is this right? The buffer's total weight is ~100, so average weight
    # is .1. So they're only sampled 10x as much as average?
    # Oh, but then the weight options would be .1 and 10. But we have the rule
    # that we can only have weights <=1, so both of these get shifted down.
    assert weights.mean() < .15
    assert weights.mean() > .05

def test_prioritized_replace():
    args = parse_args([])
    args.buffer_size = 1000
    args.replay_buffer_alpha = 1
    b = PrioritizedReplayBuffer(args)
    for i in range(900):
        # samples are (data, weight)
        b.save_samples([(i, .01)])
    for i in range(900, 1000):
        # samples are (data, weight)
        b.save_samples([(i, 1)])

    for i in range(10):
        batch = b.sample(100)
        for value, weight, key in batch:
            if weight < .1:
                b.update_priorities((key, ), (.02, ))
            else:
                b.update_priorities((key, ), (.03, ))

    batch = b.sample(1000)
    values = np.array(batch)[:, 0]
    weights = np.array(batch)[:, 1]
    keys = np.array(batch)[:, 2]
    assert values.mean() < 700
