import time
import torch
from torch import nn
from papers.muzero import player
from papers.muzero.model_storage import ModelStorage
from papers.muzero.replay_buffer import ReplayBuffer, build_batch, PrioritizedReplayBuffer
from papers.muzero.network import Network, DQNNetwork
from papers.muzero.parse_args import parse_args
from papers.muzero.tests.test_game import MockEnv, MockNetwork
from papers.muzero.player_statistics import PlayerStatistics

import numpy as np

from papers.muzero import train_step

# class MockOpt:
#     def step(self):
#         pass

def test():
    args = parse_args([])
    args.num_simulations = 50
    args.decay = .9
    args.device = torch.device("cpu")
    args.num_vec_env = 4
    args.batch_size = 4
    model_storage = ModelStorage()
    replay_buffer = ReplayBuffer(args)
    network = MockNetwork(args).to(args.device)
    player.Environment = MockEnv
    p = player.GamePlayer(args, model_storage, replay_buffer)
    p.network = network
    for i in range(20):
        p.step()

    args.device = torch.device("cuda")
    batch = replay_buffer.sample_batch()
    batch = [x.to(args.device) for x in batch]

    network = Network(args).to(args.device)
    opt = torch.optim.Adam(network.parameters())
    train_step.train_step_simple(opt, network, batch, args)


def test_simple():
    args = parse_args([])
    args.discount = .95
    args.num_vec_env = 4
    args.batch_size = 4
    args.td_steps = 16
    model_storage = ModelStorage()
    replay_buffer = ReplayBuffer(args)
    # player.Environment = MockEnv
    p = player.RandomPlayer(args, model_storage, replay_buffer)
    for i in range(100):
        p.step()

    args.device = torch.device("cuda")
    batch = replay_buffer.sample_batch()
    batch = [x.to(args.device) for x in batch]

    network = Network(args).to(args.device)
    opt = torch.optim.Adam(network.parameters())
    train_step.train_step_simple(opt, network, batch, args)


def test_sa():
    args = parse_args([])
    args.discount = .95
    args.num_vec_env = 4
    args.batch_size = 4
    args.td_steps = 16
    model_storage = ModelStorage()
    replay_buffer = ReplayBuffer(args)
    # player.Environment = MockEnv
    p = player.RandomPlayer(args, model_storage, replay_buffer)
    for i in range(100):
        p.step()

    args.device = torch.device("cuda")
    batch = replay_buffer.sample_batch()
    batch = [x.to(args.device) for x in batch]

    network = Network(args).to(args.device)
    opt = torch.optim.Adam(network.parameters())
    train_step.train_step_sa(opt, network, batch, args)



def test_policy_loss():
    n = 10
    a = torch.ones((5, n)) / n
    b = torch.ones((5, n)) / n
    # Cross entropy is KL divergence plus entropy of one of the dists
    target = 0 + np.log(n)
    loss = train_step.compute_policy_loss(a, b)
    assert torch.allclose(loss, torch.tensor(target))
