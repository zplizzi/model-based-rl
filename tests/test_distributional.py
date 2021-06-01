import time
import torch
from torch import nn
import numpy as np

from papers.muzero import player
from papers.muzero.model_storage import ModelStorage
from papers.muzero.replay_buffer import ReplayBuffer, build_batch, PrioritizedReplayBuffer
# from papers.muzero.network import Network, DQNNetwork
from papers.muzero.parse_args import parse_args
from papers.muzero.tests.test_game import MockEnv, MockNetwork
from papers.muzero.player_statistics import PlayerStatistics
from papers.muzero.algs.distributional import Trainer, DQNPlayer, DQNNetwork


def test_dqn():
    args = ["--n_framestack=4", "--actions_in_obs=False"]
    args = parse_args(args)
    args.discount = .95
    args.num_vec_env = 4
    args.batch_size = 4
    args.td_steps = 1
    args.num_unroll_steps = 1
    args.dim = 84
    args.prioritized_replay = False
    args.bootstrapping = True
    args.off_policy_target = True
    args.device = torch.device("cuda")
    stats = PlayerStatistics()
    network = DQNNetwork(args).to(args.device)
    model_storage = ModelStorage()
    # if args.prioritized_replay:
    #     replay_buffer = PrioritizedReplayBuffer(args)
    # else:
    replay_buffers = [ReplayBuffer(args)]
    p = DQNPlayer(args, model_storage, replay_buffers, stats)
    # p = player.RandomPlayer(args, model_storage, replay_buffers, stats)
    p.network = network

    # This janky structure is to let us test Trainer._train_step but not any
    # of the rest of Trainer
    class MockTrainer:
        def __init__(self):
            self.network = network
            self.target_network = network
            self.replay_buffers = replay_buffers
            self.model_storage = model_storage
            self.optimizer = torch.optim.Adam(network.parameters())
            self.args = args
    MockTrainer._train_step = Trainer._train_step

    trainer = MockTrainer()
    for i in range(100):
        p.step()

    batch = replay_buffers[0].get_batch()
    batch = {k: v.cuda() for k, v in batch.items()}

    trainer._train_step(batch)
