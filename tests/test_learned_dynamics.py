import time
import torch
from torch import nn
import numpy as np

from papers.muzero import player
from papers.muzero.model_storage import ModelStorage
from papers.muzero.replay_buffer import ReplayBuffer, build_batch, PrioritizedReplayBuffer
from papers.muzero.network import Network, DQNNetwork
from papers.muzero.parse_args import parse_args
from papers.muzero.tests.test_game import MockEnv, MockNetwork
from papers.muzero.player_statistics import PlayerStatistics
# from papers.muzero.algs.learned_dynamics import Trainer, LearnedDynamicsNetwork
from papers.muzero.algs import learned_dynamics


##

def test_dqn():
    args = ["--n_framestack=4", "--actions_in_obs=False"]
    args = parse_args(args)
    args.discount = .95
    args.num_vec_env = 4
    args.batch_size = 8
    args.td_steps = 3
    args.num_unroll_steps = 3
    args.dim = 64
    args.prioritized_replay = False
    args.bootstrapping = True
    args.off_policy_target = True
    args.all_unroll_images = True
    args.device = torch.device("cuda")
    stats = PlayerStatistics()
    network = learned_dynamics.LearnedDynamicsNetwork(args).to(args.device)
    model_storage = ModelStorage()
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args)
    else:
        replay_buffer = ReplayBuffer(args)
    p = player.RandomPlayer(args, model_storage, replay_buffer, stats)

    # This janky structure is to let us test Trainer._train_step but not any
    # of the rest of Trainer
    class MockTrainer:
        def __init__(self):
            self.network = network
            self.target_network = network
            self.replay_buffer = replay_buffer
            self.model_storage = model_storage
            self.optimizer = torch.optim.Adam(network.parameters())
            self.args = args
            self.overfitting_samples = []
    MockTrainer._train_step = learned_dynamics.Trainer._train_step

    trainer = MockTrainer()
    for i in range(100):
        p.step()

    samples = replay_buffer.get_samples()
    batch = build_batch(samples)
    batch = {k: v.cuda() for k, v in batch.items()}

    trainer._train_step(batch)

def test_train_using_model():
    args = ["--n_framestack=4", "--actions_in_obs=False"]
    args = parse_args(args)
    args.discount = .95
    args.num_vec_env = 4
    args.batch_size = 8
    args.td_steps = 3
    args.num_unroll_steps = 3
    args.dim = 64
    args.prioritized_replay = False
    args.bootstrapping = True
    args.off_policy_target = True
    args.all_unroll_images = True
    args.double_q = True
    args.device = torch.device("cuda")
    stats = PlayerStatistics()
    network = learned_dynamics.LearnedDynamicsNetwork(args).to(args.device)
    model_storage = ModelStorage()
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args)
    else:
        replay_buffer = ReplayBuffer(args)
    p = player.RandomPlayer(args, model_storage, replay_buffer, stats)

    # This janky structure is to let us test Trainer._train_step but not any
    # of the rest of Trainer
    class MockTrainer:
        def __init__(self):
            self.network = network
            self.target_network = network
            self.replay_buffer = replay_buffer
            self.model_storage = model_storage
            self.optimizer = torch.optim.Adam(network.parameters())
            self.args = args
            self.overfitting_samples = []
    MockTrainer.train_using_model = learned_dynamics.Trainer.train_using_model

    trainer = MockTrainer()
    for i in range(100):
        p.step()

    samples = replay_buffer.get_samples()
    batch = build_batch(samples)
    batch = {k: v.cuda() for k, v in batch.items()}

    trainer.train_using_model(batch)

def test_just_dqn():
    args = ["--n_framestack=4", "--actions_in_obs=False"]
    args = parse_args(args)
    args.discount = .95
    args.num_vec_env = 4
    args.batch_size = 8
    args.td_steps = 3
    args.num_unroll_steps = 3
    args.dim = 64
    args.prioritized_replay = False
    args.bootstrapping = True
    args.off_policy_target = True
    args.all_unroll_images = True
    args.double_q = True
    args.device = torch.device("cuda")
    stats = PlayerStatistics()
    network = learned_dynamics.LearnedDynamicsNetwork(args).to(args.device)
    model_storage = ModelStorage()
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args)
    else:
        replay_buffer = ReplayBuffer(args)
    p = player.RandomPlayer(args, model_storage, replay_buffer, stats)

    # This janky structure is to let us test Trainer._train_step but not any
    # of the rest of Trainer
    class MockTrainer:
        def __init__(self):
            self.network = network
            self.target_network = network
            self.replay_buffer = replay_buffer
            self.model_storage = model_storage
            self.optimizer = torch.optim.Adam(network.parameters())
            self.args = args
            self.overfitting_samples = []
    MockTrainer.train_just_dqn_3 = learned_dynamics.Trainer.train_just_dqn_3

    trainer = MockTrainer()
    for i in range(100):
        p.step()

    samples = replay_buffer.get_samples()
    batch = build_batch(samples)
    batch = {k: v.cuda() for k, v in batch.items()}

    trainer.train_just_dqn_3(batch)
