from collections import namedtuple

from papers.muzero import player
from papers.muzero.model_storage import ModelStorage
from papers.muzero.replay_buffer import ReplayBuffer
from papers.muzero.player_statistics import PlayerStatistics
from papers.muzero.network import Network
from papers.muzero.parse_args import parse_args
from papers.muzero.tests.test_game import MockNetwork

import argparse
import torch
import ray

import time

def test_player_perf_local():
    args = parse_args([])
    args.num_vec_env = 8
    # args.device = torch.device("cpu")
    model_storage = ModelStorage()
    replay_buffer = ReplayBuffer(args)
    stats = PlayerStatistics()
    network = Network(args).to(args.device)
    # network = MockNetwork(args)
    # p = player.RandomPlayer(args, model_storage, replay_buffer, stats)
    p = player.SAPlayer(args, model_storage, replay_buffer, stats)
    p.network = network
    for i in range(100):
        start = time.time()
        p.step()
        print(time.time() - start)
        print(args.num_vec_env / (time.time() - start))

def test_player_perf_remote():
    # ray.init(ignore_reinit_error=True)
    args = parse_args([])
    args.num_vec_env = 32
    # model_storage = ModelStorage()
    model_storage = ray.remote(ModelStorage).remote()
    replay_buffer = ray.remote(ReplayBuffer).remote(args)
    # replay_buffer = ReplayBuffer(args)
    stats = ray.remote(PlayerStatistics).remote()
    # stats = PlayerStatistics()
    network = Network(args).to(args.device)
    p = player.MCTSPlayer(args, model_storage, replay_buffer, stats)
    p.network = network
    for i in range(100):
        start = time.time()
        p.step()
        print(time.time() - start)
        print(args.num_vec_env / (time.time() - start))

def network_perf_test(batch_size):
    print(f"testing {batch_size}")
    args = parse_args([])

    network = Network(args).to(args.device).eval()
    network = network.half()
    hidden = torch.zeros((batch_size, 256+18, 6, 6)).cuda()
    hidden = hidden.half()
    # action = torch.ones((batch_size, )).long().cuda()
    # action_planes = torch.ones((batch_size, 18, 6, 6)).cuda()
    for i in range(100):
        start = time.time()
        # network.dynamics_eval(hidden, action)
        with torch.no_grad():
            network.dynamics_net(hidden)
        print((time.time() - start) / batch_size)
        print(batch_size / (time.time() - start))

def perf_test_pred(batch_size):
    print(f"testing {batch_size}")
    args = parse_args([])

    network = Network(args).to(args.device)
    hidden = torch.zeros((batch_size, 256, 6, 6))
    for i in range(5):
        start = time.time()
        network.prediction_eval(hidden.cuda())
        print((time.time() - start) / batch_size)
        print(batch_size / (time.time() - start))

def test_model_performance_dynamics():
    # perf_test(100)
    perf_test(1000)

def test_model_performance_pred():
    perf_test_pred(100)


def test_remote_player():
    args = parse_args([])
    args.num_workers = 4

    ray.init(args.ray_address, ignore_reinit_error=True)
    model_storage = ray.remote(ModelStorage).remote()
    replay_buffer = ray.remote(ReplayBuffer).remote(args)
    network = Network(args)
    ray.register_custom_serializer(Network, use_pickle=True)
    # ray.get(model_storage.save_network.remote(0, network.cpu()))
    ray.get(model_storage.save_network.remote(step=0, network=network))

    constructor = ray.remote(
        num_cpus=1,
        num_gpus=.001,
        # memory=3*1024**3,
    )(player.GamePlayer).remote

    workers = [
        constructor(args, model_storage, replay_buffer)
        for _ in range(args.num_workers)
    ]
    # Start the workers
    [w.run.remote() for w in workers]

    time.sleep(10000)
