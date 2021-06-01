import time
import torch
from torch import nn
from papers.muzero import player_batched
from papers.muzero.model_storage import ModelStorage
from papers.muzero.replay_buffer import ReplayBuffer
from papers.muzero.models import Network
from papers.muzero.parse_args import parse_args
from papers.muzero.tests.test_game import MockEnv, MockNetwork

import numpy as np
import ray


def test_push_model():
    args = parse_args([])
    args.num_workers = 1

    ray.init(args.ray_address, ignore_reinit_error=True)
    model_storage = ray.remote(ModelStorage).remote()
    network = Network(args)
    ray.register_custom_serializer(Network, use_pickle=True)
    ray.get(model_storage.save_network.remote(step=0, network=network))

    net2 = ray.get(model_storage.latest_network.remote())
    assert net2 is not None


def test_remote_player_cpu():
    args = parse_args([])
    args.num_workers = 1
    args.num_vec_env = 1
    args.device = torch.device("cpu")

    ray.init(args.ray_address, ignore_reinit_error=True)
    model_storage = ray.remote(ModelStorage).remote()
    replay_buffer = ray.remote(ReplayBuffer).remote(args)
    network = MockNetwork(args)
    ray.register_custom_serializer(Network, use_pickle=True)
    # ray.get(model_storage.save_network.remote(0, network.cpu()))
    ray.get(model_storage.save_network.remote(step=0, network=network))

    constructor = ray.remote(
        num_cpus=1,
        num_gpus=.001,
        # memory=3*1024**3,
    )(player_batched.GamePlayer).remote

    workers = [
        constructor(args, model_storage, replay_buffer)
        for _ in range(args.num_workers)
    ]
    [w.pull_latest_model.remote() for w in workers]
    for i in range(10):
        handles = [w.step.remote() for w in workers]
        [ray.get(h) for h in handles]


def test_remote_player_gpu():
    args = parse_args([])
    args.num_workers = 1
    args.num_vec_env = 1
    args.device = torch.device("cuda:0")

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
    )(player_batched.GamePlayer).remote

    workers = [
        constructor(args, model_storage, replay_buffer)
        for _ in range(args.num_workers)
    ]
    [w.pull_latest_model.remote() for w in workers]
    for i in range(10):
        handles = [w.step.remote() for w in workers]
        [ray.get(h) for h in handles]
