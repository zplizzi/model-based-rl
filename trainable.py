import numpy as np
import torch
import random

from lib import tracker_global as t
import ray
import ray.tune
import time

from papers.muzero.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, RoundRobinBufferPuller
from papers.muzero.model_storage import ModelStorage
from papers.muzero.player_statistics import PlayerStatistics


class BaseTrainer(ray.tune.Trainable):
    def _setup(self, config, player_cls, args):
        # path = t.download_model("e6b66eb60dd34d88b0f11da1ae9c328c", "muzero")
        # self.network.load_state_dict(torch.load(path))
        self.args = args

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=args.lr_init,
        )

        self.model_storage = ray.remote(num_gpus=.001)(ModelStorage).remote()

        # if args.prioritized_replay:
        #     self.replay_buffer = ray.remote(PrioritizedReplayBuffer).remote(
        #         args)
        # else:
        self.replay_buffers = [ray.remote(ReplayBuffer).remote(args) for _ in range(4)]
        self.buffer_puller = RoundRobinBufferPuller(self.replay_buffers)

        self.player_stats = ray.remote(PlayerStatistics).remote()
        result = self.model_storage.save_network.remote(
            step=0, network=self.network.state_dict())
        ray.get(result)
        constructor = ray.remote(
            num_cpus=.5,
            num_gpus=.001,
        )(player_cls).remote

        self.workers = [
            constructor(args, self.model_storage, self.replay_buffers,
                        self.player_stats) for _ in range(args.num_workers)
        ]

        # Start the workers
        [w.run.remote() for w in self.workers]

        t.init(
            name=self._experiment_id,
            project=args.project,
            args=args,
            group=config["group"])
        # t.watch(self.network, freq=100)

    def _train(self):
        for i in range(10):
            self.update_target_network()
            start = time.time()
            target = 10
            while time.time() - start < target:
                self.train_step()

        self.do_per_eval_step()
        return {"eval_loss": 0}

    def update_target_network(self):
        # Push the updated model to storage
        print("pushing latest model")
        self.model_storage.save_network.remote(
            step=t.i, network=self.network.state_dict())

        # update the target network
        # TODO: handle this better so we don't get silent failures maybe
        try:
            self.target_network.load_state_dict(self.network.state_dict())
        except AttributeError:
            print("target network failed to update")
            pass

    def get_batch(self):
        try:
            batch = self.buffer_puller.get_batch()
            # replay_buffer = random.choice(self.replay_buffers)
            # batch = ray.get(replay_buffer.get_batch.remote())
            # if batch is None:
            #     time.sleep(5)
            #     print("waiting for samples")
            #     return None
            batch = {k: v.cuda() for k, v in batch.items()}
        except (ValueError, IndexError):
            time.sleep(5)
            print("waiting for samples")
            return None
        return batch

    def do_per_eval_step(self):
        # Replay buffer statistics
        env_steps, train_steps = ray.get(
            self.replay_buffers[0].get_statistics.remote())
        t.add_scalar("replay_buffer/env_steps", env_steps, freq=1)
        t.add_scalar("replay_buffer/train_steps", train_steps, freq=1)

        # Worker statistics
        lengths, rewards = ray.get(self.player_stats.get_statistics.remote())
        t.add_histogram("worker/episode_length", lengths, freq=1)
        t.add_scalar(
            "worker/episode_length_mean", np.array(lengths).mean(), freq=1)
        t.add_histogram("worker/episode_reward", rewards, freq=1)
        t.add_scalar(
            "worker/episode_reward_mean", np.array(rewards).mean(), freq=1)
        video = ray.get(self.player_stats.get_video.remote())
        if video is not None:
            video = torch.stack(video)
            t.add_video("worker/video", video, freq=1, fps=7)
            # t.add_video("worker/video", video, freq=1, fps=5)

    def train_step(self):
        batch = self.get_batch()
        if batch is None:
            return

        # try:
        #     batch = self.batch
        # except AttributeError:
        #     batch = self.get_batch()
        #     if batch is None:
        #         return
        #     self.batch = batch

        # print("doing train step")
        self._train_step(batch, logging=True)

        t.log_iteration_time(self.args.batch_size, freq=300)
        t.increment_i()

        t.checkpoint_model(self.network, freq=5000)
