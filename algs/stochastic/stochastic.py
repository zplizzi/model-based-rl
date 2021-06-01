import torch
from torch import nn

import ray
import ray.tune
import sys
import time
import numpy as np
import scipy.stats
import random

from lib import tracker_global as t
from papers.muzero.lib.utils import explained_variance
from papers.muzero.trainable import BaseTrainer
from papers.muzero.player import RandomPlayer, SAPlayer
from papers.muzero.parse_args import parse_args

import collections
from collections import defaultdict

from lib.utils.utils import load_state_dict_debug, debug_grad_norm, select_from_axis
from papers.muzero.lib.distributional import plot_distribution, distributional_loss

from papers.muzero.algs.stochastic import models


class Trainer(BaseTrainer):
    def _setup(self, config):
        self.args = config["args"]
        self.network = models.Network(self.args)
        self.network = self.network.to(args.device)

        self.target_network = models.Network(self.args)
        self.target_network = self.target_network.to(args.device)

        path = t.download_model("0f9f4bdc23db409584822e7af9fc2111", "muzero")
        load_state_dict_debug(self.network, torch.load(path))
        load_state_dict_debug(self.target_network, torch.load(path))

        super()._setup(config, player_cls=RandomPlayer, args=self.args)

    def do_per_eval_step(self):
        super().do_per_eval_step()

    def _train_step(self, batch, logging=False):
        self.network.train()
        self.optimizer.zero_grad()

        images = batch["images"]
        actions = batch["actions"]
        initial_image = images[:, 0]
        final_image = images[:, -1]

        hidden = self.network.representation(initial_image)
        hidden_dynamics = hidden

        losses = defaultdict(list)
        for i in range(0, self.args.num_unroll_steps):
            # for a given i we have (state, action, etc). So we want the action
            # taken on step i, doing dynamics then decoder will give state
            # i+1
            action = actions[:, i]
            image = images[:, i + 1]
            # Get the last frame from the framestack
            # TODO: this will only work for greyscale images
            target_frame = images[:, i + 1, -1]

            # Unrolled autoencoding loss term
            # Get the code
            with torch.no_grad():
                code = torch.zeros((self.args.batch_size, models.NUM_CODE_LAYERS), device=self.args.device)
                hidden_code = self.network.dynamics(hidden_dynamics, action, code)
                code_frame = self.network.decoder_net(hidden_code)

            code_input = torch.stack((code_frame.detach(), target_frame), dim=1)
            assert code_input.shape == (self.args.batch_size, 2, self.args.dim, self.args.dim)
            code = self.network.code_net(code_input)

            # Predict frame using the code
            hidden_dynamics = self.network.dynamics(hidden_dynamics, action, code)
            decoded_dynamics = self.network.decoder_net(hidden_dynamics)
            l = nn.functional.mse_loss(decoded_dynamics,
                                       target_frame,
                                       reduction="none")
            l = l.clamp(0, .01)
            l = l.mean(-1).mean(-1)
            assert l.shape == (self.args.batch_size, )
            l = l * 5
            # For riverraid
            l = l * 1000
            losses["unrolled_autoencoder"].append(l)

            # Autoencoding loss without dynamics network
            # TODO: we could also do this at timestep i=0
            hidden_repr = self.network.representation(image)
            decoded_repr = self.network.decoder_net(hidden_repr)
            l = nn.functional.mse_loss(decoded_repr,
                                       target_frame,
                                       reduction="none")
            l = l.clamp(0, .01)
            l = l.mean(-1).mean(-1)
            l = l * 5
            # For riverraid
            l = l * 1000
            assert l.shape == (self.args.batch_size, )
            losses["direct_autoencoder"].append(l)

        # Get the loss for each type and each sample in the batch
        loss_dict = {
            k: torch.stack(values).sum(dim=0)
            for k, values in losses.items()
        }

        # Get the total loss for each item in the batch
        per_sample_loss = torch.stack(list(loss_dict.values())).sum(dim=0)
        assert per_sample_loss.shape == (self.args.batch_size, )

        loss = per_sample_loss.mean()

        loss.backward()
        # if logging and t.i % 100 == 0:
        #     grad_norm = debug_grad_norm(self.network.parameters())
        #     t.add_scalar("charts/grad_norm", grad_norm, freq=1)
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), .4)
        self.optimizer.step()

        freq = 100
        if logging and t.i % freq == 0:
            freq = 1
            t.add_scalar("loss/total", loss, freq=freq)
            for key, value in loss_dict.items():
                t.add_scalar(f"loss/{key}", value.mean(), freq=freq)
            t.add_image_grid("input",
                             initial_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_image_grid("input_final",
                             final_image[0],
                             freq=freq,
                             range=(0, 1))
            t.add_image("final_unrolled", decoded_dynamics[0], freq=freq)
            t.add_image("target frame", target_frame[0], freq=freq)
            t.add_image("final_direct", decoded_repr[0], freq=freq)
            t.add_image("code_frame", code_frame[0], freq=freq)

            t.add_histogram("code", code, freq=freq, plot_mean=True)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # ray.init(ignore_reinit_error=True, local_mode=True)
    ray.init(args.ray_address,
             ignore_reinit_error=True,
             object_store_memory=10 * 1024**3)

    config = {
        "group": "asdf",
        "args": args,
    }
    trainer = Trainer(config)

    while True:
        trainer.train()
