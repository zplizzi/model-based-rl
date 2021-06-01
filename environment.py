import numpy as np
import random
import procgen
import gym
from typing import Dict, List, Optional
from torchvision import transforms


class Environment:
    def __init__(self, args):
        self.episode_steps = 0
        self.episode_reward = 0
        self.args = args

        if "procgen" in args.game:
            # Note: this kills all randomness
            # self.env = gym.make(args.game, rand_seed=9, num_threads=1, num_levels=1)
            self.env = gym.make(args.game, num_threads=1)
        else:
            self.env = gym.make(args.game)

        # Data preprocessing for raw Atari frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((args.dim, args.dim)),
            # Converts to tensor and from [0,255] to [0,1]
            transforms.ToTensor(),
            # For a tensor in range (0, 1), this will convert to range (-1, 1)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.episode_lengths: List[int] = []
        self.episode_rewards: List[int] = []
        # self.reset()

        self.done = False
        self.steps_after_done = 0
        self.last_obs = None

    def step(self, action):
        """Perform a single step of the environment."""

        if self.done:
            if self.args.repeat_last_frame > self.steps_after_done:
                self.steps_after_done += 1
                # Act like the game isn't done, but give 0 reward
                return self.last_obs, 0, False
            else:
                # indicate the game is done
                return self.last_obs, 0, True

        step_reward = 0
        # We have the option to skip steps_to_skip steps, selecting the same
        # action this many times and returning the cumulative rewards
        # from the skipped steps and the final observation
        for _ in range(self.args.steps_to_skip):
            obs, reward, done, _ = self.env.step(action)
            # TODO: remove this
            # Reduce scale of reward, especially for bigfish
            reward = reward / 5
            # Further reduce scale of reward, for riverraid
            reward = reward / 20
            self.episode_reward += reward
            step_reward += reward
            self.episode_steps += 1

            if done:
                # self.episode_lengths.append(self.episode_steps)
                # self.episode_rewards.append(self.episode_reward)
                # self.reset()

                self.done = True
                # Stop skipping steps and just finish this step
                break

        obs = self.transform(obs)

        if done:
            if self.args.repeat_last_frame > self.steps_after_done:
                done = False
                self.steps_after_done += 1
                self.last_obs = obs

        return obs, step_reward, done

    def reset(self):
        self.steps_after_done = 0
        self.last_obs = None
        self.done = False

        self.episode_steps = 0
        self.episode_reward = 0
        obs = self.env.reset()

        if self.args.noop_on_reset:
            # perform a random number of noop steps after reset
            # Do at least n_framestack steps so that we always can make a full
            # model input
            for _ in range(np.random.randint(1, 30)):
                obs, _, done, = self.step(0)
                if done:
                    self.env.reset()

        obs = self.transform(obs)
        return obs
