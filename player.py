import numpy as np
from typing import Dict, List, Optional
import torch
import ray
import random

from papers.muzero.network import Network, DQNNetwork

from papers.muzero.game import Game
from papers.muzero.environment import Environment

import pyximport
pyximport.install()
from papers.muzero.player_cy import epsilon_greedy


class Player:
    def __init__(self, args, model_storage, replay_buffers, stats):
        torch.set_num_threads(1)
        self.replay_buffers = replay_buffers
        self.args = args
        self.model_storage = model_storage
        self.stats = stats

        self.envs = [Environment(args) for _ in range(args.num_vec_env)]

        # Need to make sure to append an initial observation to these games
        self.games = [Game(self.args) for _ in range(args.num_vec_env)]

        # Initialize environments with lots of random steps
        for game, env in zip(self.games, self.envs):
            env.reset()
            # for _ in range(1000):
            #     action = random.sample(range(args.num_actions), k=1)[0]
            #     # We step the underlying env to avoid computation overhead
            #     _, _, done, _ = env.env.step(action)
            #     if done:
            #         env.env.reset()
            # Take one real step just to get an initial obs for the game
            obs, _, game.done = env.step(0)
            game.obs.append(obs)

        self.network = None
        self.samples_to_push = []

    def pull_latest_model(self):
        try:
            state_dict, i = self.model_storage.latest_network()
        except:
            state_dict, i = ray.get(self.model_storage.latest_network.remote())
        if self.network is None:
            # self.network = Network(self.args).to(self.args.device)
            # self.network = DQNNetwork(self.args).to(self.args.device)
            self.network = self.network_cls(self.args).to(self.args.device)
        self.network.load_state_dict(state_dict)
        self.network.i = i
        # Set the network to be eval and no-grad for this whole thread
        self.network.eval()
        # torch.autograd.set_grad_enabled(False)

    def push_to_buffer(self):
        samples = [
            x for game in self.games for x in game.get_new_training_examples()
        ]
        self.samples_to_push += samples
        if len(self.samples_to_push) > 200:
            print(f"pushing {len(self.samples_to_push)} samples")
        else:
            return

        replay_buffer = random.choice(self.replay_buffers)
        try:
            replay_buffer.save_samples.remote(self.samples_to_push)
        except AttributeError:
            # We most likely have a local replay buffer
            replay_buffer.save_samples(self.samples_to_push)
        self.samples_to_push = []

    def complete_finished_games(self):
        def is_complete(game):
            """Terminiation conditions for a game."""
            return game.done or len(game.actions) > self.args.max_moves

        for i in range(len(self.games)):
            if is_complete(self.games[i]):
                try:
                    self.stats.log_finished_game(
                        *self.games[i].get_finished_statistics())
                except:
                    self.stats.log_finished_game.remote(
                        *self.games[i].get_finished_statistics())
                game = Game(self.args)
                game.obs.append(self.envs[i].reset())
                self.games[i] = game

    def step(self):
        # print("worker stepping")
        self.complete_finished_games()
        with torch.no_grad():
            actions, values, policies = self.compute()

        # Select an action and step the environment
        for game, env, action, value, policy in zip(self.games, self.envs, actions, values,
                                               policies):
            # Do the environment step
            obs, reward, game.done = env.step(action)
            # game.obs is always one step longer than the other history
            # arrays because we appended an obs at the beginning of the episode
            # So this obs really corresponds to the next step.
            game.obs.append(obs)
            game.rewards.append(reward)
            game.actions.append(action)
            game.policies.append(policy)
            game.values.append(value)

        self.push_to_buffer()


class RandomPlayer(Player):
    def __init__(self, args, model_storage, replay_buffers, stats):
        super().__init__(args, model_storage, replay_buffers, stats)

    def run(self):
        while True:
            self.step()

    def compute(self):
        actions = []
        values = []
        policies = []
        for game, env in zip(self.games, self.envs):
            action = random.sample(range(self.args.num_actions), k=1)[0]
            value = 0
            policy = torch.ones(self.args.num_actions)
            policy = policy / policy.sum()
            actions.append(action)
            policies.append(policy)
            values.append(value)
        return actions, values, policies


class SAPlayer(Player):
    def __init__(self, args, model_storage, replay_buffers, stats):
        super().__init__(args, model_storage, replay_buffers, stats)

    def run(self):
        self.pull_latest_model()
        i = 0
        while True:
            self.step()
            if i % 256 == 0:
                frames = self.games[0].get_video_frames()
                self.stats.log_video_frames.remote(frames)
            if i % 256 == 0:
                self.pull_latest_model()
            i += 1

    def get_values(self, hiddens):
        raise NotImplementedError

    def compute(self):
        # Do representation step
        images = []
        for game in self.games:
            image = game.make_image(len(game.obs) - 1)
            images.append(image)

        images = torch.stack(images)
        images = images.to(self.args.device)
        hiddens = self.network.representation(images)
        # predictions = self.network.prediction_eval(hiddens)
        # _, root_reward, _ = predictions
        # TODO: fix this such that we use the action-dependent reward instead.
        # it definitely gimps this algo to not have action-dependent rewards.
        # reward_after_root_state = root_reward

        values = self.get_values(hiddens)
        values = values.cpu()
        assert values.shape == (len(self.games), self.args.num_actions)

        # For debugging
        self.values = values

        def eps_decay(i):
            # For pong
            # const = 10000
            # linear_1 = 20000
            # linear_2 = 70000
            # For bigfish
            # const = 800
            # linear_1 = 500
            # linear_2 = 10000
            # test
            # const = 1000
            # linear_1 = 5000
            # linear_2 = 10000
            # # inst
            const = 0
            linear_1 = 0
            linear_2 = 5000

            if i < const:
                eps = 1
            elif i < const + linear_1:
                di = i - const
                eps = 1 - min(di / linear_1, .8)
            else:
                di = i - (const + linear_1)
                eps = .2 - .2 * min(di / linear_2, .95)
            return eps

        # Constant eps
        def eps_decay(i):
            return .01

        epsilon = eps_decay(self.network.i)
        if np.random.rand() < .01:
            print(f"eps is {epsilon}")

        actions = []
        root_values = []
        policies = []
        for i, game in enumerate(self.games):

            # TODO: remove me!!!!!
            if i > 8:
                # Scale eps by up to 4x, to .04
                epsilon = .01 * (i / 8) * 4

            action_values = values[i]
            assert action_values.shape == (self.args.num_actions, )
            # Use temperature to bias towards selecting most common
            # temperature = 100
            # policy = torch.nn.functional.softmax(temperature * action_values, dim=0)
            policy = action_values
            # # Apparently the torch output isn't close enough to 1 for numpy's satisfaction
            # policy_np = policy.numpy()
            # policy_np /= policy_np.sum()
            # action = np.random.choice(list(range(self.args.num_actions)),
            #                           p=policy_np)
            # action = torch.argmax(action_values)

            action = epsilon_greedy(action_values, epsilon=epsilon)

            actions.append(action)
            # This policy is the softmax of the QSA values, really just for debugging
            policies.append(policy)
            # This value is the value of the chosen action
            if self.args.bootstrapping:
                root_values.append(action_values[action])
            else:
                # Or maybe we just disable bootstrapping for simplicity
                root_values.append(0)

        return actions, root_values, policies
