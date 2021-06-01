import numpy as np
import torch

from papers.muzero.lib import compression


class Game(object):
    """A single episode of interaction with the environment."""
    def __init__(self, args) -> None:

        # For a given index, the order items are recorded should be:
        # obs[i] -> action[i] -> reward[i] -> obs[i+1]
        # policy[i] and value[i] go with obs[i]
        self.obs = []
        self.actions = []
        self.rewards = []
        # A list of normalized MCTS policies, ordered from start to finish
        self.policies = []
        # A list of the MCTS value estimates, ordered from start to finish
        self.values = []

        self.done = False
        self.args = args

        self.last_training_sample = -1

    def make_image(self, i: int):
        """ Build game specific feature planes.

        i is the index of the desired observation (plus history).
        So i=0 would be valid once the first observation has been
        recorded.

        TODO: optimize this. This cuts the speed of random rollouts from 
        550 to 275 fps. single-threaded CPU is not good at bulk image stuff.
        maybe it's too big to fit in cache? 32*96*96 is 1.2MB..
        test computer has 256k l2, 2.5m l3 per core
        """

        # This should only get called in the state where we've recorded a new
        # observation, but haven't yet recorded an action
        assert len(self.obs) == len(self.actions) + 1
        assert i >= 0

        obs = self.obs[max(0, i - self.args.n_framestack + 1):i + 1]
        # Actions lags obs by one, since we want the actions that led to
        # the obs. As a result, at i=0 it's possible to have no actions.
        actions = self.actions[max(0, i - self.args.n_framestack + 1 -
                                   1):max(0, i + 1 - 1)]

        # Pad obs with all-zero frames
        n_pad = self.args.n_framestack - len(obs)
        obs_pad = [torch.zeros((self.args.color_channels, self.args.dim, self.args.dim)) for _ in range(n_pad)]
        obs = obs_pad + obs
        # Pad actions with no-op action
        n_pad = self.args.n_framestack - len(actions)
        actions_pad = [0 for _ in range(n_pad)]
        actions = actions_pad + actions

        obs = torch.stack(obs)
        obs = obs.reshape((self.args.n_framestack * self.args.color_channels,
                           self.args.dim, self.args.dim))

        if self.args.actions_in_obs:
            # Expand actions to fill an image plane with value a/18
            action_frames = [
                torch.ones((self.args.dim, self.args.dim)) * a / self.args.num_actions for a in actions
            ]
            action_frames = torch.stack(action_frames)

            image = torch.cat([obs, action_frames], dim=0)
        else:
            image = obs

        assert image.shape == (self.args.image_channels, self.args.dim, self.args.dim), image.shape
        return image

    def make_value_target(self, i):
        """Get the discounted, bootstrapped value using data from up to
        args.td_steps into the future."""
        # There's the important question of how to handle states where
        # i is within td_steps of len(values).
        # This could either mean that the episode is over, and in this case
        # we should value states past the end of the epsisode as 0.
        # Or, it could mean that we are just partway through the game,
        # and we aren't ready to make a target yet.
        # So, we'll raise an error if the game isn't done and we request this.
        if not self.done:
            assert i < len(self.values) - self.args.td_steps

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        bootstrap_index = i + self.args.td_steps
        # The terminal value is either the discountd MCTS value estimate
        # td_steps past i, or 0 if it's past the end of the
        # game
        if bootstrap_index < len(self.values):
            value = self.values[
                bootstrap_index] * self.args.discount**self.args.td_steps
        else:
            value = 0

        # Then add the discounted reward for each step to the terminal
        for j, reward in enumerate(self.rewards[i:bootstrap_index]):
            value += reward * self.args.discount**j

        return value

    def get_training_example(self, i):
        """Get a full training example, rooted at step i.

        This has the initial observation image (incl framestack + actions),
        plus actions, values, rewards, and policies for num_unroll_steps into
        the future.
        """
        # We can't learn on an example if there's no record of what action was
        # taken at all the steps.
        # TODO: this is actually erroneous, I should generate samples
        # past the end because in eval we're going to need to have seen the
        # last few steps of the game - esp if the reward is always on the last
        # step.
        assert (i + self.args.num_unroll_steps - 1) < len(self.actions)
        images = [self.make_image(i)]
        targets = []
        priorities = []
        for j in range(self.args.num_unroll_steps):
            action = self.actions[i + j]
            value_target = self.make_value_target(i + j)
            reward = self.rewards[i + j]
            policy = self.policies[i + j]
            targets.append((action, reward, value_target, policy))
            # For now, don't do any special priority
            # TODO: is this right?
            # the target is not the same as in the training..
            # i think it's right if td_steps is right though
            # Base initial priorities on td error
            # value_pred = self.values[i + j]
            # td_error = np.abs(value_pred - value_target)
            # td_error = np.clip(td_error, .01, 1)
            # Constant initial priority
            td_error = .1
            priorities.append(td_error)
        if self.args.all_unroll_images:
            # Have an image for every timestep
            images = [self.make_image(i+j) for j in
                      range(self.args.num_unroll_steps)]
        if self.args.off_policy_target:
            # Also include the final image, ie the one generated after the
            # final action in the above sequence. This is reasonable to enable
            # along with all_unroll_images.
            target_image = self.make_image(i + self.args.td_steps)
            images.append(target_image)

        images = torch.stack(images)
        if self.args.compression:
            images = compression.pack(images.numpy())
        # else:
        #     images = images.numpy()
        # In theory, we could have a weight for each timestep within this
        # sample. But we're going to train on a full sample, so the weight
        # should if anything be like a sum of all those sub-weights.
        # Here we just use the first value, ie the error of the first transition
        priority = np.array(priorities)[0]
        sample = {"images": images, "targets": targets, "priority": priority}
        return sample

    def get_new_training_examples(self):
        """Get training examples that are newly ready since the last call.

        Will return additional samples if game.done is marked True.
        """
        # The index of the last example that's fully ready
        end_index = len(self.actions) - (self.args.num_unroll_steps - 1) - 1
        if not self.done:
            end_index -= self.args.td_steps
        if end_index < 0:
            return []

        samples = [
            self.get_training_example(i)
            for i in range(self.last_training_sample + 1, end_index + 1)
        ]

        # Clear out old observations to save memory
        # The 128 is just a healthy margin to not have to think about indexes
        # and to have enough to make videos with
        if self.args.clear_old_obs:
            dt = self.args.n_framestack + 256
            for i in range(max(0, self.last_training_sample - dt),
                           max(0, end_index - dt)):
                self.obs[i] = None

        self.last_training_sample = end_index
        return samples

    def get_finished_statistics(self):
        assert self.done
        rewards = sum(self.rewards)
        length = len(self.rewards)
        return length, rewards

    def get_video_frames(self):
        return self.obs[-256:-1:2]
        # return self.obs[-256:-1]
