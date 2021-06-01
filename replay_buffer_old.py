import ray
import random
import torch
from typing import List, Tuple
from collections import OrderedDict
# from papers.muzero.player import Game

from papers.muzero.lib import compression
from papers.muzero.lib.segment_tree import SumSegmentTree, MinSegmentTree
import numpy as np
import time

from collections import deque
import copy

def build_batch(samples):
    """
    TODO: this code is gross, because having embedded tensors in lists of
    lists makes them really hard to deal with. Not sure what the cleaner
    solution is.
    """
    batch = {}
    for key in samples[0].keys():
        key_data = []
        for sample in samples:
            if key == "images":
                images = sample["images"]
                data = torch.as_tensor(compression.unpack(images))
            else:
                data = sample[key]
            key_data.append(data)

        if key == "targets":
            targets = np.array(key_data, dtype=object)
            actions, rewards, values, policies = targets.transpose((2, 0, 1))

            batch["actions"] = torch.tensor(actions.astype(int))
            batch["rewards"] = torch.tensor(rewards.astype(np.float32))
            batch["values"] = torch.tensor(values.astype(np.float32))
            batch["policies"] = torch.stack(list(map(torch.stack, policies.tolist())))
        else:
            if type(key_data[0]) == torch.Tensor:
                batch[key] = torch.stack(key_data)
            else:
                batch[key] = torch.tensor(key_data)
    return batch


class ReplayBuffer:
    def __init__(self, args):
        self._maxsize = args.buffer_size
        self.args = args
        self._storage = deque(maxlen=self._maxsize)
        self.env_steps = 0
        self.train_steps = 0

    def __len__(self):
        return len(self._storage)

    def save_samples(self, samples):
        for sample in samples:
            # The sample as returned by Ray will be in shared memory. Shared
            # memory has somewhat limited capacity so we want it back in heap.
            # Copying will do that.
            sample = copy.deepcopy(sample)
            self._storage.append(sample)
        self.env_steps += len(samples)

    def sample(self, batch_size):
        # samples = random.sample(self._storage, k=batch_size)
        sample_indexes = np.random.randint(0, len(self._storage), size=batch_size)
        samples = [self._storage[i] for i in sample_indexes]
        # Add age, which for now is just the sample index in the buffer
        # Note that this only works with a dequeu!
        for i, sample in enumerate(samples):
            sample["age"] = sample_indexes[i]

        def add_weight_key(sample):
            sample["weight"] = 1
            sample["key"] = 0
            return sample
        # append (weight, key) to samples
        samples = [add_weight_key(s) for s in samples]
        self.train_steps += batch_size
        return samples

    def get_samples(self):
        # TODO: remove this alias method
        if len(self._storage) < self.args.batch_size:
        # if len(self._storage) < 2000:
            return None
        return self.sample(batch_size=self.args.batch_size)

    def get_statistics(self):
        return self.env_steps, self.train_steps


class ReplayBufferDict:
    def __init__(self, args):
        self._storage = OrderedDict()
        # This is a map from a ring buffer index to storage key
        self.index_to_key = {}
        self.key_to_index = {}
        self._maxsize = args.buffer_size
        self._next_key = 0
        self._next_idx = 0
        self.args = args

        self.env_steps = 0
        self.train_steps = 0

    def __len__(self):
        return len(self._storage)

    def save_samples(self, samples):
        for sample in samples:
            # The sample as returned by Ray will be in shared memory. Shared
            # memory has somewhat limited capacity so we want it back in heap.
            # Copying will do that.
            sample = copy.deepcopy(sample)
            priority = sample["priority"]
            assert type(sample) == dict
            self.add(sample, priority=priority)
        self.env_steps += len(samples)

    def add(self, data, priority):
        assert type(data) == dict
        if self._next_key < self._maxsize:
            # Storage not yet full
            self._storage[self._next_key] = data
            self.index_to_key[self._next_idx] = self._next_key
            self.key_to_index[self._next_key] = self._next_idx
        else:
            # Storage is full
            k, v = self._storage.popitem(last=False)
            del self.key_to_index[k]
            self._storage[self._next_key] = data
            self.index_to_key[self._next_idx] = self._next_key
            self.key_to_index[self._next_key] = self._next_idx

        self._next_key += 1
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        keys = random.sample(self._storage.keys(), k=batch_size)
        samples = [self._storage[key] for key in keys]
        # samples = random.sample(list(self._storage.values()), k=batch_size)

        def add_weight_key(sample):
            sample["weight"] = 1
            sample["key"] = 0
            return sample
        # append (weight, key) to samples
        samples = [add_weight_key(s) for s in samples]
        self.train_steps += batch_size
        return samples

    def get_samples(self):
        # TODO: remove this alias method
        if len(self._storage) < self.args.batch_size:
            return None
        return self.sample(batch_size=self.args.batch_size)

    def get_statistics(self):
        return self.env_steps, self.train_steps


class PrioritizedReplayBuffer(ReplayBufferDict):
    def __init__(self, args):
        super().__init__(args)
        # Amount of prioritization - 0 = none, 1 = full
        self._alpha = args.replay_buffer_alpha
        assert self._alpha > 0

        # The true capacity needs to be a power of 2
        it_capacity = 1
        while it_capacity < args.buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, data, priority):
        idx = self._next_idx
        super().add(data, priority)
        if priority is None:
            priority = self._max_priority
        self._it_sum[idx] = priority**self._alpha
        self._it_min[idx] = priority**self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=1):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage))**(-beta)

        samples = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage))**(-beta)
            weight = weight / max_weight
            key = self.index_to_key[idx]
            sample = self._storage[key]
            # sample = list(sample) + [weight, key]
            sample["weight"] = weight
            # Get the original priority, for debugging
            sample["priority"] = self._it_sum[idx]**(1 / self._alpha)
            sample["key"] = key
            samples.append(sample)
        self.train_steps += batch_size
        return samples

    def update_priorities(self, keys, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
          List of idxes of sampled transitions
        priorities: [float]
          List of updated priorities corresponding to
          transitions at the sampled idxes denoted by
          variable `idxes`.
        """
        assert len(keys) == len(priorities)
        for key, priority in zip(keys, priorities):
            assert priority > 0
            try:
                idx = self.key_to_index[key]
                assert 0 <= idx < len(self._storage)
                self._it_sum[idx] = priority**self._alpha
                self._it_min[idx] = priority**self._alpha

                self._max_priority = max(self._max_priority, priority)
            except KeyError:
                # Sample was deleted
                pass


class BufferPuller:
    """Essentially allows for async pulling from the replay buffer."""
    def __init__(self, replay_buffer):
        self.n = 10
        # deque is thread-safe
        self.batches = deque(maxlen=self.n)
        self.replay_buffer = replay_buffer
        import threading
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            if len(self.batches) < self.n:
                # batch = ray.get(self.replay_buffer.sample_batch.remote())
                try:
                    samples = ray.get(self.replay_buffer.get_samples.remote())
                except AttributeError:
                    samples = self.replay_buffer.get_samples()
                if samples is None:
                    time.sleep(.01)
                    continue
                batch = build_batch(samples)
                batch_handle = ray.put(batch)
                self.batches.append(batch_handle)
            else:
                time.sleep(.01)

    def get_batch(self):
        if len(self.batches) > 0:
            # Can't return a raw handle apparently..
            return [self.batches.popleft()]
        else:
            return None


class RoundRobinBufferPuller:
    def __init__(self, replay_buffer, num_actors, local=False):
        self.pullers = [ray.remote(BufferPuller).remote(replay_buffer)
                        for _ in range(num_actors)]
        self.num_actors = num_actors
        self.i = 0

        self.buffer_size = 10
        # deque is thread-safe
        self.batches = deque(maxlen=self.buffer_size)
        self.replay_buffer = replay_buffer
        import threading
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            if len(self.batches) < self.buffer_size:
                batch = self.fetch_batch()
                # TODO: not sure if we always want this
                batch = {k: v.cuda() for k, v in batch.items()}
                self.batches.append(batch)
            else:
                time.sleep(.01)

    def fetch_batch(self):
        while True:
            batch_handle = ray.get(self.pullers[self.i].get_batch.remote())
            if batch_handle is not None:
                break
            print(f"buffer puller {self.i} is empty")
            time.sleep(.01)
            self.i += 1
            if self.i % self.num_actors == 0:
                self.i = 0
        batch = ray.get(batch_handle[0])
        return batch

    def get_batch(self):
        while True:
            if len(self.batches) > 0:
                return self.batches.popleft()
            else:
                print("master batchpuller is starved")
                time.sleep(.01)
