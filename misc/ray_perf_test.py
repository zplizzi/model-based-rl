import ray
import random
import torch
from typing import List, Tuple
from collections import OrderedDict
# from papers.muzero.player import Game

import numpy as np
import time

from collections import deque
import copy

import logging
import base64
import pyarrow
from six import string_types

logger = logging.getLogger(__name__)

import lz4.frame


def pack(data):
    data = pyarrow.serialize(data).to_buffer().to_pybytes()
    data = lz4.frame.compress(data)
    # TODO(ekl) we shouldn't need to base64 encode this data, but this
    # seems to not survive a transfer through the object store if we don't.
    data = base64.b64encode(data).decode("ascii")
    return data


def pack_if_needed(data):
    if isinstance(data, np.ndarray):
        data = pack(data)
    return data


def unpack(data):
    data = base64.b64decode(data)
    data = lz4.frame.decompress(data)
    data = pyarrow.deserialize(data)
    return data


def unpack_if_needed(data):
    if is_compressed(data):
        data = unpack(data)
    return data


def is_compressed(data):
    return isinstance(data, bytes) or isinstance(data, string_types)

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
                data = torch.as_tensor(unpack(images))
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
    def __init__(self):
        self._storage = OrderedDict()
        # This is a map from a ring buffer index to storage key
        self.index_to_key = {}
        self.key_to_index = {}
        self._maxsize = 100000
        self._next_key = 0
        self._next_idx = 0

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
        samples = random.sample(list(self._storage.values()), k=batch_size)

        def add_weight_key(sample):
            sample["weight"] = 1
            sample["key"] = 0
            return sample
        # append (weight, key) to samples
        samples = [add_weight_key(s) for s in samples]
        return samples

    def get_samples(self):
        # TODO: remove this alias method
        if len(self._storage) < 100:
            return None
        return self.sample(batch_size=64)


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
                samples = ray.get(self.replay_buffer.get_samples.remote())
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
            # print(f"buffer puller {self.i} is empty")
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
                # print("master batchpuller is starved")
                time.sleep(.01)


def build_sample():
    images = pack(torch.rand((2, 4, 56, 56)).numpy())
    targets = [[1, 2.0, 3.0, torch.tensor([1,2,3])]]
    priority = 1
    sample = {"images": images, "targets": targets, "priority": priority}
    return sample

class SamplePusher:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def run(self):
        while True:
            sample = build_sample()
            ray.get(self.replay_buffer.save_samples.remote([sample]))
            # Push at most ~300 fps
            time.sleep(.003)

if __name__ == "__main__":
    ray.init()
    replay_buffer = ray.remote(ReplayBuffer).remote()
    worker1 = ray.remote(SamplePusher).remote(replay_buffer)
    worker2 = ray.remote(SamplePusher).remote(replay_buffer)
    worker1.run.remote()
    worker2.run.remote()
    buffer_puller = RoundRobinBufferPuller(replay_buffer, num_actors=4)
    while True:
        start = time.time()
        for i in range(100):
            batch = buffer_puller.fetch_batch()
        print((64 * 100) / (time.time() - start))
