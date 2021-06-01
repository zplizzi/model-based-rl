import collections
import random
import torch
import time
import ray
import timeout_decorator

import copy


class Storage:
    def __init__(self):
        self.buffer = collections.deque(maxlen=20000)
        self.buffer.append(ray.put("first item"))
        self.i = 0

    def save_samples(self, samples):
        for sample in samples:
            # self.buffer.append(copy.deepcopy(sample))
            self.buffer.append(sample)
        if self.i % 10 == 0:
            print(f"buffer has {len(self.buffer)} items")
        self.i += 1

    def sample_batch(self):
        batch = [self.sample() for _ in range(200)]
        return batch

    def sample(self):
        return random.sample(self.buffer, k=1)[0]


class Worker:
    def __init__(self, storage):
        self.storage = storage

    def run(self):
        while True:
            self.step()

    def step(self):
        samples = [torch.rand((32, 96, 96)) for _ in range(64)]
        # self.storage.save_samples.remote(samples)
        # handles = [ray.put(x) for x in samples]
        start = time.time()
        ray.get(self.storage.save_samples.remote(samples))
        print(time.time() - start)
        print("worker stepped")
        time.sleep(.05)
        print(self.stats)

if __name__ == "__main__":
    ray.init(object_store_memory=13.88*1024**3)
    storage = ray.remote(Storage).remote()
    workers = [ray.remote(Worker).remote(storage) for _ in range(2)]
    [w.run.remote() for w in workers]
    start = time.time()
    for i in range(1000000):
        batch = ray.get(storage.sample_batch.remote())
