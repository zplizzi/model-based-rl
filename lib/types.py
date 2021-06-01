from typing import List, Tuple, Dict, Optional, NamedTuple
import argparse
import torch

Action = int
Image = torch.Tensor
Value = float
Reward = float
UnNormalizedPolicy = torch.Tensor
NormalizedPolicy = torch.Tensor

ActionHistory = List[Action]


ImageBatch = torch.Tensor
ActionHistoryBatch = torch.Tensor

Target = Tuple[Value, Reward, NormalizedPolicy]

Batch = Tuple[ImageBatch, ActionHistoryBatch, List[Target]]

Args = argparse.Namespace

import numba
from numba import jitclass
from numba import int32, float32, deferred_type, optional

from numba.typed import Dict
import numpy as np

node_type = deferred_type()

spec = [
    ('visit_count', int32),
    ('prior', float32),
    ('value_sum', float32),
    ('children', optional(node_type[:])),
    ('hidden_state', float32[:,:,:]),
    ('reward', float32),
]

@jitclass(spec)
class Node(object):
    """Holds information for a node in the MCTS tree."""
    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        # The prior is the model policy's probability assigned to this action
        self.prior = prior
        # value_sum is the total backpropagated value of all simulations
        # through this node
        self.value_sum: float = 0
        self.children = None
        # Init with an empty tensor to make type checker/numba happy
        self.hidden_state: torch.Tensor = np.zeros((1, 1, 1), dtype=np.float32)
        self.reward: float = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        """Get the average value of the simulations through this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class Nodes:
    def __init__(self):
        n = 50
        self.visit_counts = np.zeros(n, dtype=np.int32)
        self.priors = np.zeros(n)
        self.value_sums = np.zeros(n)
        # Each node can have up to 50 children, since there's no cycles
        # actually each node can only have num_actions children, and will
        # always be either none or fully expanded.
        # TODO: record child count perhaps
        self.childrens = np.ones((n, num_actions), dtype=np.int32) * -1
        self.hidden_states = np.zeros((n, 256+8, 6, 6))
        self.rewards = np.zeros(n)


node_type.define(Node.class_type.instance_type)


MAXIMUM_FLOAT_VALUE = float('inf')


spec = [
    ('maximum', float32),
    ('minimum', float32),
]

@jitclass(spec)
class MinMaxStats(object):
    """A class that holds the min-max values of the tree and uses them to
    normalize rewards and values."""
    def __init__(self) -> None:
        self.maximum = -MAXIMUM_FLOAT_VALUE
        self.minimum = MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        # if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
        return (value - self.minimum) / (self.maximum - self.minimum)
        # return value


# TODO: remove
class NetworkOutput(NamedTuple):
    value: torch.Tensor
    reward: torch.Tensor
    # policy_logits: Dict[Action, float]
    policy_logits: torch.Tensor
    # hidden_state: List[float]
