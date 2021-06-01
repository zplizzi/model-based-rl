import math
from libc.math cimport log, sqrt
import cython
from typing import Dict, List, Optional

import numpy as np


cdef class Node(object):
    cdef public int visit_count
    cdef public float prior
    cdef public float value_sum
    cdef public list children
    cdef public hidden_state
    # The reward received when transitioning from the parent to this node
    cdef public float reward

    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        # The prior is the model policy's probability assigned to this action
        self.prior = prior
        # value_sum is the total backpropagated value of all simulations
        # through this node
        self.value_sum: float = 0
        # TODO: could become fixed size
        self.children = []
        self.hidden_state = None
        self.reward: float = 0

    cpdef expanded(self):
        return len(self.children) > 0
    
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef float value(self, discount):
        # if self.visit_count == 0:
        #     return 0
        # return self.value_sum / self.visit_count

        # Recursively find the max value in the tree
        if self.visit_count == 0:
            return self.prior
        return max([c.reward + discount * c.value(discount) for c in self.children])
    
    def add_child(self, child):
        # TODO: this is kinda weird given a child is implicitly associated
        # with an action via its index.
        self.children.append(child)

        
cdef class MinMaxStats(object):
    cdef float minimum
    cdef float maximum
    def __init__(self) -> None:
        self.maximum = -float('inf')
        self.minimum = float('inf')

    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef float normalize(self, float value):
        """return value will be in [0, 1], assuming it's within
        [minimum, maximum]."""
        if self.maximum > self.minimum:
          return (value - self.minimum) / (self.maximum - self.minimum)
        else:
          # If we've only gotten one update, min and max will be the same..
          # causing div by 0 errors
          return value

    cpdef update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

# def select_child(node: "Node", min_max_stats: "MinMaxStats"):
#     """Select the child with the highest UCB score."""
#     # print([(ucb_score(node, child, min_max_stats), action, child)
#     #     for action, child in enumerate(node.children)])
#     _, action, child = max(
#         (ucb_score(node, child, min_max_stats), action, child)
#         for action, child in enumerate(node.children))
#     return action, child


"""
Comments on the +1 in "child.visit_count + 1":
If you do that, then it will happen (eg under the uniform policy) that
some children never get visited, while others are very frequently visited.
(for a small number of simulations).
The failure mode is if we visit one node with high enough value to trigger
this, but it wasn't actually the best node - then we don't visit the
better node, or at least not until many simulations.
So - if we have the +1, we're relying on the policy to be strong enough
to avoid this behavior.
I've changed it to +.001, which avoids this issue - it basically guarantees
that all children will be visited once before revisiting any node a second
time. But obviously this isn't great in general either - if we want to search
deeply, this will make the # of simulations required exponential.
"""
# @cython.cdivision(True)
# @cython.nonecheck(False)
# cdef float ucb_score(Node parent, Node child, MinMaxStats min_max_stats):
#     # cdef float pb_c = log((parent.visit_count + 19652 + 1) / 19652) + 1.25
#     cdef float pb_c = 1
#     pb_c *= sqrt(parent.visit_count) / (child.visit_count + .001)
#     cdef float prior_score = pb_c * child.prior
#     cdef float value_score = min_max_stats.normalize(child.value())
#     return prior_score + value_score


# # We expand a node using the value, reward and policy prediction obtained from
# # the neural network.
# def expand_node(node: Node, reward, policy_logits, hidden,
#                 num_actions: int):
#     """Given a leaf node, expand it to have a child for each potential action.
#     Initialize the node with the network policy and value predictions."""
#     # _, reward, policy_logits = network_output
#     node.hidden_state = hidden
#     node.reward = reward
#     # Get the un-normalized policy, and take softmax
#     policy = np.exp(np.array(policy_logits))
#     policy = policy / policy.sum()
#     for i in range(len(policy)):
#         node.children.append(Node(policy[i]))

# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, reward, q_values, hidden,
                num_actions: int):
    """Given a leaf node, expand it to have a child for each potential action.
    Initialize the node with the network policy and value predictions."""
    # _, reward, policy_logits = network_output
    node.hidden_state = hidden
    node.reward = reward
    for i in range(len(q_values)):
        node.children.append(Node(q_values[i]))



# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, discount: float,
                  min_max_stats: MinMaxStats):
    """Backpropagate up the tree.
        `search_path` should be a list of nodes from root to child, inclusive
        `value` is the value prediction at the child
    """
    for i, node in enumerate(search_path[::-1]):
        # if node.visit_count == 0:
        #   old_value = 0
        # else:
        #   old_value = node.value_sum / node.visit_count

        node.visit_count += 1
        # Node value is based on average of visits through the node
        # node.value_sum += value

        # Node value is based on best-ever-seen value
        # Note this is highly prone to maximization bias
        # also if we initially predict a high value, and then all children
        # have lower value, we won't adjust down.
        # The fix for that specific issue is to initialize children with the
        # q(s, a) value, so the child can get adjusted down. And then select
        # argmax from all children when doing backup, instead of keeping
        # current best.
        # Tricky though because how to give child an initial value? Need special
        # logic.
        # if value > old_value:
        #   # New value better than old, overwrite
        #   value = value
        # else:
        #   # Propagate up the same old value since the new one is worse
        #   # Could also just break
        #   value = old_value
        # node.value_sum = value * node.visit_count

        # min_max_stats.update(node.value())
        # value = node.reward + discount * value

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(args, node: Node):
    """Add dirichlet noise to the priors of each child of `node`."""
    # actions = list(node.children.keys())
    noise = np.random.dirichlet([args.root_dirichlet_alpha] * args.num_actions)
    frac = args.root_exploration_fraction
    for a, n in enumerate(noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# def select_action(args, node: "Node", training_step: int):
#     """Sample an action at a node based on the visit count of each action."""
#     visit_counts = [(child.visit_count, action)
#                     for action, child in enumerate(node.children)]

#     def visit_softmax_temperature(training_steps):
#         if training_steps < 500e3:
#             return 1.0
#         elif training_steps < 750e3:
#             return 0.5
#         else:
#             return 0.25

#     # Softmax temperature
#     t = visit_softmax_temperature(training_step)

#     # Sample
#     counts, actions = zip(*visit_counts)
#     c = np.array(counts)**(1 / t)
#     ps = c / c.sum()
#     action = np.random.choice(actions, p=ps)
#     return action


def epsilon_greedy(policy, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(range(len(policy)))
    else:
        action = np.argmax(policy)
    return action

