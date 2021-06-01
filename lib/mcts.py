import numpy as np
import torch
import copy
import random

import pyximport
pyximport.install()
# from papers.muzero.player_cy import Node, MinMaxStats, select_child, \
#     expand_node, backpropagate, add_exploration_noise, epsilon_greedy
from papers.muzero.player_cy import Node, MinMaxStats, \
    expand_node, backpropagate, add_exploration_noise, epsilon_greedy


class MCTS:
    # def __init__(self, root: "Node", actions: List[Action]):
    def __init__(self, root):
        self.root = root
        self.min_max_stats = MinMaxStats()
        # self.actions = actions


def batch_mcts_step(hiddens, network, args):
    # q_values = network.policy_net(hiddens)
    roots = []
    assert len(hiddens) == args.batch_size
    for hidden in hiddens:
        root = Node(0)
        # _, policy_logits = prediction

        # For now, just use a uniform policy
        # exp() of any uniform vector is safe
        policy_logits = torch.zeros(args.num_actions)

        # The reward for transitioning to the parent doesn't matter
        reward = 0
        expand_node(root, reward, policy_logits, hidden, args.num_actions)
        # add_exploration_noise(args, root)
        # roots.append(MCTS(root, game.actions))
        roots.append(MCTS(root))

    assert len(roots) == args.batch_size

    for _ in range(args.num_simulations):
        model_inputs = []
        for mcts in roots:
            # history = copy.copy(mcts.actions)
            history = []
            node = mcts.root
            search_path = [node]

            while node.expanded():
                # Select based on UCB score
                # action, node = select_child(node, mcts.min_max_stats)
                # Select uniformly
                action = random.choice(range(len(node.children)))
                node = node.children[action]
                history.append(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the
            # next hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            model_inputs.append((parent.hidden_state, history[-1]))
            mcts.current_node = node
            mcts.current_search_path = search_path

        hiddens, actions = zip(*model_inputs)
        assert len(hiddens) == args.batch_size, len(hiddens)
        assert len(actions) == args.batch_size, len(actions)
        hiddens = torch.stack(hiddens)
        actions = torch.tensor(actions).to(args.device)
        hiddens, rewards = network.dynamics(hiddens, actions)
        # q_valuess = network.policy_net(hiddens)
        q_valuess = network.policy(hiddens)
        rewardss = network.reward_net(hiddens).reshape((args.batch_size, args.num_actions, 100))
        # Lower bound estimate
        # rewardss = rewardss[:, :, :10].mean(dim=2)
        rewardss = rewardss.mean(dim=2)
        assert len(q_valuess) == args.batch_size
        for hidden, reward, q_values, mcts, rewards in zip(hiddens, rewards,
                                                    q_valuess, roots, rewardss):
            # Reward is the reward after taking the action
            # rewards is the rewards of all actions from the new state
            # value, policy_logits = prediction
            # q_values = prediction
            value = q_values.max()

            # For now, just use a uniform policy
            # exp() of any uniform vector is safe
            # TODO: use UCB1 instead of PUCB with constant prior
            # policy_logits = torch.zeros(args.num_actions)

            # TODO: it seems like a problem if we select actions in a manner
            # differently in MCTS than the actual game. If we do this, how
            # are the MCTS values going to represent the true values?
            # expand_node(mcts.current_node, reward, policy_logits, hidden,
            #             args.num_actions)

            # Get the value of the children as if we were there (ie without
            # rewards in the transition to that state, and inversely discounted)
            child_values = (q_values - rewards) / args.discount
            expand_node(mcts.current_node, reward, child_values.cpu().detach(), hidden,
                        args.num_actions)

            child_node = mcts.current_search_path[-1]
            child_node.visit_count += 1
            backpropagate(mcts.current_search_path, value, args.discount,
                          mcts.min_max_stats)

    # We need the root node to use later
    root_nodes = [mcts.root for mcts in roots]

    root_values = []
    action_valuess = []
    visit_countss = []
    for root in root_nodes:
        # values = [child.value() for child in root.children]
        values = [args.discount * child.value(args.discount) + child.reward for child in root.children]
        values = torch.tensor(values)

        visit_counts = [child.visit_count for child in root.children]
        visit_counts = torch.tensor(visit_counts)

        value = root.value(args.discount)

        root_values.append(value)
        action_valuess.append(values)
        visit_countss.append(visit_counts)

    return {
        "root_value": torch.tensor(root_values),
        "action_values": torch.stack(action_valuess),
        "visit_counts": torch.stack(visit_countss),
        "root_nodes": root_nodes,
    }


def batch_mcts_step_dist(hiddens, network, args):
    # q_values = network.policy_net(hiddens)
    roots = []
    assert len(hiddens) == args.batch_size
    for hidden in hiddens:
        root = Node(0)
        # _, policy_logits = prediction

        # For now, just use a uniform policy
        # exp() of any uniform vector is safe
        policy_logits = torch.zeros(args.num_actions)

        # The reward for transitioning to the parent doesn't matter
        reward = 0
        expand_node(root, reward, policy_logits, hidden, args.num_actions)
        # add_exploration_noise(args, root)
        roots.append(MCTS(root))

    assert len(roots) == args.batch_size

    for _ in range(args.num_simulations):
        model_inputs = []
        for mcts in roots:
            history = []
            node = mcts.root
            search_path = [node]

            while node.expanded():
                action, node = select_child(node, mcts.min_max_stats)
                history.append(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the
            # next hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            model_inputs.append((parent.hidden_state, history[-1]))
            mcts.current_node = node
            mcts.current_search_path = search_path

        hiddens, actions = zip(*model_inputs)
        assert len(hiddens) == args.batch_size, len(hiddens)
        assert len(actions) == args.batch_size, len(actions)
        hiddens = torch.stack(hiddens)
        actions = torch.tensor(actions).to(args.device)
        hiddens, rewards = network.dynamics(hiddens, actions)
        # q_valuess = network.policy_net(hiddens)
        q_valuess = network.policy(hiddens)
        assert len(q_valuess) == args.batch_size
        for hidden, reward, q_values, mcts in zip(hiddens, rewards,
                                                    q_valuess, roots):
            # value, policy_logits = prediction
            # q_values = prediction
            value = q_values.max()

            # For now, just use a uniform policy
            # exp() of any uniform vector is safe
            # TODO: use UCB1 instead of PUCB with constant prior
            policy_logits = torch.zeros(args.num_actions)

            # TODO: it seems like a problem if we select actions in a manner
            # differently in MCTS than the actual game. If we do this, how
            # are the MCTS values going to represent the true values?
            expand_node(mcts.current_node, reward, policy_logits, hidden,
                        args.num_actions)

            backpropagate(mcts.current_search_path, value, args.discount,
                          mcts.min_max_stats)

    # We need the root node to use later
    root_nodes = [mcts.root for mcts in roots]

    root_values = []
    action_valuess = []
    visit_countss = []
    for root in root_nodes:
        # values = [child.value() for child in root.children]
        values = [args.discount * child.value() + child.reward for child in root.children]
        values = torch.tensor(values)

        visit_counts = [child.visit_count for child in root.children]
        visit_counts = torch.tensor(visit_counts)

        value = root.value()

        root_values.append(value)
        action_valuess.append(values)
        visit_countss.append(visit_counts)

    return {
        "root_value": torch.tensor(root_values),
        "action_values": torch.stack(action_valuess),
        "visit_counts": torch.stack(visit_countss),
        "root_nodes": root_nodes,
    }
