import argparse
import torch

import gym

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def num_actions(game):
    env = gym.make(game)
    n = env.action_space.n
    print(f"env has {n} actions")
    return n

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    # parser.add_argument('--game', default="PongNoFrameskip-v4")
    # parser.add_argument('--game', default="procgen:procgen-zackbigfish-v0")
    # parser.add_argument('--game', default="procgen:procgen-zackchaser-v0")
    parser.add_argument('--game', default="RiverraidNoFrameskip-v4")
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--greyscale', default=True, type=str2bool)
    parser.add_argument('--actions_in_obs', default=False, type=str2bool)
    parser.add_argument('--ray_address', default=None, type=str)
    parser.add_argument('--project', default="muzero", type=str)
    parser.add_argument('--device', default="cuda:0")

    # Ray stuff - ignore
    parser.add_argument('--node-ip-address', default=None, type=str)
    parser.add_argument('--object-store-name', default=None, type=str)
    parser.add_argument('--raylet-name', default=None, type=str)
    parser.add_argument('--redis-address', default=None, type=str)
    parser.add_argument('--temp-dir', default=None, type=str)

    # Muzero config
    # TODO: make sure all these are used somewhere

    # Env settings
    # parser.add_argument('--action_space_size', default=18, type=int)
    parser.add_argument('--max_moves', default=27000, type=int)  # 30 mins at repeat 4
    parser.add_argument('--noop_on_reset', default=True, type=str2bool)
    parser.add_argument('--repeat_last_frame', default=0, type=int)

    # Player settings
    parser.add_argument('--clear_old_obs', default=True, type=str2bool)



    parser.add_argument('--discount', default=.997, type=float)
    parser.add_argument('--root_dirichlet_alpha', default=.25, type=float)
    parser.add_argument('--num_simulations', default=50, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_vec_env', default=1, type=int)
    parser.add_argument('--td_steps', default=10, type=int)
    # parser.add_argument('--num_workers', default=350, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--lr_init', default=.05, type=float)
    parser.add_argument('--lr_decay_steps', default=350e3, type=float)
    parser.add_argument('--lr_decay_rate', default=.1, type=float)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--root_exploration_fraction', default=.25, type=float)
    parser.add_argument('--training_steps', default=int(1e6), type=int)
    parser.add_argument('--window_size', default=int(1e6), type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--num_unroll_steps', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--pb_c_base', default=19652, type=float)
    parser.add_argument('--pb_c_init', default=1.25, type=float)
    parser.add_argument('--n_framestack', default=4, type=int)
    parser.add_argument('--steps_to_skip', default=4, type=int)
    parser.add_argument('--local', default=False, type=str2bool)
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--bootstrapping', default=False, type=str2bool)

    # DQN options
    parser.add_argument('--off_policy_target', default=False, type=str2bool)
    parser.add_argument('--all_unroll_images', default=False, type=str2bool)
    parser.add_argument('--double_q', default=False, type=str2bool)

    # Replay buffer options
    parser.add_argument('--replay_buffer_alpha', default=.7, type=float)
    parser.add_argument('--prioritized_replay', default=False, type=str2bool)
    parser.add_argument('--compression', default=True, type=str2bool)


    args = parser.parse_args(args)
    args.device = torch.device(args.device)

    args.num_actions = num_actions(args.game)

    if args.greyscale:
        args.color_channels = 1
    else:
        args.color_channels = 3
    args.image_channels = args.n_framestack * (args.color_channels + args.actions_in_obs)
    return args
