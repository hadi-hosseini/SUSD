import datetime
import os
import socket
import sys
import gymnasium as gym
import torch
import numpy as np
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.modules.parameter_module import ParameterModule


from garage.experiment.experiment import get_metadata
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import (CentralizedWrapper,
                                                               DownstreamCentralizedWrapper,
                                                               SequentialDSWrapper)
from envs.mp.particle import Particle

import global_context

EXP_DIR = 'exp'
g_start_time = int(datetime.datetime.now().timestamp())


def get_run_env_dict():
    d = {}
    d['timestamp'] = datetime.datetime.now().timestamp()
    d['hostname'] = socket.gethostname()
    if 'SLURM_JOB_ID' in os.environ:
        d['slurm_job_id'] = int(os.environ['SLURM_JOB_ID'])
    if 'SLURM_PROCID' in os.environ:
        d['slurm_procid'] = int(os.environ['SLURM_PROCID'])
    if 'SLURM_RESTART_COUNT' in os.environ:
        d['slurm_restart_count'] = int(os.environ['SLURM_RESTART_COUNT'])

    git_root_path, metadata = get_metadata()
    # get_metadata() does not decode git_root_path.
    d['git_root_path'] = git_root_path.decode('utf-8') if git_root_path is not None else None
    d['git_commit'] = metadata.get('githash')
    d['launcher'] = metadata.get('launcher')

    return d

def get_exp_name(args):
    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    exp_name += '_' + args.env
    exp_name += '_' + args.algo

    return exp_name, exp_name_prefix

def get_log_dir(args):
    exp_name, exp_name_prefix = get_exp_name(args)
    assert len(exp_name) <= os.pathconf('/', 'PC_NAME_MAX')
    # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
    log_dir = os.path.realpath(os.path.join(EXP_DIR, args.run_group, exp_name))
    assert not os.path.exists(log_dir), f'The following path already exists: {log_dir}'

    return log_dir


def make_env(args, max_path_length):
    if args.env == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)
    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100)
    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder  # Only support pixel-based environments
        if args.env == 'dmc_cheetah':
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_humanoid':
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif args.env == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        # assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)

    elif args.env == "kitchen_franka":
        from envs.mujoco.kitchen_franka import KitchenFranka
        from gymnasium_robotics.envs.franka_kitchen import KitchenEnv

        all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
        custom_order = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,     # Robot
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Switches
                28, 29, 30, 49, 50, 51,                                           # Cabinets
                31, 52,                                                          # Microwave
                32, 33, 34, 35, 36, 37, 38, 53, 54, 55, 56, 57, 58               # Kettle
        ]
        
        base_env = KitchenEnv(
            tasks_to_complete=all_tasks,
            terminate_on_tasks_completed=False,
            render_mode="rgb_array"
        )

        env = KitchenFranka(base_env, custom_order=custom_order)

    elif args.env == "fetch":
        from envs.mujoco.fetch import FetchEnvironment

        custom_order = [
                    0, 1, 2,      # Gripper position
                    9, 10,        # Finger joint positions
                    20, 21, 22,   # Gripper linear velocities
                    23, 24,       # Finger linear velocities
                    3, 4, 5,      # Puck global position
                    6, 7, 8,      # Puck relative position to gripper
                    11, 12, 13,   # Puck global rotation (Euler angles)
                    14, 15, 16,   # Puck relative linear velocity
                    17, 18, 19    # Puck angular velocity
                ]

        base_env = gym.make('FetchPickAndPlace-v3', max_episode_steps=150, render_mode="rgb_array")
        env = FetchEnvironment(base_env, custom_order=custom_order)

    elif args.env == "particle":        
        env = simple_heterogenous_v3.parallel_env(
            render_mode= "rgb_array",
            max_cycles=1000,
            continuous_actions=True,
            local_ratio=0,
            N=10,
            img_encoder=None)

        env = CentralizedWrapper(env, simplify_action_space=True)

        distances = list(range(0, 10))       # 0–9
        agent_info = list(range(10, 50))     # 10–49
        station_info = list(range(50, 70))   # 50–69

        custom_order = []

        for i in range(10):
            custom_order.append(distances[i])                       
            custom_order.extend(agent_info[i*4:(i+1)*4])            
            custom_order.extend(station_info[i*2:(i+1)*2])

        env = Particle(env, custom_order, (512, 480))
        env.reset(seed=args.seed)

    else:
        raise NotImplementedError
    
    
    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    return env


def make_q_function(input_dim, action_dim, master_dims, nonlinearity, alpha):
    qf1 = ContinuousMLPQFunctionEx(
            obs_dim=input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
    log_alpha = ParameterModule(torch.Tensor([np.log(alpha)]))
    return qf1, log_alpha
