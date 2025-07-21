from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import os
import akro
import imageio


from gym import utils
import gymnasium as gym
import gymnasium_robotics

import torch
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait

import os
os.environ["MUJOCO_GL"] = "egl"

class FetchEnvironment(gym.Wrapper, MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, *args, custom_order=None, **kwargs):        
        super().__init__(*args, **kwargs)
        self.last_state = None
        self.last_ob = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.custom_order = custom_order
        self.ob_info = dict(
            type='state',
            shape=(25,),  # hardcode shape, don't rely on observation_space property
        )
        
    @staticmethod
    def rearrange_vector(vec, custom_order):
        if isinstance(vec, torch.Tensor):
            indices = torch.tensor(custom_order, device=vec.device, dtype=torch.long)
            return vec[indices]
        elif isinstance(vec, np.ndarray):
            return vec[custom_order]
        elif isinstance(vec, list):
            return [vec[i] for i in custom_order]
        else:
            raise TypeError("Unsupported type for vec. Must be torch.Tensor, numpy.ndarray, or list.")

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)

    def get_state(self, state):
        vector = np.asarray(state)
        if self.custom_order is not None:
            vector = self.rearrange_vector(vector, self.custom_order)
        return vector

    def reset(self):
        state, _ = super().reset()
        ob = self.get_state(state['observation'])


        self.last_state = state
        self.last_ob = ob
        
        return ob
    
    def step(self, action, render=False):
        next_state, reward, terminated, truncated, info = super().step(action)

        done = terminated or truncated
        ob = self.get_state(next_state['observation'])

        coords = self.last_state['observation'][:2].copy()
        next_coords = next_state['observation'][:2].copy()

        info['coordinates'] = coords
        info['next_coordinates'] = next_coords
        info['ori_obs'] = self.last_state['observation']
        info['next_ori_obs'] = next_state['observation']

        if render:
            info['render'] = self.render().transpose(2, 0, 1)

        self.last_state = next_state
        self.last_ob = ob

        return ob, reward, done, info
    

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
        eval_metrics = {}

        # goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
        # sum_successes = 0

        # for i, goal_name in enumerate(goal_names):
        #     goal_key = f'metric_success_task_relevant/goal_{i}'
        #     success = 0
        #     for traj in trajectories:
        #         env_infos = traj['env_infos']
        #         # Case 1: dict of lists
        #         if isinstance(env_infos, dict):
        #             vals = env_infos.get(goal_key, [0])
        #             success = max(success, max(vals))
        #         # Case 2: list of dicts
        #         elif isinstance(env_infos, list):
        #             vals = [info.get(goal_key, 0) for info in env_infos if isinstance(info, dict)]
        #             if vals:
        #                 success = max(success, max(vals))
        #     eval_metrics[f'KitchenTask{goal_name}'] = success
        #     sum_successes += success

        # eval_metrics[f'KitchenOverall'] = sum_successes
        return eval_metrics


# Create base environment
# base_env = gym.make('FetchPickAndPlace-v3', max_episode_steps=150)

# # Optional: a custom observation index order
# custom_order = np.arange(25)  # identity mapping for now

# # Wrap it with your custom FetchEnvironment
# env = FetchEnvironment(base_env, custom_order=custom_order)

# # Reset the environment
# obs = env.reset()
# print("Initial observation shape:", obs.shape)

# # Take a random action
# action = env.action_space.sample()
# next_obs, reward, done, info = env.step(action)

# print("Next observation shape:", next_obs.shape)
# print("Reward:", reward)
# print("Done:", done)
# print("Info keys:", info.keys())