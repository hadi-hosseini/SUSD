from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import os
import akro

from gym import utils
import gymnasium as gym
import gymnasium_robotics

import torch
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait
from gymnasium_robotics.envs.franka_kitchen import KitchenEnv


# env = gym.make(
#     'FrankaKitchen-v1',
#     tasks_to_complete=['microwave', 'kettle'],
#     terminate_on_tasks_completed=True,
# )

class KitchenFranka(KitchenEnv):
    def __init__(self, *args, custom_order=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_state = None
        self.last_ob = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.custom_order = custom_order
        self.ob_info = dict(
            type='state',
            shape=(59,),  # hardcode shape, don't rely on observation_space property
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

    def get_state(self, state):
        if isinstance(state, dict) and 'state_vector' in state:
            vector = state['state_vector']
        else:
            vector = state
        
        vector = np.asarray(vector)
        assert vector.shape == (59,), f"Expected state vector shape (59,), got {vector.shape}"

        return vector

    def step(self, action):
        return super().step(action)


kf = KitchenFranka(custom_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40, 41, 42, 43, 44, 45, 46, 47, 48, 29, 30, 31, 49, 50, 51, 32, 52, 33, 34, 35, 36, 37, 38, 39, 53, 54, 55, 56, 57, 58])
print(kf.observation_space)
print(kf.action_space)
print(kf.custom_order)
