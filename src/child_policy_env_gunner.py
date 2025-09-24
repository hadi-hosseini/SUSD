from collections import defaultdict

import akro
import gym.spaces.utils
import numpy as np
import torch

from garage.envs import EnvSpec
from garage.torch.distributions import TanhNormal
from src.dusdi_utils import Actor

from iod.utils import get_torch_concat_obs


class ChildPolicyEnvGunner(gym.Wrapper):
    def __init__(
            self,
            env,
            cp_dict,
            cp_action_range,
            cp_unit_length,
            cp_multi_step,
            cp_num_truncate_obs,
            cp_omit_obs_idxs=None,
            cp_multitask = False,
            cp_discrete = False
    ):
        super().__init__(env)

        mode = 0 # 0:susd   1:dsd  2:dusdi
        self.mode = mode

        if mode == 2: # dusdi
            self.child_policy = Actor("state", 38, 6, 20, 1024, True, [-10, 2], "moma2D")
            self.child_policy.load_state_dict(cp_dict)
            self.cp_dim_action = 5
            self.cp_N = 4
            self.cp_discrete = True

        else:
            self.cp_dim_action = cp_dict['dim_option']
            self.child_policy = cp_dict['policy']
            self.cp_discrete = cp_dict['discrete']

            if mode == 0: # susd
                self.cp_N = getattr(cp_dict, 'N', 4) # susd (3)
                self.cp_discrete = cp_discrete # DISCRETE

            else: # dsd-baselines
                self.cp_N = getattr(cp_dict, 'N', 1) # baselines

        self.child_policy.eval()
        
        self.cp_action_range = cp_action_range
        self.cp_unit_length = cp_unit_length
        self.cp_multi_step = cp_multi_step
        self.cp_num_truncate_obs = cp_num_truncate_obs
        self.cp_omit_obs_idxs = cp_omit_obs_idxs
        self.cp_multitask = cp_multitask

        self.observation_space = self.env.observation_space

        if self.cp_discrete:
            self.action_space = akro.Box(low=0, high=1, shape=(self.cp_dim_action * self.cp_N,), dtype=np.int8)
        else:
            self.action_space = akro.Box(low=-1., high=1., shape=(self.cp_dim_action * self.cp_N,))

    @property
    def spec(self):
        return EnvSpec(action_space=self.action_space,
                       observation_space=self.observation_space)


    def get_full_state(self, obs):
        full_obs = np.concatenate([obs, self.env.get_additional_states()])
        return full_obs
    
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.cycle_reward = 0
        self.step_count = 0
        self.last_obs = observation
        return self.get_full_state(observation)


    def step(self, cp_action, **kwargs):
        cp_action_norm = np.linalg.norm(cp_action)
        cp_action = cp_action.copy()
        if not self.cp_discrete:
            if self.cp_unit_length:
                cp_action = cp_action / cp_action_norm
            else:
                cp_action = cp_action * self.cp_action_range
        else:
            cp_action = (cp_action > 0.5).astype(np.int8) 

        for i in range(self.cp_multi_step):
            self.step_count += 1
            cp_obs = self.last_obs
            cp_obs = torch.as_tensor(cp_obs)
            if self.cp_num_truncate_obs > 0:
                cp_obs = cp_obs[:-self.cp_num_truncate_obs]
            if self.cp_omit_obs_idxs is not None:
                cp_obs[self.cp_omit_obs_idxs] = 0

            cp_action = torch.as_tensor(cp_action)
            cp_input = get_torch_concat_obs(cp_obs, cp_action, dim=0).float()

            # dusdi
            if self.mode == 2:
                action_dist = self.child_policy(cp_input.unsqueeze(dim=0))
                action = action_dist.mean.detach().numpy()
                action = action[0]


            else: # First try to use mode
                if hasattr(self.child_policy._module, 'forward_mode'):
                    # Beta
                    action = self.child_policy.get_mode_actions(cp_input.unsqueeze(dim=0))[0]
                else:
                    # Tanhgaussian
                    action_dist = self.child_policy(cp_input.unsqueeze(dim=0))[0]
                    action = action_dist.mean.detach().numpy()
                action = action[0]  

            # Assume that the range of the variable 'action' (= the output from self.child_policy) is [-1, 1]
            # This assumption is probably true as of now (since we only use (scaled) Beta or TanhGaussian policy)
            lb, ub = self.env.action_space.low, self.env.action_space.high
            action = lb + (action + 1) * (0.5 * (ub - lb))
            action = np.clip(action, lb, ub)

            observation, r, done, info = self.env.step(action, **kwargs)

            self.cycle_reward += r
            if self.step_count % self.cp_multi_step == 0:
                reward = self.cycle_reward / self.cp_multi_step
                self.cycle_reward = 0
            else:
                reward = 0

            self.last_obs = observation

        return self.get_full_state(observation), reward, done, info
    

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
        eval_metrics = {}
        return eval_metrics
