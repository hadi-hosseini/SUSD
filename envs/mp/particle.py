from PIL import Image
import numpy as np
import torch

from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper
from envs.mujoco.mujoco_utils import MujocoTrait


class Particle(MujocoTrait, CentralizedWrapper):
    def __init__(self, env, custom_order, frame_size):
        super().__init__(env)
        self.env = env
        self.frame_size = frame_size
        self.custom_order = custom_order
        self.last_obs = None

    def render(self, mode='human'):
        frame = self._env.render()
        img = Image.fromarray(frame)
        img = img.resize(self.frame_size, Image.BILINEAR)


        margin_size = 10
        new_width = self.frame_size[0] + 2 * margin_size
        new_height = self.frame_size[1] + 2 * margin_size
        new_img = Image.new("RGB", (new_width, new_height), color=(0, 0, 0))  # black margin
        new_img.paste(img, (margin_size, margin_size))
        resized_frame_with_margin = np.array(new_img)

        return resized_frame_with_margin
    

    def permute_state(self, state):
        vector = np.asarray(state)
        if self.custom_order is not None:
            vector = self.rearrange_vector(vector, self.custom_order)
        return vector
    

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

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        obs = self.permute_state(obs)
        self.last_obs = obs

        return obs
    
    def step(self, action, render=False):
        obs, rewards, done, infos = self.env.step(action)
        obs = self.permute_state(obs)

        coords = self.last_obs[3:5].copy()
        next_coords = obs[3:5].copy()

        infos['coordinates'] = coords
        infos['next_coordinates'] = next_coords
        infos['ori_obs'] = self.last_obs
        infos['next_ori_obs'] = obs

        if render:
            infos['render'] = self.render().transpose(2, 0, 1)

        self.last_obs = obs

        return obs, rewards, done, infos
    

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