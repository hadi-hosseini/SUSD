from envs.elden_kitchen.kitchen import Kitchen
from envs.elden_kitchen.kitchen_downstream import KitchenDownStreamTask

import copy
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

import torch
import imageio
import numpy as np

import os
os.environ["MUJOCO_GL"] = "egl"

from envs.mujoco.mujoco_utils import MujocoTrait
from contextlib import contextmanager

class EldenKitchen(GymWrapper, MujocoTrait):
    def __init__(self, env, custom_order=None):
        super().__init__(env)
        self.custom_order = custom_order
        self.last_obs = None

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

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = self.permute_state(obs)
        self.last_obs = obs
        return obs

    def step(self, action, render=False):
        next_obs, reward, done, info = super().step(action)
        next_obs = self.permute_state(next_obs)

        #### THIS SHOULD BE FIXED LATER!!!!
        coords = self.last_obs[3:5].copy()
        next_coords = next_obs[3:5].copy()
        info['coordinates'] = coords
        info['next_coordinates'] = next_coords
        info['ori_obs'] = self.last_obs
        info['next_ori_obs'] = next_obs

        self.last_obs = next_obs
        return next_obs, reward, done, info

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
        eval_metrics = {}

        return eval_metrics
    

def elden_kitchen(reward_scale=0.0, horizon=50, render=False, downstream_task=False): # reward_sacle = 1.0 is used for downstream task
    manipulation_env_params = {
                "robots": "UR5e",
                "controller_name": "OSC_POSITION",
                "gripper_types": "RethinkGripper",
                "control_freq": 20,
                "reward_scale": reward_scale,
                "block_env_params": {
                    "dynamics_keys": ["robot0_eef_pos", "robot0_eef_vel", "robot0_gripper_qpos", "robot0_gripper_qvel",
                                    "mov0_pos", "mov0_quat"],
                    "policy_additional_keys": [],
                    "horizon": horizon,
                    "num_movable_objects": 1,
                    "cube_x_range": [-0.25, 0.25],
                    "cube_y_range": [-0.25, 0.25],
                    "table_full_size": [0.8, 1.2, 0.05],
                    "table_offset": [0.0, 0.0, 0.8],
                    "normalization_range": [[-0.5, -0.5, 0.7], [0.5, 0.5, 1.1]]
                },
                "kitchen_env_params": {
                    "horizon": horizon,
                    "dynamics_keys": ["robot0_eef_pos", "robot0_gripper_qpos",
                                    "butter_pos", "butter_quat", "butter_melt_status",
                                    "meatball_pos", "meatball_cook_status", "meatball_overcooked",
                                    "pot_pos", "pot_quat",
                                    "stove_pos", "target_pos",
                                    "button_pos", "button_joint_qpos"],
                    "policy_additional_keys": ["robot0_eef_vel", "robot0_gripper_qvel",
                                            "butter_to_robot0_eef_pos", "butter_to_robot0_eef_quat", "butter_grasped",
                                            "meatball_to_robot0_eef_pos", "meatball_grasped",
                                            "pot_to_robot0_eef_pos", "pot_to_robot0_eef_quat", "pot_grasped",
                                            "stove_to_robot0_eef_pos", "target_to_robot0_eef_pos",
                                            "pot_handle_pos", "pot_handle_to_robot0_eef_pos",
                                            "button_handle_pos", "button_handle_to_robot0_eef_pos", "button_touched"],
                    "butter_x_range": [-0.25, -0.15],
                    "butter_y_range": [-0.3, -0.2],
                    "meatball_x_range": [-0.15, -0.05],
                    "meatball_y_range": [-0.3, -0.2],
                    "pot_x_range": [-0.20, -0.10],
                    "pot_y_range": [-0.10, -0.00],
                    "button_x_range": [-0.2, -0.05],
                    "button_y_range": [0.15, 0.3],
                    "stove_x_range": [0.05, 0.25],
                    "stove_y_range": [-0.225, -0.1],
                    "target_x_range": [0.05, 0.25],
                    "target_y_range": [0.1, 0.2],
                    "table_full_size": [0.8, 1.2, 0.05],
                    "table_offset": [0.0, 0.0, 0.8],
                    "normalization_range": [[-0.5, -0.6, 0.7], [0.5, 0.6, 1.2]]
                }
            }

    env_name = "kitchen"

    env_kwargs = copy.deepcopy(manipulation_env_params[env_name + "_env_params"])
    env_kwargs.pop("dynamics_keys")
    env_kwargs.pop("policy_additional_keys")

    if downstream_task:
        env = KitchenDownStreamTask(robots=manipulation_env_params["robots"],
                    controller_configs=load_controller_config(default_controller=manipulation_env_params["controller_name"]),
                    gripper_types=manipulation_env_params["gripper_types"],
                    has_renderer=False,
                    has_offscreen_renderer=render,
                    use_camera_obs=False,
                    ignore_done=False,
                    control_freq=manipulation_env_params["control_freq"],
                    reward_scale=manipulation_env_params["reward_scale"],
                    **env_kwargs)
    else:
        env = Kitchen(robots=manipulation_env_params["robots"],
                        controller_configs=load_controller_config(default_controller=manipulation_env_params["controller_name"]),
                        gripper_types=manipulation_env_params["gripper_types"],
                        has_renderer=False,
                        has_offscreen_renderer=render,
                        use_camera_obs=False,
                        ignore_done=False,
                        control_freq=manipulation_env_params["control_freq"],
                        reward_scale=manipulation_env_params["reward_scale"],
                        **env_kwargs)
                    
    return env


@contextmanager
def kitchen_env(custom_order, reward_scale=1.0, horizon=50, render=True):
    env = None
    try:
        base_env = elden_kitchen(reward_scale=reward_scale, horizon=horizon, render=render)
        env = EldenKitchen(base_env, custom_order=custom_order)
        yield env
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Error closing env: {e}")


# env = elden_kitchen(reward_scale=1.0, horizon=50.0, render=True)

# custom_order = list(range(0, 128))
# env = EldenKitchen(env, custom_order=custom_order)

# print(env.action_space)
# print(env.observation_space)

# frames = []
# obs = env.reset()

# for i in range(50):
#     frame = env.render()
#     frames.append(frame)
    
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)

#     print(f"Step {i}:")
#     print(f"  Reward: {reward}")
#     print(f"  Done: {done}")
        
#     if done:
#         print("Episode finished!")
#         obs = env.reset()
#         break


# video_path = "envs/elden_kitchen/kitchen.mp4"
# imageio.mimsave(video_path, frames, fps=30)
# print(f"🎞️ Video saved to: {video_path}")

# env.close()