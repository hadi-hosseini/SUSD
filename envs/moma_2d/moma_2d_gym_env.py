"""
This is a gym env of MoMa2DEnv
"""

import gym
from gym import spaces
import matplotlib.pyplot as plt

import numpy as np
import torch
import math
from envs.mujoco.mujoco_utils import MujocoTrait



def to_one_hot(idx, num_classes):
	one_hot = np.zeros(num_classes, dtype=int)
	if idx < 0:
		return one_hot
	one_hot[idx] = 1
	return one_hot


class MoMa2DGymEnv(MujocoTrait, gym.Env):
	"""Custom Environment that follows gym interface"""
	def __init__(self, max_step, custom_order, show_empty=None):
		super(MoMa2DGymEnv, self).__init__()
		# We hard-code these values
		limit = 4.99
		stochastic = False

		self.max_step = max_step
		self.stochastic = stochastic
		self.custom_order = custom_order
		self.last_obs = None
		self.last_agent_pos = None
		self.action_range = np.array([0.3, 0.3, 0.15, 0.15, 0.15, 0.8], dtype=np.float32)
		self.limit = limit
		self.arm_pos_limit = limit / 3
		self.gripper_limit = 1.0
		self.view_limit = np.pi

		self.action_space = spaces.Box(low=-self.action_range, high=self.action_range, shape=(6,), dtype=np.float32)

		obs_low = np.array([0]*17 + [-self.view_limit], dtype=np.float32)
		obs_high = np.array([1]*12 + [self.limit]*4 + [self.gripper_limit, self.view_limit], dtype=np.float32)
		self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(18,), dtype=np.float32)

		# Visualize rotated to left by 90 degrees
		# [4v, 4a, 0, 2a, 2v],
		# [3a, 2, 1a, 4, 3a],
		# [0, 2a, 0, 4a, 0],
		# [1a, 1, 3a, 3, 1a],
		# [3v, 4a, 0, 2a, 1v],

		self.arm_map = np.array([
			[0, 3, 0, 1, 0],
			[4, 0, 2, 0, 4],
			[0, 1, 0, 3, 0],
			[2, 0, 4, 0, 2],
			[0, 3, 0, 1, 0],
		])

		# Arm Location:
		self.base_map = np.array([
			[0, 0, 0, 0, 0],
			[0, 1, 0, 2, 0],
			[0, 0, 0, 0, 0],
			[0, 3, 0, 4, 0],
			[0, 0, 0, 0, 0],
		])
		self.reset()

	def get_grasp_obj(self, arm_pos):
		int_arm_pos = arm_pos.astype(int)
		arm_idx = self.arm_map[int_arm_pos[0], int_arm_pos[1]]
		return to_one_hot(arm_idx - 1, 4)

	def get_base_obj(self, agent_pos):
		int_pos = agent_pos.astype(int)
		base_idx = self.base_map[int_pos[0], int_pos[1]]
		return to_one_hot(base_idx - 1, 4)

	def get_view_obj(self, view):
		# We assume that 0 is facing up
		# First: the target object can only be in the wrapperrant that the agent is facing
		if view > 0:
			if view > np.pi / 2:
				target = np.array([self.limit, 0])
				idx = 0
			else:
				target = np.array([self.limit, self.limit])
				idx = 1
		else:
			if view < -np.pi / 2:
				target = np.array([0, 0])
				idx = 2
			else:
				target = np.array([0, self.limit])
				idx = 3

		# Step 1: calculate the angle between target and agent pos
		relative_target = target - self.agent_pos
		angle = math.atan2(relative_target[0], relative_target[1])
		d_angle = np.abs(angle - view)
		if d_angle > np.pi / 10:
			idx = -1
		return to_one_hot(idx, 4)
	

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


	def step(self, action, render=False):
		"""
		No reward for this env
		"""
		action = action.flatten()
		base_action = action[:2]
		arm_action = action[2:4]
		gripper_action = action[4]
		head_action = action[5]

		if self.stochastic:
			raise NotImplementedError
			NOISE_LEVEL = 1
			self.agent_pos += np.random.normal(scale=NOISE_LEVEL, size=self.agent_pos.shape[0])

		# Update agent state
		self.agent_pos = np.clip(self.agent_pos + base_action, 0, self.limit)
		self.relative_arm_pos = np.clip(self.relative_arm_pos + arm_action, -self.arm_pos_limit, self.arm_pos_limit)
		self.arm_pos = np.clip(self.agent_pos + self.relative_arm_pos, 0, self.limit)
		self.view = np.clip(self.view + head_action, -self.view_limit, self.view_limit)
		self.gripper_location = np.clip(self.gripper_location + gripper_action, 0, self.gripper_limit)

		if self.gripper_location >= 0.5:
			# Simulate grasping an object
			self.arm_item = self.get_grasp_obj(self.arm_pos)

		self.base_item = self.get_base_obj(self.agent_pos)
		self.view_item = self.get_view_obj(self.view)

		self.step_count += 1
		done = self.step_count >= self.max_step

		observation = [*self.base_item, *self.arm_item, *self.view_item,
					   *self.agent_pos, *self.arm_pos, self.gripper_location, self.view].copy()

		self.agent_traj.append(observation)

		reward = self.get_reward()

		observation = self.permute_state(observation)
		observation = observation.reshape(-1) # add

		coords = self.last_agent_pos
		next_coords = self.agent_pos
		
		infos = {}
		infos['coordinates'] = coords
		infos['next_coordinates'] = next_coords
		infos['ori_obs'] = self.last_obs
		infos['next_ori_obs'] = observation

		if render:
			infos['render'] = self.render().transpose(2, 0, 1)

		self.last_obs = observation
		self.last_agent_pos = self.agent_pos

		return observation, reward, done, infos

	# This function can be rewrote by downstream tasks
	def get_reward(self):
		return 0

	def reset(self, seed=None):
		self.step_count = 0
		self.agent_traj = []
		self.agent_pos = np.array([self.limit / 2, self.limit / 2], dtype=np.float32)
		self.arm_pos = self.agent_pos
		self.relative_arm_pos = np.array([0.0, 0.0], dtype=np.float32)
		self.gripper_location = 0.5
		self.view = 0.0
		self.arm_item = np.zeros(4, dtype=int)

		if self.gripper_location >= 0.5:
			# Simulate grasping an object
			self.arm_item = self.get_grasp_obj(self.arm_pos)

		self.base_item = self.get_base_obj(self.agent_pos)
		self.view_item = self.get_view_obj(self.view)
		obs = [*self.base_item, *self.arm_item, *self.view_item,
					   *self.agent_pos, *self.arm_pos, self.gripper_location, self.view].copy()


		
		obs = self.permute_state(obs)
		obs = obs.reshape(-1) # add
		
		self.last_obs = obs
		self.last_agent_pos = self.agent_pos

		

		return obs

	def save_traj(self, fn):
		plt.clf()
		plt.xlim(0, self.limit)
		plt.ylim(0, self.limit)
		agent_traj = np.array(self.agent_traj)
		skip = 1
		plt.plot(agent_traj[::skip, 12], agent_traj[::skip, 13])
		plt.plot(agent_traj[::skip, 14], agent_traj[::skip, 15], c='r')
		plt.title(self.step_count)
		# plt.show()
		plt.savefig(fn)

	def render(self, mode='rgb_array', block=False):
		plt.clf()
		frame1 = plt.gca()
		frame1.axes.xaxis.set_ticklabels([])
		frame1.axes.yaxis.set_ticklabels([])
		plt.xlim(0, self.limit)
		plt.ylim(0, self.limit)
		plt.scatter(self.agent_pos[0], self.agent_pos[1], marker="*", s=600)
		plt.scatter(self.arm_pos[0], self.arm_pos[1], c='r', marker="d", s=150)

		plt.scatter(self.limit - 0.5, self.limit - 0.5, marker="x", s=400)

		view_scale = 0.8
		plt.plot([self.agent_pos[0], self.agent_pos[0] + view_scale * math.sin(self.view)],
				 [self.agent_pos[1], self.agent_pos[1] + view_scale * math.cos(self.view)] , c='b')
		plt.title(self.step_count)
		if mode == 'human':
			plt.show(block=block)
			plt.pause(0.00001)
			return None
		elif mode == 'rgb_array':
			fig = plt.gcf()
			fig.canvas.draw()
			rgba_buf = fig.canvas.buffer_rgba()
			w, h = fig.canvas.get_width_height()
			rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].copy()
			return rgba_arr
		else:
			raise NotImplementedError


	def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
		eval_metrics = {}
		return eval_metrics