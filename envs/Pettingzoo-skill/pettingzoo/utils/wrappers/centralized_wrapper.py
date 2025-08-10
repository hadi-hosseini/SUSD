"""
Convert mpe to gym environment
"""

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np
import gymnasium
import torch


class CentralizedWrapper(gym.Env):
	def __init__(self, env, simplify_action_space=True):
		self._env = env
		self.simplify_action_space = simplify_action_space
		self.initialize_action_space()
		self.observation_space = env.state_space
		self.agent_name = self._env.possible_agents[0]
		# assert self._env.unwrapped.local_ratio == 0, "local_ratio must be 0"

	def initialize_action_space(self):
		dict_act_space = self._env.action_spaces
		low_action_range = []
		high_action_range = []

		if self.simplify_action_space:
			for val in dict_act_space.values():
				# Just 2D action space which controls the x and y velocity
				assert isinstance(val, gymnasium.spaces.Box)
				low_action_range.append(np.ones(2) * -1)
				high_action_range.append(np.ones(2))
			low_action_range = np.concatenate(low_action_range)
			high_action_range = np.concatenate(high_action_range)
		else:
			for val in dict_act_space.values():
				assert isinstance(val, gymnasium.spaces.Box)
				low_action_range.append(val.low)
				high_action_range.append(val.high)
			low_action_range = np.concatenate(low_action_range)
			high_action_range = np.concatenate(high_action_range)

		self.action_space = spaces.Box(
			low=low_action_range, high=high_action_range, shape=low_action_range.shape, dtype=np.float32)

	def reset(self, seed=None):
		_, _ = self._env.reset(seed)
		return self._env.state()

	def action_transform(self, action):
		if self.simplify_action_space:
			tf_action = np.zeros(5, dtype=np.float32)
			tf_action[0] = 0.5
			if action[0] > 0:
				tf_action[1] = action[0]
				tf_action[2] = 0
			else:
				tf_action[1] = 0
				tf_action[2] = -action[0]

			if action[1] > 0:
				tf_action[3] = action[1]
				tf_action[4] = 0
			else:
				tf_action[3] = 0
				tf_action[4] = -action[1]
		else:
			tf_action = action
		return tf_action

	def step(self, action):
		# Loop through each agent and assign action
		# We assume each agent has the same action space
		actions = np.split(action, len(self._env.agents))
		actions = {agent:self.action_transform(act)  for agent, act in zip(self._env.agents, actions)}
		_, rewards, terminations, truncations, infos = self._env.step(actions)

		done = terminations[self.agent_name] or truncations[self.agent_name]
		rewards = rewards[self.agent_name]
		return self._env.state(), rewards, done, infos

	def render(self, mode='human'):
		return self._env.render()

	# # THIS IS A HACK (It's compatible for DUSDi not us)
	# def plot_prediction_net(self, agent, cfg, step=0, device="cuda", anti=False, SHOW=False):
	# 	assert agent.domain == "particle"
	# 	N = cfg.env.particle.N
	# 	# So we want to test: for each vector, what are the predicted skill
	# 	vectors = np.linspace(0.05, 2.00, 16)
	# 	possible_vectors = [[v] for v in vectors]
	# 	prediction = []
	# 	for vec in possible_vectors:
	# 		with torch.no_grad():
	# 			test_obs = torch.tensor(vec*N, device=device, dtype=torch.float32)
	# 			if anti:
	# 				predicted_z = torch.softmax(agent.anti_diayn(None, torch.tensor(test_obs, device=device)).
	# 											reshape(cfg.agent.skill_channel, -1),
	# 											dim=-1)
	# 			else:
	# 				predicted_z = torch.softmax(agent.diayn(None, torch.tensor(test_obs, device=device)).
	# 											reshape(cfg.agent.skill_channel, -1),
	# 											dim=-1)
	# 		prediction.append(predicted_z.cpu().numpy())

	# 	apd = f"_step_{step}"
	# 	if anti:
	# 		apd += "_anti"

	# 	import sys
	# 	np.set_printoptions(threshold=sys.maxsize)

	# 	# Save in the right format (group by subskill)
	# 	text = np.array2string(np.array(prediction).swapaxes(0, 1), precision=2, suppress_small=True)
	# 	with open(f"pred{apd}.txt", "w") as text_file:
	# 		text_file.write(text)

	def __getattr__(self, name):
		return getattr(self._env, name)

class DownstreamCentralizedWrapper(CentralizedWrapper):
	"""
	Centralized wrapper that is responsible for downstream tasks
	Takes in a list of landmark ids that are used to generate reward
	Defines the food poison environment
	"""
	def __init__(self, env, landmark_id, N, factorize, simplify_action_space=True):
		self._env = env
		self.N = N
		self.factored = factorize
		self.distance_threshold = 0.6
		# We want to have binary indicator for each episode / each timestep
		# close or far from the landmark
		self.landmark_id = landmark_id
		self.simplify_action_space = simplify_action_space

		self.initialize_parameters()
		self.initialize_action_space()
		self.initialize_state_space()

	def initialize_parameters(self):
		self.cycle_step = 50
		self.agent_name = self._env.possible_agents[0]
		assert self._env.unwrapped.local_ratio == 0, "local_ratio must be 0"

	def initialize_state_space(self):
		state_dim = self._env.state_space.shape[0] + self.N + 1 # We have an additional indicator variable, plus time counter
		self.observation_space = spaces.Box(
			low=-np.float32(np.inf),
			high=+np.float32(np.inf),
			shape=(state_dim,),
			dtype=np.float32,
		)

	def step(self, action):
		self.step_count += 1.0
		# Loop through each agent and assign action
		# We assume each agent has the same action space
		actions = np.split(action, len(self._env.agents))
		actions = {agent:self.action_transform(act)  for agent, act in zip(self._env.agents, actions)}
		_, rewards, terminations, truncations, infos = self._env.step(actions)

		done = terminations[self.agent_name] or truncations[self.agent_name]

		state = self._env.state()
		reward = self.get_reward(state)

		if self.step_count % self.cycle_step == 0:
			self.ds_state_update()

		return state, reward, done, infos

	def reset(self, seed=None):
		self._env.reset(seed)
		self.step_count = 0.0
		self.downstream_reset()
		return self._env.state()

	def get_reward(self, state):
		dist_list = state[:self.N]
		reward = np.zeros_like(self.landmark_id, dtype=np.float32)
		for idx, ids in enumerate(self.landmark_id):
			binary = self.binary_indicator[ids]
			dist = dist_list[ids]
			if binary == 0:
				if dist < self.distance_threshold:
					reward[idx] += 1
				else:
					reward[idx] -= 1
			else:
				if dist > self.distance_threshold:
					reward[idx] += 1
				else:
					reward[idx] -= 1
		if not self.factored:
			reward = np.sum(reward)
		return reward

	def ds_state_update(self):
		self.binary_indicator = np.random.randint(2, size=10)

	def downstream_reset(self):
		self.binary_indicator = np.random.randint(2, size=10)

	def get_end_skill_reward(self, obs=None, meta_action=None):
		return [0]

	# Defines additional states needed for the upper policy
	def get_additional_states(self, obs=None):
		return np.concatenate([self.binary_indicator, [self.step_count / self.cycle_step]])


class SequentialDSWrapper(DownstreamCentralizedWrapper):
	"""
	Defines the sequential interaction environment
	"""
	def __init__(self, env, N, agent_sequence=[0, 1, 2], simplify_action_space=True):
		self._env = env
		self.N = N
		self.distance_threshold = 0.6

		self.agent_sequence = agent_sequence
		self.simplify_action_space = simplify_action_space

		self.initialize_parameters()
		self.initialize_action_space()
		self.initialize_state_space()

	# This happens to be the same as the previous wrapper, but it is not always the case
	def initialize_state_space(self):
		state_dim = self._env.state_space.shape[0] + self.N + 1
		self.observation_space = spaces.Box(
			low=-np.float32(np.inf),
			high=+np.float32(np.inf),
			shape=(state_dim,),
			dtype=np.float32,
		)

	def get_reward(self, state):
		if self.progress_idx == len(self.agent_sequence):
			reward = 10
		else:
			dist_list = state[:self.N]
			reward = 0
			for idx in range(self.N):
				# if idx in [5, 8]:
				# 	continue
				binary = self.curren_idx[idx]
				dist = dist_list[idx]
				if binary == 0:
					if dist > self.distance_threshold:
						reward += 0
					else:
						reward -= 0.1
				else:
					# Ok here is the problem -> after update
					if dist < self.distance_threshold:
						reward += 0
						self.charge_counter += 1
					else:
						reward -= 0.1
		return reward

	def ds_state_update(self):
		if self.progress_idx < len(self.agent_sequence) and self.charge_counter > 40:
			# switch to next target
			self.progress_idx += 1
			self.charge_counter = 0
			self.curren_idx = np.zeros(self.N)
			if self.progress_idx < len(self.agent_sequence):
				self.curren_idx[self.agent_sequence[self.progress_idx]] = 1


	def downstream_reset(self):
		self.progress_idx = 0
		self.charge_counter = 0
		self.curren_idx = np.zeros(self.N)
		self.curren_idx[self.agent_sequence[self.progress_idx]] = 1

	def get_end_skill_reward(self, obs=None, meta_action=None):
		return [0]

	# Defines additional states needed for the upper policy
	def get_additional_states(self, obs=None):
		return np.concatenate([self.curren_idx, [self.step_count / self.cycle_step]])