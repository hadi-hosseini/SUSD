import akro

from garage.envs import EnvSpec


import gym
import gymnasium.spaces as gymn_spaces

def gymnasium_to_gym_space(space):
    if 'gym.spaces' in str(type(space)):
        # Already a gym space, return as is
        return space
    
    if isinstance(space, gymn_spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gymn_spaces.Discrete):
        return gym.spaces.Discrete(n=space.n)
    elif isinstance(space, gymn_spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gymn_spaces.MultiBinary):
        return gym.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gymn_spaces.Tuple):
        return gym.spaces.Tuple(tuple(gymnasium_to_gym_space(s) for s in space.spaces))
    elif isinstance(space, gymn_spaces.Dict):
        return gym.spaces.Dict({k: gymnasium_to_gym_space(s) for k, s in space.spaces.items()})
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")

class AkroWrapperTrait:
    @property
    def spec(self):
        # print("Action space type:", type(self.action_space))
        # print("Observation space type:", type(self.observation_space))
        gym_action_space = gymnasium_to_gym_space(self.action_space)
        gym_obs_space = gymnasium_to_gym_space(self.observation_space)
        # exit()
        return EnvSpec(action_space=akro.from_gym(gym_action_space),
                       observation_space=akro.from_gym(gym_obs_space))

