import torch
import numpy as np
from downstream_tasks.ant_multi_goals import AntMultiGoalsEnv
# import gym
# import numpy as np
# from stable_baselines3 import SAC
# from gym.wrappers import Monitor
# from envs.mujoco.ant_env import AntEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained option_policy
checkpoint = torch.load('exp/Debug/sd000_1752248887_ant_metra/option_policy19000.pt')
option_policy = checkpoint['policy']
option_policy.to(device)

# Load pretrained phi encoder 
checkpoint = torch.load('exp/Debug/sd000_1752248887_ant_metra/traj_encoder19000.pt')
traj_encoder = checkpoint['traj_encoder']
traj_encoder.to(device)

option_policy.eval()
traj_encoder.eval()


# Run up the Downstream Task
env = AntMultiGoalsEnv()

obs = env.reset()
print("Initial observation shape:", obs.shape)
print("Initial goal:", env.current_goal)

# Run one episode
done = False
step = 0
while not done:  # hard limit to avoid infinite loops
    action = env.action_space.sample()  # sample random action
    obs, reward, done, info = env.step(action)

    position = info["coordinates"]  # current x, y position
    goal = info["current_goal"]     # current goal position

    print(f"Step {step:3d}: "
          f"action={np.round(action, 2)} "
          f"pos=({position[0]:.2f}, {position[1]:.2f}) "
          f"goal=({goal[0]:.2f}, {goal[1]:.2f}) "
          f"reward={reward:.2f}, done={done}")
    
    step += 1

env.close()


