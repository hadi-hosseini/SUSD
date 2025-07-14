import torch
import gym
import numpy as np
from stable_baselines3 import SAC
from gym.wrappers import Monitor
from envs.mujoco.ant_env import AntEnv

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



