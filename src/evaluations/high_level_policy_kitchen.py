import os
import numpy as np
import torch
# import gym
# from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats


import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import imageio
import time
from tqdm import tqdm
import csv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium_robotics.envs.franka_kitchen import KitchenEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

mode = "eval" # ["train", "plot", "eval"]
algo = "dsd" # ["dsd", "metra"]

if algo == "dsd":
    option_policy_checkpoint_path = 'exp/Debug/sd000_1752773936_kitchen_franka_metra/option_policy5000.pt'
    traj_encoder_checkpoint_path = 'exp/Debug/sd000_1752773936_kitchen_franka_metra/traj_encoder5000.pt'

elif algo == "metra": 
    option_policy_checkpoint_path = '/home/hadi/RL/LDG/METRA/exp/Debug/sd000_1752257820_ant_metra/option_policy17000.pt'    
    traj_encoder_checkpoint_path = '/home/hadi/RL/LDG/METRA/exp/Debug/sd000_1752257820_ant_metra/traj_encoder17000.pt'

csv_path = f"results/high_level_{algo}_kitchen_5000.csv"
option_ckpt = torch.load(option_policy_checkpoint_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]

env = KitchenEnv(
    tasks_to_complete=all_tasks,
    terminate_on_tasks_completed=True,
    render_mode="rgb_array"
)
max_steps = 280  # Set your max steps per episode here
env = TimeLimit(env, max_episode_steps=max_steps)

skill_dim = 25 # N=5, d=5
max_skill_steps = 20 # maximum number of steps for each z (25)


class SkillWrapperEnv(gym.Env):
    def __init__(self, env, option_policy, traj_encoder, skill_dim, max_skill_steps, device='cpu'):
        super().__init__()
        self.env = env
        self.option_policy = option_policy.to(device).eval()
        self.traj_encoder = traj_encoder.to(device).eval()
        self.device = device
        self.skill_dim = skill_dim
        self._max_skill_steps = max_skill_steps 
        self.current_obs = None

        self.observation_space = env.observation_space.spaces['observation']
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(skill_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs['observation']
        return self.current_obs, info

    def step(self, skill_z):
        skill_z = torch.tensor(skill_z, dtype=torch.float32).unsqueeze(0).to(self.device)
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._max_skill_steps):
            if isinstance(self.current_obs, dict):
                obs_tensor = torch.tensor(self.current_obs['observation'], dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            input_tensor = torch.cat([obs_tensor, skill_z], dim=-1)

            with torch.no_grad():
                action_np, _ = self.option_policy.get_action(input_tensor)
            action = action_np[0]

            self.current_obs, reward, terminated, truncated, info  = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                break


        completed_tasks = info.get("episode_task_completions", [])
        # print("Completed tasks:", completed_tasks)
        info['total_reward'] = total_reward
        info['total_completed_tasks'] = len(completed_tasks)
        info['completed_tasks'] = completed_tasks


        if isinstance(self.current_obs, dict):
            return self.current_obs['observation'], total_reward, terminated, truncated, info
        else:
            return self.current_obs, total_reward, terminated, truncated, info


def train():
    log_dir = f"logs/sac_high_level_{algo}_5000_kitchen"
    # log_dir = "logs/test"
    model_dir = os.path.join(log_dir, "models")
    tensorboard_log_dir = os.path.join(log_dir, "tb")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    wrapped_env = DummyVecEnv([lambda: SkillWrapperEnv(env, option_policy, traj_encoder, skill_dim, max_skill_steps, device)])
    new_logger = configure(folder=tensorboard_log_dir, format_strings=["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(
        net_arch=[1024, 1024],
    )

    sac_model = SAC(
        policy="MlpPolicy",
        env=wrapped_env,
        learning_rate=1e-4,
        buffer_size=int(1e6),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=50,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=tensorboard_log_dir  # Enables TB metrics
    )
    sac_model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix="sac_highlevel_kitchen",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    class RewardLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.total_reward = 0.0
            self.completed_tasks = 0.0

        def _on_step(self) -> bool:
            done = self.locals.get('dones', [False])[0]
            info = self.locals.get('infos', [{}])[0]

            if 'total_reward' in info:
                self.total_reward += info['total_reward'] 
                self.completed_tasks += info['total_completed_tasks']

            if done:
                self.logger.record('custom/total_cumulative_reward', self.total_reward)
                self.logger.record('custom/completed_tasks', self.completed_tasks)

            return True
    
    reward_callback = RewardLoggingCallback()
    sac_model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, reward_callback])
    sac_model.save(os.path.join(model_dir, "sac_highlevel_final"))

def eval(env, max_duration=30.0, max_skill_steps=10):
    log_dir = f"logs/sac_high_level_{algo}_5000_kitchen"
    snapshot_version = "sac_highlevel_kitchen_120000_steps.zip"
    model_dir = os.path.join(log_dir, "models", snapshot_version)

    sac_model = SAC.load(model_dir, device=device)
    wrapped_env = SkillWrapperEnv(env, option_policy, traj_encoder, skill_dim, max_skill_steps, device)

    time_reward_log = []
    record_video = False
    done = True
    start_time = time.time()
    last_log_time = start_time
    cumulative_reward = 0.0
    frames = []
    # diverse_tasks = set()

    while time.time() - start_time < max_duration:
        if done:
            obs, _ = wrapped_env.reset()
            done = False
        else:
            z, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(z)
            done = terminated or truncated
            # diverse_tasks.update(info['completed_tasks'])
            cumulative_reward += reward

            if record_video:
                frame = env.render(mode="rgb_array")
                frames.append(frame)

        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            elapsed = current_time - start_time
            time_reward_log.append((elapsed, cumulative_reward))
            # time_reward_log.append((elapsed, len(diverse_tasks)))
            last_log_time = current_time

    print(f"Total accumulated reward: {cumulative_reward:.2f}")

    if record_video:
        video_path = f"eval_sac_highlevel_ant_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")

    return time_reward_log


def run_multiple_seeds(num_runs=8, max_duration=50.0, max_skill_steps=10):
    all_logs = []
    csv_rows = []
    
    for seed in tqdm(range(num_runs)):
        print(f"Running seed {seed}...")
        env = KitchenEnv(
            tasks_to_complete=all_tasks,
            terminate_on_tasks_completed=True,
            render_mode="rgb_array",
        )
        # env.seed(seed)
                
        time_reward_log = eval(env, max_duration=max_duration, max_skill_steps=max_skill_steps)
        all_logs.append(time_reward_log)

        for time_val, reward in time_reward_log:
            csv_rows.append({'seed': seed, 'time': time_val, 'cumulative_reward': reward})


    fieldnames = ['seed', 'time', 'cumulative_reward']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n📁 Logs saved to {csv_path}")
    return all_logs


def plot_multiple_methods_cumulative_reward(logs_by_method, max_duration, dt=1.0, confidence=0.95, save_path=None):
    common_times = np.arange(0, max_duration + dt, dt)

    plt.figure(figsize=(10, 6))

    for method, all_logs in logs_by_method.items():
        interp_rewards = []
        for log in all_logs:
            times, rewards = zip(*log)
            f = interp1d(times, rewards, kind='previous', bounds_error=False,
                         fill_value=(rewards[0], rewards[-1]))
            interp_rewards.append(f(common_times))
        
        interp_rewards = np.array(interp_rewards)
        mean_rewards = np.mean(interp_rewards, axis=0)
        sem = stats.sem(interp_rewards, axis=0)
        margin = sem * stats.t.ppf((1 + confidence) / 2., interp_rewards.shape[0] - 1)

        # Plot mean and confidence interval
        plt.plot(common_times, mean_rewards, label=method)
        plt.fill_between(common_times, mean_rewards - margin, mean_rewards + margin, alpha=0.2)

    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Cumulative Reward')
    plt.title('Average Cumulative Tasks over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.show()

def load_logs_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    all_logs = []

    for seed, group in df.groupby("seed"):
        sorted_group = group.sort_values("time")
        log = list(zip(sorted_group["time"], sorted_group["cumulative_reward"]))
        all_logs.append(log)

    return all_logs

if mode == "train":
    train()
elif mode == "eval":
    run_multiple_seeds(num_runs=8, max_duration=50.0, max_skill_steps=10)
elif mode == "plot":
    logs_5000 = load_logs_from_csv("results/high_level_dsd_kitchen_5000.csv")
    logs_10000 = load_logs_from_csv("results/high_level_dsd_kitchen_10000.csv")
    logs_15000 = load_logs_from_csv("results/high_level_dsd_kitchen_15000.csv")

    logs_by_method = {
        "5000": logs_5000,
        "10000": logs_10000,
        "15000": logs_15000
    }

    plot_multiple_methods_cumulative_reward(
        logs_by_method,
        max_duration=50.0,
        dt=1.0,
        save_path=f"results/high_level_kitchen_comparison_ours.png"
    )