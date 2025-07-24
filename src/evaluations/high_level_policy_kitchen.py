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
from stable_baselines3.common.buffers import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
NUM_TASKS = len(all_tasks)

mode = "train" # ["train", "plot", "eval"]
algo = "metra" # ["dsd", "metra", "csd"]

if algo == "dsd":
    option_policy_checkpoint_path = 'dsd_models/q/option_policy8000.pt'
    traj_encoder_checkpoint_path = 'dsd_models/q/traj_encoder8000.pt'

elif algo == "metra": 
    option_policy_checkpoint_path = 'final_models/METRA/option_policy40000.pt'    
    traj_encoder_checkpoint_path = 'final_models/METRA/traj_encoder40000.pt'

elif algo == "csd": 
    option_policy_checkpoint_path = 'final_models/CSD/option_policy40000.pt'    
    traj_encoder_checkpoint_path = 'final_models/CSD/traj_encoder40000.pt'

csv_path = f"final_models/TEST/high_level_{algo}_kitchen.csv"
option_ckpt = torch.load(option_policy_checkpoint_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]

env = KitchenEnv(
    tasks_to_complete=all_tasks,
    terminate_on_tasks_completed=True,
    render_mode="rgb_array"
)
max_steps = 200  # Set your max steps per episode here
env = TimeLimit(env, max_episode_steps=max_steps)

skill_dim = 2 # N=5, d=5
max_skill_steps = 10

all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

task_to_onehot = {
    task: np.eye(len(all_tasks))[i]
    for i, task in enumerate(all_tasks)
}

custom_order = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,     # Panda Arm and Gripper States
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Burners and Overhead Light
                29, 30, 31, 49, 50, 51,                                           # Cabinets (Slide + Left + Right Hinge)
                32, 52,                                                          # Microwave Door
                33, 34, 35, 36, 37, 38, 39, 53, 54, 55, 56, 57, 58               # Kettle
        ]


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
        self.all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
        self.num_tasks = len(self.all_tasks)


        obs_space = env.observation_space.spaces['observation']
        obs_low = obs_space.low
        obs_high = obs_space.high
        self.observation_space = gym.spaces.Box(low=np.concatenate([obs_low, np.zeros(self.num_tasks)]), high=np.concatenate([obs_high, np.ones(self.num_tasks)]), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(skill_dim,), dtype=np.float32)

    def reset(self, task_idx=None, **kwargs):
        if task_idx is None:
            task_idx = np.random.randint(self.num_tasks)

        self.task_idx = task_idx
        self.current_goal_onehot = np.eye(self.num_tasks)[task_idx]
        self.current_task_name = self.all_tasks[task_idx]

        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs['observation'] if isinstance(obs, dict) else obs
        self.current_obs = rearrange_vector(self.current_obs, custom_order)

        return self._get_augmented_obs(self.current_obs), info
    

    def _get_augmented_obs(self, obs):
        return np.concatenate([obs, self.current_goal_onehot], axis=-1).astype(np.float32)

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
        total_reward = total_reward * ([self.current_task_name] == completed_tasks)
        if total_reward == 1:
            print("Completed tasks:", completed_tasks)
            print(self.current_task_name)
            print(total_reward)
            print(60*'-')
        info['total_reward'] = total_reward
        info['total_completed_tasks'] = len(completed_tasks)
        info['completed_tasks'] = completed_tasks
        info['task_idx'] = self.task_idx

        if isinstance(self.current_obs, dict):
            obs_out = self.current_obs['observation']
        else:
            obs_out = self.current_obs
        
        obs_out = rearrange_vector(obs_out, custom_order)

        return self._get_augmented_obs(obs_out), total_reward, terminated, truncated, info


class SACWithMinBufferSize(SAC):
    def __init__(self, *args, min_buffer_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_buffer_size = min_buffer_size

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Skip training if buffer not large enough
        if self.replay_buffer.size() < self.min_buffer_size:
            # Optionally print or log this event
            # print(f"Skipping training: replay buffer size {self.replay_buffer.size()} < {self.min_buffer_size}")
            return
        # Otherwise call original train
        super().train(gradient_steps, batch_size)

class BalancedTaskReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device,
                 n_tasks: int, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, **kwargs)
        self.task_indices = np.empty((buffer_size,), dtype=np.int32)
        self.n_tasks = n_tasks

    def add(self, obs, next_obs, action, reward, done, infos):
        task_idx = infos[0]['task_idx']
        super().add(obs, next_obs, action, reward, done, infos)
        self.task_indices[self.pos - 1] = task_idx

    def sample(self, batch_size: int, env=None):
        batch_per_task = batch_size // self.n_tasks
        remainder = batch_size % self.n_tasks

        indices = []
        for task in range(self.n_tasks):
            task_indices = np.where(self.task_indices[:self.size()] == task)[0]
            if len(task_indices) == 0:
                continue
            k = batch_per_task + (1 if task < remainder else 0)
            sampled = np.random.choice(task_indices, size=min(k, len(task_indices)), replace=False)
            indices.extend(sampled)

        indices = np.array(indices)
        return self._get_samples(indices, env)


def train():
    log_dir = f"final_models/TEST/sac_high_level_{algo}_kitchen"
    # log_dir = "logs/TEST"
    model_dir = os.path.join(log_dir, "models")
    tensorboard_log_dir = os.path.join(log_dir, "tb")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    wrapped_env = DummyVecEnv([lambda: SkillWrapperEnv(env, option_policy, traj_encoder, skill_dim, max_skill_steps, device)])
    new_logger = configure(folder=tensorboard_log_dir, format_strings=["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(
        net_arch=[1024, 1024],
    )

    sac_model = SACWithMinBufferSize(
        policy="MlpPolicy",
        env=wrapped_env,
        replay_buffer_class=BalancedTaskReplayBuffer,
        replay_buffer_kwargs=dict(n_tasks=NUM_TASKS),  # pass your number of tasks
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
        tensorboard_log=tensorboard_log_dir,
        min_buffer_size=1
    )
    sac_model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=4000,
        save_path=model_dir,
        name_prefix="sac_highlevel_kitchen",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    class RewardLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_reward = 0.0
            self.episode_completed_tasks = 0.0

        def _on_step(self) -> bool:
            done = self.locals.get('dones', [False])[0]
            info = self.locals.get('infos', [{}])[0]

            if 'total_reward' in info:
                self.episode_reward += info['total_reward'] 
                self.episode_completed_tasks += info['total_completed_tasks']

            if done:
                self.logger.record('custom/episode_reward', self.episode_reward)
                self.logger.record('custom/episode_completed_tasks', self.episode_completed_tasks)

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