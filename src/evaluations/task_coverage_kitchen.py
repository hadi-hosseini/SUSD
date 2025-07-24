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



import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

mode = "plot" # ["plot", "eval"]
algo = "dsd" # ["dsd", "metra"]

if algo == "dsd":
    option_policy_checkpoint_path = 'final_models/DSD/option_policy6000.pt'
    traj_encoder_checkpoint_path = 'final_models/DSD/traj_encoder6000.pt'

elif algo == "metra": 
    option_policy_checkpoint_path = 'final_models/METRA/option_policy40000.pt'    
    traj_encoder_checkpoint_path = 'final_models/METRA/traj_encoder40000.pt'

elif algo == "csd":
    option_policy_checkpoint_path = 'final_models/CSD/option_policy40000.pt'    
    traj_encoder_checkpoint_path = 'final_models/CSD/traj_encoder40000.pt'

elif algo == "lsd":
    option_policy_checkpoint_path = 'final_models/LSD/option_policy40000.pt'    
    traj_encoder_checkpoint_path = 'final_models/LSD/traj_encoder40000.pt'

elif algo == "diayn":
    option_policy_checkpoint_path = 'final_models/DIAYN/option_policy40000.pt'    
    traj_encoder_checkpoint_path = 'final_models/DIAYN/traj_encoder40000.pt'

csv_path = f"final_models/COVERAGE/task_coverage_{algo}_kitchen.csv"
option_ckpt = torch.load(option_policy_checkpoint_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]
option_policy = option_policy.to(device).eval()
traj_encoder = traj_encoder.to(device).eval()

env = KitchenEnv(
    tasks_to_complete=all_tasks,
    terminate_on_tasks_completed=True,
    render_mode="rgb_array"
)
max_steps = 200  # Set your max steps per episode here
env = TimeLimit(env, max_episode_steps=max_steps)

skill_dim = 10 # N=5, d=5


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


def eval(env, seed):

    log = []
    record_video = False
    done = True
    frames = []
    unique_tasks = set()
    steps = 0
    z_period = 200

    while steps <= 1e4:
        if done:
            obs, _ = env.reset(seed=seed)
            obs = obs['observation']
            done = False
            random_z = np.random.randn(1, skill_dim)
            random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
        else:
            if steps % z_period ==0:
                random_z = np.random.randn(1, skill_dim)
                random_z = torch.tensor(random_z, dtype=torch.float32).to(device)

            obs = rearrange_vector(obs, custom_order)
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            input_tensor = torch.cat([obs, random_z], dim=-1)
            with torch.no_grad():
                action_np, _ = option_policy.get_action(input_tensor)
            action = action_np[0]

            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs['observation']
            steps += 1
            done = terminated or truncated
            unique_tasks.update(info['episode_task_completions'])
            print(unique_tasks)

            if record_video:
                frame = env.render()
                frames.append(frame)

            log.append((steps, len(unique_tasks)))

    print(f"completed unique tasks: {len(unique_tasks):.2f}")

    if record_video:
        video_path = f"eval_sac_highlevel_kitchen_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")

    return log


def run_multiple_seeds(num_runs=8):
    all_logs = []
    csv_rows = []
    
    for seed in tqdm(range(num_runs)):
        print(f"Running seed {seed}...")
        env = KitchenEnv(
            tasks_to_complete=all_tasks,
            terminate_on_tasks_completed=True,
            render_mode="rgb_array",
        )
        env = TimeLimit(env, max_episode_steps=max_steps)
                
        time_reward_log = eval(env, seed)
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

    plt.xlabel('Steps')
    plt.ylabel('Task Coverage')
    plt.title('Average Task Coverage over Steps')
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


if mode == "eval":
    run_multiple_seeds(num_runs=8)
elif mode == "plot":
    dsd_logs = load_logs_from_csv("final_models/COVERAGE/task_coverage_dsd_kitchen.csv")
    dsd2_logs = load_logs_from_csv("final_models/COVERAGE/task_coverage_dsd2_kitchen.csv")
    metra_logs = load_logs_from_csv("final_models/COVERAGE/task_coverage_metra_kitchen.csv")
    csd_logs = load_logs_from_csv("final_models/COVERAGE/task_coverage_csd_kitchen.csv")

    logs_by_method = {
        "DSD_Q": dsd_logs,
        "DSD_NORM": dsd2_logs,
        "METRA": metra_logs,
        "CSD": csd_logs
    }

    plot_multiple_methods_cumulative_reward(
        logs_by_method,
        max_duration=1e4,
        dt=1.0,
        save_path=f"final_models/COVERAGE/task_coverage_kitchen_comparison_ours.png"
    )