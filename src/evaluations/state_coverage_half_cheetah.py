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

from envs.mujoco.half_cheetah_env import HalfCheetahEnv


import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "plot" # ["plot", "eval"]
algo = "lsd" # ["susd", "metra", "csd", "diayn", "lsd"]
skill_dim = 2

if algo == "susd":
    option_policy_checkpoint_path = f'final_models/half_cheetah/SUSD/option_policy10000.pt'
    traj_encoder_checkpoint_path = f'final_models/half_cheetah/SUSD/traj_encoder10000.pt'

elif algo == "metra": 
    option_policy_checkpoint_path = 'final_models/half_cheetah/METRA/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/half_cheetah/METRA/traj_encoder10000.pt'

elif algo == "csd":
    option_policy_checkpoint_path = 'final_models/half_cheetah/CSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/half_cheetah/CSD/traj_encoder10000.pt'

elif algo == "lsd":
    option_policy_checkpoint_path = 'final_models/half_cheetah/LSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/half_cheetah/LSD/traj_encoder10000.pt'

elif algo == "diayn":
    option_policy_checkpoint_path = 'final_models/half_cheetah/DIAYN/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/half_cheetah/DIAYN/traj_encoder10000.pt'


csv_path = f"final_models/half_cheetah/COVERAGE/state_coverage_{algo}_half_cheetah.csv"
option_ckpt = torch.load(option_policy_checkpoint_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]
option_policy = option_policy.to(device).eval()
traj_encoder = traj_encoder.to(device).eval()

env = HalfCheetahEnv(render_hw=100, fixed_initial_state=True)
max_steps = 200


def eval(env):

    log = []
    record_video = False
    done = True
    frames = []
    steps = 0
    z_period = 200
    unique_xs = set()

    while steps <= 1e4:
        if done:
            obs = env.reset()
            done = False
            random_z = np.random.randn(1, skill_dim)
            random_z /= np.linalg.norm(random_z)
            random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
        else:
            if steps % z_period ==0:
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
                random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
                obs = env.reset() # reset the environment each 200 steps

            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            input_tensor = torch.cat([obs, random_z], dim=-1)
            with torch.no_grad():
                action_np, _ = option_policy.get_action(input_tensor)
            action = action_np[0]

            obs, reward, done, info = env.step(action)
            steps += 1

            x = env.sim.data.qpos[0]
            x = round(x, 2)
            unique_xs.add(x)

            if record_video:
                frame = env.render()
                frames.append(frame)

            log.append((steps, len(unique_xs)))

    print(f"unique X: {len(unique_xs):.2f}")

    if record_video:
        video_path = f"eval_state_coverage_half_cheetah_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")

    return log


def run_multiple_seeds(num_runs=8):
    all_logs = []
    csv_rows = []
    
    for seed in tqdm(range(num_runs)):
        print(f"Running seed {seed}...")
        env = HalfCheetahEnv(render_hw=100, fixed_initial_state=True)
        env.seed(seed)
                
        time_reward_log = eval(env)
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
    plt.ylabel('State Coverage')
    plt.title('Average State Coverage over Steps')
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
    susd_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/state_coverage_susd_half_cheetah.csv")
    metra_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/state_coverage_metra_half_cheetah.csv")
    csd_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/state_coverage_csd_half_cheetah.csv")
    lsd_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/state_coverage_lsd_half_cheetah.csv")

    logs_by_method = {
        "SUSD": susd_logs,
        "METRA": metra_logs,
        "CSD": csd_logs,
        "LSD": lsd_logs,
    }

    plot_multiple_methods_cumulative_reward(
        logs_by_method,
        max_duration=1e4,
        dt=1.0,
        save_path=f"final_models/half_cheetah/COVERAGE/state_coverage_half_cheetah_comparison_ours.png"
    )