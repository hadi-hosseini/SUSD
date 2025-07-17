import torch
import numpy as np
import imageio
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
from tqdm import tqdm
import pandas as pd
import csv

from downstream_tasks.ant_multi_goals import AntMultiGoalsEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# variables 
algo = "dsd" # ["dsd", "metra", "lsd", "csd", "diayn"]
num_runs = 8
max_duration = 50
max_steps = 200

if algo == "dsd":
    option_policy_checkpoint_path = 'exp/Debug/sd000_1752248887_ant_metra/option_policy19000.pt'
    traj_encoder_checkpoint_path = 'exp/Debug/sd000_1752248887_ant_metra/traj_encoder19000.pt'
    csv_path = "results/dsd.csv"

elif algo == "metra": 
    option_policy_checkpoint_path = '/home/hadi/RL/LDG/METRA/exp/Debug/sd000_1752257820_ant_metra/option_policy17000.pt'    
    traj_encoder_checkpoint_path = '/home/hadi/RL/LDG/METRA/exp/Debug/sd000_1752257820_ant_metra/traj_encoder17000.pt'
    csv_path = "results/metra.csv"


# Load pretrained option_policy
checkpoint = torch.load(option_policy_checkpoint_path)
option_policy = checkpoint['policy']
option_policy.to(device)

# Load pretrained phi encoder 
checkpoint = torch.load(traj_encoder_checkpoint_path)
traj_encoder = checkpoint['traj_encoder']
traj_encoder.to(device)

option_policy.eval()
traj_encoder.eval()

def zero_shot_eval(env, max_duration=30.0, max_steps=200):

    obs = env.reset()

    done = False
    cumulative_reward = 0.0
    # frames = []

    start_time = time.time()
    last_log_time = start_time
    time_reward_log = []
    episode_step_counts = []
    steps = 0

    
    while time.time() - start_time < max_duration:
        if done:
            obs = env.reset()
            done = False
            steps = 0

        while not done and steps < max_steps and (time.time() - start_time < max_duration):
            # ---- 1. Get current state and goal ----
            current_pos = env.sim.data.qpos.flat[:2]
            goal = env.current_goal

            # ---- 2. Encode current and goal to skill space ----
            s_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            g_tensor = torch.from_numpy(np.copy(obs)).float().unsqueeze(0).to(device)
            g_tensor[:, 0] = torch.tensor(goal[0])  # Overwrite x
            g_tensor[:, 1] = torch.tensor(goal[1])  # Overwrite y

            if algo == "dsd":
                phi_s = traj_encoder(s_tensor).detach()
                phi_g = traj_encoder(g_tensor).detach()
            else:
                phi_s = traj_encoder(s_tensor).mean.detach()
                phi_g = traj_encoder(g_tensor).mean.detach()

            z = phi_g - phi_s
            z /= torch.norm(z, dim=-1, keepdim=True) + 1e-12

            # ---- 3. Get action from option policy ----
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(torch.float32).to(z.device).unsqueeze(0)


            input_tensor = torch.cat([obs] +  [z], dim=1)
            action_np, _ = option_policy.get_action(input_tensor)
            action = action_np[0]

            # ---- 4. Step environment ----
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            steps += 1

            # frame = env.render(mode="rgb_array")
            # frames.append(frame)

            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                elapsed = current_time - start_time
                time_reward_log.append((elapsed, cumulative_reward))
                last_log_time = current_time

            # print(f"Step {step:3d}: "
            #     f"action={np.round(action, 2)} "
            #     f"pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) "
            #     f"goal=({goal[0]:.2f}, {goal[1]:.2f}) "
            #     f"reward={reward:.2f}, done={done}")
        
        episode_step_counts.append(steps)

    env.close()

    # video_path = "results/zero_shot_ant_run.mp4"
    # imageio.mimsave(video_path, frames, fps=30)
    # print(f"\n✅ Cumulative Reward: {cumulative_reward:.2f}")
    # print(f"🎞️ Video saved to: {video_path}")

    return time_reward_log, episode_step_counts


def run_multiple_seeds(num_runs=8, max_duration=30, max_steps=200):
    all_logs = []
    csv_rows = []
    
    for seed in tqdm(range(num_runs)):
        print(f"Running seed {seed}...")
        env = AntMultiGoalsEnv(render_hw=256)
        env.seed(seed)
        
        normalizer_mean, normalizer_std = get_normalizer_preset(f'ant_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        
        time_reward_log, _ = zero_shot_eval(env, max_duration=max_duration, max_steps=max_steps)
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
    plt.title('Average Cumulative Reward over Time')
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


## train the methods
# all_logs = run_multiple_seeds(num_runs, max_duration, max_steps)


# plot the results
# metra_logs = load_logs_from_csv("results/metra.csv")
# dsd_logs = load_logs_from_csv("results/dsd.csv")

# logs_by_method = {
#     "METRA": metra_logs,
#     "DSD": dsd_logs
# }

# plot_multiple_methods_cumulative_reward(
#     logs_by_method,
#     max_duration=max_duration,
#     dt=1.0,
#     save_path="results/zero_shot_comparison.png"
# )