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

from downstream_tasks.half_cheetah_multi_goals import HalfCheetahGoal
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

algo = "csd" # ["susd", "metra", "lsd", "csd", "diayn"]
num_runs = 8 
max_duration = 2e4 # steps
max_steps = 200
mode = "plot" # ["eval", "plot"]

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


# Load pretrained option_policy
csv_path = f"final_models/half_cheetah/COVERAGE/zero_shot_{algo}_half_cheetah.csv"
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
    cumulative_reward = 0.0
    frames = []
    record_video = False

    time_reward_log = []
    steps = 0
    total_steps = 0
    
    while total_steps < max_duration: 

        if steps >= max_steps:
            obs = env.reset()
            steps = 0

        goal = env.current_goal

        s_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        g_tensor = torch.from_numpy(np.copy(obs)).float().unsqueeze(0).to(device)
        g_tensor[:, 0] = torch.tensor(goal)  

        g_tensor = env._apply_normalize_obs(g_tensor.cpu()).float().to('cuda')
        s_tensor = env._apply_normalize_obs(s_tensor.cpu()).float().to('cuda')

        if algo == "susd":
            phi_s = traj_encoder(s_tensor).detach()
            phi_g = traj_encoder(g_tensor).detach()
        else:
            phi_s = traj_encoder(s_tensor).mean.detach()
            phi_g = traj_encoder(g_tensor).mean.detach()

        z = phi_g - phi_s
        z /= torch.norm(z, dim=-1, keepdim=True) + 1e-12

        if isinstance(obs, np.ndarray):
            obs = env._apply_normalize_obs(obs)
            obs = torch.from_numpy(obs).to(torch.float32).to(z.device).unsqueeze(0)


        input_tensor = torch.cat([obs] +  [z], dim=1)
        action_np, _ = option_policy.get_action(input_tensor)
        action = action_np[0]

        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        steps += 1
        total_steps += 1

        if record_video:
            frame = env.render(mode="rgb_array")
            frames.append(frame)

        time_reward_log.append((total_steps, cumulative_reward))

        # print(f"Step={steps}, pos=({env.sim.data.qpos[0]:.2f}), goal=({goal:.2f}), reward={reward:.2f}")

    env.close()

    # print(cumulative_reward)
    # exit()

    if record_video:
        video_path = "results/zero_shot_half_cheetah_run.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"\n✅ Cumulative Reward: {cumulative_reward:.2f}")
        print(f"🎞️ Video saved to: {video_path}")

    return time_reward_log


def run_multiple_seeds(num_runs=8, max_duration=30, max_steps=200):
    all_logs = []
    csv_rows = []
    
    for seed in tqdm(range(num_runs)):
        print(f"Running seed {seed}...")
        env = HalfCheetahGoal(render_hw=256)
        
        normalizer_mean, normalizer_std = get_normalizer_preset(f'half_cheetah_preset')
        env = consistent_normalize(env, normalize_obs=False, mean=normalizer_mean, std=normalizer_std)
        
        time_reward_log = zero_shot_eval(env, max_duration=max_duration, max_steps=max_steps)
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

def smooth_rewards(values, alpha=0.6, context=5):
    smoothed = []
    for i, v in enumerate(values):
        if i == 0:
            smoothed.append(v)
        else:
            # Exponential smoothing
            smoothed_val = alpha * v + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_val)
    smoothed = np.array(smoothed)

    # Apply rolling mean with context window (without dropping at the ends)
    if context > 1:
        smoothed_context = []
        for i in range(len(smoothed)):
            start = max(0, i - context + 1)
            window = smoothed[start:i+1]
            smoothed_context.append(np.mean(window))
        smoothed = np.array(smoothed_context)

    return smoothed

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

        mean_rewards = smooth_rewards(mean_rewards, alpha=0.6, context=200)

        # Plot mean and confidence interval
        plt.plot(common_times, mean_rewards, label=method)
        plt.fill_between(common_times, mean_rewards - margin, mean_rewards + margin, alpha=0.2)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Average Cumulative Reward')
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
    run_multiple_seeds(num_runs=num_runs, max_duration=max_duration, max_steps=200)
elif mode == "plot":
    susd_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/zero_shot_susd_half_cheetah.csv")
    metra_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/zero_shot_metra_half_cheetah.csv")
    csd_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/zero_shot_csd_half_cheetah.csv")
    lsd_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/zero_shot_lsd_half_cheetah.csv")
    # diayn_logs = load_logs_from_csv("final_models/half_cheetah/COVERAGE/zero_shot_diayn_half_cheetah.csv")


    logs_by_method = {
        "SUSD": susd_logs,
        "METRA": metra_logs,
        "CSD": csd_logs,
        "LSD": lsd_logs,
        # "DIAYN": diayn_logs,
    }

    plot_multiple_methods_cumulative_reward(
        logs_by_method,
        max_duration=2e4,
        dt=1.0,
        save_path=f"final_models/half_cheetah/COVERAGE/zero_shot_half_cheetah_comparison_ours.png"
    )
