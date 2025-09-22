import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats

import imageio
from tqdm import tqdm
import csv

from src.dusdi_utils import Actor

import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "plot" # ["plot", "eval"]
algo = "diayn" # ["csd", "metra", "lsd", "diayn", "susd", "dusdi"]
skill_dim = 2

if algo == "susd":
    option_policy_checkpoint_path = f'final_models/gunner/SUSD/option_policy10000.pt'
    traj_encoder_checkpoint_path = f'final_models/gunner/SUSD/traj_encoder10000.pt'
    skill_dim = 6 # N=3 & d=2

elif algo == "metra": 
    option_policy_checkpoint_path = 'final_models/gunner/METRA/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/gunner/METRA/traj_encoder10000.pt'

elif algo == "csd":
    option_policy_checkpoint_path = 'final_models/gunner/CSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/gunner/CSD/traj_encoder10000.pt'

elif algo == "lsd":
    option_policy_checkpoint_path = 'final_models/gunner/LSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/gunner/LSD/traj_encoder10000.pt'

elif algo == "diayn":
    option_policy_checkpoint_path = 'final_models/gunner/DIAYN/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/gunner/DIAYN/traj_encoder10000.pt'

elif algo == "dusdi":
    option_policy_checkpoint_path = 'final_models/gunner/DUSDI/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/gunner/DUSDI/traj_encoder10000.pt'
    skill_dim = 20 # N = 3 & D = 5


csv_path = f"final_models/gunner/COVERAGE/state_coverage_{algo}_gunner.csv"

if algo == "dusdi":
    option_policy = Actor("state", 38, 6, 20, 1024, True, [-10, 2], "moma2D")
    cp_dict = torch.load(option_policy_checkpoint_path, map_location='cpu')
    option_policy.load_state_dict(cp_dict)
else:
    option_ckpt = torch.load(option_policy_checkpoint_path)
    option_policy = option_ckpt["policy"]

option_policy = option_policy.to(device).eval()

if algo == "dusdi":
    custom_order = list(range(18))
else:
    custom_order = [0, 1, 2, 3, 12, 13,
                        4, 5, 6, 7, 14, 15, 16,
                        8, 9, 10, 11, 17]

def create_gunner_env(seed=0):
    from envs.moma_2d.moma_2d_gym_env import MoMa2DGymEnv
    if algo == "dusdi":
        custom_order = list(range(18))
    else:
        custom_order = [0, 1, 2, 3, 12, 13,
                        4, 5, 6, 7, 14, 15, 16,
                        8, 9, 10, 11, 17] # base, arm, view (ORIGINAL)
    env = MoMa2DGymEnv(max_step=1000, custom_order=custom_order)
    env.reset()
    return env

def random_one_hot_concat(N, d):
    import random
    indices = [random.choice(range(d)) for _ in range(N)]
    one_hot = np.zeros((N, d), dtype=int)
    one_hot[np.arange(N), indices] = 1
    return one_hot.reshape(1, -1)

def eval(env):
    log = []
    record_video = False
    done = True
    frames = []
    steps = 0
    z_period = 200
    unique_pairs= set()

    while steps <= 1e5:
        if done:
            obs = env.reset()
            done = False
            if algo == "dusdi":
                random_z = random_one_hot_concat(N=4, d=5)
            else:
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
            random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
        else:
            if steps % z_period ==0:
                if algo == "dusdi":
                    random_z = random_one_hot_concat(N=4, d=5)
                else:
                    random_z = np.random.randn(1, skill_dim)
                    random_z /= np.linalg.norm(random_z)
                random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
                obs = env.reset() # RESET EACH 200 STEPS

            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            input_tensor = torch.cat([obs, random_z], dim=-1)
            with torch.no_grad():
                if algo == "dusdi":
                    action_dist = option_policy(input_tensor)
                    action_np = action_dist.mean.detach().cpu().numpy()
                else:
                    action_np, _ = option_policy.get_action(input_tensor)
            action = action_np[0]

            obs, _, done, info = env.step(action)
            steps += 1

            x, y = env.agent_pos
            pair = (round(x, 2), round(y, 2))
            unique_pairs.add(pair)

            if record_video:
                frame = env.render()
                frames.append(frame)

            log.append((steps, len(unique_pairs)))

    print(f"unique pairs: {len(unique_pairs)}")

    if record_video:
        video_path = f"eval_state_coverage_ant_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")

    return log


def run_multiple_seeds(num_runs=8):
    all_logs = []
    csv_rows = []
    
    for iteration in tqdm(range(num_runs)):
        print(f"Iteration {iteration}...")
        env = create_gunner_env()
                
        time_reward_log = eval(env)
        all_logs.append(time_reward_log)

        for time_val, unique_steps in time_reward_log:
            csv_rows.append({'iter': iteration, 'time': time_val, 'unique_pairs': unique_steps})


    fieldnames = ['iter', 'time', 'unique_pairs']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n📁 Logs saved to {csv_path}")
    return all_logs


def smooth_curve(values, alpha=0.6):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return np.array(smoothed)


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

def plot_multiple_methods_unique_steps(logs_by_method, max_duration, dt=1.0, confidence=0.95, save_path=None):
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
    plt.ylabel('State Coverage')
    plt.title('Average State Coverage (2D-Gunner)')
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

    for iter, group in df.groupby("iter"):
        sorted_group = group.sort_values("time")
        log = list(zip(sorted_group["time"], sorted_group["unique_pairs"]))
        all_logs.append(log)

    return all_logs


if mode == "eval":
    run_multiple_seeds(num_runs=8)
elif mode == "plot":
    susd_logs = load_logs_from_csv("final_models/gunner/COVERAGE/state_coverage_susd_gunner.csv")
    metra_logs = load_logs_from_csv("final_models/gunner/COVERAGE/state_coverage_metra_gunner.csv")
    csd_logs = load_logs_from_csv("final_models/gunner/COVERAGE/state_coverage_csd_gunner.csv")
    lsd_logs = load_logs_from_csv("final_models/gunner/COVERAGE/state_coverage_lsd_gunner.csv")
    # diayn_logs = load_logs_from_csv("final_models/gunner/COVERAGE/state_coverage_diayn_gunner.csv")
    dusdi_logs = load_logs_from_csv("final_models/gunner/COVERAGE/state_coverage_dusdi_gunner.csv")

    logs_by_method = {
        "SUSD": susd_logs,
        "METRA": metra_logs,
        "CSD": csd_logs,
        "LSD": lsd_logs,
        # "DIAYN": diayn_logs,
        "DUSDI": dusdi_logs,
    }

    plot_multiple_methods_unique_steps(
        logs_by_method,
        max_duration=1e5,
        dt=1.0,
        save_path=f"final_models/gunner/COVERAGE/state_coverage_gunner.png"
    )