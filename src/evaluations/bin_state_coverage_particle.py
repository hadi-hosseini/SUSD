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

from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper
from envs.mp.particle import Particle
from src.dusdi_utils import Actor

import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "eval" # ["plot", "eval"]
algo = "dusdi" # ["csd", "metra", "lsd", "diayn", "susd", "dusdi"]
skill_dim = 2

if algo == "susd":
    option_policy_checkpoint_path = f'final_models/particle/SUSD/option_policy10000.pt'
    traj_encoder_checkpoint_path = f'final_models/particle/SUSD/traj_encoder10000.pt'
    skill_dim = 20 # N=10 & d=2

elif algo == "metra": 
    option_policy_checkpoint_path = 'final_models/particle/METRA/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/particle/METRA/traj_encoder10000.pt'

elif algo == "csd":
    option_policy_checkpoint_path = 'final_models/particle/CSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/particle/CSD/traj_encoder10000.pt'

elif algo == "lsd":
    option_policy_checkpoint_path = 'final_models/particle/LSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/particle/LSD/traj_encoder10000.pt'

elif algo == "diayn":
    option_policy_checkpoint_path = 'final_models/particle/DIAYN/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/particle/DIAYN/traj_encoder10000.pt'

elif algo == "dusdi":
    option_policy_checkpoint_path = 'final_models/particle/DUSDI/option_policy10000.pt'    
    traj_encoder_checkpoint_path = 'final_models/particle/DUSDI/traj_encoder10000.pt'
    skill_dim = 50 # N = 10 & D = 5


csv_path = f"final_models/particle/COVERAGE/bin_state_coverage_{algo}_particle_sum.csv"

if algo == "dusdi":
    option_policy = Actor("state", 120, 20, 50, 1024, True, [-10, 2], "particle")
    cp_dict = torch.load(option_policy_checkpoint_path, map_location='cpu')
    option_policy.load_state_dict(cp_dict)
else:
    option_ckpt = torch.load(option_policy_checkpoint_path)
    option_policy = option_ckpt["policy"]

option_policy = option_policy.to(device).eval()

if algo == "dusdi":
    custom_order = list(range(0, 70))
    agent_positions = {0: (12, 13), 1: (16, 17), 2: (20, 21), 3: (24, 25), 4: (28, 29), 5: (32, 33), 6: (36, 37), 7: (40, 41), 8: (44, 45), 9: (48, 49)}
else:
    distances = list(range(0, 10))       # 0–9
    agent_info = list(range(10, 50))     # 10–49
    station_info = list(range(50, 70))   # 50–69

    custom_order = []

    for i in range(10):
        custom_order.append(distances[i])                       
        custom_order.extend(agent_info[i*4:(i+1)*4])           
        custom_order.extend(station_info[i*2:(i+1)*2])

    agent_positions = {0: (3, 4), 1: (10, 11), 2: (17, 18), 3: (24, 25), 4: (31, 32), 5: (38, 39), 6: (45, 46), 7: (52, 53), 8: (59, 60), 9: (66, 67)}


def create_particle_env():
    env = simple_heterogenous_v3.parallel_env(
            render_mode= "rgb_array",
            max_cycles=1000,
            continuous_actions=True,
            local_ratio=0,
            N=10,
            img_encoder=None)

    env = CentralizedWrapper(env, simplify_action_space=True)
    env = Particle(env, custom_order, (512, 480))
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

    num_bins = 150
    x_edges = np.linspace(-1, 1, num_bins + 1)
    y_edges = np.linspace(-1, 1, num_bins + 1)
    grid_counts_list = [np.zeros((num_bins, num_bins), dtype=int) for _ in range(10)]

    while steps <= 1e5:
        if done:
            obs = env.reset()
            done = False
            if algo == "dusdi":
                random_z = random_one_hot_concat(N=10, d=5)
            else:
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
            random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
        else:
            if steps % z_period ==0:
                if algo == "dusdi":
                    random_z = random_one_hot_concat(N=10, d=5)
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

            for i in range(10):
                x_index, y_index = agent_positions[i]
                x = obs[x_index]
                y = obs[y_index]
                pair = (round(x, 2), round(y, 2))
                x_bin = np.digitize(pair[0], x_edges) - 1
                y_bin = np.digitize(pair[1], y_edges) - 1

                if x_bin < 0:
                    x_bin = 0
                if y_bin < 0:
                    y_bin = 0
                if x_bin >= num_bins:
                    x_bin = num_bins - 1
                if y_bin >= num_bins:
                    y_bin = num_bins - 1

                grid_counts_list[i][x_bin][y_bin] += 1
                
            if record_video:
                frame = env.render()
                frames.append(frame)
            

            # min_cell = min(len(s) for row in grid_sets for s in row)
            cov_list = []
            for i in range(10):
                cov = sum(1 for row in grid_counts_list[i] for s in row if s != 0) / (num_bins * num_bins)
                cov_list.append(cov)

            log.append((steps, np.mean(cov_list)))


    cov_list = []
    for i in range(10):
        cov = sum(1 for row in grid_counts_list[i] for s in row if s != 0) / (num_bins * num_bins)
        cov_list.append(cov)

    print(f"min cell pairs: {np.mean(cov_list)}")

    ### save the result
    # np.savez(f"{algo}_grid_counts_list.npz", *grid_counts_list)
    # exit()

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
        env = create_particle_env()
                
        time_reward_log = eval(env)
        all_logs.append(time_reward_log)

        for time_val, unique_steps in time_reward_log:
            csv_rows.append({'iter': iteration, 'time': time_val, 'sum_unique_pairs': unique_steps})


    fieldnames = ['iter', 'time', 'sum_unique_pairs']
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
    plt.rcParams.update({'font.size': 18})


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
    plt.ylabel('Bin Coverage')
    plt.title('Average Bin Coverage (Multi-Particle)- 22500 Bins')
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
        log = list(zip(sorted_group["time"], sorted_group["sum_unique_pairs"]))
        all_logs.append(log)

    return all_logs


if mode == "eval":
    run_multiple_seeds(num_runs=8)
elif mode == "plot":
    susd_logs = load_logs_from_csv("final_models/particle/COVERAGE/bin_state_coverage_susd_particle_sum.csv")
    metra_logs = load_logs_from_csv("final_models/particle/COVERAGE/bin_state_coverage_metra_particle_sum.csv")
    csd_logs = load_logs_from_csv("final_models/particle/COVERAGE/bin_state_coverage_csd_particle_sum.csv")
    lsd_logs = load_logs_from_csv("final_models/particle/COVERAGE/bin_state_coverage_lsd_particle_sum.csv")
    # diayn_logs = load_logs_from_csv("final_models/particle/COVERAGE/state_coverage_diayn_particle.csv")
    dusdi_logs = load_logs_from_csv("final_models/particle/COVERAGE/bin_state_coverage_dusdi_particle_sum.csv")

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
        save_path=f"final_models/particle/COVERAGE/bin_state_coverage_particle.png"
    )