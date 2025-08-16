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

import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "eval" # ["plot", "eval"]
algo = "csd" # ["csd", "metra", "lsd", "diyan", "susd"]
skill_dim = 2

if algo == "susd":
    option_policy_checkpoint_path = f'final_models/particle/SUSD/option_policy6000.pt'
    traj_encoder_checkpoint_path = f'final_models/particle/SUSD/traj_encoder6000.pt'
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

csv_path = f"final_models/particle/COVERAGE/state_coverage_{algo}_particle.csv"
option_ckpt = torch.load(option_policy_checkpoint_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]
option_policy = option_policy.to(device).eval()
traj_encoder = traj_encoder.to(device).eval()

distances = list(range(0, 10))       # 0–9
agent_info = list(range(10, 50))     # 10–49
station_info = list(range(50, 70))   # 50–69

custom_order = []

for i in range(10):
    custom_order.append(distances[i])                       
    custom_order.extend(agent_info[i*4:(i+1)*4])            
    custom_order.extend(station_info[i*2:(i+1)*2])


def create_particle_env(seed=0):
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

max_steps = 50


def eval(env, seed):

    log = []
    record_video = False
    done = True
    frames = []
    steps = 0
    z_period = 50
    unique_pairs = set()

    while steps <= 1e4:
        if done:
            obs = env.reset(seed)
            done = False
            random_z = np.random.randn(1, skill_dim)
            random_z /= np.linalg.norm(random_z)
            random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
        else:
            if steps % z_period ==0:
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
                random_z = torch.tensor(random_z, dtype=torch.float32).to(device)

            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            input_tensor = torch.cat([obs, random_z], dim=-1)
            with torch.no_grad():
                action_np, _ = option_policy.get_action(input_tensor)
            action = action_np[0]

            obs, _, done, info = env.step(action)
            steps += 1

            x, y = info['next_coordinates']
            pair = (round(x, 2), round(y, 2))
            unique_pairs.add(pair)

            if record_video:
                frame = env.render()
                frames.append(frame)

            log.append((steps, len(unique_pairs)))

    print(f"unique pairs: {len(unique_pairs):.2f}")

    if record_video:
        video_path = f"eval_state_coverage_ant_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")

    return log


def run_multiple_seeds(num_runs=8):
    all_logs = []
    csv_rows = []
    
    for seed in tqdm(range(num_runs)):
        print(f"Running seed {seed}...")
        env = create_particle_env(seed)
                
        time_reward_log = eval(env, seed)
        all_logs.append(time_reward_log)

        for time_val, unique_steps in time_reward_log:
            csv_rows.append({'seed': seed, 'time': time_val, 'unique_steps': unique_steps})


    fieldnames = ['seed', 'time', 'unique_steps']
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

def plot_multiple_methods_unique_steps(logs_by_method, max_duration, dt=1.0, confidence=0.95, save_path=None):
    common_times = np.arange(0, max_duration + dt, dt)

    plt.figure(figsize=(10, 6))

    for method, all_logs in logs_by_method.items():
        interp_rewards = []
        for log in all_logs:
            times, rewards = zip(*log)
            times = np.array(times)
            rewards = np.array(rewards)
            
            interp = np.interp(common_times, times, rewards)
            interp_rewards.append(interp)
        
        interp_rewards = np.array(interp_rewards)
        mean_rewards = np.mean(interp_rewards, axis=0)

        mean_rewards = smooth_curve(mean_rewards, alpha=0.15)

        plt.plot(common_times, mean_rewards, label=method, linewidth=0.8)

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
        log = list(zip(sorted_group["time"], sorted_group["unique_steps"]))
        all_logs.append(log)

    return all_logs


if mode == "eval":
    run_multiple_seeds(num_runs=8)
elif mode == "plot":
    susd_logs = load_logs_from_csv("final_models/particle/COVERAGE/state_coverage_susd_particle.csv")
    metra_logs = load_logs_from_csv("final_models/particle/COVERAGE/state_coverage_metra_particle.csv")
    csd_logs = load_logs_from_csv("final_models/particle/COVERAGE/state_coverage_csd_particle.csv")
    # lsd_logs = load_logs_from_csv("final_models/particle/COVERAGE/state_coverage_lsd_particle.csv")


    logs_by_method = {
        "SUSD": susd_logs,
        "METRA": metra_logs,
        "CSD": csd_logs,
        # "LSD": lsd_logs
    }

    plot_multiple_methods_unique_steps(
        logs_by_method,
        max_duration=1e4,
        dt=1.0,
        save_path=f"final_models/particle/COVERAGE/state_coverage_particle_comparison_ours.png"
    )