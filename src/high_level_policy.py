import os
import gym
import numpy as np
import torch
from gym import spaces
import imageio
import time
from tqdm import tqdm
import csv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from iod.utils import get_normalizer_preset
from garagei.envs.consistent_normalized_env import consistent_normalize
from downstream_tasks.ant_multi_goals import AntMultiGoalsEnv 
from src.zero_shot_goal_reaching import plot_multiple_methods_cumulative_reward, load_logs_from_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "plot" # ["train", "plot", "eval"]
env_name = "ant" # ["ant", "kitchen_franka"]
algo = "metra" # ["dsd", "metra"]

if algo == "dsd":
    option_policy_checkpoint_path = 'exp/Debug/sd000_1752248887_ant_metra/option_policy19000.pt'
    traj_encoder_checkpoint_path = 'exp/Debug/sd000_1752248887_ant_metra/traj_encoder19000.pt'

elif algo == "metra": 
    option_policy_checkpoint_path = '/home/hadi/RL/LDG/METRA/exp/Debug/sd000_1752257820_ant_metra/option_policy17000.pt'    
    traj_encoder_checkpoint_path = '/home/hadi/RL/LDG/METRA/exp/Debug/sd000_1752257820_ant_metra/traj_encoder17000.pt'

csv_path = f"results/high_level_{algo}.csv"
option_ckpt = torch.load(option_policy_checkpoint_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]

env = AntMultiGoalsEnv(render_hw=256)
mean, std = get_normalizer_preset("ant_preset")
env = consistent_normalize(env, normalize_obs=True, mean=mean, std=std)

skill_dim = 12 # N=6, d=2
max_skill_steps = 10 # maximum number of steps for each z (25)


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

        self.observation_space = env.observation_space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(skill_dim,), dtype=np.float32)

    def reset(self):
        self.current_obs = self.env.reset()
        return self.current_obs

    def step(self, skill_z):
        skill_z = torch.tensor(skill_z, dtype=torch.float32).unsqueeze(0).to(self.device)
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._max_skill_steps):
            obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            input_tensor = torch.cat([obs_tensor, skill_z], dim=-1)

            with torch.no_grad():
                action_np, _ = self.option_policy.get_action(input_tensor)
            action = action_np[0]

            self.current_obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break

        return self.current_obs, total_reward, done, info


def train():
    log_dir = f"logs/sac_high_level_{algo}"
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
        name_prefix="sac_highlevel_ant",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    class RewardLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.total_reward = 0.0
            self.total_steps = 0
            self.num_tasks = 0
            self.episode_reward = 0.0

        def _on_step(self) -> bool:
            reward = self.locals.get('rewards')[0]
            done = self.locals.get('dones')[0]

            if reward is not None:
                self.total_reward += reward
                self.total_steps += 1
                self.episode_reward += reward

            if done:
                self.num_tasks += 1
                avg_reward_per_task = self.total_reward / (self.num_tasks + 1e-8)

                self.logger.record('custom/total_cumulative_reward', self.total_reward)
                self.logger.record('custom/average_reward_per_task', avg_reward_per_task)
                self.logger.record('custom/num_tasks', self.num_tasks)
                self.logger.record('custom/episode_reward', self.episode_reward)

                self.episode_reward = 0.0

            return True

    
    reward_callback = RewardLoggingCallback()
    sac_model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, reward_callback])
    sac_model.save(os.path.join(model_dir, "sac_highlevel_final"))

def eval(env, max_duration=30.0, max_skill_steps=10):
    log_dir = f"logs/sac_high_level_{algo}"
    snapshot_version = "sac_highlevel_ant_200000_steps.zip"
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

    while time.time() - start_time < max_duration:
        if done:
            obs = wrapped_env.reset()
            done = False
        else:
            z, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, done, _ = wrapped_env.step(z)
            cumulative_reward += reward

            if record_video:
                frame = env.render(mode="rgb_array")
                frames.append(frame)

        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            elapsed = current_time - start_time
            time_reward_log.append((elapsed, cumulative_reward))
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
        env = AntMultiGoalsEnv(render_hw=256)
        env.seed(seed)
        
        normalizer_mean, normalizer_std = get_normalizer_preset(f'ant_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        
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

if mode == "train":
    train()
elif mode == "eval":
    run_multiple_seeds(num_runs=8, max_duration=50.0, max_skill_steps=10)
elif mode == "plot":
    metra_logs = load_logs_from_csv("results/high_level_metra.csv")
    dsd_logs = load_logs_from_csv("results/high_level_dsd.csv")

    logs_by_method = {
        "METRA": metra_logs,
        "DSD": dsd_logs
    }

    plot_multiple_methods_cumulative_reward(
        logs_by_method,
        max_duration=50.0,
        dt=1.0,
        save_path=f"results/high_level_{env_name}_comparison.png"
    )