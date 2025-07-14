import os
import gym
import numpy as np
import torch
from gym import spaces
import imageio

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from iod.utils import get_normalizer_preset
from garagei.envs.consistent_normalized_env import consistent_normalize
from downstream_tasks.ant_multi_goals import AntMultiGoalsEnv 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

option_ckpt = torch.load("exp/Debug/sd000_1752248887_ant_metra/option_policy19000.pt")
traj_ckpt = torch.load("exp/Debug/sd000_1752248887_ant_metra/traj_encoder19000.pt")
option_policy = option_ckpt["policy"]
traj_encoder = traj_ckpt["traj_encoder"]

env = AntMultiGoalsEnv(render_hw=256)
mean, std = get_normalizer_preset("ant_preset")
env = consistent_normalize(env, normalize_obs=True, mean=mean, std=std)

skill_dim = 12 # N=6, d=2
max_steps_per_goal = 25 # maximum number of steps for each z


class SkillWrapperEnv(gym.Env):
    def __init__(self, env, option_policy, traj_encoder, skill_dim, max_steps_per_goal, device='cpu'):
        super().__init__()
        self.env = env
        self.option_policy = option_policy.to(device).eval()
        self.traj_encoder = traj_encoder.to(device).eval()
        self.device = device
        self.skill_dim = skill_dim
        self._max_skill_steps = max_steps_per_goal 
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
    log_dir = "logs/sac_high_level"
    model_dir = os.path.join(log_dir, "models")
    tensorboard_log_dir = os.path.join(log_dir, "tb")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    wrapped_env = DummyVecEnv([lambda: SkillWrapperEnv(env, option_policy, traj_encoder, skill_dim, max_steps_per_goal, device)])

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
            self.episode_reward = 0.0

        def _on_step(self) -> bool:
            reward = self.locals.get('reward')
            done = self.locals.get('done')

            if reward is not None:
                self.episode_reward += reward

            if done:
                self.logger.record('custom/episode_reward', self.episode_reward)
                self.episode_reward = 0.0  # reset for next episode
            return True
    
    reward_callback = RewardLoggingCallback()
    sac_model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, reward_callback])
    sac_model.save(os.path.join(model_dir, "sac_highlevel_final"))

def eval():
    sac_model = SAC.load("sac_high_level_ant_multi_goals", device=device)
    wrapped_env = SkillWrapperEnv(env, option_policy, traj_encoder, skill_dim, max_steps_per_goal, device)

    num_eval_episodes = 20
    successes = []
    rewards = []
    video_frames = [] 
    record_video = True

    for ep in range(num_eval_episodes):
        obs = wrapped_env.reset()
        done = False
        ep_reward = 0.0
        ep_success = 0
        ep_frames = []

        while not done:
            z, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, done, info = wrapped_env.step(z)
            ep_reward += reward
            if "current_goal" in info:
                ep_success += 1

            if record_video:
                frame = env.render(mode="rgb_array")
                ep_frames.append(frame)

        rewards.append(ep_reward)
        successes.append(ep_success)
        print(f"Episode {ep+1}/{num_eval_episodes} - Reward: {ep_reward:.2f} - Goals Reached: {ep_success}")

        if record_video:
            video_frames.extend(ep_frames)

    avg_reward = np.mean(rewards)
    avg_success = np.mean(successes)
    print(f"\n✅ Evaluation over {num_eval_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Goals Reached: {avg_success:.2f} per episode")

    if record_video:
        video_path = "eval_sac_highlevel_ant.mp4"
        imageio.mimsave(video_path, video_frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")


train()