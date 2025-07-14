import torch
import gym
import numpy as np
from stable_baselines3 import SAC
from gym.wrappers import Monitor
from envs.mujoco.ant_env import AntEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained option_policy and freeze weights
checkpoint = torch.load('exp/Debug/sd000_1752248887_ant_metra/option_policy3000.pt')
option_policy = checkpoint['policy']
option_policy.to(device)

option_policy.eval()
for param in option_policy.parameters():
    param.requires_grad = False


class HierarchicalEnvWrapper(gym.Env):
    def __init__(self, env, option_policy, skill_dim, skill_duration=20):
        super().__init__()
        self.env = env
        self.option_policy = option_policy
        self.skill_duration = skill_duration

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(skill_dim,), dtype=np.float32)  # for continuous skill_dim
        self.observation_space = env.observation_space

        self.current_step = 0
        self.current_skill = None
        self.current_obs = None  # store latest obs

    def reset(self):
        self.current_step = 0
        self.current_skill = None
        obs = self.env.reset()
        self.current_obs = obs
        return obs

    def step(self, action):
        self.current_skill = action
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.skill_duration):
            obs = self.current_obs
            obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(device)
            skill_tensor = torch.tensor(self.current_skill).float().unsqueeze(0).to(device)

            policy_input = torch.cat([obs_tensor, skill_tensor], dim=-1)

            with torch.no_grad():
                act = self.option_policy(policy_input)
                act = act.cpu().numpy()

            obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated
            total_reward += reward
            self.current_obs = obs

            if done:
                break

        return obs, total_reward, done, info


if __name__ == "__main__":
    base_env = AntEnv(render_hw=100)

    # Wrap with Monitor to save videos, recordings saved to ./videos/
    video_dir = "./videos"
    monitored_env = Monitor(base_env, video_dir, video_callable=lambda episode_id: True, force=True)

    skill_dim = 12
    skill_duration = 25

    hier_env = HierarchicalEnvWrapper(monitored_env, option_policy, skill_dim, skill_duration)

    sac_kwargs = dict(
        learning_rate=1e-4,
        buffer_size=int(1e5),
        batch_size=256,
        gamma=0.99,
        tau=0.995,
        ent_coef=0.01,
        train_freq=1,
        gradient_steps=100,
        verbose=1,
        seed=42,
        device='auto',
    )

    model = SAC('MlpPolicy', hier_env, **sac_kwargs)

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    model.save("high_level_policy_sac")

    obs = hier_env.reset()
    done = False

    while not done:
        skill, _ = model.predict(obs)
        obs, reward, done, info = hier_env.step(skill)
        # No need to call render, video will be saved automatically

    print(f"Video saved to {video_dir}")
