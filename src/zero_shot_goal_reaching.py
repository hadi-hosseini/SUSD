from argparse import Namespace
import torch
import numpy as np
import imageio

from downstream_tasks.ant_multi_goals import AntMultiGoalsEnv
from src.factorization import factorize_environment
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained option_policy
checkpoint = torch.load('exp/Debug/sd000_1752248887_ant_metra/option_policy19000.pt')
option_policy = checkpoint['policy']
option_policy.to(device)

# Load pretrained phi encoder 
checkpoint = torch.load('exp/Debug/sd000_1752248887_ant_metra/traj_encoder19000.pt')
traj_encoder = checkpoint['traj_encoder']
traj_encoder.to(device)

option_policy.eval()
traj_encoder.eval()

# Run up the Downstream Task
env = AntMultiGoalsEnv(render_hw=256)

normalizer_mean, normalizer_std = get_normalizer_preset(f'ant_preset')
env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)


obs = env.reset()

args = Namespace(env="ant")
partitions = factorize_environment(args)

# Method - Zero-shot Goal Reaching
done = False
step = 0
max_steps = 500
cumulative_reward = 0.0
frames = []

while not done and step < max_steps:
    # ---- 1. Get current state and goal ----
    current_pos = env.sim.data.qpos.flat[:2]
    goal = env.current_goal

    # ---- 2. Encode current and goal to skill space ----
    s_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    g_tensor = torch.from_numpy(np.copy(obs)).float().unsqueeze(0).to(device)
    g_tensor[:, 0] = torch.tensor(goal[0])  # Overwrite x
    g_tensor[:, 1] = torch.tensor(goal[1])  # Overwrite y

    phi_s = traj_encoder(s_tensor).detach()
    phi_g = traj_encoder(g_tensor).detach()

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

    frame = env.render(mode="rgb_array")
    frames.append(frame)

    print(f"Step {step:3d}: "
          f"action={np.round(action, 2)} "
          f"pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) "
          f"goal=({goal[0]:.2f}, {goal[1]:.2f}) "
          f"reward={reward:.2f}, done={done}")

    step += 1

env.close()

video_path = "zero_shot_ant_run.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"\n✅ Cumulative Reward: {cumulative_reward:.2f}")
print(f"🎞️ Video saved to: {video_path}")


