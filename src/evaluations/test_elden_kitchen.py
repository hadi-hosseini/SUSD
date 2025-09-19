import os
import numpy as np
import torch
import imageio

from envs.elden_kitchen.elden_kitchen import kitchen_env

import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "eval" # ["plot", "eval"]
algo = "susd" # ["csd", "metra", "lsd", "diyan", "susd"]
skill_dim = 2

if algo == "susd":
    high_option_policy_path = f'/home/hadi/rl/SUSD/exp/HRL_SUSD_elden_BiP/sd000_1757762930_elden_kitchen_sac/option_policy9000.pt'
    traj_encoder_checkpoint_path = f'/home/hadi/rl/SUSD/exp/HRL_SUSD_elden_BiP/sd000_1757762930_elden_kitchen_sac/traj_encoder9000.pt'

    low_level_policy_path = 'final_models/elden_kitchen/SUSD/option_policy10000.pt'    
    skill_dim = 14 # N=7 & d=2

elif algo == "metra": 
    high_option_policy_path = 'final_models/elden_kitchen/METRA/option_policy6000.pt'    
    traj_encoder_checkpoint_path = 'final_models/elden_kitchen/METRA/traj_encoder6000.pt'

elif algo == "csd":
    high_option_policy_path = 'final_models/elden_kitchen/CSD/option_policy6000.pt'    
    traj_encoder_checkpoint_path = 'final_models/elden_kitchen/CSD/traj_encoder6000.pt'

elif algo == "lsd":
    high_option_policy_path = 'final_models/elden_kitchen/LSD/option_policy6000.pt'    
    traj_encoder_checkpoint_path = 'final_models/elden_kitchen/LSD/traj_encoder6000.pt'

elif algo == "diayn":
    high_option_policy_path = 'final_models/elden_kitchen/DIAYN/option_policy6000.pt'    
    traj_encoder_checkpoint_path = 'final_models/elden_kitchen/DIAYN/traj_encoder6000.pt'


high_policy = torch.load(high_option_policy_path)
traj_ckpt = torch.load(traj_encoder_checkpoint_path)

low_policy = torch.load(low_level_policy_path)

high_option_policy = high_policy["policy"]
low_option_policy = low_policy["policy"]
traj_encoder = traj_ckpt["traj_encoder"]
high_option_policy = high_option_policy.to(device).eval()
low_level_policy = low_option_policy.to(device).eval()
traj_encoder = traj_encoder.to(device).eval()

custom_order = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 0, 1, 2, 3] # 29 arm + 4 don't know
custom_order += [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101, 102, 103, 104, 105, 106]  # 22 pot
custom_order += [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # 18 butter
custom_order += [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56] # 19 meatball
custom_order += [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 107, 108, 109, 110, 111, 112] # 22 button
custom_order += [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86] # 14 stove
custom_order += [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # 14 target 


def test_elden_policy(env):
    record_video = True
    done = True
    frames = []
    steps = 0
    max_steps = 1000

    while steps <= max_steps:
        if done:
            obs = env.reset()
            done = False
        else:
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                skill_np, _ = high_option_policy.get_action(obs)
                skill_torch = torch.tensor(skill_np, dtype=torch.float32).to(device)
                input_tensor = torch.cat([obs, skill_torch], dim=-1)

                for _ in range(5):
                    action_np, _ = low_option_policy.get_action(input_tensor)
                    cp_action_norm = np.linalg.norm(action_np[0])
                    action_np = action_np[0] / cp_action_norm
                    lb, ub = env.action_space.low, env.action_space.high
                    action = lb + (action_np + 1) * (0.5 * (ub - lb))
                    action = np.clip(action, lb, ub)
                    obs, reward, done, info = env.step(action)
                    print(f"Step {steps}:")
                    print(f"  Reward: {reward}")
                    print(f"  Done: {done}")
                    if reward:
                        return

            if done:
                print("Episode finished!")
                obs = env.reset()
                break

            steps += 1

            if record_video:
                frame = env.render()
                frames.append(frame)

    if record_video:
        video_path = f"test_elden_kitchen_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")


with kitchen_env(custom_order=custom_order, reward_scale=1.0, horizon=1000, render=True, downstream_task=1) as env:
    test_elden_policy(env)