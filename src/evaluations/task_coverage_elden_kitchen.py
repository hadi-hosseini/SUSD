import os
import numpy as np
import torch
import imageio

from envs.elden_kitchen.elden_kitchen import kitchen_env

from src.dusdi_utils import Actor

import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "eval" # ["plot", "eval"]
algo = "susd" # ["susd", "csd", "metra", "lsd", "dusdi"]
skill_dim = 2

if algo == "susd":
    skill_policy_path = 'final_models/elden_kitchen/SUSD/option_policy10000.pt'    
    skill_dim = 14 # N=7 & d=2

elif algo == "metra": 
    skill_policy_path = 'final_models/elden_kitchen/METRA/option_policy10000.pt'    

elif algo == "csd":
    skill_policy_path = 'final_models/elden_kitchen/CSD/option_policy10000.pt'    

elif algo == "lsd":
    skill_policy_path = 'final_models/elden_kitchen/LSD/option_policy10000.pt'    

elif algo == "diayn":
    skill_policy_path = 'final_models/elden_kitchen/DIAYN/option_policy10000.pt'    

elif algo == "dusdi":
    skill_policy_path = 'final_models/elden_kitchen/DUSDI/option_policy10000.pt'


if algo == "dusdi":
    low_option_policy = Actor("state", 177, 4, 35, 1024, True, [-10, 2], "elden")
    cp_dict = torch.load(skill_policy_path, map_location='cpu')
    low_option_policy.load_state_dict(cp_dict)
    skill_dim = 35

else:
    low_policy = torch.load(skill_policy_path)
    low_option_policy = low_policy["policy"]

low_level_policy = low_option_policy.to(device).eval()

custom_order = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 0, 1, 2, 3] # 29 arm + 4 don't know
custom_order += [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101, 102, 103, 104, 105, 106]  # 22 pot
custom_order += [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # 18 butter
custom_order += [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56] # 19 meatball
custom_order += [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 107, 108, 109, 110, 111, 112] # 22 button
custom_order += [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86] # 14 stove
custom_order += [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # 14 target 


def rearrange_vector(vec, custom_order):
    if isinstance(vec, torch.Tensor):
        indices = torch.tensor(custom_order, device=vec.device, dtype=torch.long)
        return vec[indices]
    elif isinstance(vec, np.ndarray):
        return vec[custom_order]
    elif isinstance(vec, list):
        return [vec[i] for i in custom_order]
    else:
        raise TypeError("Unsupported type for vec. Must be torch.Tensor, numpy.ndarray, or list.")

def random_one_hot_concat(N, d):
    import random
    indices = [random.choice(range(d)) for _ in range(N)]
    one_hot = np.zeros((N, d), dtype=int)
    one_hot[np.arange(N), indices] = 1
    return one_hot.reshape(1, -1)

def task_coverage_elden_policy(env):
    record_video = False
    done = True
    frames = []
    steps = 0
    solved_counter = 0
    max_steps = 1e4

    while steps <= max_steps:
        if done:
            obs = env.reset()
            done = False

            if algo == "dusdi":
                random_z = random_one_hot_concat(N=7, d=5)
            else:    
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
                # random_z = np.random.randn(1, skill_dim)
            random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
        else:            
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            input_tensor = torch.cat([obs, random_z], dim=-1)

            with torch.no_grad():
                if algo == "dusdi":
                    action_dist = low_option_policy(input_tensor)
                    action = action_dist.mean.detach().cpu().numpy()
                    action = action[0]
                else:
                    action_np, _ = low_option_policy.get_action(input_tensor)
                    action = action_np[0]
            
            obs, reward, done, info = env.step(action)
            # print(f"Step {steps}:")
            # print(f"  Reward: {reward}")
            # print(f" Done: {done}")
            if reward:
                solved_counter += 1

            if done:
                obs = env.reset()
                break

            steps += 1

            if steps % 50 == 0:
                done = True

            if record_video:
                frame = env.render()
                frames.append(frame)


    if record_video:
        video_path = f"test_elden_kitchen_{algo}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎞️ Video saved to: {video_path}")

    return solved_counter


with kitchen_env(custom_order=custom_order, reward_scale=1.0, horizon=1e4, render=False, downstream_task=1) as env:
    c = 0
    all_rewards = []
    for _ in range(8):
        r = task_coverage_elden_policy(env)
        all_rewards.append(r)
        print(f"Numbers of Done is: {r}")
        c += r
    mean_r = np.mean(all_rewards)
    std_r = np.std(all_rewards)
    print(f"Mean: {mean_r:.4f}, Std: {std_r:.4f}")


# 1: BiP
# 2: MiP
# 3: PoS
# 6: PoT