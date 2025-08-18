import numpy as np
import imageio
from envs.moma_2d.moma_2d_gym_env import MoMa2DGymEnv

custom_order = [0, 1, 2, 3, 12, 13, 4, 5, 6, 7, 14, 15, 16, 8, 9, 10, 11, 17] # base, arm, view
env = MoMa2DGymEnv(max_step=1000, custom_order=custom_order)
s = env.reset()

frames = [] 

for i in range(100):
    a = env.action_space.sample()
    s, r, done, info = env.step(a)

    # Capture frame
    frame = env.render(mode="rgb_array")
    frames.append(frame)

    if done:
        s = env.reset()

env.close()

video_name = "moma2d_video.mp4"
imageio.mimsave(video_name, frames, fps=30)

print(f"Video saved as {video_name}")
