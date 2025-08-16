import torch

from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import (CentralizedWrapper,
                                                               DownstreamCentralizedWrapper,
                                                               SequentialDSWrapper)
import wrapper
from custom_env.hierarchical_env_wrapper import FlatEnvWrapper
from particle import Particle


seed = 0
N = 10
render_mode = "rgb_array"
simplify_action_space = True
frame_size = (512, 480)

distances = list(range(0, 10))       # 0–9
agent_info = list(range(10, 50))     # 10–49
station_info = list(range(50, 70))   # 50–69

custom_order = []

for i in range(10):
    custom_order.append(distances[i])                       
    custom_order.extend(agent_info[i*4:(i+1)*4])            
    custom_order.extend(station_info[i*2:(i+1)*2])  

# env = simple_heterogenous_v3.parallel_env(
#     render_mode=render_mode,
#     max_cycles=1000,
#     continuous_actions=True,
#     local_ratio=0,
#     N=N,
#     img_encoder=None)

# env = CentralizedWrapper(env, simplify_action_space=simplify_action_space)
# env = Particle(env, custom_order, frame_size)
# obs = env.reset(seed=seed)

# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)

# video_filename = "random_actions_imageio.mp4"
# fps = 30
# max_steps = 50
# import imageio

# writer = imageio.get_writer(video_filename, fps=fps)

# for step in range(max_steps):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
    
#     print(f"Step {step+1}")
#     print("Info:", info)
#     print("Reward:", reward)
#     print("Done:", done)
#     print()
    
#     frame = env.render()
#     writer.append_data(frame)
    
#     if done:
#         print("Episode done at step", step+1)
#         break

# writer.close()
# print(f"Video saved as {video_filename}")


#### downstream task for multiparticle
N = 10
landmark_id = range(N)
low_level_step = 50
factorize = False
env = simple_heterogenous_v3.parallel_env(
        render_mode='rgb_array',
        max_cycles=1000,
        continuous_actions=True,
        local_ratio=0,
        N=10
    )

env = DownstreamCentralizedWrapper(env, landmark_id=range(10), N=10, factorize=False, simplify_action_space=True)
# env = FlatEnvWrapper(env, low_level_step)
env = Particle(env, custom_order, frame_size)
init_obs = env.reset(seed=0)

# if ds_task == "poison_s":
#     agent_list = [1]
# elif ds_task == "poison_m":
#     agent_list = [0, 2, 4, 6, 9]
# elif ds_task == "poison_l":
#     agent_list = [0, 1, 2, 3, 4, 6, 7, 9]
