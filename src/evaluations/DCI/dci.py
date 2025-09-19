# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
import scipy
# from six.moves import range
from sklearn import ensemble

from src.dusdi_utils import Actor


def compute_dci(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err = compute_importance_gbt(mus_train, ys_train)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores


def compute_importance_gbt(x_train, y_train):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)

  train_loss = []
  for i in range(num_factors):
    model = ensemble.GradientBoostingRegressor()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
  return importance_matrix, np.mean(train_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

def dci(fn):
    import numpy as np
    import glob

    for filename in glob.glob(fn):
        ts = np.load(filename)

    skill = ts.get("skill")
    obs = ts.get("obs")
    n_points = obs.shape[0]
    print(f"evaluating on {n_points} data")

    code = obs
    
    x = code.T
    y = skill.T

    # x = discretize(x, 2)
    # y = discretize(y, 2)

    print(compute_dci(x, y))


def discretize(X, num_bins=5):
    X_discrete = np.zeros_like(X, dtype=int)
    for i in range(X.shape[0]):
        # np.digitize returns bin index (1..num_bins)
        bins = np.linspace(X[i, :].min(), X[i, :].max(), num_bins+1)
        X_discrete[i, :] = np.digitize(X[i, :], bins) - 1  # convert to 0-based index
    return X_discrete


import torch
from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper
from envs.mp.particle import Particle
from tqdm import tqdm

import os
os.environ["MUJOCO_GL"] = "egl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = "dusdi" # ["csd", "metra", "lsd", "diyan", "susd", "dusdi"]
env_name = "particle" # ["elden", "particle"]
skill_dim = 2

if algo == "csd":
    if env_name == "elden":
        option_policy_checkpoint_path = f'final_models/elden_kitchen/SUSD/option_policy10000.pt'
        skill_dim = 14 # elden
    else:
        option_policy_checkpoint_path = f'final_models/particle/SUSD/option_policy10000.pt'
        skill_dim = 20 # N=10 & d=2 particle

elif algo == "dusdi":
    if env_name == "elden":
        option_policy_checkpoint_path = f'final_models/elden_kitchen/DUSDI/option_policy10000.pt'
        skill_dim= 35 # elden N=7 & d=5 
    else:
        option_policy_checkpoint_path = f'final_models/particle/DUSDI/option_policy10000.pt'
        skill_dim = 50 # N=10 & d=5

elif algo == "metra": 
    if env_name == "elden":
        option_policy_checkpoint_path = 'final_models/elden_kitchen/METRA/option_policy10000.pt' 
    else:   
        option_policy_checkpoint_path = 'final_models/particle/METRA/option_policy10000.pt'    

elif algo == "csd":
    if env_name == "elden":
        option_policy_checkpoint_path = 'final_models/elden_kitchen/CSD/option_policy10000.pt'    
    else:
        option_policy_checkpoint_path = 'final_models/particle/CSD/option_policy10000.pt'    

elif algo == "lsd":
    if env_name == "elden":
        option_policy_checkpoint_path = 'final_models/elden_kitchen/LSD/option_policy10000.pt'    
    else:
        option_policy_checkpoint_path = 'final_models/particle/LSD/option_policy10000.pt'    

elif algo == "diayn":
    if env_name == "elden":
        option_policy_checkpoint_path = 'final_models/elden_kitchen/DIAYN/option_policy10000.pt'    
    else:
        option_policy_checkpoint_path = 'final_models/particle/DIAYN/option_policy10000.pt'    


if algo == "dusdi":
    if env_name == "elden":
        option_policy = Actor("state", 177, 4, 35, 1024, True, [-10, 2], "elden")
    else:
        option_policy = Actor("state", 120, 20, 50, 1024, True, [-10, 2], "particle")
    cp_dict = torch.load(option_policy_checkpoint_path, map_location='cpu')
    option_policy.load_state_dict(cp_dict)
    option_policy = option_policy.to(device).eval()
else:
    option_ckpt = torch.load(option_policy_checkpoint_path)
    option_policy = option_ckpt["policy"]
    option_policy = option_policy.to(device).eval()


def create_particle_env(seed=0):

    distances = list(range(0, 10))       # 0–9
    agent_info = list(range(10, 50))     # 10–49
    station_info = list(range(50, 70))   # 50–69

    custom_order = []

    for i in range(10):
        custom_order.append(distances[i])                       
        custom_order.extend(agent_info[i*4:(i+1)*4])            
        custom_order.extend(station_info[i*2:(i+1)*2])

    env = simple_heterogenous_v3.parallel_env(
            render_mode= "rgb_array",
            max_cycles=1000,
            continuous_actions=True,
            local_ratio=0,
            N=10,
            img_encoder=None)

    env = CentralizedWrapper(env, simplify_action_space=True)
    env = Particle(env, custom_order, (512, 480))

    print(env.observation_space)
    exit()

    return env


def create_elden_env(seed=0):
    from envs.elden_kitchen.elden_kitchen import elden_kitchen, EldenKitchen
    env = elden_kitchen(reward_scale=0.0, horizon=50, render=False) # reward_scale = 0.0 is used for USD
    custom_order = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 0, 1, 2, 3] # 29 arm + 4 don't know
    custom_order += [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101, 102, 103, 104, 105, 106]  # 22 pot
    custom_order += [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # 18 butter
    custom_order += [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56] # 19 meatball
    custom_order += [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 107, 108, 109, 110, 111, 112] # 22 button
    custom_order += [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86] # 14 stove
    custom_order += [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # 14 target 
    
    env = EldenKitchen(env, custom_order=custom_order) 
    return env


def eval_and_save(fn):
    done = True
    steps = 0
    z_period = 50
    seed = 0
    if env_name == "elden":
        env = create_elden_env(seed)
    else:
        env = create_particle_env(seed)
    obs_list = []
    skill_list = []

    with tqdm(total=int(100000), desc="Evaluating env") as pbar:
        while steps <= 100000: # 100K rollout steps
            if done or steps % 250 == 0:
                obs = env.reset()
                done = False
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
                random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
                steps += 1
            else:
                if steps % z_period ==0:
                    random_z = np.random.randn(1, skill_dim)
                    random_z /= np.linalg.norm(random_z)
                    random_z = torch.tensor(random_z, dtype=torch.float32).to(device)

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
                pbar.update(1)

            # Save observation and skill
            obs_list.append(obs)                 # shape: [state_dim]
            skill_list.append(random_z.cpu().numpy()[0])  # shape: [skill_dim]

        # Convert lists to arrays
        obs_array = np.array(obs_list)         # [num_steps, state_dim]
        skill_array = np.array(skill_list)     # [num_steps, skill_dim]

    np.savez(fn, obs=obs_array, skill=skill_array)
    print(f"Saved {obs_array.shape[0]} steps to {fn}")


fn = f"src/evaluations/DCI/test_disentanglement/{algo}.npz"
eval_and_save(fn)
dci(fn)