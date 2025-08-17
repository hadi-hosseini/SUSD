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


def compute_dci(mus_train, ys_train, partition):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err = compute_importance_gbt(mus_train, ys_train, partition)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores


def compute_importance_gbt(x_train, y_train, partition):
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


    partition = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]


    x = code.T
    y = skill.T

    # x = discretize(x, 2)
    # y = discretize(y, 2)

    print(compute_dci(x, y, partition))


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

algo = "susd" # ["csd", "metra", "lsd", "diyan", "susd"]
skill_dim = 2

if algo == "susd":
    option_policy_checkpoint_path = f'../../../final_models/particle/SUSD/option_policy6000.pt'
    traj_encoder_checkpoint_path = f'../../../final_models/particle/SUSD/traj_encoder6000.pt'
    skill_dim = 20 # N=10 & d=2

elif algo == "metra": 
    option_policy_checkpoint_path = '../../../final_models/particle/METRA/option_policy10000.pt'    
    traj_encoder_checkpoint_path = '../../../final_models/particle/METRA/traj_encoder10000.pt'

elif algo == "csd":
    option_policy_checkpoint_path = '../../../final_models/particle/CSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = '../../../final_models/particle/CSD/traj_encoder10000.pt'

elif algo == "lsd":
    option_policy_checkpoint_path = '../../../final_models/particle/LSD/option_policy10000.pt'    
    traj_encoder_checkpoint_path = '../../../final_models/particle/LSD/traj_encoder10000.pt'

elif algo == "diayn":
    option_policy_checkpoint_path = '../../../final_models/particle/DIAYN/option_policy10000.pt'    
    traj_encoder_checkpoint_path = '../../../final_models/particle/DIAYN/traj_encoder10000.pt'


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


def eval_and_save(fn):
    done = True
    steps = 0
    z_period = 50
    seed = 0
    env = create_particle_env(seed)
    obs_list = []
    skill_list = []

    with tqdm(total=int(50000), desc="Evaluating env") as pbar:
        while steps <= 50000:
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
                pbar.update(1)

            # Save observation and skill
            obs_list.append(obs)                 # shape: [state_dim]
            skill_list.append(random_z.cpu().numpy()[0])  # shape: [skill_dim]

        # Convert lists to arrays
        obs_array = np.array(obs_list)         # [num_steps, state_dim]
        skill_array = np.array(skill_list)     # [num_steps, skill_dim]

    np.savez(fn, obs=obs_array, skill=skill_array)
    print(f"Saved {obs_array.shape[0]} steps to {fn}")


fn = f"test_disentanglement/{algo}.npz"
eval_and_save(fn)

dci(fn)
