import torch
import torch.nn as nn
import numpy as np

from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx

def get_gaussian_module_construction(args,
                                     *,
                                     hidden_sizes,
                                     const_std=False,
                                     hidden_nonlinearity=torch.relu,
                                     w_init=torch.nn.init.xavier_uniform_,
                                     init_std=1.0,
                                     min_std=1e-6,
                                     max_std=None,
                                     **kwargs):
    module_kwargs = dict()
    if const_std:
        module_cls = GaussianMLPModuleEx
        module_kwargs.update(dict(
            learn_std=False,
            init_std=init_std,
        ))
    else:
        module_cls = GaussianMLPIndependentStdModuleEx
        module_kwargs.update(dict(
            std_hidden_sizes=hidden_sizes,
            std_hidden_nonlinearity=hidden_nonlinearity,
            std_hidden_w_init=w_init,
            std_output_w_init=w_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
        ))

    module_kwargs.update(dict(
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_w_init=w_init,
        output_w_init=w_init,
        std_parameterization='exp',
        bias=True,
        spectral_normalization=args.spectral_normalization,
        **kwargs,
    ))
    return module_cls, module_kwargs


def factorize_environment(args):
    if args.env == "ant":
        state_factorization_points = [0, 3, 7, 15, 18, 21, 29]
        if args.env == "franka_kitchen":
            custom_order = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,     # Panda Arm and Gripper States
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Burners and Overhead Light
                29, 30, 31, 49, 50, 51,                                           # Cabinets (Slide + Left + Right Hinge)
                32, 52,                                                          # Microwave Door
                33, 34, 35, 36, 37, 38, 39, 53, 54, 55, 56, 57, 58               # Kettle
            ]
            partition_start_indices = [0, 18, 37, 43, 45]
    return state_factorization_points


class PartitionedTrajectoryEncoder(nn.Module):
    def __init__(self, args, partition_points, master_dims, nonlinearity, output_dim, module_cls_factory):
        super().__init__()
        self.partition_points = partition_points
        self.encoders = nn.ModuleList()

        for i in range(len(partition_points) - 1):
            start, end = partition_points[i], partition_points[i + 1]
            local_input_dim = end - start

            module_cls, module_kwargs = module_cls_factory(args=args, 
                                                           master_dims=master_dims, 
                                                           nonlinearity=nonlinearity, 
                                                           input_dim=local_input_dim,
                                                           output_dim=output_dim)

            self.encoders.append(module_cls(**module_kwargs))

    def forward(self, obs):
        outputs = []
        for i in range(len(self.partition_points) - 1):
            start, end = self.partition_points[i], self.partition_points[i + 1]
            local_obs = obs[:, start:end]
            dist = self.encoders[i](local_obs)
            local_encoded = dist.mean
            outputs.append(local_encoded)

        final_encoding = torch.cat(outputs, dim=-1)
        return final_encoding


def module_cls_factory(args, master_dims, nonlinearity, input_dim, output_dim):
    return get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=input_dim,
        output_dim=output_dim,
    )