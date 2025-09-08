import torch
import torch.nn as nn

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
        state_factorization_points = [0, 29]
    elif args.env == "kitchen_franka":
        # state_factorization_points = [0, 18, 38, 44, 46, 59]
        state_factorization_points = [0, 59]
    elif args.env == "fetch":
        state_factorization_points = [0, 11, 25]
    elif args.env == "particle":
        state_factorization_points = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    elif args.env == "gunner":
        state_factorization_points = [0, 6, 13, 18]
    elif args.env == "elden_kitchen":
        ##### original ordering: 142 
        # object-state - 4: [robot0_eef_vel (3), button_joint_qpos (1)], [0-3]
        # object-state (pot) - 16: [pot_pos (3), pot_quat (4), pot_to_robot0_eef_pos (3), pot_to_robot0_eef_quat (4), pot_grasped (1), pot_touched (1)], [4, 19]
        # object-state (butter) - 18: [butter_pos (3), butter_quat (4), butter_to_robot0_eef_pos (3), butter_to_robot0_eef_quat (4), butter_grasped (1), butter_touched (1), butter_melt_status (1), butter_in_pot (1)] [20, 37]
        # object-state (meatball) - 19: [meatball_pos (3), meatball_quat (4), meatball_to_robot0_eef_pos (3), meatball_to_robot0_eef_quat (4), meatball_grasped (1), meatball_touched (1), meatball_cook_status (1), meatball_overcooked (1), meatball_in_pot (1)] [38, 56]
        # object-state (button) - 16: [button_pos (3), button_quat (4), button_to_robot0_eef_pos (3), button_to_robot0_eef_quat (4), button_grasped (1), button_touched (1)], [57, 72]
        # object-state (stove) - 14: [stove_pos (3), stove_quat (4), stove_to_robot0_eef_pos (3), stove_to_robot0_eef_quat (4)], [73, 86]
        # object-state (target) - 14: [target_pos (3), target_quat (4), target_to_robot0_eef_pos (3), target_to_robot0_eef_quat (4)], [87, 100]
        # object-state (pot) - 6: [pot_handle_pos (3), pot_handle_to_robot0_eef_pos (3)], [101, 106]
        # object-state (button) - 6: [button_handle_pos (3), button_handle_to_robot0_eef_pos (3)], [107, 112]
        # robot0_proprio-state - 29: [ robot0_joint_pos_cos (6), robot0_joint_pos_sin (6), robot0_joint_vel (6), robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_qpos (2), robot0_gripper_qvel (2)]. [113, 141]
        # [0, 4, 20, 38, 57, 73, 87, 101, 107, 113, 142]

        ##### custom order
        # robot0_proprio-state (robot) - 29: [ robot0_joint_pos_cos (6), robot0_joint_pos_sin (6), robot0_joint_vel (6), robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_qpos (2), robot0_gripper_qvel (2)]
        # object-state (robot) - 4: [robot0_eef_vel (3), button_joint_qpos (1)]
        # object-state (pot) - 16: [pot_pos (3), pot_quat (4), pot_to_robot0_eef_pos (3), pot_to_robot0_eef_quat (4), pot_grasped (1), pot_touched (1)]
        # object-state (pot) - 6: [pot_handle_pos (3), pot_handle_to_robot0_eef_pos (3)]
        # object-state (butter) - 18: [butter_pos (3), butter_quat (4), butter_to_robot0_eef_pos (3), butter_to_robot0_eef_quat (4), butter_grasped (1), butter_touched (1), butter_melt_status (1), butter_in_pot (1)]
        # object-state (meatball) - 19: [meatball_pos (3), meatball_quat (4), meatball_to_robot0_eef_pos (3), meatball_to_robot0_eef_quat (4), meatball_grasped (1), meatball_touched (1), meatball_cook_status (1), meatball_overcooked (1), meatball_in_pot (1)]
        # object-state (button) - 16: [button_pos (3), button_quat (4), button_to_robot0_eef_pos (3), button_to_robot0_eef_quat (4), button_grasped (1), button_touched (1)]
        # object-state (button) - 6: [button_handle_pos (3), button_handle_to_robot0_eef_pos (3)]
        # object-state (stove) - 14: [stove_pos (3), stove_quat (4), stove_to_robot0_eef_pos (3), stove_to_robot0_eef_quat (4)]
        # object-state (target) - 14: [target_pos (3), target_quat (4), target_to_robot0_eef_pos (3), target_to_robot0_eef_quat (4)]

        # custom_order = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 0, 1, 2, 3] # 29 arm + 4 don't know
        # custom_order += [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101, 102, 103, 104, 105, 106]  # 22 pot
        # custom_order += [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # 18 butter
        # custom_order += [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56] # 19 meatball
        # custom_order += [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 107, 108, 109, 110, 111, 112] # 22 button
        # custom_order += [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86] # 14 stove
        # custom_order += [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # 14 target 

        state_factorization_points = [0, 33, 55, 73, 92, 114, 128, 142] # robot, pot, butter, meatball, button, stove, target
    return state_factorization_points



class PartitionedTrajectoryEncoderWithInputFactor0(nn.Module):
    def __init__(self, args, partition_points, master_dims, nonlinearity, output_dim, module_cls_factory):
        super().__init__()
        self.partition_points = partition_points
        self.encoders = nn.ModuleList()

        for i in range(len(partition_points) - 1):
            start, end = partition_points[i], partition_points[i + 1]
            local_input_dim = end - start

            if i >= 1:
                factor0_dim = partition_points[1] - partition_points[0]
                module_cls, module_kwargs = module_cls_factory(args=args, 
                                                            master_dims=master_dims, 
                                                            nonlinearity=nonlinearity, 
                                                            input_dim=local_input_dim + factor0_dim,
                                                            output_dim=output_dim)
            else:
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
            if i >= 1:
                factor0_obs = obs[:, self.partition_points[0]: self.partition_points[1]]
                local_obs = obs[:, start:end]
                combined_obs = torch.cat([factor0_obs, local_obs], dim=1)
                dist = self.encoders[i](combined_obs)
            else:
                local_obs = obs[:, start:end]
                dist = self.encoders[i](local_obs)
            local_encoded = dist.mean
            outputs.append(local_encoded)

        final_encoding = torch.cat(outputs, dim=-1)
        return final_encoding


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