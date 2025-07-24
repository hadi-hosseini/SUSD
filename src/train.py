from dataclasses import dataclass, field
from typing import Optional, List
import os
import torch
import numpy as np
import functools

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

from iod.sac import SAC
from iod.ppo import PPO
from src.child_policy_env import ChildPolicyEnv
from src.conf import DSDConfig
from src.utils import get_exp_name, get_log_dir
from envs.mujoco.downstream_kitchen import KitchenEnv

from garagei.experiment.option_local_runner import OptionLocalRunner
from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garage.torch.modules import MLPModule
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.replay_buffer.path_buffer_ex import PathBufferEx


@dataclass
class DSDHighLevelConfig(DSDConfig):
    cp_path: Optional[str] = None
    cp_path_idx: Optional[int] = None  # For exp name
    cp_multi_step: int = 1
    cp_unit_length: int = 0

    downstream_reward_type: str = 'esparse'
    downstream_num_goal_steps: int = 50

    goal_range: float = 50.0

@dataclass
class METRAHighLevelKitchenConfig(DSDHighLevelConfig):
    max_path_length: int = 8
    dim_option: int = 2
    n_parallel: int = 1 # 4 is better
    algo: str = "sac"
    n_epochs_per_eval: int = 100
    n_epochs_per_save: int = 0
    n_epochs_per_pt_save: int = 0
    n_epochs_per_pkl_update: int = 0
    n_epochs: int = 200001 # 16000 is better
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-50, 50, -50, 50])
    trans_optimization_epochs: int = 50
    te_trans_optimization_epochs: int = 50
    sac_replay_buffer: int = 1
    sac_max_buffer_size: int = 1000000
    joint_train: int = 1
    te_only_last_frame: int  = 0
    goal_range: float = 7.5
    alpha: float = 0.1
    cp_multi_step: int = 25
    downstream_reward_type: str = "esparse"
    downstream_num_goal_steps: int = 50
    cp_path: str = "exp/path/option_policy10000.pt"
    cp_path_idx: int = 0
    cp_unit_length: int = 1



hl_kitchen_config = METRAHighLevelKitchenConfig()


@wrap_experiment(log_dir=get_log_dir(hl_kitchen_config), name=get_exp_name(hl_kitchen_config)[0])
def run(ctxt=None):
    if hl_kitchen_config.n_thread is not None:
        torch.set_num_threads(hl_kitchen_config.n_thread)

    
    def _finalize_lr(lr):
        if lr is None:
            lr = hl_kitchen_config.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr


    set_seed(hl_kitchen_config.seed)

    runner = OptionLocalRunner(ctxt)


    all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


    env = KitchenEnv(
        tasks_to_complete=all_tasks,
        terminate_on_tasks_completed=True,
        render_mode="rgb_array"
    )

    custom_order = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,     # Panda Arm and Gripper States
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Burners and Overhead Light
                29, 30, 31, 49, 50, 51,                                           # Cabinets (Slide + Left + Right Hinge)
                32, 52,                                                          # Microwave Door
                33, 34, 35, 36, 37, 38, 39, 53, 54, 55, 56, 57, 58               # Kettle
    ]



    max_path_length = hl_kitchen_config.max_path_length
    if hl_kitchen_config.cp_path is not None:
        max_path_length *= hl_kitchen_config.cp_multi_step


    contextualized_make_env = functools.partial(lambda: env, args=hl_kitchen_config, max_path_length=max_path_length)
    env = contextualized_make_env()

    if hl_kitchen_config.cp_path is not None:
        cp_path = hl_kitchen_config.cp_path
        if not os.path.exists(cp_path):
            import glob
            cp_path = glob.glob(cp_path)[0]
        cp_dict = torch.load(cp_path, map_location='cpu')

        env = ChildPolicyEnv(
            env,
            cp_dict,
            cp_action_range=1.5,
            cp_unit_length=hl_kitchen_config.cp_unit_length,
            cp_multi_step=hl_kitchen_config.cp_multi_step,
            cp_num_truncate_obs=0,
        )

    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim
    module_obs_dim = obs_dim



    if hl_kitchen_config.algo in ['sac', 'ppo']:
        policy_q_input_dim = module_obs_dim # obs_dim (should be fixed)


    if hl_kitchen_config.algo == "sac":
        pass

    device = torch.device('cuda' if hl_kitchen_config.use_gpu else 'cpu')
    master_dims = [hl_kitchen_config.model_master_dim] * hl_kitchen_config.model_master_num_layers


    if hl_kitchen_config.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif hl_kitchen_config.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None



    if hl_kitchen_config.algo == "sac":
        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        log_alpha = ParameterModule(torch.Tensor([np.log(hl_kitchen_config.alpha)]))

        vf = MLPModule(
            input_dim=policy_q_input_dim,
            output_dim=1,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            layer_normalization=hl_kitchen_config.critic_layer_norm,
        )

        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(hl_kitchen_config.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(hl_kitchen_config.sac_lr_a)},
            ]),
            'vf': torch.optim.Adam([
                {'params': vf.parameters(), 'lr': _finalize_lr(hl_kitchen_config.lr_op)},
            ]),
        })

    elif hl_kitchen_config.algo == 'ppo':
        vf = MLPModule(
            input_dim=policy_q_input_dim,
            output_dim=1,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            layer_normalization=hl_kitchen_config.critic_layer_norm,
        )
        optimizers = ({
            'vf': torch.optim.Adam([
                {'params': vf.parameters(), 'lr': _finalize_lr(hl_kitchen_config.lr_op)},
            ]),
        })


    optimizer = OptimizerGroupWrapper(
            optimizers=optimizers,
            max_optimization_epochs=None,
        )


    algo_kwargs = dict(
            env_name=hl_kitchen_config.env,
            algo=hl_kitchen_config.algo,
            env_spec=env.spec,
            # option_policy=option_policy,
            # traj_encoder=traj_encoder,
            # skill_dynamics=skill_dynamics,
            # dist_predictor=dist_predictor,
            # dual_lam=dual_lam,
            optimizer=optimizer,
            alpha=hl_kitchen_config.alpha,
            max_path_length=hl_kitchen_config.max_path_length,
            n_epochs_per_eval=hl_kitchen_config.n_epochs_per_eval,
            n_epochs_per_log=hl_kitchen_config.n_epochs_per_log, 
            n_epochs_per_tb=hl_kitchen_config.n_epochs_per_log, 
            n_epochs_per_save=hl_kitchen_config.n_epochs_per_save, 
            n_epochs_per_pt_save=hl_kitchen_config.n_epochs_per_pt_save, 
            n_epochs_per_pkl_update=hl_kitchen_config.n_epochs_per_eval if hl_kitchen_config.n_epochs_per_pkl_update is None else hl_kitchen_config.n_epochs_per_pkl_update,
            dim_option=hl_kitchen_config.dim_option,
            N = hl_kitchen_config.N,
            num_random_trajectories=hl_kitchen_config.num_random_trajectories,
            num_video_repeats=hl_kitchen_config.num_video_repeats,
            eval_record_video=hl_kitchen_config.eval_record_video,
            video_skip_frames=hl_kitchen_config.video_skip_frames,
            eval_plot_axis=hl_kitchen_config.eval_plot_axis,
            name='sac',
            device=device,
            sample_cpu=hl_kitchen_config.sample_cpu,
            num_train_per_epoch=1,
            sd_batch_norm=hl_kitchen_config.sd_batch_norm,
            # skill_dynamics_obs_dim=skill_dynamics_obs_dim,
            trans_minibatch_size=hl_kitchen_config.trans_minibatch_size,
            trans_optimization_epochs=hl_kitchen_config.trans_optimization_epochs,
            discount=hl_kitchen_config.sac_discount,
            discrete=hl_kitchen_config.discrete,
            unit_length=hl_kitchen_config.unit_length,
        )
    

    replay_buffer = PathBufferEx(capacity_in_transitions=int(hl_kitchen_config.sac_max_buffer_size), pixel_shape=None)

    
    sac_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=hl_kitchen_config.sac_tau,
        scale_reward=hl_kitchen_config.sac_scale_reward,
        target_coef=hl_kitchen_config.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=hl_kitchen_config.sac_min_buffer_size,
        inner=hl_kitchen_config.inner,

        num_alt_samples=hl_kitchen_config.num_alt_samples,
        split_group=hl_kitchen_config.split_group,

        # dual_reg=hl_kitchen_config.dual_reg,
        # dual_slack=hl_kitchen_config.dual_slack,
        # dual_dist=hl_kitchen_config.dual_dist,

        pixel_shape=None,
        # partition_points=partition_points

    )

    if hl_kitchen_config.algo == "sac":
        algo = SAC(**sac_args)
        
    elif hl_kitchen_config.algo == 'ppo':
        algo = PPO(
            **algo_kwargs,
            vf=vf,
            gae_lambda=0.95,
            ppo_clip=0.2,
        )
    

    if hl_kitchen_config.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)

    runner.setup(
            algo=algo,
            env=env,
            make_env=contextualized_make_env,
            sampler_cls=OptionMultiprocessingSampler,
            sampler_args=dict(n_thread=hl_kitchen_config.n_thread),
            n_workers=hl_kitchen_config.n_parallel,
        )

    algo.option_policy.to(device)
    runner.train(n_epochs=hl_kitchen_config.n_epochs, batch_size=hl_kitchen_config.traj_batch_size)

