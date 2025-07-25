from dataclasses import dataclass, field
from typing import Optional, List
import os
import torch
import torch.multiprocessing as mp
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
from downstream_tasks.downstream_kitchen import DownstreamKitchen

from garagei.experiment.option_local_runner import OptionLocalRunner
from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garage.torch.modules import MLPModule
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx
from garagei.torch.utils import xavier_normal_ex
from garage.torch.distributions import TanhNormal
from garagei.torch.policies.policy_ex import PolicyEx



if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'

@dataclass
class DSDHighLevelConfig(DSDConfig):
    cp_path: Optional[str] = None
    cp_path_idx: Optional[int] = None  # For exp name
    cp_multi_step: int = 1
    cp_unit_length: int = 0
    cp_multitask: int = 0

    downstream_reward_type: str = 'esparse'
    downstream_num_goal_steps: int = 50

    goal_range: float = 50.0

@dataclass
class METRAHighLevelKitchenConfig(DSDHighLevelConfig):
    run_group: str = "HIGH_LEVEL"
    max_path_length: int = 8 # 8 (original value)
    dim_option: int = 2
    n_parallel: int = 4 # 4 is better
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
    sac_min_buffer_size: int = 1 # 512
    joint_train: int = 1
    te_only_last_frame: int  = 0
    goal_range: float = 7.5
    alpha: float = 0.1
    cp_multi_step: int = 20 # 25 (original value)
    downstream_reward_type: str = "esparse"
    downstream_num_goal_steps: int = 50
    cp_path: str = "/home/hadi/RL/LDG/DSD/final_models/kitchen/CSD/option_policy40000.pt" # this should be corrected 
    cp_path_idx: int = 0
    cp_unit_length: int = 1

    env: str = "test_kitchen"
    cp_multitask: int = 7


all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
custom_order = [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,     # Panda Arm and Gripper States
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Burners and Overhead Light
                    29, 30, 31, 49, 50, 51,                                           # Cabinets (Slide + Left + Right Hinge)
                    32, 52,                                                          # Microwave Door
                    33, 34, 35, 36, 37, 38, 39, 53, 54, 55, 56, 57, 58               # Kettle
        ]


def make_env(max_path_length):
    from gymnasium.wrappers import TimeLimit
    env =  DownstreamKitchen(
    tasks_to_complete=all_tasks,
    terminate_on_tasks_completed=True,
    render_mode="rgb_array",
    custom_order=custom_order)

    if args.cp_path is not None:
        cp_path = args.cp_path
        if not os.path.exists(cp_path):
            import glob
            cp_path = glob.glob(cp_path)[0]
        cp_dict = torch.load(cp_path, map_location='cpu')


    env = ChildPolicyEnv(
            env,
            cp_dict,
            cp_action_range=1.5,
            cp_unit_length=args.cp_unit_length,
            cp_multi_step=args.cp_multi_step,
            cp_num_truncate_obs=0,
            cp_multitask=len(all_tasks)
        )
    return env


args = METRAHighLevelKitchenConfig()


@wrap_experiment(log_dir=get_log_dir(args), name=get_exp_name(args)[0])
def run(ctxt=None):
    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)
    
    def _finalize_lr(lr):
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr


    set_seed(args.seed)
    runner = OptionLocalRunner(ctxt)


    max_path_length = args.max_path_length
    if args.cp_path is not None:
        max_path_length *= args.cp_multi_step

    contextualized_make_env = functools.partial(make_env, max_path_length=max_path_length)
    env = contextualized_make_env()


    # if args.cp_path is not None:
    #     cp_path = args.cp_path
    #     if not os.path.exists(cp_path):
    #         import glob
    #         cp_path = glob.glob(cp_path)[0]
    #     cp_dict = torch.load(cp_path, map_location='cpu')

        # env = ChildPolicyEnv(
        #     env,
        #     cp_dict,
        #     cp_action_range=1.5,
        #     cp_unit_length=args.cp_unit_length,
        #     cp_multi_step=args.cp_multi_step,
        #     cp_num_truncate_obs=0,
        #     cp_multitask=len(all_tasks)
        # )


    if args.algo in ['sac', 'ppo']:
        if args.env == "test_kitchen":
            policy_q_input_dim = env.observation_space.shape[0] + args.cp_multitask
            action_dim = env.action_space.shape[0]


    device = torch.device('cuda' if args.use_gpu else 'cpu')
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None


    if args.algo == "sac":
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
        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

        optimizers = ({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ]),
        })

    elif args.algo == 'ppo':
        vf = MLPModule(
            input_dim=policy_q_input_dim,
            output_dim=1,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            layer_normalization=args.critic_layer_norm,
        )
        optimizers = ({
            'vf': torch.optim.Adam([
                {'params': vf.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })


    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))
    module_kwargs = dict(
            hidden_sizes=master_dims,
            layer_normalization=False,
        )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)

    module_cls = GaussianMLPTwoHeadedModuleEx
    module_kwargs.update(dict(
        max_std=np.exp(2.),
        normal_distribution_cls=TanhNormal,
        output_w_init=functools.partial(xavier_normal_ex, gain=1.),
        init_std=1.,
    ))
    policy_module = module_cls(
        input_dim=policy_q_input_dim,
        output_dim=action_dim,
        **module_kwargs
    )
    option_info = {
        'dim_option': args.dim_option,
    }
    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )

    policy_kwargs['module'] = policy_module
    option_policy = PolicyEx(**policy_kwargs) 

    optimizers.update({
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
    })


    optimizer = OptimizerGroupWrapper(
            optimizers=optimizers,
            max_optimization_epochs=None,)
    

    import torch.nn as nn
    traj_encoder = nn.Identity()

    algo_kwargs = dict(
            env_name=args.env,
            algo=args.algo,
            env_spec=env.spec,
            option_policy=option_policy,
            traj_encoder=traj_encoder,
            skill_dynamics=None,
            dist_predictor=None,
            dual_lam=dual_lam,
            optimizer=optimizer,
            alpha=args.alpha,
            max_path_length=args.max_path_length,
            n_epochs_per_eval=args.n_epochs_per_eval,
            n_epochs_per_log=args.n_epochs_per_log, 
            n_epochs_per_tb=args.n_epochs_per_log, 
            n_epochs_per_save=args.n_epochs_per_save, 
            n_epochs_per_pt_save=args.n_epochs_per_pt_save, 
            n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
            dim_option=args.dim_option,
            N = args.N,
            num_random_trajectories=args.num_random_trajectories,
            num_video_repeats=args.num_video_repeats,
            eval_record_video=args.eval_record_video,
            video_skip_frames=args.video_skip_frames,
            eval_plot_axis=args.eval_plot_axis,
            name='sac',
            device=device,
            sample_cpu=args.sample_cpu,
            num_train_per_epoch=1,
            sd_batch_norm=args.sd_batch_norm,
            skill_dynamics_obs_dim=policy_q_input_dim,
            trans_minibatch_size=args.trans_minibatch_size,
            trans_optimization_epochs=args.trans_optimization_epochs,
            discount=args.sac_discount,
            discrete=args.discrete,
            unit_length=args.unit_length,
        )
    

    replay_buffer = PathBufferEx(capacity_in_transitions=int(args.sac_max_buffer_size), pixel_shape=None)
    
    sac_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        pixel_shape=None,
        multitask=args.cp_multitask
    )

    if args.algo == "sac":
        algo = SAC(**algo_kwargs, **sac_args)
        
    elif args.algo == 'ppo':
        algo = PPO(
            **algo_kwargs,
            vf=vf,
            gae_lambda=0.95,
            ppo_clip=0.2,
        )
    

    if args.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)

    runner.setup(
            algo=algo,
            env=env,
            make_env=contextualized_make_env,
            sampler_cls=OptionMultiprocessingSampler,
            sampler_args=dict(n_thread=args.n_thread),
            n_workers=args.n_parallel,
        )

    algo.option_policy.to(device)
    runner.train(n_epochs=args.n_epochs, batch_size=args.traj_batch_size)


if __name__ == '__main__':
    mp.set_start_method(START_METHOD)
    run()
