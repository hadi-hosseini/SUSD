#!/usr/bin/env python3
import tempfile

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import wandb

import functools
import os
import platform
import torch.multiprocessing as mp

from dotenv import load_dotenv
load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from iod.dsd import DSD
from iod.dads import DADS

from src.utils import get_exp_name, get_log_dir, make_env
from src.factorization import get_gaussian_module_construction, factorize_environment, PartitionedTrajectoryEncoder, module_cls_factory
from src.conf import METRAAntConfig, METRAKitchenConfig

if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'

# args = METRAAntConfig()
args = METRAKitchenConfig()

@wrap_experiment(log_dir=get_log_dir(args), name=get_exp_name(args)[0])
def run(ctxt=None):
    if args.use_wandb:
        if wandb_api_key:
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.run_group, name=get_exp_name()[0],
                    config=vars(args), dir=wandb_output_dir)

    dowel.logger.log('ARGS: ' + str(args))

    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)

    runner = OptionLocalRunner(ctxt)

    # args.resume = True
    # args.resume_path = "exp/Debug/sd000_1752602706_kitchen_franka_metra"
    if args.resume:
        dowel.logger.log(f"Resuming from checkpoint: {args.resume_path}")
        restored_train_args = runner.restore(
            from_dir=args.resume_path,
            make_env=functools.partial(make_env, args=args, max_path_length=args.max_path_length),
            from_epoch='last',
        )

        # set new saving arguments 
        runner._algo.n_epochs_per_pkl_update = 1000 # params
        runner._algo.n_epochs_per_save = 1000 # phi encoder
        runner._algo.n_epochs_per_pt_save = 1000 # option policy

        runner.train(n_epochs=restored_train_args.n_epochs, batch_size=restored_train_args.batch_size)
        return



    max_path_length = args.max_path_length
    contextualized_make_env = functools.partial(make_env, args=args, max_path_length=max_path_length)
    env = contextualized_make_env()

    example_ob = env.reset()

    if args.encoder:
        if hasattr(env, 'ob_info'):
            if env.ob_info['type'] in ['hybrid', 'pixel']:
                pixel_shape = env.ob_info['pixel_shape']
        else:
            pixel_shape = (64, 64, 3)
    else:
        pixel_shape = None

    device = torch.device('cuda' if args.use_gpu else 'cpu')
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None
    


    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim

    if args.encoder:
        def make_encoder(**kwargs):
            return Encoder(pixel_shape=pixel_shape, **kwargs)

        def with_encoder(module, encoder=None):
            if encoder is None:
                encoder = make_encoder()

            return WithEncoder(encoder=encoder, module=module)

        example_encoder = make_encoder()
        module_obs_dim = example_encoder(torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
    else:
        module_obs_dim = obs_dim

    
    partition_points = factorize_environment(args)
    args.N = len(partition_points) - 1
    dowel.logger.log(f'observation space: {obs_dim}, action space: {action_dim}, partition_points: {partition_points},  #factors: {args.N}')

    option_info = {
        'dim_option': args.dim_option,
    }

    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )

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

    policy_q_input_dim = module_obs_dim + args.dim_option * args.N  
    policy_module = module_cls(
        input_dim=policy_q_input_dim,
        output_dim=action_dim,
        **module_kwargs
    )

    if args.encoder:
        policy_module = with_encoder(policy_module)

    policy_kwargs['module'] = policy_module
    option_policy = PolicyEx(**policy_kwargs) #  π(a∣s,z), O + Nd -> A

    output_dim = args.dim_option
    traj_encoder = PartitionedTrajectoryEncoder(
        args=args,
        partition_points=partition_points,
        master_dims=master_dims,
        nonlinearity=nonlinearity,
        output_dim=output_dim,
        module_cls_factory=module_cls_factory
    ) # π(z | s), O -> Nd

    if args.encoder:
        if args.spectral_normalization:
            te_encoder = make_encoder(spectral_normalization=True)
        else:
            te_encoder = None
        traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)


    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=obs_dim,
        output_dim=obs_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    if args.dual_dist == 's2_from_s':
        dist_predictor = module_cls(**module_kwargs)
    else:
        dist_predictor = None

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    # Skill dynamics do not support pixel obs
    sd_dim_option = args.dim_option
    skill_dynamics_obs_dim = obs_dim
    skill_dynamics_input_dim = skill_dynamics_obs_dim + sd_dim_option
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        const_std=args.sd_const_std,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        input_dim=skill_dynamics_input_dim,
        output_dim=skill_dynamics_obs_dim,
        min_std=0.3,
        max_std=10.0,
    )

    if args.algo == 'dads':
        skill_dynamics = module_cls(**module_kwargs)
    else:
        skill_dynamics = None

    def _finalize_lr(lr):
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
        'traj_encoder': torch.optim.Adam([
            {'params': traj_encoder.parameters(), 'lr': _finalize_lr(args.lr_te)},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': _finalize_lr(args.dual_lr)},
        ]),
    }

    if skill_dynamics is not None:
        optimizers.update({
            'skill_dynamics': torch.optim.Adam([
                {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })

    if dist_predictor is not None:
        optimizers.update({
            'dist_predictor': torch.optim.Adam([
                {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    replay_buffer = PathBufferEx(capacity_in_transitions=int(args.sac_max_buffer_size), pixel_shape=pixel_shape)

    if args.algo in ['metra', 'dads']:
        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            qf1 = with_encoder(qf1)
        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            qf2 = with_encoder(qf2)
        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))
        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })

    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_kwargs = dict(
        env_name=args.env,
        algo=args.algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        skill_dynamics=skill_dynamics,
        dist_predictor=dist_predictor,
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
        name='METRA',
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        sd_batch_norm=args.sd_batch_norm,
        skill_dynamics_obs_dim=skill_dynamics_obs_dim,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length,
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        inner=args.inner,

        num_alt_samples=args.num_alt_samples,
        split_group=args.split_group,

        dual_reg=args.dual_reg,
        dual_slack=args.dual_slack,
        dual_dist=args.dual_dist,

        pixel_shape=pixel_shape,
    )

    if args.algo == 'metra':
        algo = DSD(
            **algo_kwargs,
            **skill_common_args,
        )
    elif args.algo == 'dads':
        algo = DADS(
            **algo_kwargs,
            **skill_common_args,
        )
    else:
        raise NotImplementedError


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
