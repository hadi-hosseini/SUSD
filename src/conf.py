from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DSDConfig:
    # === Run and environment configuration ===
    run_group: str = 'Debug'  # Name for grouping runs (e.g., experiment name)
    normalizer_type: str = 'off'  # choices: ['off', 'preset']
    encoder: int = 0  # 0: no encoder, non-zero: pixel encoder enabled

    env: str = 'maze'  # choices: ['maze', 'half_cheetah', 'ant', 'dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'kitchen']
    frame_stack: Optional[int] = None  # Number of frames stacked for pixel obs

    max_path_length: int = 200  # Max trajectory length per episode

    # === Hardware and sampling settings ===
    use_gpu: int = 1  # 1: use GPU, 0: CPU only
    sample_cpu: int = 1  # 1: sampling on CPU, 0: sampling on device (GPU)
    seed: int = 0  # Random seed for reproducibility
    n_parallel: int = 4  # Number of parallel env workers
    n_thread: int = 1  # Number of CPU threads for sampler

    # === Training schedule and batch sizes ===
    n_epochs: int = 1000000  # Total number of training epochs
    traj_batch_size: int = 8  # Batch size (number of trajectories per update)
    trans_minibatch_size: int = 256  # Minibatch size for transition optimization
    trans_optimization_epochs: int = 200  # Optimization epochs for transitions

    # === Logging, saving, and evaluation frequencies ===
    n_epochs_per_eval: int = 125  # Frequency to evaluate policy (in epochs)
    n_epochs_per_log: int = 25  # Frequency to log training metrics
    n_epochs_per_save: int = 1000  # Frequency to save checkpoints
    n_epochs_per_pt_save: int = 1000  # Frequency to save PyTorch model state
    n_epochs_per_pkl_update: Optional[int] = None  # Frequency to update pickled files (default: n_epochs_per_eval)

    # === Resume and checkpointing ===
    resume: bool = False  # Whether to resume training from a saved checkpoint
    resume_path: Optional[str] = None  # Path to checkpoint directory (e.g., exp/Debug/exp-name)

    # === Evaluation and visualization settings ===
    num_random_trajectories: int = 48  # Number of random trajectories per iteration
    num_video_repeats: int = 2  # Number of video repeats during evaluation
    eval_record_video: int = 1  # 1: record eval videos, 0: disable
    eval_plot_axis: Optional[List[float]] = None  # Axis limits for evaluation plots
    video_skip_frames: int = 1  # Number of frames to skip in eval video recording

    # === Algorithm and model architecture ===
    dim_option: int = 2  # Dimensionality of latent option space
    N: Optional[int] = None # Number of state factors
    algo: str = 'metra'  # choices: ['metra', 'dads']

    # === Use Weights & Biases logging ===
    use_wandb: bool = False  # Whether to enable wandb logging (True/False)
    wandb_project: str = 'DSD'  # WandB project name
    wandb_entity: str = 'hadi-hosseini0171-sharif-university-of-technology'  # WandB entity / username

    # === Learning rates and optimizer settings ===
    common_lr: float = 1e-4  # Default learning rate for optimizers
    lr_op: Optional[float] = None  # Learning rate for option policy optimizer
    lr_te: Optional[float] = None  # Learning rate for trajectory encoder optimizer

    # === SAC (Soft Actor-Critic) specific parameters ===
    alpha: float = 0.01  # Entropy regularization temperature
    sac_tau: float = 5e-3  # Soft target update rate
    sac_lr_q: Optional[float] = None  # Learning rate for Q-functions
    sac_lr_a: Optional[float] = None  # Learning rate for actor
    sac_discount: float = 0.99  # Discount factor
    sac_scale_reward: float = 1.0  # Reward scaling
    sac_target_coef: float = 1.0  # Target entropy coefficient
    sac_min_buffer_size: int = 10000  # Minimum replay buffer size before updates
    sac_max_buffer_size: int = 300000  # Max replay buffer size

    # === Model architecture details ===
    spectral_normalization: int = 0  # 1: enable spectral norm, 0: disable
    model_master_dim: int = 1024  # Number of units per master network layer
    model_master_num_layers: int = 2  # Number of master network layers
    model_master_nonlinearity: Optional[str] = None  # choices: ['relu', 'tanh', None]

    # === Skill dynamics parameters ===
    sd_const_std: int = 1  # 1: fix std, 0: learn std
    sd_batch_norm: int = 1  # 1: enable batch norm, 0: disable
    num_alt_samples: int = 100  # Number of alternative samples for training
    split_group: int = 65536  # Group size for splitting computations

    # === Skill option and training flags ===
    discrete: int = 0  # 1: discrete skills, 0: continuous
    inner: int = 1  # Use inner loop training or not
    unit_length: int = 1  # Unit length constraints on continuous skills


    # === Dual regularization parameters ===
    dual_reg: int = 1  # 1: enable dual regularization, 0: disable
    dual_lam: float = 30  # Lambda coefficient for dual regularization
    dual_slack: float = 1e-3  # Slack parameter for dual regularization
    dual_lr: Optional[float] = None  # Learning rate for dual regularization optimizer
    dual_dist: str = 'one'  # Distance metric choice; options: 'l2', 's2_from_s', 'one'


# --- Specific Configs ---

# METRA on state-based Ant (2-D skills)
@dataclass
class METRAAntConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'ant'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 1
    normalizer_type: str = 'preset'
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-50, 50, -50, 50])
    trans_optimization_epochs: int = 50
    n_epochs_per_log: int = 100
    n_epochs_per_eval: int = 1000
    n_epochs_per_save: int = 10000
    sac_max_buffer_size: int = 1000000
    algo: str = 'metra'
    discrete: int = 0
    dim_option: int = 2

# LSD on state-based Ant (2-D skills)
@dataclass
class LSDAntConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'ant'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 1
    normalizer_type: str = 'preset'
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-50, 50, -50, 50])
    trans_optimization_epochs: int = 50
    n_epochs_per_log: int = 100
    n_epochs_per_eval: int = 1000
    n_epochs_per_save: int = 10000
    sac_max_buffer_size: int = 1000000
    algo: str = 'metra'
    dual_reg: int = 0
    spectral_normalization: int = 1
    discrete: int = 0
    dim_option: int = 2

# DADS on state-based Ant (2-D skills)
@dataclass
class DADSAntConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'ant'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 1
    normalizer_type: str = 'preset'
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-50, 50, -50, 50])
    trans_optimization_epochs: int = 50
    n_epochs_per_log: int = 100
    n_epochs_per_eval: int = 1000
    n_epochs_per_save: int = 10000
    sac_max_buffer_size: int = 1000000
    algo: str = 'dads'
    inner: int = 0
    unit_length: int = 0
    dual_reg: int = 0
    discrete: int = 0
    dim_option: int = 2

# DIAYN on state-based Ant (2-D skills)
@dataclass
class DIAYNAntConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'ant'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 1
    normalizer_type: str = 'preset'
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-50, 50, -50, 50])
    trans_optimization_epochs: int = 50
    n_epochs_per_log: int = 100
    n_epochs_per_eval: int = 1000
    n_epochs_per_save: int = 10000
    sac_max_buffer_size: int = 1000000
    algo: str = 'metra'
    inner: int = 0
    unit_length: int = 0
    dual_reg: int = 0
    discrete: int = 0
    dim_option: int = 2

# METRA on state-based HalfCheetah (16 skills)
@dataclass
class METRAHalfCheetahConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'half_cheetah'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 1
    normalizer_type: str = 'preset'
    trans_optimization_epochs: int = 50
    n_epochs_per_log: int = 100
    n_epochs_per_eval: int = 1000
    n_epochs_per_save: int = 10000
    sac_max_buffer_size: int = 1000000
    algo: str = 'metra'
    discrete: int = 1
    dim_option: int = 16


# METRA on pixel-based Quadruped (4-D skills)
@dataclass
class METRADmcQuadrupedConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'dmc_quadruped'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 4
    normalizer_type: str = 'off'
    video_skip_frames: int = 2
    frame_stack: Optional[int] = 3
    sac_max_buffer_size: int = 300000
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-15, 15, -15, 15])
    algo: str = 'metra'
    trans_optimization_epochs: int = 200
    n_epochs_per_log: int = 25
    n_epochs_per_eval: int = 125
    n_epochs_per_save: int = 1000
    n_epochs_per_pt_save: int = 1000
    discrete: int = 0
    dim_option: int = 4
    encoder: int = 1
    sample_cpu: int = 0

# METRA on pixel-based Humanoid (2-D skills)
@dataclass
class METRADmcHumanoidConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'dmc_humanoid'
    max_path_length: int = 200
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 4
    normalizer_type: str = 'off'
    video_skip_frames: int = 2
    frame_stack: Optional[int] = 3
    sac_max_buffer_size: int = 300000
    eval_plot_axis: Optional[List[float]] = field(default_factory=lambda: [-15, 15, -15, 15])
    algo: str = 'metra'
    trans_optimization_epochs: int = 200
    n_epochs_per_log: int = 25
    n_epochs_per_eval: int = 125
    n_epochs_per_save: int = 1000
    n_epochs_per_pt_save: int = 1000
    discrete: int = 0
    dim_option: int = 2
    encoder: int = 1
    sample_cpu: int = 0

# METRA on pixel-based Kitchen (24 skills)
@dataclass
class METRAKitchenConfig(DSDConfig):
    run_group: str = 'Debug'
    env: str = 'kitchen'
    max_path_length: int = 50
    seed: int = 0
    traj_batch_size: int = 8
    n_parallel: int = 4
    normalizer_type: str = 'off'
    num_video_repeats: int = 1
    frame_stack: Optional[int] = 3
    sac_max_buffer_size: int = 100000
    algo: str = 'metra'
    sac_lr_a: float = -1
    trans_optimization_epochs: int = 100
    n_epochs_per_log: int = 25
    n_epochs_per_eval: int = 250
    n_epochs_per_save: int = 1000
    n_epochs_per_pt_save: int = 1000
    discrete: int = 1
    dim_option: int = 24
    encoder: int = 1
    sample_cpu: int = 0
