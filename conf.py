from dataclasses import dataclass
from typing import Optional

@dataclass
class MetraConfig:
    # Determines how often the policy is evaluated during training;
    # every N epochs, the policy is tested and its reward measured.
    # Set to 0 to disable evaluation during training.
    n_epochs_per_eval: int = 125

    # Total number of training epochs to run;
    # controls how long the training process lasts.
    n_epochs: int = 1000000

    # Determines how often training logs are recorded;
    # every N epochs, training diagnostics and metrics are logged.
    n_epochs_per_log: int = 25

    # Determines how often the full experiment state (model, optimizer, stats) is saved;
    # every N epochs, a complete snapshot is stored.
    n_epochs_per_save: int = 1000

    # Determines how often only the model weights are saved as .pt files;
    # every N epochs, PyTorch checkpoint files are written.
    n_epochs_per_pt_save: int = 1000

    # Determines how often the configuration and metadata are updated as a .pkl file;
    # can be set to None to disable or defaults to the evaluation frequency if left unspecified.
    n_epochs_per_pkl_update: Optional[int] = None

    # Number of trajectories (episodes) to sample in each training epoch;
    # higher values increase batch diversity but also computational cost.
    traj_batch_size: int = 8

    # Maximum number of environment steps per trajectory (episode);
    # controls the length of each sampled path during training and evaluation.
    max_path_length: int = 200

    # Indicates whether to use discrete or continuous skill representations;
    # set to 1 for discrete one-hot skills, 0 for continuous vector skills.
    discrete: int = 0

    # Dimension of the skill (option) vector used in the policy;
    # controls the size of the skill embedding space for exploration and learning.
    dim_option: int = 2

    # Number of optimization steps to apply to internal components.
    trans_optimization_epochs: int = 200
