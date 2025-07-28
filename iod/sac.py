import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
from garage.misc import tensor_utils
import copy

from iod.utils import to_np_object_arr, FigManager, get_option_colors, record_video, draw_2d_gaussians, get_torch_concat_obs


class SAC(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,

            multitask,
            exp_name,

            pixel_shape=None,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

        self.multitask = multitask
        self.exp_name = exp_name

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }
    

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):
        if self.multitask > 0: # make one-hot vectors as goals
            batch_size = runner._train_args.batch_size
            random_indices = np.random.randint(0, self.multitask, size=(batch_size))
            random_goals = np.eye(self.multitask)[random_indices] 
            flat_random_goals = random_goals.reshape(batch_size, self.multitask)
            extras = self._generate_option_extras(flat_random_goals)

        else:  # single task
            extras = [{} for _ in range(runner._train_args.batch_size)]

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _train_once_inner(self, path_data, runner):

        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_op(tensors, v)

        print("Train Modules")
        return tensors

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
            )

        # LossSacp should be updated here because Q functions are changed by optimizers.
        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    def _update_loss_qf(self, tensors, v):
        if self.multitask > 0: # rectify rewards based on the tasks that have been solved
            coeff_reward = torch.sum(v['options'] * v['episode_task_completions'], axis=1) # kitchen_franka
            rewards = (coeff_reward >= 1).float()

            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
            next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

            sac_utils.update_loss_qf(
                self, tensors, v,
                obs=processed_cat_obs,
                actions=v['actions'],
                next_obs=next_processed_cat_obs,
                dones=v['dones'],
                rewards=rewards * self._reward_scale_factor,
                policy=self.option_policy,
            )
        else:
            processed_cat_obs = self.option_policy.process_observations(v['obs'])
            next_processed_cat_obs = self.option_policy.process_observations(v['next_obs'])

            sac_utils.update_loss_qf(
                self, tensors, v,
                obs=processed_cat_obs,
                actions=v['actions'],
                next_obs=next_processed_cat_obs,
                dones=v['dones'],
                rewards=v['rewards'] * self._reward_scale_factor,
                policy=self.option_policy,
            )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        if self.multitask > 0:
            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        else:
            processed_cat_obs = self.option_policy.process_observations(v['obs'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _evaluate_policy(self, runner, **kwargs):
        if self.multitask > 0:
            random_indices = np.random.randint(0, self.multitask, size=(self.num_random_trajectories,))
            flat_random_goals = np.eye(self.multitask)[random_indices].reshape(self.num_random_trajectories, self.multitask) # (batch, multitask_one_hot vector)

            random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(flat_random_goals),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),)

            for i in range(len(random_trajectories)): # franka kitchen
                rewards_i = []
                for j in range(len(random_trajectories[i]['rewards'])):
                    reward_ij = (flat_random_goals[i] * random_trajectories[i]['env_infos']['episode_task_completions'][j]).sum()
                    rewards_i.append(reward_ij)
                    if np.isclose(reward_ij, 1.0):
                        break

                pad_len = len(random_trajectories[i]['rewards']) - len(rewards_i)
                rewards_i.extend([0.0] * pad_len)
                random_trajectories[i]['rewards'] = np.array(rewards_i)

        else:
            random_trajectories = self._get_trajectories(
                runner,
                sampler_key='option_policy',
                extras=[{} for _ in range(self.num_random_trajectories)],
                worker_update=dict(
                    _render=False,
                    _deterministic_initial_state=False,
                    _deterministic_policy=True,
                ),
                env_update=dict(_action_noise_std=None),
            )

        # with FigManager(runner, 'TrajPlot_RandomZ') as fm:
        #     runner._env.render_trajectories(
        #         random_trajectories, np.zeros((self.num_random_trajectories, 3)), self.eval_plot_axis, fm.ax
        #     )

        eval_option_metrics = {}
        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)
