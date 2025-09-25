import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy
import matplotlib.pyplot as plt
import os

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

class SUSD(IOD):
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
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,

            pixel_shape=None,
            partition_points,
            exp_name,
            susd_dist_norm,
            susd_input_factor0,
            q1_list,
            # log_alpha_list,
            susd_q_function,
            susd_ablation_mode,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.susd_q_function = susd_q_function
        if self.susd_q_function:
            self.qf1_list = [qf1.to(self.device) for qf1 in q1_list]
            # self.log_alpha_list = [log_alpha.to(self.device) for log_alpha in log_alpha_list]
            self.target_qf1_list = [copy.deepcopy(qf1) for qf1 in self.qf1_list]

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape
    
        self.partition_points = partition_points

        self.csd_reward_logs = []
        self.te_losses = []
        self.mus = []
        self.csd_logs = []
        self.do_print = False
        self.early_stopping = []
        self.early_stopping_with_names = []
        self.q_values = []

        self.exp_name = exp_name
        self.susd_dist_norm = susd_dist_norm
        self.susd_input_factor0 = susd_input_factor0
        self.susd_ablation_mode = susd_ablation_mode


        self.adaptive_base = 0.0

        assert self._trans_optimization_epochs is not None

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):
        batch_size = runner._train_args.batch_size

        if self.discrete:
            skill_list = []

            for _ in range(self.N):
                indices = np.random.randint(0, self.dim_option, runner._train_args.batch_size)                
                one_hot = np.eye(self.dim_option)[indices]                
                skill_list.append(one_hot)

            skills = np.concatenate(skill_list, axis=1)
            extras = self._generate_option_extras(skills)

            
        else:
            random_options = np.random.randn(batch_size, self.N * self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True) 
            extras = self._generate_option_extras(random_options)

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

            if self.susd_ablation_mode == 3:
                csd_distances = data.pop('csd_distances')

                alpha = 0.1
                csd_mean = csd_distances.mean().item()
                self.adaptive_base = (1 - alpha) * self.adaptive_base + alpha * csd_mean
                print(csd_distances)
                print(self.adaptive_base)
                print(csd_mean)
            
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list

                self.replay_buffer.add_path(path)
                
                if self.susd_ablation_mode == 3 and csd_distances[i] > self.adaptive_base: # oversample good samples in the buffer
                    oversample_factor = 2 * int(csd_distances[i]/self.adaptive_base)
                    print(oversample_factor)
                    for _ in range(oversample_factor):
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

        if self.susd_ablation_mode == 3: # if oversampling is activate, then calculate the CSD for each rollout
            csd_distances = self._compute_csd(path_data) 
            path_data["csd_distances"] = csd_distances.detach().cpu().numpy()

        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data, runner)

        return tensors

    def _train_components(self, epoch_data, runner):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for i in range(self._trans_optimization_epochs):
            if i == 0 and runner.step_itr % 50 == 0:
                self.do_print = True

            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()


            self._optimize_te(tensors, v, runner)
            self._update_rewards(tensors, v)
            self._optimize_op(runner, tensors, v)
        
        print("Train Modules")
        return tensors

    def _optimize_te(self, tensors, internal_vars, runner):
        self._update_loss_te(tensors, internal_vars, runner)

        losses_te = tensors['LossTe']
        te_keys = [f'traj_encoder_{i}' for i in range(len(losses_te))]
        self._gradient_descent(losses_te, optimizer_keys=te_keys)

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(tensors['LossDualLam'], optimizer_keys=['dual_lam'],)

            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],)
                
        if self.susd_q_function:
            loss_qfn = [tensors[f'LossQf1_{i}'] for i in range(self.N)]
            qfn_keys = [f'qf_{i}' for i in range(self.N)]
            self._gradient_descent(loss_qfn, optimizer_keys=qfn_keys)

    def _optimize_op(self, runner, tensors, internal_vars):
        self._update_loss_qf(runner, tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

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

        # if self.susd_q_function:
        #     self._update_loss_alpha_N(tensors, internal_vars)
        #     for i in range(self.N):
        #         self._gradient_descent(
        #             tensors[f'LossAlpha_{i}'],
        #             optimizer_keys=[f'log_alpha_{i}'],
        #         )
        #     sac_utils.update_targets_N(self)

        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs)
            next_z = self.traj_encoder(next_obs)
            target_z = next_z - cur_z

            if self.discrete:
                batch_size, _ = target_z.shape
                options_reshaped = v['options'].view(batch_size, self.N, self.dim_option)
                masks = options_reshaped - options_reshaped.mean(dim=2, keepdim=True)
                masks = (masks * self.dim_option) / (self.dim_option - 1 if self.dim_option != 1 else 1)
                target_z_reshaped = target_z.view(batch_size, self.N, self.dim_option)
                rewards = (target_z_reshaped * options_reshaped).sum(dim=2)  # shape: [batch_size, N]

            else:
                batch_size, _ = target_z.shape
                target_z_reshaped = target_z.view(batch_size, self.N, self.dim_option)
                options_reshaped = v['options'].view(batch_size, self.N, self.dim_option)
                rewards = (target_z_reshaped * options_reshaped).sum(dim=2)  # shape: [batch_size, N]
            
                # rewards = (target_z * v['options']).sum(dim=1)

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])


        v['rewards'] = rewards

    def _partition_dist_predictor(self, obs):
        s2_dist = self.dist_predictor(obs)
        s2_dist_mean = s2_dist.mean
        s2_dist_std = s2_dist.stddev
        
        mean_partitions = [s2_dist_mean[:, start:end] for start, end in zip(self.partition_points[:-1], self.partition_points[1:])]
        std_partitions = [s2_dist_std[:, start:end] for start, end in zip(self.partition_points[:-1], self.partition_points[1:])]

        return mean_partitions, std_partitions

    def _compute_csd(self, path_data):
        epoch_data = self._flatten_data(path_data)
        data = {}
        for key, value in epoch_data.items():
            data[key] = value

        obs = data['obs']
        next_obs = data['next_obs']
        s2_dist = self.dist_predictor(obs)
        s2_dist_mean = s2_dist.mean
        s2_dist_std = s2_dist.stddev
        scaling_factor = 1. / s2_dist_std
        geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
        normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
        normalized_csd = torch.square((next_obs - obs) - s2_dist_mean) * normalized_scaling_factor
        csd_distances = torch.mean(normalized_csd, dim=1)
        return csd_distances


    def _update_loss_te(self, tensors, v, runner):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()

            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':

                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean

                if self.do_print:
                    mean_partitions, _ = self._partition_dist_predictor(obs)
                    self.mus.append((runner.step_itr, [mean_partitions[i].norm(p=2, dim=1).mean().detach().cpu().numpy().item() for i in range(len(self.partition_points) - 1)]))

                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                normalized_csd = torch.square((next_obs - obs) - s2_dist_mean) * normalized_scaling_factor

                if self.susd_ablation_mode == 1 or self.susd_ablation_mode == 3: # just CSD weight or Oversampling
                    csd_distances = torch.mean(normalized_csd, dim=1)
                elif self.susd_ablation_mode == 0 or self.susd_ablation_mode == 2: # SUSD
                    if self.susd_dist_norm:
                        if self.env_name == "kitchen_franka":
                            csd_distances = [(59.0 * normalized_csd[:, start:end])/(end - start) for start, end in zip(self.partition_points[:-1], self.partition_points[1:])]
                    else:
                        csd_distances = [normalized_csd[:, start:end] for start, end in zip(self.partition_points[:-1], self.partition_points[1:])]
                    csd_distances = [torch.sum(csd_distance, dim=1)/normalized_csd.shape[1] for csd_distance in csd_distances]
                    csd_distances = torch.stack(csd_distances, dim=1)


                if self.do_print:
                    if self.susd_ablation_mode == 1: # just CSD weight
                        self.csd_logs.append((runner.step_itr, [csd_distances.mean().detach().cpu().numpy().item() for i in range(len(self.partition_points) - 1)]))
                    elif self.susd_ablation_mode == 0: # SUSD
                        self.csd_logs.append((runner.step_itr, [csd_distances[:, i].mean().detach().cpu().numpy().item() for i in range(len(self.partition_points) - 1)]))

                v.update({'csd_distances': csd_distances})

            else:
                raise NotImplementedError

            cst_penalty = torch.ones_like(x[:, 0]) - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards.sum(dim=1) + dual_lam.detach() * cst_penalty
            te_objs = [te_obj for _ in range(len(self.partition_points) - 1)]
            cst_penalty = [cst_penalty]

            v.update({
                'cst_penalty': cst_penalty,
            })
        else:
            te_obj = rewards

        loss_te = []
        for te_obj in te_objs:
            loss_te_i = -te_obj.mean()
            loss_te.append(loss_te_i)

        if self.do_print:
            # self.do_print = False
            self.te_losses.append((runner.step_itr, [loss_te[i].detach().cpu().numpy().item() for i in range(len(self.partition_points) - 1)]))


        tensors.update({
            'LossTe': loss_te
        })

        if self.susd_q_function:
            next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])
            sac_utils.update_loss_qf_N(
                self, tensors, v,
                actions=v['actions'],
                next_obs=next_processed_cat_obs,
                dones=v['dones'],
                rewards=v['rewards'] * torch.sqrt(v['csd_distances']),
                policy=self.option_policy,
            )

    def plot_csd_reward_logs(self, runner):
        if len(self.csd_reward_logs) == 0:
            return

        epochs, csd_values = zip(*self.csd_reward_logs)
        epochs = np.array(epochs)
        csd_values =  np.array(csd_values)

        os.makedirs(f'results/{self.exp_name}', exist_ok=True)

        for i in range(csd_values.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                csd_values[:, i],
                label=f'Factor {i}',
                marker='o',
                markersize=3,
                linewidth=1
            )

            ax.set_xlabel('Epoch')
            ax.set_ylabel('CSD Value')
            ax.set_title(f'CSD over Epochs')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            csd_plot_path = f'results/{self.exp_name}/csd_plot_epoch_{runner.step_itr}_reward.png'
            fig.savefig(csd_plot_path)
            plt.close(fig)

    def plot_csd_logs(self, runner):
        if len(self.csd_logs) == 0:
            return

        epochs, csd_values = zip(*self.csd_logs)
        epochs = np.array(epochs)
        csd_values =  np.array(csd_values)
        
        os.makedirs(f'results/{self.exp_name}', exist_ok=True)

        for i in range(csd_values.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                csd_values[:, i],
                label=f'Factor {i}',
                marker='o',
                markersize=3,
                linewidth=1
            )

            ax.set_xlabel('Epoch')
            ax.set_ylabel('CSD Value')
            ax.set_title(f'CSD for Factor {i} over Epochs')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # Save each factor's plot separately
            csd_plot_path = f'results/{self.exp_name}/csd_plot_epoch_{runner.step_itr}_factor_{i}.png'
            fig.savefig(csd_plot_path)
            plt.close(fig)

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        cst_penalty = v['cst_penalty'][0]
        loss_dual_lam = log_dual_lam * (cst_penalty.detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, runner, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

        if self.susd_q_function:
            rewards = torch.zeros((v['rewards'].shape[0]), device=v['rewards'].device)
            # print(rewards.shape)
            rewards_logs = []
            for i in range(self.N):
                start = self.partition_points[i]
                end = self.partition_points[i + 1]
                start_option = i * self.dim_option
                end_option = (i + 1) * self.dim_option
                reward_i = self.qf1_list[i](self._get_concat_obs(self.option_policy.process_observations(v['obs'][:, start:end]), v['options'][:, start_option:end_option]), v['actions'])
                reward_i = reward_i.view(-1)
                rewards_logs.append(reward_i)
                rewards = rewards + reward_i 
            if self.do_print:
                self.do_print = False
                self.q_values.append((runner.step_itr, [rewards_logs[i].mean().detach().cpu().numpy().item() for i in range(len(self.partition_points) - 1)]))

        else:
            if self.susd_ablation_mode == 1: # just CSD weight
                rewards = v['rewards'].sum(dim=1)
                rewards = rewards * torch.sqrt(v['csd_distances'])
            elif self.susd_ablation_mode == 0: # SUSD method
                rewards = v['rewards'] * torch.sqrt(v['csd_distances'])
                rewards = rewards.sum(dim=1)
            elif self.susd_ablation_mode == 3 or self.susd_ablation_mode == 2: # Oversampling
                rewards = v['rewards'].sum(dim=1)

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=rewards * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    # def _update_loss_alpha_N(self, tensors, v):
        # sac_utils.update_loss_alpha_N(
        #     self, tensors, v,
        # )   

    def plot_early_stopping(self, early_stopping):
        unique_tasks, step_iters = zip(*early_stopping)

        plt.figure(figsize=(8, 5))
        plt.plot(step_iters, unique_tasks, marker='o', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Unique Completed Tasks')
        plt.title('Unique Task Coverage over Time')
        plt.grid(True)
        plt.tight_layout()

        save_path = f"results/{self.exp_name}/task_coverage.png"

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Early Stopping Plot Saved to: {save_path}")
        else:
            plt.show()

        plt.close()


    def plot_q_decomposition(self, runner):
        if len(self.q_values) == 0:
            return

        epochs, q_values = zip(*self.q_values)
        epochs = np.array(epochs)
        q_values =  np.array(q_values)
        
        os.makedirs(f'results/{self.exp_name}', exist_ok=True)

        for i in range(q_values.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                q_values[:, i],
                label=f'Factor {i}',
                marker='o',
                markersize=3,
                linewidth=1
            )

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Q Value')
            ax.set_title(f'Q Value for Factor {i} over Epochs')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # Save each factor's plot separately
            csd_plot_path = f'results/{self.exp_name}/q_plot_epoch_{runner.step_itr}_q_{i}.png'
            fig.savefig(csd_plot_path)
            plt.close(fig)

    def plot_mus(self, runner):
        if len(self.mus) == 0:
            return

        epochs, mus = zip(*self.mus)
        epochs = np.array(epochs)
        mus =  np.array(mus)
        
        os.makedirs(f'results/{self.exp_name}', exist_ok=True)

        for i in range(mus.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                mus[:, i],
                label=f'Factor {i}',
                marker='o',
                markersize=3,
                linewidth=1
            )

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Norm 2 Mu')
            ax.set_title(f'Mu for Factor {i} over Epochs')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # Save each factor's plot separately
            csd_plot_path = f'results/{self.exp_name}/mu_plot_epoch_{runner.step_itr}_mu_{i}.png'
            fig.savefig(csd_plot_path)
            plt.close(fig)

    def plot_te_losses(self, runner):
        if len(self.te_losses) == 0:
            return

        epochs, te_losses = zip(*self.te_losses)
        epochs = np.array(epochs)
        te_losses =  -1 * np.array(te_losses)  # Scale if desired
        
        os.makedirs(f'results/{self.exp_name}', exist_ok=True)

        for i in range(te_losses.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                te_losses[:, i],
                label=f'Factor {i}',
                marker='o',
                markersize=3,
                linewidth=1
            )

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Phi Loss')
            ax.set_title(f'Phi for Factor {i} over Epochs')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # Save each factor's plot separately
            csd_plot_path = f'results/{self.exp_name}/phi_plot_epoch_{runner.step_itr}_phi_{i}.png'
            fig.savefig(csd_plot_path)
            plt.close(fig)

    def get_completed_task_names(self, mask):
        task_names = ['BB', 'TB', 'LS', 'SC', 'HC', 'MI', 'KE']
        return [name for done, name in zip(mask, task_names) if done == 1]

    def plot_early_stopping_with_names(self, early_stopping_with_names):
        unique_tasks = [entry[0] for entry in early_stopping_with_names]
        task_names = [entry[1] for entry in early_stopping_with_names]
        step_iters = [entry[2] for entry in early_stopping_with_names]

        plt.figure(figsize=(10, 6))
        plt.plot(step_iters, unique_tasks, marker='o', linestyle='-', color='steelblue')

        for i, (step, count, names) in enumerate(zip(step_iters, unique_tasks, task_names)):
            label = "\n".join(names)  
            offset = 15 if i % 2 == 0 else -25 
            plt.annotate(
                label,
                (step, count),
                textcoords="offset points",
                xytext=(0, offset),
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7)
            )

        plt.xlabel('Epochs')
        plt.ylabel('Completed Tasks')
        plt.title('Task Coverage Over Time')
        plt.grid(True)
        plt.xticks(step_iters, rotation='vertical')
        plt.tight_layout()

        save_path = f"results/{self.exp_name}/task_coverage_with_names.png"
        plt.savefig(save_path)
        print(f"Early Stopping Plot Saved to: {save_path}")

        plt.close()

    def _evaluate_policy(self, runner):

        if self.discrete:

            skill_list = []
            for _ in range(self.N):
                indices = np.random.randint(0, self.dim_option, self.dim_option)                
                one_hot = np.eye(self.dim_option)[indices]                
                skill_list.append(one_hot)
            skills = np.concatenate(skill_list, axis=1)

            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(skills[i])
                    colors.append(i)
            random_options = np.array(random_options)


            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)

        else:
            random_options = np.random.randn(self.num_random_trajectories, self.N * self.dim_option)

            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=1, keepdims=True)
            
            random_option_colors = get_option_colors(random_options.reshape(self.num_random_trajectories, -1) * 4)

        flat_random_options = random_options.reshape(self.num_random_trajectories, self.N * self.dim_option)

        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(flat_random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        data = self.process_samples(random_trajectories)

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
                runner._env.render_trajectories(
                    random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
                )


        from sklearn.decomposition import PCA
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(dtype=torch.float32, device=self.device) for ob in data['obs']])
        
        option_dists = self.traj_encoder(last_obs)
        option_means = option_dists.detach().cpu().numpy()
        pca = PCA(n_components=2)
        option_means_2d = pca.fit_transform(option_means)
        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
                draw_2d_gaussians(
                    option_means_2d,
                    [[0.5, 0.5]] * len(option_means_2d),
                    option_colors,
                    fm.ax,
                    fill=True,
                    use_adaptive_axis=True,
                    alpha=1.0
                )
        
        eval_option_metrics = {}


        if not self.env_name == "elden_kitchen":
            if self.eval_record_video:
                if self.discrete:
                    skill_list = []
                    for _ in range(self.N):
                        indices = np.random.randint(0, self.dim_option, self.dim_option + 1)                
                        one_hot = np.eye(self.dim_option)[indices]                
                        skill_list.append(one_hot)
                    video_options = np.concatenate(skill_list, axis=1)

                    video_options = video_options.repeat(self.num_video_repeats, axis=0)
                else:
                    if self.dim_option * self.N == 2:
                        video_options = np.random.randn(9, self.N * self.dim_option)
                        if self.unit_length:
                            video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                        flat_random_options = video_options.reshape(9, self.N * self.dim_option)
                    else:
                        video_options = np.random.randn(9, self.N * self.dim_option)
                        if self.unit_length:
                            video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                        flat_random_options = video_options.reshape(9, self.N * self.dim_option)

                    video_options = flat_random_options.repeat(self.num_video_repeats, axis=0)
                video_trajectories = self._get_trajectories(
                    runner,
                    sampler_key='local_option_policy',
                    extras=self._generate_option_extras(video_options),
                    worker_update=dict(
                        _render=True,
                        _deterministic_policy=True,
                    ),
                )
                record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

            eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
            with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
                log_performance_ex(
                    runner.step_itr,
                    TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                    discount=self.discount,
                    additional_records=eval_option_metrics,
                )
            self._log_eval_metrics(runner)

        # self.plot_csd_logs(runner)
        # self.plot_csd_reward_logs(runner)
        # self.plot_te_losses(runner)
        # self.plot_mus(runner)
        self.plot_q_decomposition(runner)


        #### plot the task coverage for franka kitchen
        if self.env_name == "kitchen_franka":
            done_tasks = np.zeros_like(data['episode_task_completions'][-1][0])
            for arr in data['episode_task_completions']:
                done_tasks = np.maximum(done_tasks, arr[-1])

            task_names = self.get_completed_task_names(done_tasks)
            task_coverage = done_tasks.sum()
            self.early_stopping.append((task_coverage, runner.step_itr))
            self.early_stopping_with_names.append((task_coverage, task_names, runner.step_itr))
            self.plot_early_stopping(self.early_stopping)
            self.plot_early_stopping_with_names(self.early_stopping_with_names)



