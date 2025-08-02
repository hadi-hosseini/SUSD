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

class DSD(IOD):
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
            susd_mode,
            susd_temperature,
            exp_name,
            susd_dist_norm,
            susd_csd,

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
        self.susd_mode = susd_mode

        self.csd_logs = []
        self.do_print = False
        self.early_stopping = []
        self.early_stopping_with_names = []
        self.susd_temperature = susd_temperature
        self.exp_name = exp_name
        self.counter = 0
        self.susd_dist_norm = susd_dist_norm
        self.susd_csd = susd_csd

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
            random_indices = np.random.randint(0, self.dim_option, size=(batch_size, self.N))
            random_options = np.eye(self.dim_option)[random_indices]  # Shape: (batch_size, N, dim_option)
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
            self._optimize_op(tensors, v)
        
        print("Train Modules")
        return tensors


    def _optimize_te(self, tensors, internal_vars, runner):
        self._update_loss_te(tensors, internal_vars, runner)


        losses_te = tensors['LossTe']
        te_keys = [f'traj_encoder_{i}' for i in range(len(losses_te))]
        self._gradient_descent(losses_te, optimizer_keys=te_keys)

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)

            for i in range(len(self.dual_lam)):
                self._gradient_descent(tensors[f'LossDualLam_{i}'], optimizer_keys=[f'dual_lam_{i}'])

            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],)

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

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

        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs)
            next_z = self.traj_encoder(next_obs)
            target_z = next_z - cur_z

            if self.discrete:
                batch_size = v['options'].shape[0]
                option_vec = v['options'].reshape(batch_size, self.N, self.dim_option)  # (batch_size, N, dim_option)
                per_factor_mean = option_vec.mean(dim=2, keepdim=True)  # (batch_size, N, 1)
                masks = (option_vec - per_factor_mean) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                masks = masks.reshape(batch_size, self.N * self.dim_option)  # back to (batch_size, total_option_dim)
                rewards = (target_z * masks).sum(dim=1)

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
    
    def _csd_loss(self, obs, next_obs, s2_dist_mean, s2_dist_std):
        scaling_factor = 1. / s2_dist_std
        geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
        normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
        cst_dist = torch.mean(torch.square((next_obs - obs) - s2_dist_mean) * normalized_scaling_factor, dim=1)
        return cst_dist
    
    def _calculate_one_minus_q(self, obs, next_obs, s2_dist_mean, s2_dist_std):
        csd_loss = self._csd_loss(obs, next_obs, s2_dist_mean, s2_dist_std)
        q = torch.exp(-csd_loss)
        one_minus_q = 1 - q
        return one_minus_q      
    
    def _csd_loss_clip(self, obs, next_obs, s2_dist_mean, s2_dist_std):
        scaling_factor = 1. / s2_dist_std
        geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
        normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
        cst_dist = torch.mean(torch.square((next_obs - obs) - s2_dist_mean) * normalized_scaling_factor, dim=1)
        cst_dist_clipped = torch.clamp(cst_dist, min=0.0, max=0.05)
        return cst_dist_clipped

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
            dual_lam = [dual.param.exp() for dual in self.dual_lam]
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':

                mean_partitions, std_partitions = self._partition_dist_predictor(obs)

                csd_distances = []
                if self.susd_mode == 1:
                    # original
                    for i, (s2_dist_mean, s2_dist_std) in enumerate(zip(mean_partitions, std_partitions)):
                        start = self.partition_points[i]
                        end = self.partition_points[i + 1]
                        obs_i = x[:, start:end]
                        next_obs_i = y[:, start:end]
                        csd_distance = self._csd_loss(obs=obs_i, next_obs=next_obs_i, s2_dist_mean=s2_dist_mean, s2_dist_std=s2_dist_std)
                        csd_distances.append(csd_distance)
                    csd_distances = torch.stack(csd_distances, dim=1) # (batch_size, N)
 
                elif self.susd_mode == 2:
                    # normalize
                    for i, (s2_dist_mean, s2_dist_std) in enumerate(zip(mean_partitions, std_partitions)):
                        start = self.partition_points[i]
                        end = self.partition_points[i + 1]
                        obs_i = x[:, start:end]
                        next_obs_i = y[:, start:end]
                        csd_distance = self._csd_loss(obs=obs_i, next_obs=next_obs_i, s2_dist_mean=s2_dist_mean, s2_dist_std=s2_dist_std)
                        if self.susd_dist_norm:
                            csd_distance = csd_distance / (end - start)
                        csd_distances.append(csd_distance) # each element is (batch_size)
                    csd_distances = torch.stack(csd_distances, dim=1) # (batch_size, N)
                    csd_distances = csd_distances / csd_distances.sum(dim=1, keepdim=True) # (batch_size, N)
                    
                elif self.susd_mode == 3:
                    # clip
                    for i, (s2_dist_mean, s2_dist_std) in enumerate(zip(mean_partitions, std_partitions)):
                        start = self.partition_points[i]
                        end = self.partition_points[i + 1]
                        obs_i = x[:, start:end]
                        next_obs_i = y[:, start:end]
                        csd_distance = self._csd_loss_clip(obs=obs_i, next_obs=next_obs_i, s2_dist_mean=s2_dist_mean, s2_dist_std=s2_dist_std)
                        csd_distances.append(csd_distance)
                    csd_distances = torch.stack(csd_distances, dim=1) # (batch_size, N)

                elif self.susd_mode == 4:
                    # 1 - q
                    for i, (s2_dist_mean, s2_dist_std) in enumerate(zip(mean_partitions, std_partitions)):
                        start = self.partition_points[i]
                        end = self.partition_points[i + 1]
                        obs_i = x[:, start:end]
                        next_obs_i = y[:, start:end]
                        csd_distance = self._calculate_one_minus_q(obs=obs_i, next_obs=next_obs_i, s2_dist_mean=s2_dist_mean, s2_dist_std=s2_dist_std)
                        csd_distances.append(csd_distance)
                    csd_distances = torch.stack(csd_distances, dim=1) # (batch_size, N)

                elif self.susd_mode == 5:
                    # softmax with temperature
                    for i, (s2_dist_mean, s2_dist_std) in enumerate(zip(mean_partitions, std_partitions)):
                        start = self.partition_points[i]
                        end = self.partition_points[i + 1]
                        obs_i = x[:, start:end]
                        next_obs_i = y[:, start:end]
                        csd_distance = self._csd_loss(obs=obs_i, next_obs=next_obs_i, s2_dist_mean=s2_dist_mean, s2_dist_std=s2_dist_std)
                        csd_distances.append(csd_distance)
                    csd_distances = torch.stack(csd_distances, dim=1) # (batch_size, N)
                    csd_distances = torch.softmax(csd_distances / self.susd_temperature, dim=1) # (batch_size, N)


                if self.do_print:
                    self.do_print = False
                    self.csd_logs.append((runner.step_itr, 
                                          [csd_distances[0][i].detach().cpu().numpy().item() for i in range(len(self.partition_points) - 1)]))

                v.update({'csd_distances': csd_distances})

            else:
                raise NotImplementedError


            ### pure csd 
            if self.susd_csd:
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((next_obs - obs) - s2_dist_mean) * normalized_scaling_factor, dim=1)
                v['csd_reward'] = cst_dist

                # cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
                # cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
                # te_obj = rewards + dual_lam[0].detach() * cst_penalty
                # te_objs = [te_obj]
                # cst_penalty = [cst_penalty]

            #### me
            cst_penalty = []
            te_objs = []

            for i in range(len(self.partition_points) - 1):
                start = i * self.dim_option
                end = (i+1) * self.dim_option
                cst_penalty_i = csd_distances[:, i] - torch.square(phi_y[:, start:end] - phi_x[:, start:end]).mean(dim=1)
                cst_penalty_i = torch.clamp(cst_penalty_i, max=self.dual_slack)
                te_obj_i = rewards[:, i] + dual_lam[i].detach() * cst_penalty_i
                cst_penalty.append(cst_penalty_i)
                te_objs.append(te_obj_i) 

            
            v.update({
                'cst_penalty': cst_penalty,
            })
        else:
            te_obj = rewards

        loss_te = []
        for te_obj in te_objs:
            loss_te_i = -te_obj.mean()
            loss_te.append(loss_te_i)

        tensors.update({
            'LossTe': loss_te
        })


    def plot_csd_logs(self, runner, min_csd=None, max_csd=None):
        if len(self.csd_logs) == 0:
            return

        epochs, csd_values = zip(*self.csd_logs)
        epochs = np.array(epochs)
        csd_values = 1000 * np.array(csd_values)  # Scale if desired

        # Clip values if bounds are provided
        if min_csd is not None or max_csd is not None:
            # Use default bounds if needed
            if min_csd is None:
                min_csd = -np.inf
            if max_csd is None:
                max_csd = np.inf
            csd_values = np.clip(csd_values, min_csd, max_csd)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(csd_values.shape[1]):
            ax.plot(
                epochs,
                csd_values[:, i],
                label=f'Factor {i}',
                marker='o',
                markersize=3,
                linewidth=1
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('CSD Value (×1e3)')
        ax.set_title('CSD per Factor over Epochs')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        csd_plot_path = f'results/{self.exp_name}/csd_plot_epoch_{runner.step_itr}.png'
        os.makedirs(os.path.dirname(csd_plot_path), exist_ok=True)
        fig.savefig(csd_plot_path)
        plt.close(fig)

    def _update_loss_dual_lam(self, tensors, v):
        assert len(v['cst_penalty']) == len(self.dual_lam)

        dual_lams = []
        loss_dual_lams = []

        for i, (dual_lam_module, cst_penalty_i) in enumerate(zip(self.dual_lam, v['cst_penalty'])):
            log_dual_lam_i = dual_lam_module()                           # log(λ_i)
            dual_lam_i = log_dual_lam_i.exp()                            # λ_i
            loss_dual_lam_i = log_dual_lam_i * cst_penalty_i.detach().mean()  # λ_i * E[cst]

            dual_lams.append(dual_lam_i)
            loss_dual_lams.append(loss_dual_lam_i)

            # Save each one individually
            tensors.update({
                f'DualLam_{i}': dual_lam_i,
                f'LossDualLam_{i}': loss_dual_lam_i,
            })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

        #### define the reward of the low-level policy 
        if self.susd_csd:
            rewards = v['rewards'].sum(dim=1) * v['csd_reward']
        else:
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

    def plot_early_stopping(self, early_stopping):
        unique_tasks, step_iters = zip(*early_stopping)

        plt.figure(figsize=(8, 5))
        plt.plot(step_iters, unique_tasks, marker='o', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Unique Completed Tasks')
        plt.title('Unique Task Coverage over Time')
        plt.grid(True)
        plt.tight_layout()

        save_path = f"results/{self.exp_name}"

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Early Stopping Plot Saved to: {save_path}")
        else:
            plt.show()

        plt.close()

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
        plt.xticks(step_iters)
        plt.tight_layout()

        save_path = f"results/{self.exp_name}_with_names"
        plt.savefig(save_path)
        print(f"Early Stopping Plot Saved to: {save_path}")

        plt.close()

    def _evaluate_policy(self, runner):

        if self.discrete:
            eye_options = np.eye(self.dim_option)  # (dim_option, dim_option)
            random_options = []
            colors = []

            for i in range(self.dim_option):
                num_trajs = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                one_hot = eye_options[i]  # shape (dim_option,)
                for _ in range(num_trajs):
                    # Repeat the one-hot N times → shape (N, dim_option)
                    multi_factor_option = np.tile(one_hot, (self.N, 1))  # (N, dim_option)
                    random_options.append(multi_factor_option)
                    colors.append(i)

            random_options = np.stack(random_options, axis=0)  # (num_trajs, N, dim_option)
            colors = np.array(colors)

            from matplotlib import cm
            cmap_name = 'tab10' if self.dim_option <= 10 else 'tab20'
            cmap = cm.get_cmap(cmap_name)
            random_option_colors = np.array([cmap(c)[:3] for c in colors])

        else:
            # random_option = np.random.randn(1, self.N, self.dim_option)
            # random_option /= np.linalg.norm(random_option, axis=-1, keepdims=True)
            # random_options = [random_option.copy()]

            # for i in range(self.num_random_trajectories - 1):
            #     new_random_option = random_option.copy()

            #     time_idx = i % self.N
            #     new_random_option[0, time_idx, :] = np.random.randn(self.dim_option)
            #     new_random_option /= np.linalg.norm(new_random_option, axis=-1, keepdims=True)
            #     random_options.append(new_random_option)
            
            # random_options = np.vstack(random_options)

            #### just one factor activate for z
            # activate_factor = self.counter % self.N
            # self.counter += 1
            # random_options = np.zeros((self.num_random_trajectoriesrajectories, self.N, self.dim_option))
            # random_options[:, activate_factor, :] = np.random.randn(self.num_random_trajectories, self.dim_option)

           #### main code 
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

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )


        from sklearn.decomposition import PCA
        data = self.process_samples(random_trajectories)
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

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option * self.N == 2:
                    # radius = 1. if self.unit_length else 1.5
                    # video_options = []
                    # for angle in [3, 2, 1, 4]:
                    #     video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    # video_options.append([0, 0])
                    # for angle in [0, 5, 6, 7]:
                    #     video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    # video_options = np.array(video_options)

                    video_options = np.random.randn(9, self.N * self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                    flat_random_options = video_options.reshape(9, self.N * self.dim_option)
                else:
                    # random_option = np.random.randn(1, self.N, self.dim_option)
                    # random_option /= np.linalg.norm(random_option, axis=-1, keepdims=True)
                    # random_options = [random_option.copy()]

                    # for i in range(17):
                    #     new_random_option = random_option.copy()

                    #     time_idx = i % self.N
                    #     new_random_option[0, time_idx, :] = np.random.randn(self.dim_option)
                    #     new_random_option /= np.linalg.norm(new_random_option, axis=-1, keepdims=True)
                    #     random_options.append(new_random_option)
                    
                    # random_options = np.vstack(random_options)
                    # flat_random_options = random_options.reshape(18, self.N * self.dim_option)

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

        self.plot_csd_logs(runner, 0, 5000)


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

