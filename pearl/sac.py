from collections import OrderedDict
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import losses, optimizers
# import torch.optim as optim

from pearl import tf2_util as tfu
from pearl.core.eval_util import create_stats_ordered_dict
from pearl.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optimizers.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        # self.qf_criterion = losses.MSE
        self.vf_criterion = losses.MSE
        # self.vib_criterion = losses.MSE
        # self.l2_reg_criterion = losses.MSE
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        self.qf1, self.qf2, self.vf, self.target_vf = nets[1:]

        self.policy_optimizer = optimizer_class(lr=policy_lr)
        self.qf_optimizer = optimizer_class(lr=qf_lr)
        self.vf_optimizer = optimizer_class(lr=vf_lr)
        self.context_optimizer = optimizer_class(lr=context_lr)

        # self.init_weight()

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def init_weight(self):
        obs_dim = self.agent.policy.input_size - self.latent_dim
        self.agent(tf.zeros([1, obs_dim]), tf.zeros([1, self.latent_dim]))
        self.qf1(tf.zeros([1, self.qf1.input_size]))
        self.qf2(tf.zeros([1, self.qf2.input_size]))
        self.vf(tf.zeros([1, self.vf.input_size]))
        self.target_vf(tf.zeros([1, self.target_vf.input_size]))

    def training_mode(self, mode):
        pass
        # for net in self.networks:
        #     net.train(mode)

    ##### Data handling #####
    def sample_data(self, indices, encoder=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = tfu.np_to_tf2_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
            else:
                batch = tfu.np_to_tf2_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            if encoder and self.sparse_rewards:
                # in sparse reward settings, only the encoder is trained with sparse reward
                r = batch['sparse_rewards'][None, ...]
            else:
                r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = tf.concat(obs, axis=0)
        actions = tf.concat(actions, axis=0)
        rewards = tf.concat(rewards, axis=0)
        next_obs = tf.concat(next_obs, axis=0)
        terms = tf.concat(terms, axis=0)
        return [obs, actions, rewards, next_obs, terms]

    def prepare_encoder_data(self, obs, act, rewards):
        ''' prepare context for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        task_data = tf.concat([obs, act, rewards], axis=2)
        return task_data

    def prepare_context(self, idx):
        ''' sample context from replay buffer and prepare it '''
        batch = tfu.np_to_tf2_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        context = self.prepare_encoder_data(obs, act, rewards)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, _, _ = mini_batch
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)
            self._take_step(indices, context)

            # stop backprop
            self.agent.stopgrad_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, tf.stop_gradient(task_z))
        q2 = self.qf2(obs, actions, tf.stop_gradient(task_z))
        min_q = tf.minimum(q1, q2)
        return min_q

    def _update_target_network(self):
        tfu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)

        # run inference in networks
        with tf.GradientTape(persistent=True) as tape:
            policy_outputs, task_z = self.agent(obs, context)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

            # flattens out the task dimension
            t, b, _ = obs.shape
            obs = tf.reshape(obs, [t * b, -1])
            actions = tf.reshape(actions, [t * b, -1])
            next_obs = tf.reshape(next_obs, [t * b, -1])

            # Q and V networks
            # encoder will only get gradients from Q nets
            q1_pred = self.qf1(obs, actions, task_z)
            q2_pred = self.qf2(obs, actions, task_z)
            v_pred = self.vf(obs, tf.stop_gradient(task_z))
            # get targets for use in V and Q updates
            target_v_values = tf.stop_gradient(self.target_vf(next_obs, task_z))

            # KL constraint on z if probabilistic
            if self.use_information_bottleneck:
                kl_div = self.agent.compute_kl_div()
                kl_loss = self.kl_lambda * kl_div

            # qf and encoder update (note encoder does not get grads from policy or vf)
            rewards_flat = tf.reshape(rewards, [self.batch_size * num_tasks, -1])
            # scale rewards for Bellman update
            rewards_flat = rewards_flat * self.reward_scale
            terms_flat = tf.reshape(terms, [self.batch_size * num_tasks, -1])
            q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
            qf_loss = tf.reduce_mean((q1_pred - q_target) ** 2) + tf.reduce_mean((q2_pred - q_target) ** 2)

            context_loss = kl_loss + qf_loss

            # compute min Q on the new actions
            min_q_new_actions = self._min_q(obs, new_actions, task_z)

            # vf update
            v_target = min_q_new_actions - log_pi
            vf_loss = self.vf_criterion(v_pred, tf.stop_gradient(v_target))

            # policy update
            # n.b. policy update includes dQ/da
            log_policy_target = min_q_new_actions

            policy_loss = tf.reduce_mean(log_pi - log_policy_target)

            mean_reg_loss = self.policy_mean_reg_weight * tf.reduce_mean(policy_mean ** 2)
            std_reg_loss = self.policy_std_reg_weight * tf.reduce_mean(policy_log_std ** 2)
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * tf.reduce_mean(
                tf.reduce_sum(pre_tanh_value**2, axis=1))
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss


        grad_context_loss = tape.gradient(context_loss, self.agent.context_encoder.trainable_variables)
        grad_qf1_loss = tape.gradient(qf_loss, self.qf1.trainable_variables)
        grad_qf2_loss = tape.gradient(qf_loss, self.qf2.trainable_variables)

        self.qf_optimizer.apply_gradients(zip(grad_qf1_loss, self.qf1.trainable_variables))
        self.qf_optimizer.apply_gradients(zip(grad_qf2_loss, self.qf2.trainable_variables))
        self.context_optimizer.apply_gradients(zip(grad_context_loss, self.agent.context_encoder.trainable_variables))

        grad_vf_loss = tape.gradient(vf_loss, self.vf.trainable_variables)
        self.vf_optimizer.apply_gradients(zip(grad_vf_loss, self.vf.trainable_variables))
        self._update_target_network()

        grad_policy_loss = tape.gradient(policy_loss, self.agent.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grad_policy_loss, self.agent.policy.trainable_variables))

        del tape

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(self.agent.z_means[0].numpy()))
                z_sig = np.mean(self.agent.z_vars[0].numpy())
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = kl_div.numpy()
                self.eval_statistics['KL Loss'] = kl_loss.numpy()

            self.eval_statistics['QF Loss'] = np.mean(qf_loss.numpy())
            self.eval_statistics['VF Loss'] = np.mean(vf_loss.numpy())
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss.numpy())
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                q1_pred.numpy(),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                v_pred.numpy(),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                log_pi.numpy(),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                policy_mean.numpy(),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                policy_log_std.numpy(),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.agent.policy,
            vf=self.vf,
            target_vf=self.target_vf,
            context_encoder=self.agent.context_encoder,
        )
        # snapshot = OrderedDict(
        #     qf1=self.qf1.get_weights(),
        #     qf2=self.qf2.get_weights(),
        #     policy=self.agent.policy.get_weights(),
        #     vf=self.vf.get_weights(),
        #     target_vf=self.target_vf.get_weights(),
        #     context_encoder=self.agent.context_encoder.get_weights(),
        # )
        return snapshot
