
from copy import deepcopy
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
import tensorflow_probability.python.distributions as tfd


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = tf.clip_by_value(sigmas_squared, 1e-7, np.inf)
    sigma_squared = 1. / tf.reduce_sum(tf.math.reciprocal(sigmas_squared), axis=0)
    mu = sigma_squared * tf.reduce_sum(mus / sigmas_squared, axis=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = tf.reduce_mean(mus, axis=0)
    sigma_squared = tf.reduce_mean(sigmas_squared, axis=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(Model):
    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs.get('recurrent', False)
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.z = tf.zeros([1, self.latent_dim])
        self.z_means = tf.zeros([1, self.latent_dim])
        self.z_vars = tf.zeros([1, self.latent_dim])

        self.clear_z()
        self.init_weights()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = tf.zeros([num_tasks, self.latent_dim])
        if self.use_ib:
            var = tf.ones([num_tasks, self.latent_dim])
        else:
            var = tf.zeros([num_tasks, self.latent_dim])
        self.z_means = deepcopy(mu)
        self.z_vars = deepcopy(var)
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def stopgrad_z(self):
        ''' disable backprop through z '''
        self.z = tf.stop_gradient(self.z)
        if self.recurrent:
            self.context_encoder.hidden = tf.stop_gradient(self.context_encoder.hidden)

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = tf.convert_to_tensor(o[None, None, ...], dtype=tf.float32)
        a = tf.convert_to_tensor(a[None, None, ...], dtype=tf.float32)
        r = tf.convert_to_tensor(np.array([r])[None, None, ...], dtype=tf.float32)
        data = tf.concat([o, a, r], axis=2)
        if self.context is None:
            self.context = data
        else:
            self.context = tf.concat([self.context, data], axis=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = tfd.Normal(tf.zeros(self.latent_dim), tf.ones(self.latent_dim))
        posteriors = [tfd.Normal(mu, tf.math.sqrt(var)) for mu, var in zip(tf.unstack(self.z_means), tf.unstack(self.z_vars))]
        kl_divs = [tfd.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = tf.reduce_sum(tf.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = tf.reshape(params, [context.shape[0], -1, self.context_encoder.output_size])
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = tf.math.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(tf.unstack(mu), tf.unstack(sigma_squared))]
            self.z_means = tf.stack([p[0] for p in z_params])
            self.z_vars = tf.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = tf.reduce_mean(params, axis=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            # posteriors = [tfd.Normal(m, tf.math.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            # z = [d.rsample() for d in posteriors]
            z = [m + tf.math.sqrt(s) * tfd.Normal(tf.zeros_like(m), tf.ones_like(s)).sample()
                 for m, s in zip(tf.unstack(self.z_means), tf.unstack(self.z_vars))]
            self.z = tf.stack(z)
        else:
            self.z = deepcopy(self.z_means)

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = deepcopy(self.z)
        obs = tf.convert_to_tensor(obs[None], dtype=tf.float32)
        in_ = tf.concat([obs, z], axis=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def call(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = deepcopy(self.z)

        t, b, _ = obs.shape
        obs = tf.reshape(obs, [t * b, -1])
        task_z = [tf.stack([z for _ in range(b)]) for z in task_z]
        task_z = tf.concat(task_z, axis=0)

        # run policy, get log probs and new actions
        in_ = tf.concat([obs, tf.stop_gradient(task_z)], axis=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(self.z_means[0].numpy()))
        z_sig = np.mean(self.z_vars[0].numpy())
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    def init_weights(self):
        for net in self.networks:
            net(tf.zeros([1, net.input_size]))

    @property
    def networks(self):
        return [self.context_encoder, self.policy]



