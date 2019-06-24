
from pearl.envs import ENVS
from pearl.envs.env import NormalizedBoxEnv
from pearl.policy import TanhGaussianPolicy
from pearl.network import FlattenMLP, MLPEncoder   # , RecurrentEncoder
from pearl.sac import PEARLSoftActorCritic
from pearl.agent import PEARLAgent
from pearl.launcher import setup_logger
from pearl.configs.default import default_config

from timeseries.config import Config
from timeseries.model import TSModel
from timeseries.data_process import DataScheduler
from timeseries.rl import MyEnv

import os
import pathlib
import numpy as np


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def main():
    ts_configs = Config()
    # get data for all assets and dates
    ds = DataScheduler(ts_configs, is_infocode=False)
    ds.test_end_idx = ds.base_idx + 1000

    ii = 0

    model = TSModel(ts_configs)
    ts_configs.f_name = 'ts_model_test_factor1.0'
    if os.path.exists(ts_configs.f_name):
        model.load_model(ts_configs.f_name)

    # ds.set_idx(6250)
    ds.train(model,
             train_steps=ts_configs.train_steps,
             eval_steps=10,
             save_steps=200,
             early_stopping_count=30,
             model_name=ts_configs.f_name)

    # env = MyEnv(model, data_scheduler=ds, configs=ts_configs, trading_costs=0.001)

    env_name = 'korea-stock'
    variant = default_config
    variant['env_name'] = env_name
    #
    # with open("./configs/{}.json".format(env_name)) as f:
    #     exp_params = json.load(f)
    # variant = deep_update_dict(exp_params, variant)

    env = NormalizedBoxEnv(ENVS[variant['env_name']](model, ds, ts_configs, **variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    # encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    encoder_model = MLPEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )
    qf1 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    target_vf = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['env_params']['n_tasks_dict']['train']]),
        eval_tasks=list(tasks[-variant['env_params']['n_tasks_dict']['eval']:]),
        nets=[agent, qf1, qf2, vf, target_vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    def example():
        train_tasks = list(tasks[:variant['env_params']['n_tasks_dict']['train']])
        meta_batch = 64
        indices = np.random.choice(train_tasks, meta_batch)


    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()
