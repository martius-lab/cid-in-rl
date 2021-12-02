import argparse
import random
import pathlib
import os
import sys
from datetime import datetime

import gin
import numpy as np
import torch
import tqdm

from cid.memory import EpisodicReplayMemory
from cid.utils.logger import ConfigurableLogger

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more information')
parser.add_argument('--log-dir', default='./logs',
                    help='Path to logging directory')
parser.add_argument('--no-logging-subdir', action='store_true', default=False,
                    help='Directly use given log-dir as logging directory')
parser.add_argument('--render', action='store_true',
                    help='Render the environment')
parser.add_argument('--seed', type=int, help='Seed for RNG')
parser.add_argument('--store-buffer', action='store_true',
                    help='Store buffer to disk')
parser.add_argument('--use-warmup-agent', action='store_true',
                    help='Use warmup agent instead of normal agent')
parser.add_argument('--episodes', type=int, default=1,
                    help='Number of episodes to execute')
parser.add_argument('--agent-path', help='Path to stored agent')
parser.add_argument('config', help='Path to gin configuration file')
parser.add_argument('config_args', nargs='*',
                    help='Optional configuration settings')


def play(rollout_gen,
         agent,
         episode_len,
         n_episodes,
         logger,
         replay_memory=None,
         save_path=None,
         render=False):
    logger.log_message(f'Executing {n_episodes} episodes')

    for _ in tqdm.trange(n_episodes):
        rollout_gen.reset()
        episode, stats = rollout_gen.rollout(agent,
                                             episode_len,
                                             evaluate=True,
                                             render=render)
        if replay_memory is not None:
            replay_memory.store_episodes(episode)

        logger.store(**stats)

    if replay_memory is not None:
        stats = replay_memory.report_statistics()
        logger.store(**stats)

        path = save_path + f'/memory.npy'
        logger.log_message(f'Saving memory to {path}')
        replay_memory.save(path)

    logger.dump_tabular(n_episodes)


@gin.configurable(blacklist=['args'])
def main(args, experiment_name, env_fn, agent_fn, rollout_gen_cls,
         replay_memory_cls, episode_len, warmup_agent_fn=None,
         model_setup_fn=None, test_memory_cls=None, n_workers=1,
         experiment_group=None, seed=None, assign_model_to_env=False):
    save_path = None
    if args.store_buffer:
        time = datetime.now()
        save_path = pathlib.Path(args.log_dir)

        if not args.no_logging_subdir:
            if experiment_group is not None:
                save_path /= experiment_group
            save_path /= f'{time:%Y-%m-%d-%H-%M-%S}_{experiment_name}_play'
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'settings.gin', 'w') as f:
            f.write(gin.config_str())

    logger = ConfigurableLogger(save_path, log_to_tensorboard=False)

    if args.seed is not None:
        seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32 - 1)

    if args.verbose:
        logger.log_message(f'Running on host "{os.uname().nodename}"')
        logger.log_message(f'Random seed is {seed}')
        logger.log_message(f'Gin configuration:\n\n{gin.config_str()}')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = env_fn()
    env.seed(seed)
    rollout_gen = rollout_gen_cls(env)

    if not args.use_warmup_agent:
        agent = agent_fn(rollout_gen.observation_space,
                         rollout_gen.action_space)
    else:
        agent = warmup_agent_fn(rollout_gen.observation_space,
                                rollout_gen.action_space)

    if args.agent_path is not None:
        logger.log_message(f'Loading agent from {args.agent_path}')
        agent.load(args.agent_path)

    example = rollout_gen.example_transition

    if model_setup_fn is not None:
        model, _ = model_setup_fn(example)
        if assign_model_to_env:
            env.set_model(model)

    replay_memory = None
    if args.store_buffer:
        replay_memory = EpisodicReplayMemory(example,
                                             size=args.episodes,
                                             episode_len=episode_len)

        if args.verbose:
            msg = 'Example transition:\n'
            msg += '\n'.join(f'- {key}: {val.shape}, {val.dtype}'
                             for key, val in example.items())
            logger.log_message(msg)

    play(rollout_gen, agent, episode_len, args.episodes,
         logger=logger,
         replay_memory=replay_memory,
         save_path=None if save_path is None else str(save_path),
         render=args.render)

    rollout_gen.close()


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    gin.parse_config_files_and_bindings([args.config], args.config_args,
                                        skip_unknown=['train', 'test'])
    main(args)
