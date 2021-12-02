import argparse
import random
import os
import pathlib
import sys
from datetime import datetime

import gin
import numpy as np
import torch
import tqdm

from cid.rollouts import ParallelRolloutGenerator
from cid.utils import update_dict_of_lists
from cid.utils.logger import ConfigurableLogger


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more information')
parser.add_argument('-n', '--dry', action='store_true',
                    help='If true, do not log to disk')
parser.add_argument('--log-dir', default='./logs',
                    help='Path to logging directory')
parser.add_argument('--no-logging-subdir', action='store_true', default=False,
                    help='Directly use given log-dir as logging directory')
parser.add_argument('--no-tqdm', action='store_true', default=False,
                    help='Disable tqdm (for cluster use)')
parser.add_argument('--render', action='store_true',
                    help='Render the environment')
parser.add_argument('--seed', type=int, help='Seed for RNG')
parser.add_argument('config', help='Path to gin configuration file')
parser.add_argument('config_args', nargs='*',
                    help='Optional configuration settings')


@gin.configurable(blacklist=['rollout_gen', 'agent', 'episode_len',
                             'test_memory'])
def test(rollout_gen, agent, episode_len, n_episodes=1, test_memory=None):
    test_stats = {}
    for _ in range(n_episodes):
        rollout_gen.reset()
        episodes, stats = rollout_gen.rollout(agent,
                                              episode_len,
                                              evaluate=True,
                                              render=False)
        update_dict_of_lists(test_stats, stats)
        if test_memory is not None:
            test_memory.store_episodes(episodes)

    return {key + 'Test': value for key, value in test_stats.items()}


@gin.configurable(blacklist=['rollout_gen',
                             'agent',
                             'replay_memory',
                             'episode_len',
                             'warmup_agent',
                             'model_trainer',
                             'test_memory',
                             'logger',
                             'save_path',
                             'render'])
def train(rollout_gen,
          agent,
          replay_memory,
          episode_len,
          n_iterations,
          n_updates_per_step,
          batch_size,
          logger,
          buffer_warmup=0,
          warmup_agent=None,
          model_trainer=None,
          model_train_kwargs=None,
          train_model_every=100,
          train_model_schedule=None,
          test_memory=None,
          evaluate_every=50,
          log_every=50,
          save_every=100,
          save_buffer_every=None,
          save_path=None,
          render=False):
    if evaluate_every > 0:
        assert log_every % evaluate_every == 0, \
            '`log_every must be a multiple of `evaluate_every`'
    if model_trainer is not None:
        assert log_every % train_model_every == 0, \
            '`log_every must be a multiple of `train_model_every`'

    if warmup_agent is None:
        warmup_agent = agent

    if model_train_kwargs is None:
        model_train_kwargs = {}

    for step in tqdm.trange(n_iterations):
        rollout_gen.reset()

        agent_to_use = agent if step >= buffer_warmup else warmup_agent

        episodes, stats = rollout_gen.rollout(agent_to_use,
                                              episode_len,
                                              evaluate=False,
                                              render=render)
        logger.store(**stats)

        replay_memory.store_episodes(episodes)

        if step >= buffer_warmup - 1:
            if model_trainer is not None:
                if train_model_schedule is None:
                    train = ((step - (buffer_warmup - 1))
                             % train_model_every == 0)
                    train_kwargs = {} if train else None
                else:
                    train, train_kwargs = train_model_schedule(step)

                if train:
                    train_kwargs.update(model_train_kwargs)
                    logger.log_message(f'Step {step}: Training model')
                    stats = model_trainer.train(replay_memory, **train_kwargs)
                    logger.store(**stats)

            stats = agent.update_parameters(replay_memory,
                                            batch_size,
                                            n_updates_per_step)
            logger.store(**stats)

        if save_path is not None:
            if (step + 1) % save_every == 0 or step + 1 == n_iterations:
                path = save_path + f'/policy_{step + 1}.pth'
                logger.log_message(f'Step {step}: Saving policy to {path}')
                agent.save(path)
                if model_trainer is not None:
                    path = model_trainer.save(save_path, step)
                    logger.log_message((f'Step {step}: Saving model trainer '
                                        f'to {path}'))
            if (save_buffer_every is not None and
                    (step + 1) % save_buffer_every == 0):
                path = save_path + f'/memory_{step + 1}.npy'
                logger.log_message(f'Step {step}: Saving memory to {path}')
                replay_memory.save(path)
                if test_memory is not None:
                    path = save_path + f'/test_memory_{step + 1}.npy'
                    test_memory.save(path)

        if evaluate_every > 0 and (step + 1) % evaluate_every == 0:
            stats = test(rollout_gen, agent, episode_len,
                         test_memory=test_memory)
            logger.store(**stats)

        if step >= buffer_warmup - 1 and (step + 1) % log_every == 0:
            stats = replay_memory.report_statistics()
            logger.store(**stats)
            if test_memory is not None:
                stats = test_memory.report_statistics()
                stats = {f'Test{key}': value for key, value in stats.items()}
                logger.store(**stats)

            logger.log_tabular('Time', f'{datetime.now():%Y/%m/%d %H:%M:%S}',
                               log_to_tensorboard=False)
            logger.log_tabular('Step', step + 1, log_to_tensorboard=False)
            logger.dump_tabular(step + 1)


@gin.configurable(blacklist=['args'])
def main(args, experiment_name, env_fn, agent_fn, rollout_gen_cls,
         replay_memory_cls, episode_len, warmup_agent_fn=None,
         model_setup_fn=None, test_memory_cls=None,
         n_workers=1, memory_needs_model=True,
         experiment_group=None, seed=None):
    if args.dry:
        save_path = None
    else:
        time = datetime.now()
        save_path = pathlib.Path(args.log_dir)

        if not args.no_logging_subdir:
            if experiment_group is not None:
                save_path /= experiment_group
            dir_name = f'{time:%Y-%m-%d-%H-%M-%S}_{experiment_name}'
            save_path /= dir_name
        else:
            dir_name = save_path.name

        count = 2
        while save_path.is_dir():
            save_path = save_path.with_name(f"{dir_name}_{count}")
            count += 1

        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'settings.gin', 'w') as f:
            f.write(gin.config_str())

    if args.no_tqdm:
        tqdm.tqdm = lambda *args, **kwargs: iter(*args)
        tqdm.trange = lambda *args, **kwargs: range(*args)

    logger = ConfigurableLogger(save_path)

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

    if n_workers > 1:
        assert warmup_agent_fn is None, \
            'Can not use warmup agent with `n_workers > 1` currently'

        def generator_fn(env):
            return rollout_gen_cls(env)

        rollout_gen = ParallelRolloutGenerator(n_workers, env_fn, agent_fn,
                                               generator_fn, seed,
                                               gin.config_str())
    else:
        env = env_fn()
        env.seed(seed)
        rollout_gen = rollout_gen_cls(env)

    agent = agent_fn(rollout_gen.observation_space, rollout_gen.action_space)

    warmup_agent = None
    if warmup_agent_fn is not None:
        warmup_agent = warmup_agent_fn(rollout_gen.observation_space,
                                       rollout_gen.action_space)

    example = rollout_gen.example_transition

    if model_setup_fn is not None:
        model, model_trainer = model_setup_fn(example)
    else:
        model = None
        model_trainer = None

    if model is not None and memory_needs_model:
        replay_memory_extra_args = {'model': model}
    else:
        replay_memory_extra_args = {}

    replay_memory = replay_memory_cls(example,
                                      episode_len=episode_len,
                                      **replay_memory_extra_args)

    if test_memory_cls is not None:
        test_memory = test_memory_cls(example, episode_len=episode_len,
                                      **replay_memory_extra_args)
    else:
        test_memory = None

    if model is not None and hasattr(agent.unwrapped, 'set_model'):
        agent.unwrapped.set_model(model)

    if args.verbose:
        msg = 'Example transition:\n'
        msg += '\n'.join(f'- {key}: {val.shape}, {val.dtype}'
                         for key, val in example.items())
        logger.log_message(msg)

    train(rollout_gen, agent, replay_memory, episode_len,
          warmup_agent=warmup_agent,
          model_trainer=model_trainer,
          test_memory=test_memory,
          logger=logger,
          save_path=None if save_path is None else str(save_path),
          render=args.render)

    rollout_gen.close()


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    gin.parse_config_files_and_bindings([args.config], args.config_args)
    main(args)
