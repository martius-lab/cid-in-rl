"""Script to train models on collected trajectories"""
import argparse
import datetime
import random
import pathlib
import sys
from datetime import datetime

import gin
import numpy as np
import torch
import tqdm

from cid.influence_estimation import make_model_trainer, utils
from cid.memory import EpisodicReplayMemory
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
parser.add_argument('--seed', type=int, help='Seed for RNG')
parser.add_argument('--memory-path', help='Path to stored memory')
parser.add_argument('--val-memory-path',
                    help='Path to stored validation memory')
parser.add_argument('config', help='Path to gin configuration file')
parser.add_argument('config_args', nargs='*',
                    help='Optional configuration settings')


def _get_example(env_fn, rollout_gen_cls):
    env = env_fn()
    rollout_gen = rollout_gen_cls(env)
    example = rollout_gen.example_transition
    rollout_gen.close

    return example


def _build_report(losses, metrics, prefix=''):
    report = {prefix + 'loss': np.mean(losses)}
    if 'mse' in metrics:
        p99, p100 = np.percentile(metrics['mse'], (99, 100))
        report[prefix + 'mse'] = metrics['mse'].mean()
        report[prefix + 'mse_p99'] = p99
        report[prefix + 'mse_p100'] = p100
    if 'bin_acc' in metrics:
        report[prefix + 'bin_acc'] = metrics['bin_acc'].mean()
    if 'var' in metrics:
        p01, p50, p99 = np.percentile(metrics['var'], (1, 50, 99))
        report[prefix + 'var'] = metrics['var'].mean()
        report[prefix + 'var_p01'] = p01
        report[prefix + 'var_p50'] = p50
        report[prefix + 'var_p99'] = p99
    if 'pred_norm' in metrics:
        p50, p99 = np.percentile(metrics['pred_norm'], (50, 99))
        report[prefix + 'pred'] = metrics['pred_norm'].mean()
        report[prefix + 'pred_p50'] = p50
        report[prefix + 'pred_p99'] = p99
    if 'log_likelihood' in metrics:
        report[prefix + 'log_likelihood'] = metrics['log_likelihood'].mean()
    if 'var_mse' in metrics:
        p99 = np.percentile(metrics['var_mse'], 99)
        report[prefix + 'var_mse'] = metrics['var_mse'].mean()
        report[prefix + 'var_mse_p99'] = p99

    return report


def update_best_metrics(best_metrics, best_epochs, metrics, epoch):
    for name, value in metrics.items():
        if name not in best_metrics:
            best_metrics[name] = value
            best_epochs[name] = epoch
        else:
            best_metrics[name] = min(best_metrics[name], value)
            if best_metrics[name] == value:
                best_epochs[name] = epoch


@gin.configurable(blacklist=['model_trainer', 'replay_memory', 'logger',
                             'val_replay_memory'])
def train(model_trainer, replay_memory, logger, n_epochs, log_every=5,
          n_total_updates=None, bootstrap=False, val_replay_memory=None,
          val_split=None, validate_every=1, n_workers=0, save_path=None,
          save_prefix=None, best_metric=None, early_stopping_epochs=None):
    assert (val_replay_memory is None) or (val_split is None), \
        'Validation can be done through replay memory or splitting, not both'
    splits = None
    if val_split is not None:
        splits = [1 - val_split, val_split]

    train_dataloader = model_trainer.create_dataloader(replay_memory,
                                                       bootstrap=bootstrap,
                                                       n_workers=n_workers,
                                                       split_ratios=splits,
                                                       split_idx=0)
    val_dataloader = None
    if val_replay_memory is not None:
        val_dataloader = model_trainer.create_dataloader(val_replay_memory,
                                                         bootstrap=bootstrap,
                                                         shuffle=False,
                                                         n_workers=n_workers,
                                                         val_dataset=True)
    elif splits is not None:
        val_dataloader = model_trainer.create_dataloader(replay_memory,
                                                         bootstrap=bootstrap,
                                                         n_workers=n_workers,
                                                         split_ratios=splits,
                                                         split_idx=1,
                                                         val_dataset=True)

    if isinstance(train_dataloader, list):
        bs = train_dataloader[0].batch_size
        n_samples = len(train_dataloader[0].dataset)
    else:
        bs = train_dataloader.batch_size
        n_samples = len(train_dataloader.dataset)

    if n_total_updates is not None:
        n_epochs = int(np.ceil(n_total_updates * bs / n_samples))
    if isinstance(log_every, float):
        log_every = max(int(np.floor(log_every * n_epochs)), 1)

    logger.log_message((f'Training model on {n_samples} transitions per epoch '
                        f'for {n_epochs} epochs with batch size {bs}'))

    best_metrics = {}
    best_epochs = {}
    updates_since_improvement = 0
    early_stopping = False

    for epoch in tqdm.trange(n_epochs):
        losses, metrics = model_trainer.train_epoch(train_dataloader,
                                                    return_metrics=True)

        logger.store(**_build_report(losses, metrics))

        if val_dataloader is not None and ((epoch + 1) % validate_every == 0
                                           or epoch == 0
                                           or epoch + 1 == n_epochs):
            losses, metrics = model_trainer.validate(val_dataloader,
                                                     return_metrics=True)
            report = _build_report(losses, metrics, 'val_')
            logger.store(**report)

            update_best_metrics(best_metrics, best_epochs, report, epoch)
            if save_path is not None and best_metric is not None:
                if best_epochs[best_metric] == epoch:
                    updates_since_improvement = 0
                    best_val = best_metrics[best_metric]
                    path = save_path / f'{save_prefix}_best.pth'
                    s = (f'Saving best model to {path} at epoch {epoch + 1} '
                         f'as `{best_metric}` improved to {best_val:.6f}')
                    logger.log_message(s)
                    model_trainer.save(str(path))
                elif early_stopping_epochs is not None:
                    updates_since_improvement += 1
                    if updates_since_improvement >= early_stopping_epochs:
                        early_stopping = True

        logger.log_tabular('Step', epoch + 1, log_to_tensorboard=False)
        log_stdout = (epoch + 1) % log_every == 0 or epoch + 1 == n_epochs
        if log_stdout:
            logger.log_tabular('Time', f'{datetime.now():%Y/%m/%d %H:%M:%S}',
                               log_to_tensorboard=False)
        logger.dump_tabular(epoch + 1, log_stdout=log_stdout)

        if early_stopping:
            s = (f'Stopping in epoch {epoch + 1} as metric `{best_metric}'
                 f'={best_val:.6f}` has not improved for '
                 f'{early_stopping_epochs} validations')
            logger.log_message(s)
            break


@gin.configurable(blacklist=['args'])
def main(args, experiment_name, episode_len, env_fn, rollout_gen_cls,
         model_classes, dataset_classes, memory_size=10000,
         experiment_group=None, seed=None, memory_path=None,
         val_memory_path=None):
    if memory_path is None:
        memory_path = args.memory_path
    assert memory_path is not None, 'No memory path given'
    if val_memory_path is None:
        val_memory_path = args.val_memory_path

    if args.dry:
        save_path = None
    else:
        time = datetime.now()
        save_path = pathlib.Path(args.log_dir)

        if not args.no_logging_subdir:
            if experiment_group is not None:
                save_path /= experiment_group
            save_path /= f'{time:%Y-%m-%d-%H-%M-%S}_{experiment_name}'
        save_path.mkdir(parents=True, exist_ok=True)
        settings_path = save_path / 'settings.gin'
        with open(settings_path, 'w') as f:
            f.write(gin.config_str())

    logger = ConfigurableLogger(save_path)

    if args.seed is not None:
        seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32 - 1)

    if args.verbose:
        logger.log_message(gin.config_str())
        logger.log_message(f'Random seed is {seed}')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    replay_memory = EpisodicReplayMemory({},
                                         size=memory_size,
                                         episode_len=episode_len)
    replay_memory.load(memory_path)

    val_replay_memory = None
    if val_memory_path is not None:
        val_replay_memory = EpisodicReplayMemory({},
                                                 size=memory_size,
                                                 episode_len=episode_len)
        val_replay_memory.load(val_memory_path)

    trainers = {}
    for name, model_cls in model_classes.items():
        trainers[name] = make_model_trainer(None,
                                            model_cls,
                                            dataset_classes[name],
                                            replay_memory=replay_memory)

    for name, trainer in trainers.items():
        logger.log_message(f'Training model `{name}`')
        logger.new_csv_file(f'data_{name}.csv')
        train(trainer, replay_memory, logger,
              val_replay_memory=val_replay_memory,
              save_path=save_path,
              save_prefix=f'trainer_{name}')

        if save_path is not None:
            path = save_path / f'trainer_{name}.pth'
            logger.log_message(f'Saving trained model `{name}` to {path}')
            trainer.save(str(path))

    if save_path is not None:
        eval_settings_path = utils.convert_train_settings(settings_path)
        if args.verbose:
            print((f'Converted train settings {settings_path} to '
                   f'eval settings {eval_settings_path}'))


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    gin.parse_config_files_and_bindings([args.config], args.config_args)
    main(args)
