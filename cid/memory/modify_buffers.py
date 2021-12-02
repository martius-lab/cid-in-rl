"""Script to modify stored buffers"""
import argparse
import random
import pathlib
import sys

import gin
import numpy as np
import torch
import tqdm

from cid.memory import EpisodicReplayMemory

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
parser.add_argument('--seed', default=0, help='Random seed')
parser.add_argument('--start', default=0, help='Start from this episode index')
parser.add_argument('--end', type=int, help='End at this episode index')
parser.add_argument('--subsample', type=int,
                    help='Subsample this many episodes')
parser.add_argument('--stratify', type=int,
                    help='With subsample, groups to sample balanced from')
parser.add_argument('--add-state-noise', action='store_true',
                    help='Add state noise')
parser.add_argument('output_path', help='Path to joined buffer')
parser.add_argument('memory_paths', nargs='+',
                    help='Path to buffers to store')


def main(args):
    memories = []

    for path in args.memory_paths:
        memory = EpisodicReplayMemory({}, size=1, episode_len=1)
        memory.load(path)
        memories.append(memory)

    episode_len = None
    keys = None
    shapes = None
    dtypes = None

    for memory in memories:
        if episode_len is None:
            episode_len = memory.episode_len
            keys = list(memory._buffer.keys())
            shapes = {key: memory._buffer[key].shape for key in keys}
            dtypes = {key: memory._buffer[key].dtype for key in keys}
        else:
            assert episode_len == memory.episode_len
            assert keys == list(memory._buffer.keys())

    if args.verbose:
        for path, memory in zip(args.memory_paths, memories):
            print((f'Buffer {path} with current size '
                   f'{memory.current_size} contains'))
            for key, value in memory._buffer.items():
                print(f'  {key}: {value.shape}')

    example = {key: np.zeros(shape[2:], dtype=dtypes[key])
               for key, shape in shapes.items()}
    if args.add_state_noise:
        example['s_noise'] = example['s'].copy()

    total_size = 0
    episodes_per_memory = []
    for memory in memories:
        end = memory.current_size if args.end is None else args.end
        end = min(memory.current_size, end)
        start = args.start
        if start >= end:
            raise ValueError('start_index >= end_index')

        if args.subsample is not None:
            if args.stratify is not None:
                group_size = (end - start) // args.stratify
                samples_per_group = args.subsample // args.stratify
                indices = [np.random.choice(group_size, samples_per_group,
                                            replace=False) + pos
                           for pos in range(start, end, group_size)]
                indices = np.concatenate(indices)
            else:
                indices = np.random.choice(end - start, args.subsample,
                                           replace=False) + start
        else:
            indices = slice(start, end)

        episodes = {key: val[indices]
                    for key, val in memory._buffer.items()}
        if args.add_state_noise:
            episodes['s_noise'] = np.random.randn(*episodes['s'].shape)

        total_size += len(next(iter(episodes.values())))
        episodes_per_memory.append(episodes)

    joint_memory = EpisodicReplayMemory(example,
                                        size=total_size,
                                        episode_len=episode_len - 1)
    for episodes in episodes_per_memory:
        joint_memory.store_episodes(episodes)
    joint_memory.save(args.output_path)


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)
