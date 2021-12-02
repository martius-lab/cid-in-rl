"""Script to evaluate models with influence ground truth labels"""
import argparse
import pathlib
import random
import sys

import gin
import numpy as np
import torch
from sklearn import metrics as metrics_

from cid import influence_estimation
from cid.influence_estimation import utils
from cid.memory import EpisodicReplayMemory
from cid.memory.score_based import ScoreBasedReplayMemory

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more information')
parser.add_argument('--seed', type=int, help='Seed for RNG')
parser.add_argument('--convert-settings', action='store_true',
                    help='Convert training settings to eval settings')
parser.add_argument('-o', '--output-path', help='Path for output file')
parser.add_argument('memory_path', help='Path to memory')
parser.add_argument('model_path', help='Path to model checkpoint')
parser.add_argument('config', help='Path to eval config')
parser.add_argument('config_args', nargs='*',
                    help='Optional configuration settings')


def get_metrics(scores, labels):
    fpr, tpr, roc_thresholds = metrics_.roc_curve(labels, scores)
    roc_auc = metrics_.roc_auc_score(labels, scores)
    precision, recall, pr_thresholds = metrics_.precision_recall_curve(labels,
                                                                       scores)
    average_precision = metrics_.average_precision_score(labels, scores)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores[np.isnan(f1_scores)] = 0
    best_f1_thresh = pr_thresholds[np.argmax(f1_scores)]

    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'ap': average_precision,
        'roc_th': roc_thresholds,
        'pr_th': pr_thresholds,
        'f1': f1_scores,
        'best_f1_thresh': best_f1_thresh
    }


@gin.configurable(blacklist=['args'])
def main(args, episode_len, scorer_cls, influence_label='control', seed=None):
    if args.seed is not None:
        seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32 - 1)

    if args.verbose:
        print(gin.config_str())
        print(f'Random seed is {seed}')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    np.seterr(all="ignore")  # Ignore invalid value in divide errors

    model_path = pathlib.Path(args.model_path)
    if not model_path.exists():
        raise ValueError(f'Model path {model_path} does not exist')

    load_paths = {}
    for model_name, filename in zip(('full_model',
                                     'full_model',
                                     'capped_model'),
                                    ('trainer_full_model_best.pth',
                                     'trainer_full_model.pth',
                                     'trainer_capped_model.pth')):
        if (model_path / filename).exists() and model_name not in load_paths:
            load_paths[model_name] = str(model_path / filename)
            if args.verbose:
                print(f'Using model {load_paths[model_name]} for {model_name}')

    if len(load_paths) == 0:
        raise ValueError(f'Found no model to load in {model_path}')

    memory = EpisodicReplayMemory({}, size=1, episode_len=episode_len)
    memory.load(args.memory_path)

    scorer = influence_estimation.make_pretrained_scorer(example=None,
                                                         load_paths=load_paths,
                                                         scorer_cls=scorer_cls,
                                                         replay_memory=memory)
    scorer = scorer[0]

    score_memory = ScoreBasedReplayMemory({},
                                          size=1,
                                          episode_len=episode_len,
                                          model=scorer)
    score_memory.load(args.memory_path, verbose=args.verbose)

    scores = score_memory.scores.reshape(-1)
    labels = score_memory.get_labels(influence_label).reshape(-1)

    metrics = get_metrics(scores, labels)

    decisions = scores > metrics['best_f1_thresh']

    if args.output_path:
        output_path = pathlib.Path(args.output_path)
    else:
        output_path = model_path / f'influence_{influence_label}_metrics.npy'

    print(f'{output_path.stem}.',
          f'AUC: {metrics["roc_auc"]:.4f}, AP: {metrics["ap"]:.4f}',
          f'th: {metrics["best_f1_thresh"]:.5f}\n',
          metrics_.classification_report(labels, decisions, digits=4))

    np.save(output_path,
            {**metrics, 'scores': scores},
            allow_pickle=True)


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])

    if args.convert_settings:
        utils.convert_train_settings(args.config, args.output_path)
    else:
        gin.parse_config_files_and_bindings([args.config], args.config_args)
        main(args)
