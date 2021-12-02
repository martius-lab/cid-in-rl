import gin
import numpy as np
import torch

from cid.influence_estimation import (model_trainers,
                                      schedules,
                                      transition_scorers)
from cid.memory import EpisodicReplayMemory


@gin.configurable(blacklist=['example'])
def make_scorer_and_model_trainer(example, model_classes, dataset_classes,
                                  scorer_cls,
                                  model_trainer_cls=model_trainers.ModelTrainer):
    dims = {name: _infer_dims_from_example(example, dataset_cls)
            for name, dataset_cls in dataset_classes.items()}

    models = {name: model_cls(inp_dim=_get_dim(dims[name][0]),
                              outp_dim=_get_dim(dims[name][1]))
              for name, model_cls in model_classes.items()}

    trainers = {name: model_trainer_cls(model, dataset_classes[name])
                for name, model in models.items()}

    trainer = model_trainers.OnlineModelTrainer(trainers)
    scorer = scorer_cls(**models)

    return scorer, trainer


@gin.configurable(blacklist=['example', 'replay_memory'])
def make_pretrained_scorer(example, model_classes, dataset_classes, load_paths,
                           scorer_cls,
                           replay_memory=None):
    if replay_memory is None:
        dims = {name: _infer_dims_from_example(example, dataset_cls)
                for name, dataset_cls in dataset_classes.items()}
    else:
        dims = {name: _infer_dims_from_memory(replay_memory, dataset_cls)
                for name, dataset_cls in dataset_classes.items()}

    models = {name: model_cls(inp_dim=_get_dim(dims[name][0]),
                              outp_dim=_get_dim(dims[name][1]))
              for name, model_cls in model_classes.items()}

    for name, path in load_paths.items():
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        for load_fn in (
                # Checkpoints from `ModelTrainer`
                lambda: models[name].load_state_dict(state_dict['model']),
                # Checkpoints directly from model
                lambda: models[name].load_state_dict(state_dict)):
            try:
                load_fn()
                break
            except Exception:
                pass
        else:
            raise ValueError(f'Unknown model checkpoint format `{path}`')

    scorer = scorer_cls(**models)

    return scorer, None


@gin.configurable(blacklist=['example', 'replay_memory'])
def make_model_trainer(example, model_cls, dataset_cls,
                       model_trainer_cls=model_trainers.ModelTrainer,
                       replay_memory=None):
    if replay_memory is None:
        dims = _infer_dims_from_example(example, dataset_cls)
    else:
        dims = _infer_dims_from_memory(replay_memory, dataset_cls)

    model = model_cls(inp_dim=_get_dim(dims[0]), outp_dim=_get_dim(dims[1]))

    return model_trainer_cls(model, dataset_cls)


def _get_dim(dims):
    if isinstance(dims, dict):
        return dims
    else:
        return dims[0]


def _infer_dims_from_example(example, dataset_cls):
    replay_memory = EpisodicReplayMemory(example, size=1, episode_len=1)
    replay_memory.store_episodes({key: np.repeat(value[None], 2, axis=0)[None]
                                 for key, value in example.items()})
    return _infer_dims_from_memory(replay_memory, dataset_cls)


def _infer_dims_from_memory(replay_memory, dataset_cls):
    keys = dataset_cls(memory=[]).required_keys
    transitions = replay_memory.to_transition_sequence(keys)
    dataset = dataset_cls(transitions)

    return dataset.shapes


__all__ = [make_scorer_and_model_trainer,
           make_pretrained_scorer,
           make_model_trainer,
           schedules,
           transition_scorers]
