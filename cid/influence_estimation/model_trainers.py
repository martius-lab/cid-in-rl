import os
import pickle
from collections import abc, defaultdict
from itertools import chain
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Union

import gin
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

from cid import models as models_
from cid.influence_estimation import losses as losses_
from cid.influence_estimation.datasets import (
    SeededRandomSampler, SeededWeightedRandomSampler)
from cid.memory import BaseReplayMemory


@gin.configurable
class ModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 dataset_cls: torch.utils.data.Dataset,
                 batch_size: int,
                 loss_fn: Callable[[Tensor, Tensor],
                                   Tensor] = losses_.mse_loss,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Dict[Any, Any] = None,
                 optimizer_param_groups: Dict[str, Callable[[torch.nn.Module],
                                              Iterable[torch.Tensor]]] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 l1_reg: float = 0.0,
                 reg_losses: Dict[str, float] = None,
                 outp_transform: Callable[[Any], Any] = None,
                 loss_scheduler_cls=None,
                 collate_fn=None,
                 eval_options: List[str] = None,
                 eval_bin_accuracy=False,
                 metric_fn=losses_.mse_loss,
                 val_dataset_cls: torch.utils.data.Dataset = None,
                 freeze_params: Dict[int, Callable[[torch.nn.Module],
                                     Iterable[torch.Tensor]]] = None,
                 unfreeze_params: Dict[int, Callable[[torch.nn.Module],
                                       Iterable[torch.Tensor]]] = None):
        self._model = model
        self._dataset_cls = dataset_cls
        self._val_dataset_cls = val_dataset_cls
        if loss_scheduler_cls is not None:
            self._loss_scheduler = loss_scheduler_cls(loss_fn)
            self._loss_fn = self._loss_scheduler
        else:
            self._loss_scheduler = None
            self._loss_fn = loss_fn
        self._l1_reg = l1_reg
        self._reg_losses = reg_losses
        self._batch_size = batch_size
        self._outp_transform = outp_transform
        self._collate_fn = collate_fn
        self._metric_fn = metric_fn

        if freeze_params is None:
            freeze_params = {}
        self._freeze_params_by_step = freeze_params
        assert isinstance(freeze_params, dict)
        for step, val in self._freeze_params_by_step.items():
            if callable(val):
                self._freeze_params_by_step[step] = [val]
        if unfreeze_params is None:
            unfreeze_params = {}
        self._unfreeze_params_by_step = unfreeze_params
        assert isinstance(unfreeze_params, dict)
        for step, val in self._unfreeze_params_by_step.items():
            if callable(val):
                self._unfreeze_params_by_step[step] = [val]

        if eval_options is None:
            eval_options = []
        self._eval_options = set(eval_options)
        for option in self._eval_options:
            assert option in ('bin_accuracy', 'pred_norm', 'variance',
                              'log_variance', 'std', 'log_likelihood',
                              'variance_mse')

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if optimizer_param_groups is not None:
            params = []
            for param_group in optimizer_param_groups:
                assert 'param_fn' in param_group, \
                    'Must specify `param_fn` when using param groups'
                param_group['params'] = param_group['param_fn'](self._model)
                del param_group['param_fn']
                params.append(param_group)
        else:
            params = model.parameters()
        self._optimizer = optimizer_cls(params, lr,
                                        weight_decay=weight_decay,
                                        **optimizer_kwargs)

        if hasattr(self._model, 'set_optimizer'):
            self._model.set_optimizer(self._optimizer)

        dummy_dataset = dataset_cls(memory=[])
        self._memory_keys = dummy_dataset.required_keys
        self._filter_fn = dummy_dataset.sequence_filter
        if val_dataset_cls is not None:
            dummy_dataset = val_dataset_cls(memory=[])
            self._val_memory_keys = dummy_dataset.required_keys
            self._val_filter_fn = dummy_dataset.sequence_filter

        self._n_train_steps = 0

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def batch_size(self):
        return self._batch_size

    def create_dataloader(self, replay_memory: BaseReplayMemory,
                          n_samples=None,
                          p_latest_episodes=None,
                          n_latest_episodes=None,
                          seed=None, n_workers=0,
                          shuffle=True, weighted=False, bootstrap=False,
                          split_ratios=None, split_idx=0,
                          force_to_dataset_size=False,
                          val_dataset=False) \
            -> torch.utils.data.DataLoader:
        if p_latest_episodes is None:
            assert n_latest_episodes is None, \
                ('`p_latest_episodes` and `n_latest_episodes` must both be '
                 '`None` or not `None`')

        if val_dataset and self._val_dataset_cls is not None:
            keys = self._val_memory_keys
            filter_fn = self._val_filter_fn
            T_sequence = replay_memory.to_transition_sequence(keys,
                                                              filter_fn,
                                                              split_ratios,
                                                              split_idx)
            dataset = self._val_dataset_cls(T_sequence)
        else:
            keys = self._memory_keys
            T_sequence = replay_memory.to_transition_sequence(keys,
                                                              self._filter_fn,
                                                              split_ratios,
                                                              split_idx)
            dataset = self._dataset_cls(T_sequence)

        if not shuffle:
            sampler = None
        if seed is None:
            if weighted:
                sampler = WeightedRandomSampler(np.ones((len(dataset),)),
                                                len(dataset),
                                                replacement=True)
            elif bootstrap:
                n_subset = n_samples if n_samples is not None else len(dataset)
                subset = np.random.choice(len(dataset), n_subset, replace=True)
                sampler = SubsetRandomSampler(subset)
            else:
                sampler = None
        elif n_samples is None:
            sampler = SeededRandomSampler(dataset, seed)
        else:
            n_dataset = len(dataset)
            if force_to_dataset_size:
                n_samples = min(n_samples, n_dataset)

            if p_latest_episodes is None and n_latest_episodes is None:
                n_latest_episodes = T_sequence.n_episodes
                p_latest_episodes = 1

            idx = T_sequence.get_start_index_of_n_latest_episodes(
                n_latest_episodes)  # Looks ugly, is actually correct by PEP8!
            n_latest_samples = n_dataset - idx

            weights = np.empty((n_dataset,))
            if n_latest_samples != n_dataset:
                weights[:idx] = ((1 - p_latest_episodes)
                                 / (n_dataset - n_latest_samples))
            weights[idx:] = p_latest_episodes / n_latest_samples
            sampler = SeededWeightedRandomSampler(weights,
                                                  int(n_samples),
                                                  replacement=True,
                                                  seed=seed)

        if shuffle:
            shuffle = sampler is None

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 self._batch_size,
                                                 sampler=sampler,
                                                 shuffle=shuffle,
                                                 num_workers=n_workers,
                                                 collate_fn=self._collate_fn)

        return dataloader

    def train_epoch(self, dataloader: torch.utils.data.DataLoader,
                    return_metrics=False,
                    progress=None):
        if not self._model.training:
            self._model.train()

        if self._loss_scheduler is not None:
            self._loss_scheduler.train()

        losses = []
        if return_metrics:
            metrics = defaultdict(list)

        if progress is not None:
            data = tqdm.tqdm(dataloader, desc=progress)
        else:
            data = dataloader

        for inp, target in data:
            if self._n_train_steps in self._freeze_params_by_step:
                param_fns = self._freeze_params_by_step[self._n_train_steps]
                _set_req_grad_from_param_fns(self._model, param_fns,
                                             requires_grad=False)
            if self._n_train_steps in self._unfreeze_params_by_step:
                param_fns = self._unfreeze_params_by_step[self._n_train_steps]
                _set_req_grad_from_param_fns(self._model, param_fns,
                                             requires_grad=True)

            inp = _to_float(inp)
            target = _to_float(target)

            pred = self._model(inp)
            loss = self._loss_fn(pred, target).mean()

            total_loss = loss

            l1_loss = None
            if self._l1_reg > 0:
                l1_loss = losses_.compute_l1_penalty(self._model)
                total_loss += self._l1_reg * l1_loss
            module_losses = None
            if self._reg_losses is not None:
                module_losses = losses_.collect_module_losses(self._model)
                for name, weight in self._reg_losses.items():
                    total_loss += weight * module_losses[name]

            # Set the gradients to `None`. We do this instead of calling
            # `optimizer.zero_grad()`, as optimizers with memory like Adam will
            # keep updating weights with zero grad, even if they did not
            # receive a gradient. This interferes with parameter freezing.
            for group in self._optimizer.param_groups:
                for p in group['params']:
                    p.grad = None

            total_loss.backward()
            self._optimizer.step()

            if self._loss_scheduler is not None:
                self._loss_scheduler.step()

            losses.append(loss.detach().cpu().numpy())
            if return_metrics:
                self._update_metrics(metrics, pred, target,
                                     l1_loss, module_losses)
            self._n_train_steps += 1

        if return_metrics:
            metrics = {key: np.concatenate(m) for key, m in metrics.items()}
            return losses, metrics
        else:
            return losses

    def validate(self, dataloader: torch.utils.data.DataLoader,
                 return_data=False,
                 return_metrics=False):
        if self._model.training:
            self._model.eval()
        if self._loss_scheduler is not None:
            self._loss_scheduler.eval()

        losses = []
        if return_data:
            predictions = []
            targets = []
        if return_metrics:
            metrics = defaultdict(list)

        for inp, target in dataloader:
            inp = _to_float(inp)
            target = _to_float(target)

            with torch.no_grad():
                pred = self._model(inp)
                loss = self._loss_fn(pred, target)

            losses.append(loss.cpu().numpy())

            if return_data:
                if self._outp_transform is not None:
                    pred_transformed = self._outp_transform(pred)
                predictions.append(_to_numpy(pred_transformed))
                targets.append(_to_numpy(target))

            if return_metrics:
                self._update_metrics(metrics, pred, target, training=False)

        res = [np.concatenate(losses)]

        if return_data:
            res.append(_concat(predictions))
            res.append(_concat(targets))
        if return_metrics:
            res.append({key: np.concatenate(m) for key, m in metrics.items()})

        return res

    @torch.no_grad()
    def _update_metrics(self,
                        metrics: DefaultDict[str, List[np.ndarray]],
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        l1_loss: torch.Tensor = None,
                        module_losses: Dict[str, torch.Tensor] = None,
                        training=True):
        if self._outp_transform is not None:
            pred = self._outp_transform(pred)

        metric_fn = self._metric_fn
        if isinstance(pred, dict):
            metric_fn = losses_.factorized_mse_loss
            val = next(iter(pred.values()))
            if isinstance(val, (tuple, list)):
                pred = {name: p[0] for name, p in pred.items()}
        elif isinstance(pred, (tuple, list)):
            if len(pred) > 1:
                if 'variance' in self._eval_options:
                    var = pred[1].detach().cpu().numpy()
                    metrics['var'].append(var)
                elif 'log_variance' in self._eval_options:
                    logvar = pred[1].detach().cpu().numpy()
                    metrics['var'].append(np.exp(logvar))
                elif 'std' in self._eval_options:
                    std = pred[1].detach().cpu().numpy()
                    metrics['var'].append(std**2)
                if 'log_likelihood' in self._eval_options:
                    ll = losses_.gaussian_log_likelihood_loss(pred, target)
                    metrics['log_likelihood'].append(ll.cpu().numpy())
            pred = pred[0]  # Hacky; assume 1st entry is mean pred

        mse = metric_fn(pred, target).cpu().numpy()
        metrics['mse'].append(mse)

        if 'var' in metrics and 'variance_mse' in self._eval_options:
            mse_ = ((pred - target)**2).cpu().numpy()
            var_mse = np.mean((metrics['var'][-1] - mse_)**2, axis=-1)
            metrics['var_mse'].append(var_mse)

        if 'bin_accuracy' in self._eval_options:
            bin_acc = (torch.sigmoid(pred) > 0.5) == target
            metrics['bin_acc'].append(bin_acc.cpu().numpy())

        if 'pred_norm' in self._eval_options:
            pred_norm = torch.norm(pred, p=2, dim=-1).cpu().numpy()
            metrics['pred_norm'].append(pred_norm)

        if self._l1_reg > 0 and l1_loss is not None:
            l1_loss = np.atleast_1d(l1_loss.detach().cpu().numpy())
            metrics['l1'].append(l1_loss)

        if self._reg_losses is not None and module_losses is not None:
            for name in self._reg_losses:
                module_loss = module_losses[name].detach().cpu()
                module_loss = np.atleast_1d(module_loss.numpy())
                metrics[f'reg_{name}'].append(module_loss)

    def save(self, path: str) -> str:
        if not path.endswith('.pth'):
            path += '.pth'

        state = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }
        torch.save(state, path)

        return path


@gin.configurable
class EnsembleTrainer:
    def __init__(self, model: models_.Ensemble, *args, **kwargs):
        assert isinstance(model, models_.Ensemble)
        self._model = model
        self._trainers = [ModelTrainer(sub_model, *args, **kwargs)
                          for sub_model in model.models]

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def batch_size(self):
        return self._trainers[0].batch_size

    def create_dataloader(self, replay_memory: BaseReplayMemory,
                          n_samples=None,
                          p_latest_episodes=None,
                          n_latest_episodes=None,
                          seed=None, n_workers=0,
                          shuffle=True, weighted=False, bootstrap=False,
                          split_ratios=None, split_idx=0,
                          force_to_dataset_size=True) \
            -> List[torch.utils.data.DataLoader]:
        seeds = [seed + idx if seed is not None else None
                 for idx in range(len(self._trainers))]

        loaders = [trainer.create_dataloader(replay_memory,
                                             n_samples,
                                             p_latest_episodes,
                                             n_latest_episodes,
                                             seed,
                                             n_workers,
                                             shuffle,
                                             weighted,
                                             bootstrap,
                                             split_ratios,
                                             split_idx,
                                             force_to_dataset_size)
                   for seed, trainer in zip(seeds, self._trainers)]

        return loaders

    def train_epoch(self, dataloader: Iterable[torch.utils.data.DataLoader],
                    return_metrics=False,
                    progress=None):
        assert isinstance(dataloader, abc.Iterable)

        losses_per_model = []
        if return_metrics:
            metrics_per_model = []

        for loader, trainer in zip(dataloader, self._trainers):
            if return_metrics:
                losses, metrics = trainer.train_epoch(loader,
                                                      return_metrics,
                                                      progress)
                metrics_per_model.append(metrics)
            else:
                losses = trainer.train_epoch(loader, return_metrics)

            losses_per_model.append(losses)

        # Careful, losses and metrics returned are not aligned with each other
        # as each dataloader shuffles differently
        losses = np.stack(losses_per_model, axis=0)

        if return_metrics:
            metrics = {key: np.stack([metrics[key]
                                     for metrics in metrics_per_model])
                       for key in metrics_per_model[0]}

            return losses, metrics
        else:
            return losses

    def validate(self, dataloader: torch.utils.data.DataLoader,
                 return_data=False,
                 return_metrics=False):
        losses_per_model = []
        if return_data:
            predictions_per_model = []
            targets_per_model = []
        if return_metrics:
            metrics_per_model = []

        for loader, trainer in zip(dataloader, self._trainers):
            res = trainer.validate(loader, return_data, return_metrics)

            losses_per_model.append(res[0])
            if return_data:
                predictions_per_model.append(res[1])
                targets_per_model.append(res[2])
                if return_metrics:
                    metrics_per_model.append(res[3])
            elif return_metrics:
                metrics_per_model.append(res[1])

        losses = np.stack(losses_per_model, axis=0)

        res = [losses]

        if return_data:
            assert not isinstance(predictions_per_model[0], dict), \
                'Dict prediction case not yet implemented here'
            if isinstance(predictions_per_model[0], (list, tuple)):
                predictions = [np.stack([p[i] for p in predictions_per_model])
                               for i in range(len(predictions_per_model[0]))]
            else:
                predictions = np.stack(predictions_per_model, axis=0)

            res.append(predictions)
            res.append(np.stack(targets_per_model, axis=0))
        if return_metrics:
            res.append(np.stack(metrics_per_model, axis=0))

        return res

    def save(self, path: str) -> str:
        if not path.endswith('.pth'):
            path += '.pth'

        state = {
            'model': self._model.state_dict(),
            'optimizers': [trainer._optimizer.state_dict()
                           for trainer in self._trainers]
        }
        torch.save(state, path)

        return path


@gin.configurable(blacklist=['trainers'])
class OnlineModelTrainer:
    def __init__(self, trainers: Dict[str, ModelTrainer],
                 warmup_epochs: Union[int, Dict[str, int]],
                 epochs_per_step: Union[int, Dict[str, int]],
                 batch_mode: bool = False,
                 samples_per_epoch: int = None,
                 p_latest_episodes: float = None,
                 n_latest_episodes: int = None,
                 n_warmup_steps: int = 1,
                 n_dataloader_workers: int = 0,
                 rescore_batch_size: int = 100,
                 rescore_n_latest_episodes: int = None,
                 full_rescore_every: int = 1):
        self._trainers = trainers
        self._warmup_epochs = warmup_epochs
        self._epochs_per_step = epochs_per_step
        self._samples_per_epoch = samples_per_epoch
        self._p_latest_episodes = p_latest_episodes
        self._n_latest_episodes = n_latest_episodes
        self._n_warmup_steps = n_warmup_steps
        self._n_workers = n_dataloader_workers
        self._rescore_batch_size = rescore_batch_size
        self._rescore_n_latest_episodes = rescore_n_latest_episodes
        self._full_rescore_every = full_rescore_every
        self._batch_mode = batch_mode

        self._train_steps = 0

    def train(self, replay_memory: BaseReplayMemory,
              rescore_transitions=True,
              eval_on_memory=False,
              batch_mode=None,
              epochs_to_train=None,
              samples_per_epoch=None,
              p_latest_episodes=None,
              n_latest_episodes=None):
        warmup = self._train_steps < self._n_warmup_steps

        if batch_mode is None:
            batch_mode = self._batch_mode
        if epochs_to_train is None:
            if warmup:
                epochs_to_train = self._warmup_epochs
            else:
                epochs_to_train = self._epochs_per_step
        if samples_per_epoch is None:
            if warmup:
                samples_per_epoch = np.inf  # Gets clamped to dataset size
            else:
                samples_per_epoch = self._samples_per_epoch
        if p_latest_episodes is None:
            p_latest_episodes = None if warmup else self._p_latest_episodes
        if n_latest_episodes is None:
            n_latest_episodes = None if warmup else self._n_latest_episodes

        if batch_mode:
            # TODO: make this work for dict
            batch_size = next(iter(self._trainers.values())).batch_size
            samples_per_epoch = epochs_to_train * batch_size
            epochs_to_train = 1

        seed = torch.random.initial_seed() + self._train_steps
        f = not batch_mode

        dataloaders = {name: trainer.create_dataloader(replay_memory,
                                                       samples_per_epoch,
                                                       p_latest_episodes,
                                                       n_latest_episodes,
                                                       seed,
                                                       self._n_workers,
                                                       force_to_dataset_size=f)
                       for name, trainer in self._trainers.items()}

        stats = {}
        for name, trainer in self._trainers.items():
            epochs = epochs_to_train
            if isinstance(epochs_to_train, dict):
                epochs = epochs_to_train[name]

            if epochs > 1:
                progress = None
                epoch_it = tqdm.trange(epochs, desc=f'Training model `{name}`')
            else:
                progress = f'Training model `{name}`'
                epoch_it = range(epochs)

            for epoch in epoch_it:
                if epoch + 1 == epochs and not eval_on_memory:
                    losses, metrics = trainer.train_epoch(dataloaders[name],
                                                          return_metrics=True,
                                                          progress=progress)
                    stats[f'Model/{name}_Loss'] = np.mean(losses)
                    stats[f'Model/{name}_MSE'] = np.mean(metrics['mse'])
                    for key, metric in metrics.items():
                        if key.startswith('reg_'):
                            stats[f'Model/{name}_{key}'] = np.mean(metric)
                else:
                    trainer.train_epoch(dataloaders[name],
                                        progress=progress)

        self._train_steps += 1

        if eval_on_memory:
            for name, trainer in self._trainers.items():
                dataset = dataloaders[name].dataset
                n_workers = self._n_workers
                loader = torch.utils.data.DataLoader(dataset,
                                                     trainer.batch_size,
                                                     shuffle=False,
                                                     num_workers=n_workers)
                losses, metrics = trainer.validate(loader, return_metrics=True)
                stats[f'Model/{name}_Loss'] = np.mean(losses)
                stats[f'Model/{name}_MSE'] = np.mean(metrics['mse'])

        if rescore_transitions:
            if (self._train_steps - 1) % self._full_rescore_every == 0:
                episodes = None
            else:
                episodes = self._rescore_n_latest_episodes
            batch_size = self._rescore_batch_size
            replay_memory.rescore_transitions(batch_size=batch_size,
                                              n_latest_episodes=episodes,
                                              verbose=True)

        return stats

    def save(self, dir_path: str, step: int) -> str:
        paths = {}
        for name, trainer in self._trainers.items():
            path = os.path.join(dir_path, f'trainer_{name}_{step}.pth')
            paths[name] = trainer.save(path)

        state = {
            'train_steps': self._train_steps,
            'trainer_paths': paths
        }

        path = os.path.join(dir_path, f'trainer_{step + 1}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(state, f)

        return path


def _to_float(data):
    if isinstance(data, dict):
        data = {k: v.float() for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        data = [v.float() for v in data]
    else:
        data = data.float()

    return data


def _to_numpy(data):
    if isinstance(data, dict):
        elem = next(iter(data.values()))
        if isinstance(elem, (list, tuple)):
            data = {k: [q.detach().cpu().numpy() for q in v]
                    for k, v in data.items()}
        else:
            data = {k: v.detach().cpu().numpy() for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        data = [v.detach().cpu().numpy() for v in data]
    else:
        data = data.detach().cpu().numpy()

    return data


def _concat(data: list, axis=0):
    elem = data[0]

    if isinstance(elem, dict):
        inner_elem = next(iter(elem.values()))
        if isinstance(inner_elem, (list, tuple)):
            data = {k: [np.concatenate([v[k][idx] for v in data], axis=axis)
                        for idx in range(len(inner_elem))]
                    for k in elem}
        else:
            data = {k: np.concatenate([d[k] for d in data], axis=axis)
                    for k in elem}
    elif isinstance(elem, (list, tuple)):
        data = [np.concatenate([v[idx] for v in data], axis=axis)
                for idx in range(len(elem))]
    else:
        data = np.concatenate(data, axis=axis)

    return data


def _set_req_grad_from_param_fns(model, param_fns, requires_grad):
    for p in chain.from_iterable((fn(model) for fn in param_fns)):
        p.requires_grad = requires_grad
