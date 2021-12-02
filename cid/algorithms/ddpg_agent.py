import copy
import enum
import itertools
from typing import Any, Dict, Iterable, Tuple, TYPE_CHECKING

import gin
import numpy as np
import torch

from cid import utils
from cid.algorithms.base import BaseAgent
from cid.algorithms.normalizer import Normalizer
from cid.memory import BaseReplayMemory

if TYPE_CHECKING:
    import gym


@gin.configurable(blacklist=['state_dim', 'action_space'])
class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient (DDPG)

    Implementation adapted from OpenAI SpinningUp (MIT License):
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py
    """
    def __init__(self, state_dim: int,
                 action_space: 'gym.Space',
                 pi_cls: type, q_cls: type,
                 gamma: float = 0.99,
                 polyak: float = 0.995,
                 pi_lr: float = 1e-3,
                 q_lr: float = 1e-3,
                 action_l2_penalty: float = 0.0,
                 action_noise: float = 0.1,
                 random_eps: float = 0.0,
                 zero_action_eps: float = 0.0,
                 observation_clip: Tuple[float, float] = None,
                 q_target_clip: Tuple[float, float] = None,
                 normalize_state: bool = False,
                 normalizer_warmup_updates: int = 0,
                 prioritized_experience_replay: bool = False):
        """
        :param state_dim: Dimensionality of state
        :param action_space: Action space
        :param pi_cls: Constructor for policy network
        :param q_cls: Constructor for Q-network
        :param gamma: Discount factor in range `(0, 1]`
        :param polyak: Interpolation factor in polyak averaging for target
            networks in range `[0, 1)`
        :param pi_lr: Learning rate for policy network
        :param q_lr: Learning rate for Q-network
        :param action_l2_penalty: Weight of L2 norm of actions penalty term
            for policy loss
        :param action_noise: Stddev for Gaussian exploration noise added to
            policy at training time
        :param random_eps: Probability to perform a random action for
            exploration at training time
        :param zero_action_eps: Probability to perform a zero action at
            training time. Counts as a random action and thus reduces the
            probability to perform a uniformly random action
        :param observation_clip: Tuple with lower and upper bound to clamp
            observations to. Can be `None` to not clamp observations
        :param q_target_clip: Tuple with lower and upper bound to clamp the
            TD target in the Q function loss. Can be `None` to not clamp TD
            target
        :param normalize_state: If `True`, preprocess states by transforming
            them to be approximately standard normal
        :param normalizer_warmup_updates: Number of batches that should
            initially be used to warmup the normalizer
        :param prioritized_experience_replay: If `True`, report TD errors
            to memory during updates for prioritized experience replay
        """
        super().__init__(state_dim, action_space)

        assert 0 < gamma <= 1
        assert 0 <= polyak < 1
        assert 0 <= random_eps <= 1
        assert action_space.high[0] != np.inf, \
            'Action space is unbounded, which is currently unsupported'
        assert observation_clip is None or isinstance(observation_clip, tuple)
        assert q_target_clip is None or isinstance(q_target_clip, tuple)
        if not normalize_state:
            assert normalizer_warmup_updates == 0

        self.gamma = gamma
        self.polyak = polyak
        self.action_l2_penalty = action_l2_penalty
        self.action_noise = action_noise
        self.random_eps = random_eps
        self.zero_action_eps = zero_action_eps
        self.observation_clip = observation_clip
        self.q_target_clip = q_target_clip
        self.prioritized_replay = prioritized_experience_replay

        self.normalizer = None
        if normalize_state:
            self.normalizer = Normalizer()
        self.normalizer_warmup_updates = normalizer_warmup_updates

        # Lower and upper limit for actions. Assumes all dimensions share the
        # same bound and that the action space is symmetrical
        self.max_activation = action_space.high[0]

        act_dim = action_space.shape[0]
        self.pi = pi_cls(inp_dim=state_dim,
                         outp_dim=act_dim,
                         outp_activation=torch.nn.Tanh,
                         outp_scaling=self.max_activation)
        self.q = q_cls(inp_dim=state_dim + act_dim, outp_dim=1)

        self.pi_target = copy.deepcopy(self.pi)
        self.q_target = copy.deepcopy(self.q)

        # Freeze target networks with respect to optimizers (only update via
        # Polyak averaging)
        for p in itertools.chain(self.pi_target.parameters(),
                                 self.q_target.parameters()):
            p.requires_grad = False

        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=q_lr)

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        if self.normalizer is not None:
            state = self.normalizer(state)

        if self.observation_clip is not None:
            state = np.clip(state,
                            self.observation_clip[0],
                            self.observation_clip[1])

        return state.astype(np.float32)

    def get_action(self, state: np.ndarray, evaluate: bool = False):
        if not evaluate:
            rand = np.random.rand()
            if rand < self.zero_action_eps:
                return np.zeros(self.action_space.shape,
                                self.action_space.dtype), True
            elif rand < self.random_eps:
                return self.action_space.sample(), True

        state = self._preprocess_state(state)

        with torch.no_grad():
            state = torch.as_tensor(state)

            a = self.pi(state)
            if not evaluate:
                a += self.action_noise * torch.randn_like(a)

            a.clamp_(-self.max_activation, self.max_activation)

        return a.numpy().squeeze(axis=0), False

    def update_parameters(self,
                          replay_memory: BaseReplayMemory,
                          batch_size: int,
                          n_updates: int) -> Dict[str, Any]:
        if self.normalizer is not None and self.normalizer.num_updates == 0:
            for _ in range(self.normalizer_warmup_updates):
                batch = replay_memory.sample_batch(batch_size)
                self.normalizer.update(batch['s0'])

        total_stats = {}
        for _ in range(n_updates):
            batch = replay_memory.sample_batch(batch_size)

            if self.normalizer is not None:
                self.normalizer.update(batch['s0'])

            if self.prioritized_replay:
                stats, td_error = self.update_step(batch, return_td_error=True)
                replay_memory.update_td_errors(batch, td_error)
            else:
                stats = self.update_step(batch)

            if stats is not None:
                utils.update_dict_of_lists(total_stats, stats)

        return total_stats

    def _compute_loss_q(self,
                        state: torch.Tensor,
                        state_next: torch.Tensor,
                        action: torch.Tensor,
                        reward: torch.Tensor,
                        done: torch.Tensor,
                        loss_weights: torch.Tensor = None):
        q = self.q(torch.cat((state, action), dim=1)).squeeze(axis=-1)

        # Bellman backup for Q function
        with torch.no_grad():
            q_inp = torch.cat((state_next, self.pi_target(state_next)), dim=1)
            q_pi_target = self.q_target(q_inp).squeeze(axis=-1)
            backup = reward + self.gamma * (1 - done) * q_pi_target
            if self.q_target_clip:
                backup.clamp_(self.q_target_clip[0], self.q_target_clip[1])

        # MSE loss against Bellman backup
        td_error = q - backup
        assert td_error.ndim == 1  # Protect against broadcasting bugs

        if loss_weights is None:
            loss_q = (td_error**2).mean()
        else:
            loss_q = (loss_weights * td_error**2).mean()

        stats = {
            'DDPG/Q': q.detach().numpy(),
            'DDPG/Q_Target': q_pi_target.numpy(),
            'DDPG/Q_Loss': loss_q.detach().numpy()
        }

        return loss_q, td_error, stats

    def _compute_loss_pi(self, state: torch.Tensor):
        a = self.pi(state)
        q_pi = self.q(torch.cat((state, a), dim=1)).squeeze(axis=-1)
        pi_loss = -q_pi.mean()

        if self.action_l2_penalty > 0:
            pi_loss += self.action_l2_penalty * torch.mean(a**2)

        return pi_loss

    def update_step(self, batch, return_td_error=False):
        stats = {}

        state = torch.from_numpy(self._preprocess_state(batch['s0']))
        state_next = torch.from_numpy(self._preprocess_state(batch['s1']))
        action = torch.from_numpy(batch['a'])
        reward = torch.from_numpy(batch['r'])
        done = torch.from_numpy(batch['d'])

        if reward.ndim == 2:
            importance_weights = reward[1]
            reward = reward[0]
            stats['DDPG/Reward'] = batch['r'][0]
            stats['DDPG/IS'] = batch['r'][1]
        else:
            importance_weights = None
            stats['DDPG/Reward'] = batch['r']

        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, td_error, stats_q = self._compute_loss_q(state, state_next,
                                                         action, reward, done,
                                                         importance_weights)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so we don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self._compute_loss_pi(state)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so we can optimize it at next DDPG step.
        for p in self.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by Polyak averaging.
        polyak_averaging(self.pi.parameters(),
                         self.pi_target.parameters(),
                         self.polyak)
        polyak_averaging(self.q.parameters(),
                         self.q_target.parameters(),
                         self.polyak)

        stats.update(stats_q)
        stats['DDPG/Pi_Loss'] = loss_pi.detach().numpy()

        if return_td_error:
            return stats, td_error.detach().numpy()
        else:
            return stats

    def get_state(self):
        state = {
            'pi': self.pi.state_dict(),
            'q': self.q.state_dict(),
            'pi_target': self.pi_target.state_dict(),
            'q_target': self.q_target.state_dict()
        }

        if self.normalizer is not None:
            state['normalizer'] = self.normalizer.state_dict()

        return state

    def set_state(self, state):
        self.pi.load_state_dict(state['pi'])
        self.q.load_state_dict(state['q'])
        self.pi_target.load_state_dict(state['pi_target'])
        self.q_target.load_state_dict(state['q_target'])
        if 'normalizer' in state:
            self.normalizer.load_state_dict(state['normalizer'])

    def save(self, path: str) -> str:
        """Save model parameters"""
        if not path.endswith('.pth'):
            path += '.pth'
        torch.save(self.get_state(), path)

        return path

    def load(self, path):
        """Load model parameters"""
        state = torch.load(path)
        self.set_state(state)


@torch.no_grad()
def polyak_averaging(params: Iterable, target_params: Iterable, scale: float):
    for p, p_targ in zip(params, target_params):
        p_targ.data.mul_(scale)
        p_targ.data.add_((1 - scale) * p.data)


class _SelectionType(enum.Enum):
    MAX = 0
    NORMALIZED = 1
    RANK = 2


@gin.configurable(blacklist=['state_dim', 'action_space'])
class ActiveExplorationDDPGAgent(DDPGAgent):
    """Adapted version of DDPGAgent for using a model to explore better"""
    def __init__(self, state_dim: int,
                 action_space: 'gym.Space',
                 pi_cls: type, q_cls: type,
                 gamma: float = 0.99,
                 polyak: float = 0.995,
                 pi_lr: float = 1e-3,
                 q_lr: float = 1e-3,
                 action_l2_penalty: float = 0.0,
                 action_noise: float = 0.1,
                 random_eps: float = 0.0,
                 active_eps: float= 0.0,
                 zero_action_eps: float = 0.0,
                 observation_clip: Tuple[float, float] = None,
                 q_target_clip: Tuple[float, float] = None,
                 normalize_state: bool = False,
                 normalizer_warmup_updates: int = 0,
                 n_candidate_actions: int = 32,
                 selection_type: _SelectionType = "max",
                 strip_last_state_dims: int = 0):
        self._active_eps = active_eps
        self._strip_last_state_dims = strip_last_state_dims
        self._n_candidate_actions = n_candidate_actions
        self._model = None

        if selection_type == 'max':
            self._selection_type = _SelectionType.MAX
        elif selection_type == 'normalized_sampling':
            self._selection_type = _SelectionType.NORMALIZED
        elif selection_type == 'ranked_sampling':
            self._selection_type = _SelectionType.RANK
        else:
            raise ValueError(f'Unknown selection type `{selection_type}`')

        super().__init__(state_dim, action_space, pi_cls, q_cls,
                         gamma, polyak, pi_lr, q_lr,
                         action_l2_penalty, action_noise, random_eps,
                         zero_action_eps, observation_clip, q_target_clip,
                         normalize_state, normalizer_warmup_updates)

    def set_model(self, model):
        if not hasattr(model, 'action_scores'):
            raise ValueError('Model is not compatible with active exploration')
        self._model = model

    def _select_action(self, state):
        if self._strip_last_state_dims > 0:
            state = state[:-self._strip_last_state_dims]

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        actions = np.random.rand(1, self._n_candidate_actions,
                                 self.action_space.shape[0]) * 2 - 1
        scores = self._model.action_scores(state.astype(np.float32),
                                           actions.astype(np.float32))[0]

        if self._selection_type == _SelectionType.NORMALIZED:
            scores = np.maximum(scores, 0)
            score_sum = np.sum(scores)
            if score_sum == 0:
                idx = np.random.choice(len(scores),
                                       p=np.ones_like(scores) / len(scores))
            else:
                idx = np.random.choice(len(scores), p=scores / score_sum)
        elif self._selection_type == _SelectionType.RANK:
            order = scores.argsort()[::-1]
            inverse_ranks = 1 / (order.argsort() + 1)
            p = inverse_ranks / np.sum(inverse_ranks)
            idx = np.random.choice(len(scores), p=p)
        else:
            idx = np.argmax(scores)

        return actions[0, idx]

    def get_action(self, state: np.ndarray, evaluate: bool = False):
        if not evaluate:
            rand = np.random.rand()
            if rand < self._active_eps:
                return self._select_action(state), True
            elif rand - self._active_eps < self.random_eps:
                return self.action_space.sample(), True

        state = self._preprocess_state(state)

        with torch.no_grad():
            state = torch.as_tensor(state)

            a = self.pi(state)
            if not evaluate:
                a += self.action_noise * torch.randn_like(a)

            a.clamp_(-self.max_activation, self.max_activation)

        return a.numpy().squeeze(axis=0), False
