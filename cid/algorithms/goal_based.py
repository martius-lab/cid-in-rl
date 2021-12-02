import numpy as np
from typing import Any, Dict, Iterable

from cid.algorithms import BaseAgent
from cid.memory import BaseReplayMemory


def make_goal_based_agent(observation_space, action_space, agent_cls):
    state_dim = (observation_space['observation'].shape[0] +
                 observation_space['desired_goal'].shape[0])

    return GoalBasedAgent(agent_cls(state_dim, action_space))


class GoalBasedAgent(BaseAgent):
    """Class wrapping an agent to provide goal-conditional input"""
    def __init__(self, agent):
        super().__init__(agent.state_dim, agent.action_space)
        self.agent = agent
        self.goal_based_memory = GoalBasedMemory()

    def get_action(self, state, evaluate=False):
        """Get action for state

        :param state: Tuple of `(observation, goal)`
        :param evaluate: If `True`, run in evaluation mode
        """
        return self.agent.get_action(np.concatenate(state, axis=0), evaluate)

    def update_parameters(self,
                          replay_memory: BaseReplayMemory,
                          batch_size: int,
                          n_updates: int) -> Dict[str, Any]:
        self.goal_based_memory.memory = replay_memory
        return self.agent.update_parameters(self.goal_based_memory,
                                            batch_size,
                                            n_updates)

    def update_step(self, batch):
        return self.agent.update_step(batch)

    def get_state(self):
        return self.agent.get_state()

    def set_state(self, state):
        return self.agent.set_state(state)

    @property
    def unwrapped(self):
        return self.agent

    def save(self, path):
        """Save model parameters"""
        return self.agent.save(path)

    def load(self, path):
        """Load model parameters"""
        return self.agent.load(path)


class GoalBasedMemory(BaseReplayMemory):
    """Class wrapping a ReplayMemory to provide goal-conditional batches"""
    __slots__ = ('memory',)

    def __init__(self, memory: BaseReplayMemory = None):
        self.memory = memory

    def current_size(self):
        return self.memory.current_size

    def store_episodes(self, episodes):
        return self.memory.store_episodes(episodes)

    def sample_batch(self, batch_size):
        batch = self.memory.sample_batch(batch_size)
        batch['s0'] = np.concatenate((batch['s0'], batch['g']), axis=1)
        batch['s1'] = np.concatenate((batch['s1'], batch['g']), axis=1)
        return batch

    def update_td_errors(self, batch, td_errors):
        return self.memory.update_td_errors(batch, td_errors)

    def report_statistics(self):
        return self.memory.report_statistics()

    def to_transition_sequence(self,
                               keys: Iterable[str] = None,
                               filter_fn=None):
        return self.memory.to_transition_sequence(keys, filter_fn)

    def __iter__(self):
        return self.memory.__iter__()

    def save(self, path):
        return self.memory.save(path)

    def load(self, path):
        return self.memory.load(path)
