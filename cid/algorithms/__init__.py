import gin

from cid.algorithms.base import BaseAgent
from cid.algorithms.ddpg_agent import (ActiveExplorationDDPGAgent,
                                       DDPGAgent)
from cid.algorithms.goal_based import make_goal_based_agent
from cid.algorithms.random_agent import RandomAgent


@gin.configurable(blacklist=['observation_space', 'action_space'])
def make_agent(observation_space, action_space, agent_cls,
               goal_based=False) -> BaseAgent:
    if goal_based:
        return make_goal_based_agent(observation_space,
                                     action_space,
                                     agent_cls)
    else:
        return agent_cls(observation_space.shape[0], action_space)


__all__ = [make_agent,
           ActiveExplorationDDPGAgent,
           BaseAgent,
           DDPGAgent,
           RandomAgent]
