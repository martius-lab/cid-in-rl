import abc
from typing import Any, Dict, Tuple


class BaseRolloutGenerator(abc.ABC):
    """Base class for rollout generators

    A rollout generator lets an agent act in an environment and returns the
    observed transitions. Different rollout generators can differ in the amount
    of information they record from the transitions.
    """
    @abc.abstractmethod
    def rollout(self,
                agent,
                n_steps: int,
                evaluate: bool = False,
                render: bool = False) -> Tuple[Dict[str, 'numpy.ndarray'],
                                               Dict[str, Any]]:
        """Perform rollout(s) in the environment and return transitions

        The generator may perform multiple rollouts and return a batch of
        transitions from all those rollouts.

        :param agent: Agent performing the rollout
        :param n_steps: Number of steps to perform, i.e. the number of actions
            that are taken
        :param evaluate: If `True`, run the agent in evaluation mode
        :param render: If `True`, render the environment before each step

        :return: Dictionary containing observed transitions. Shape of arrays
            is `(M, n_steps + 1, ...)`, where M is the number of rollouts
            performed.
        """
        ...

    @abc.abstractmethod
    def reset(self):
        """Reset underlying environment"""
        ...

    @abc.abstractmethod
    def close(self):
        """Clean-up this generator

        Call if generator is no longer used.
        """
        ...

    @property
    @abc.abstractmethod
    def example_transition(self) -> Dict[str, 'numpy.ndarray']:
        """Return a single (invalid) transition

        Useful to detect the shapes and data types of transition items this
        generator returns.

        :return: Dictionary containing a single transition. Shape of arrays is
            exactly the dimensionality of the transition items.
        """
        ...

    @property
    @abc.abstractmethod
    def transition_help(self) -> Dict[str, str]:
        """Explanation strings of items this generator returns"""
        ...

    @property
    @abc.abstractmethod
    def observation_space(self) -> 'gym.Space':
        """Observation space of environment"""
        ...

    @property
    @abc.abstractmethod
    def action_space(self) -> 'gym.Space':
        """Action space of environment"""
        ...
