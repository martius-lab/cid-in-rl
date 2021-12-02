import gin

from cid.algorithms.base import BaseAgent


@gin.configurable(blacklist=['state_dim', 'action_space'])
class RandomAgent(BaseAgent):
    """Agent performing only random exploration"""
    def __init__(self, state_dim, action_space):
        super().__init__(state_dim, action_space)

    def get_action(self, state, evaluate=False):
        return self.action_space.sample(), True

    def update_step(self, batch):
        return {}

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def save(self, path: str) -> str:
        """Save model parameters and return file path"""
        if not path.endswith('.pth'):
            path += '.pth'
        with open(path, 'w'):
            pass

        return path

    def load(self, path):
        """Load model parameters"""
        pass
