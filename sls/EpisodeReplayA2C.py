import numpy as np
from typing import List


class SAGTriple:
    def __init__(
            self,
            current_state: np.ndarray,
            current_action: int,
            next_reward: float,
    ):
        self.current_state = current_state
        self.current_action = current_action
        self.next_reward = next_reward
        self.g = None


class EpisodeReplayA2C:
    def __init__(self, n_step):
        self.replay: List[SAGTriple] = []
        self.n_step = n_step

    def __len__(self):
        return len(self.replay)

    def reset(self):
        self.replay = []

    def append(self, transition: SAGTriple):
        self.replay.append(transition)

    def _calculate_g(self, gamma: float, last_v):
        g = 0
        for i in range(1, self.n_step):
            g += self.replay[-self.n_step + i].next_reward * gamma**i
        g += last_v * gamma**self.n_step

        result_element = self.replay[0]
        result_element.g = g
        # remove the first element
        self.replay = self.replay[1:]
        return result_element

    def get_batch(self, batch_size):
        if len(self.replay) < batch_size + self.n_step:
            return None
        batch = []
        for i in range(batch_size):
            batch.append(self._calculate_g)
        return batch

    def get_states(self):
        states = []
        for t in range(len(self.replay)):
            states.append(self.replay[t].current_state)
        return np.array(states)

    def get_actions(self):
        actions = []
        for t in range(len(self.replay)):
            actions.append(self.replay[t].current_action)
        return np.array(actions)
