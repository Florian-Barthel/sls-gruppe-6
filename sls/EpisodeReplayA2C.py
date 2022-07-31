import numpy as np
from typing import List


class SAGTriple:
    def __init__(
            self,
            current_state: np.ndarray,
            current_action: int,
            next_reward: float,
            value: float
    ):
        self.current_state = current_state
        self.current_action = current_action
        self.next_reward = next_reward
        self.value = value
        self.g = None


class EpisodeReplayA2C:
    def __init__(self, n_step, gamma):
        self.replay: List[SAGTriple] = []
        self.n_step = n_step
        self.gamma = gamma
        self.value = []

    def __len__(self):
        return len(self.replay)

    def reset(self):
        self.replay = []

    def append(self, transition: SAGTriple):
        self.replay.append(transition)

    def _calculate_g(self) -> SAGTriple:
        g = 0
        terminal_found = False
        for i in range(0, self.n_step - 1):
            if self.replay[i].next_reward > 0:
                terminal_found = True
                break
            g += self.replay[i].next_reward * self.gamma**(i + 1)

        if not terminal_found:
            g += self.replay[self.n_step].value * self.gamma**self.n_step

        result_element = self.replay[0]
        result_element.g = g
        # remove the first element
        self.replay = self.replay[1:]
        return result_element

    def get_batch(self, batch_size):
        if len(self.replay) < batch_size + self.n_step:
            return [], []
        states = []
        gs = np.zeros([batch_size, 8])
        for i in range(batch_size):
            current_SAG = self._calculate_g()
            states.append(current_SAG.current_state)
            gs[i][current_SAG.current_action] = current_SAG.g

        return np.concatenate(states, axis=0), np.stack(gs)
