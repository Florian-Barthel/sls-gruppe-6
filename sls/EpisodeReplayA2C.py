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

    def __len__(self):
        return len(self.replay)

    def reset(self):
        self.replay = []

    def append(self, transition: SAGTriple):
        self.replay.append(transition)

    def _calculate_g(self) -> SAGTriple:
        g = 0
        terminal_found = False
        for i in range(self.n_step): # oder n_step -1 ?
            g += self.replay[i].next_reward * self.gamma**(i + 1)
            if self.replay[i].next_reward > 0:
                terminal_found = True
                break

        if not terminal_found:
            g += self.replay[self.n_step].value * self.gamma**self.n_step

        result_element = self.replay.pop(0)
        result_element.g = g
        return result_element

    def get_batch(self, batch_size):
        if len(self.replay) < batch_size + self.n_step:
            return [], [], [], True
        states = []
        gs = []
        action_select_one_hot = np.zeros([batch_size, 8])
        for i in range(batch_size):
            current_SAG = self._calculate_g()
            states.append(current_SAG.current_state)
            gs.append(current_SAG.g)
            action_select_one_hot[i][current_SAG.current_action] = 1

        return np.concatenate(states, axis=0), np.expand_dims(np.array(gs), axis=-1), action_select_one_hot, False
