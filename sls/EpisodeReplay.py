import numpy as np


class PGTransition:
    def __init__(
            self,
            current_state: np.ndarray,
            current_action: int,
            next_reward: float,
    ):
        self.current_state = current_state
        self.current_action = current_action
        self.next_reward = next_reward


class EpisodeReplay:
    def __init__(self):
        self.replay = []

    def __len__(self):
        return len(self.replay)

    def reset(self):
        self.replay = []

    def append(self, transition: PGTransition):
        self.replay.append(transition)

    def calculate_g(self, gamma: float):
        g = []
        for t in range(len(self.replay)):
            label = np.zeros(8)
            current_g = 0
            for k in range(t + 1, len(self.replay)):
                current_g += gamma**(k - t - 1) * self.replay[k].next_reward # k - 1?
            label[self.replay[t].current_action] = current_g
            g.append(label)
        return g

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
