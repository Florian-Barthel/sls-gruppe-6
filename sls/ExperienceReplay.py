import numpy as np
from collections import deque
import random


class Transition:
    def __init__(
            self,
            current_state: np.ndarray,
            current_action: int,
            next_reward: float,
            next_state: np.ndarray,
            done: bool
    ):
        self.current_state = current_state
        self.current_action = current_action
        self.next_reward = next_reward
        self.next_state = next_state
        self.done = done

class ExperienceReplay:
    def __init__(
            self,
            max_length: int
    ):
        self.replay_queue = deque(maxlen=max_length)

    def __len__(self):
        return len(self.replay_queue)

    def append(self, transition: Transition):
        self.replay_queue.append(transition)

    def get_random_batch(self, batch_size: int):
        return random.sample(self.replay_queue, batch_size)
