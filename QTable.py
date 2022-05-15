import numpy as np
import pandas as pd
from typing import List


class QLTable:
    def __init__(
            self,
            screen_size: int,
            num_states_x: int,
            num_states_y: int,
            actions: List
    ):
        self.num_states_x = num_states_x // 2
        self.num_states_y = num_states_y // 2
        self.state_width = screen_size // self.num_states_x
        self.state_height = screen_size // self.num_states_y
        self.actions = actions
        self.intervals = []

        self.q_table = None
        self.init_q_table()

    def init_q_table(self):
        d = {
            'state': [],
            'N': [],
            'NE': [],
            'E': [],
            'SE': [],
            'S': [],
            'SW': [],
            'W': [],
            'NW': []
        }

        start_x = (-self.num_states_x + 1) * self.state_width
        start_y = (-self.num_states_y + 1) * self.state_height
        end_x = self.state_width * (self.num_states_x + 1)
        end_y = self.state_height * (self.num_states_y + 1)
        for y in range(start_y, end_y, self.state_height):
            for x in range(start_x, end_x, self.state_width):
                self.intervals.append((y, x))
                d['state'].append(self.format_state(y, x))
                for action in self.actions:
                    d[action].append(np.random.normal(0.0, 0.1))
        self.intervals = np.array(self.intervals)
        self.q_table = pd.DataFrame(data=d)
        print(self.q_table)

    def get_current_state(self, marine_coords, beacon_coords):
        diff = beacon_coords - marine_coords
        for y, x in self.intervals:
            if diff[0] <= y and diff[1] <= x:
                return self.format_state(y, x)

    def format_state(self, y: int, x: int):
        return 'y: [{}, {}], x: [{}, {}] '.format(y - self.state_height, y, x - self.state_width, x)

    def get_max_action(self, current_state: str):
        row = self.q_table.loc[self.q_table['state'] == current_state]
        max_index = np.argmax(row.values[:, 1:])
        return self.actions[max_index]

    def update_q_value(
            self,
            current_state: str,
            current_action: str,
            next_state: str,
            reward: float,
            is_terminal: bool,
            discount_factor: float,
            learning_rate: float
    ):
        future_q_s_a = 0
        if not is_terminal:
            next_action = self.get_max_action(next_state)
            future_q_s_a = self.q_table.loc[self.q_table['state'] == next_state, next_action].values[0]

        q_s_a = self.q_table.loc[self.q_table['state'] == current_state, current_action].values[0]
        q_s_a += learning_rate * (reward + discount_factor * future_q_s_a - q_s_a)
        self.q_table.loc[self.q_table['state'] == current_state, current_action] = q_s_a

    def save_model(self, filename: str):
        self.q_table.to_pickle(filename)

    def load_model(self, filename: str):
        self.q_table = pd.read_pickle(filename)
