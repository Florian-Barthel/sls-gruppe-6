from sls.agents import AbstractAgent
import numpy as np
from QTable import QLTable


class QLAgent(AbstractAgent):
    def __init__(
            self,
            train,
            screen_size,
            num_states_x=32,
            num_states_y=32,
            discount_factor=0.9,
            learning_rate=0.2
    ):
        super(QLAgent, self).__init__(screen_size)
        self.actions = list(self._DIRECTIONS.keys())
        self.q_table = QLTable(
            screen_size=screen_size,
            num_states_x=num_states_x,
            num_states_y=num_states_y,
            actions=self.actions
        )
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.train = train
        if train:
            self.epsilon = 1
        else:
            self.epsilon = 0

        self.prev_direction = None
        self.prev_state = None
        self.prev_marine_coords = None

    def step(self, obs):
        if self._MOVE_SCREEN.id not in obs.observation.available_actions:
            return self._SELECT_ARMY
        marine = self._get_marine(obs)
        beacon = self._get_beacon(obs)
        if marine is None:
            return self._NO_OP
        marine_coords = self._get_unit_pos(marine)
        beacon_coords = self._get_unit_pos(beacon)
        reward = obs.reward
        is_terminal = reward > 0
        current_state = self.q_table.get_current_state(marine_coords, beacon_coords)

        if self.train and self.prev_state is not None:
            self.q_table.update_q_value(
                current_state=self.prev_state,
                current_action=self.prev_direction,
                next_state=current_state,
                reward=reward,
                is_terminal=is_terminal,
                discount_factor=self.discount_factor,
                learning_rate=self.learning_rate
            )

        direction = self.chose_action(current_state)
        self.prev_state = current_state
        self.prev_direction = direction
        self.prev_marine_coords = marine_coords
        return self._dir_to_sc2_action(direction, marine_coords)

    def save_model(self, filename):
        self.q_table.save_model(filename)

    def load_model(self, filename):
        self.q_table.load_model(filename)

    def chose_action(self, current_state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.q_table.get_max_action(current_state)
