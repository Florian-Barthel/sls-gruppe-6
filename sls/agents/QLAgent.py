from sls.agents import AbstractAgent
from QTable import QLTable
from sls.exploration import epsilon_greedy, boltzmann


class QLAgent(AbstractAgent):
    def __init__(
            self,
            train: bool,
            screen_size: int,
            sarsa: bool = False,
            num_states_x=32,
            num_states_y=32,
            discount_factor=0.9,
            learning_rate=0.2,
            exploration='epsilon_greedy'
    ):
        super(QLAgent, self).__init__(screen_size)
        self.actions = list(self._DIRECTIONS.keys())
        self.q_table = QLTable(
            screen_size=screen_size,
            num_states_x=num_states_x,
            num_states_y=num_states_y,
            actions=self.actions,
            sarsa=sarsa
        )
        assert exploration in ['epsilon_greedy', 'boltzmann']
        self.exploration = exploration
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.train = train

        # epsilon decay
        if train:
            self.epsilon = 1
        else:
            self.epsilon = 0

        # boltzmann
        self.temperature = 500

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
        direction = self.chose_action(current_state)

        if self.train and self.prev_state is not None:
            self.q_table.update_q_value(
                current_state=self.prev_state,
                current_action=self.prev_direction,
                next_state=current_state,
                reward=reward,
                is_terminal=is_terminal,
                discount_factor=self.discount_factor,
                learning_rate=self.learning_rate,
                next_action=direction
            )

        self.prev_state = current_state
        self.prev_direction = direction
        self.prev_marine_coords = marine_coords
        if is_terminal:
            self.prev_state = None
        return self._dir_to_sc2_action(direction, marine_coords)

    def save_model(self, filename):
        self.q_table.save_model(filename)

    def load_model(self, filename):
        self.q_table.load_model(filename)

    def chose_action(self, current_state):
        if self.exploration == 'epsilon_greedy':
            return epsilon_greedy(
                actions=self.q_table.actions,
                q_table_row=self.q_table.get_current_row(current_state),
                epsilon=self.epsilon
            )
        elif self.exploration == 'boltzmann':
            return boltzmann(
                actions=self.q_table.actions,
                q_table_row=self.q_table.get_current_row(current_state),
                temperature=self.temperature
            )
        else:
            raise NameError('unknown exploration strategy')
