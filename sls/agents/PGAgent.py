from sls.agents import AbstractAgent
from sls.EpisodeReplay import PGTransition
import numpy as np


class PGAgent(AbstractAgent):
    def __init__(
            self,
            train: bool,
            screen_size: int,
            network,
            exploration='epsilon_greedy',
    ):
        super(PGAgent, self).__init__(screen_size)
        self.actions = list(self._DIRECTIONS.keys())

        assert exploration in ['epsilon_greedy', 'boltzmann']
        self.exploration = exploration
        self.train = train
        self.screen_size = screen_size
        self.net = network

        self.prev_state = None
        self.prev_action = None

    def step(self, obs):
        if self._MOVE_SCREEN.id not in obs.observation.available_actions:
            return self._SELECT_ARMY, None
        marine = self._get_marine(obs)
        beacon = self._get_beacon(obs)
        if marine is None:
            return self._NO_OP
        marine_coords = self._get_unit_pos(marine)
        beacon_coords = self._get_unit_pos(beacon)
        reward = obs.reward
        is_terminal = reward > 0
        diff = beacon_coords - marine_coords
        current_state = np.array(diff / self.screen_size)
        assert -1 <= current_state.all() <= 1

        current_action = np.argmax(self.net.predict(np.expand_dims(current_state, axis=0))[0])

        if reward > 0:
            reward = 100
        else:
            reward = -0.1

        current_transition = None
        if self.prev_state is not None:
            current_transition = PGTransition(
                current_state=self.prev_state,
                current_action=self.prev_action,
                next_reward=reward
            )

        self.prev_state = current_state
        self.prev_action = current_action

        if is_terminal or obs.last():
            self.prev_state = None
        return self._dir_to_sc2_action(self.actions[current_action], marine_coords), current_transition

    def save_model(self, filename):
        self.net.save_model(filename)

    def load_model(self, filename):
        self.net.load_model(filename)
