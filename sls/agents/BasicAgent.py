from sls.agents import AbstractAgent
import numpy as np


class BasicAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(BasicAgent, self).__init__(screen_size)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)
            beacon_coords = self._get_unit_pos(beacon)
            diff = beacon_coords - marine_coords
            direction = np.sign(diff)
            d = ''
            for dir_name, dir_coord in self._DIRECTIONS.items():
                if direction[0] == dir_coord[0] and direction[1] == dir_coord[1]:
                    d = dir_name
                    break

            return self._dir_to_sc2_action(d, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
