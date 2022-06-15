from sls.agents import AbstractAgent
from sls.ExperienceReplay import ExperienceReplay, Transition
from sls.NeuralNet import Network
import numpy as np


class DDQNAgent(AbstractAgent):
    def __init__(
            self,
            train: bool,
            screen_size: int,
            discount_factor: float,
            exploration='epsilon_greedy',
            min_replay_size=6000,
            batch_size=32, # best 32
            train_interval=1
    ):
        super(DDQNAgent, self).__init__(screen_size)
        self.actions = list(self._DIRECTIONS.keys())

        assert exploration in ['epsilon_greedy', 'boltzmann']
        self.exploration = exploration
        self.discount_factor = discount_factor
        self.train = train
        self.screen_size = screen_size
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size

        self.prev_action = None
        self.prev_state = None
        self.prev_marine_coords = None
        self.loss = 0

        self.experience_replay = ExperienceReplay(100000)
        self.net = Network()

        # epsilon decay
        if train:
            self.epsilon = 1
        else:
            self.epsilon = 0

        # boltzmann
        self.temperature = 500
        self.index = 0
        self.replay_length = 0
        self.train_interval = train_interval

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
        diff = beacon_coords - marine_coords
        current_state = np.array(diff / self.screen_size)
        assert -1 <= current_state.all() <= 1

        # only predict if necessary
        if np.random.uniform() < self.epsilon:
            current_action = np.random.choice(range(len(self.actions)))
        else:
            current_action = np.argmax(self.net.predict_train_model(np.expand_dims(current_state, axis=0))[0])

        self.index += 1
        if self.train and self.prev_state is not None:
            current_transition = Transition(
                current_state=self.prev_state,
                current_action=self.prev_action,
                next_reward=reward,
                next_state=current_state,
                done=is_terminal
            )
            self.experience_replay.append(current_transition)
            self.replay_length = len(self.experience_replay)
            if self.replay_length > self.min_replay_size and self.index % self.train_interval == 0:
                transition_batch = self.experience_replay.get_random_batch(self.batch_size)
                x = []
                actions = []
                next_states = []
                for transition in transition_batch:
                    x.append(transition.current_state)
                    actions.append(transition.current_action)
                    next_states.append(transition.next_state)

                x = np.array(x)
                actions = np.array(actions)
                next_states = np.array(next_states)

                y_train = self.net.predict_train_model(x)
                y_target = self.net.predict_target_model(x)
                next_rows_train = self.net.predict_train_model(next_states)
                next_rows_target = self.net.predict_target_model(next_states)

                for i, transition in enumerate(transition_batch):
                    if transition.done:
                        y_train[i, actions[i]] = transition.next_reward
                        y_target[i, actions[i]] = transition.next_reward
                    else:
                        y_train[i, actions[i]] = transition.next_reward + self.discount_factor * np.max(next_rows_target[i])
                        y_target[i, actions[i]] = transition.next_reward + self.discount_factor * np.max(next_rows_train[i])

                self.loss = self.net.train_step_train_model(x=x, y=y_train)
                self.loss += self.net.train_step_target_model(x=x, y=y_target)

        self.prev_state = current_state
        self.prev_action = current_action
        self.prev_marine_coords = marine_coords
        if is_terminal or obs.last():
            self.prev_state = None
        return self._dir_to_sc2_action(self.actions[current_action], marine_coords)

    def save_model(self, filename):
        self.net.save_model(filename)

    def load_model(self, filename):
        self.net.load_model(filename)

    def update_target_model(self):
        pass
