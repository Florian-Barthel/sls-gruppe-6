from sls.agents import AbstractAgent
from sls.NeuralNetCNN import Network
import numpy as np
from baselines.deepq import PrioritizedReplayBuffer


class CNNAgent(AbstractAgent):
    def __init__(
            self,
            train: bool,
            screen_size: int,
            discount_factor: float,
            exploration='epsilon_greedy',
            min_replay_size=6000,
            batch_size=32, # best 32
            epsilon_replay=1e-6,
            beta=0.6,
            beta_inc=5e-6
    ):
        self.epsilon_replay = epsilon_replay
        self.replay_buffer = PrioritizedReplayBuffer(size=100000, alpha=0.4)
        super(CNNAgent, self).__init__(screen_size)
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
        self.beta = beta
        self.beta_inc = beta_inc
        self.net = Network()
        self.replay_length = 0

        # epsilon decay
        if train:
            self.epsilon = 1
        else:
            self.epsilon = 0

        # boltzmann
        self.temperature = 500

    def step(self, obs):
        if self._MOVE_SCREEN.id not in obs.observation.available_actions:
            return self._SELECT_ARMY
        marine = self._get_marine(obs)
        if marine is None:
            return self._NO_OP
        marine_coords = self._get_unit_pos(marine)
        reward = obs.reward
        is_terminal = reward > 0
        current_state = np.expand_dims(np.array(obs[3].feature_screen['unit_density']), axis=-1).astype(np.float32)
        if np.random.uniform() < self.epsilon:
            current_action = np.random.choice(range(len(self.actions)))
        else:
            current_action = np.argmax(self.net.predict_train_model(np.expand_dims(current_state, axis=0))[0])

        if self.train and self.prev_state is not None:
            self.replay_buffer.add(
                obs_t=self.prev_state,
                action=self.prev_action,
                reward=reward,
                obs_tp1=current_state,
                done=is_terminal
            )
            self.replay_length = len(self.replay_buffer)
            if self.replay_length > self.min_replay_size:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, indices = self.replay_buffer.sample(self.batch_size, beta=self.beta)
                self.beta = min(self.beta_inc + self.beta, 1.0)

                y = self.net.predict_train_model(obs_batch)
                train_model_prediction = y.copy()
                next_rows_train = self.net.predict_train_model(next_obs_batch)
                next_rows_target = self.net.predict_target_model(next_obs_batch)

                for i in range(np.shape(obs_batch)[0]):
                    if done_mask[i]:
                        y[i, act_batch[i]] = rew_batch[i]
                    else:
                        max_index_train = np.argmax(next_rows_train[i])
                        y[i, act_batch[i]] = rew_batch[i] + self.discount_factor * next_rows_target[i][max_index_train]

                separated_loss = self.net.mse_numpy(x=train_model_prediction, y=y)
                priority = separated_loss + self.epsilon_replay
                self.loss = self.net.train_step_train_model(x=obs_batch, y=y)
                self.replay_buffer.update_priorities(indices, priority)

        self.prev_state = current_state
        self.prev_action = current_action
        if is_terminal or obs.last():
            self.prev_state = None
        return self._dir_to_sc2_action(self.actions[current_action], marine_coords)

    def save_model(self, filename):
        self.net.save_model(filename)

    def load_model(self, filename):
        self.net.load_model(filename)

    def update_target_model(self):
        self.net.update_target_model()
