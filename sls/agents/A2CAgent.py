from sls.agents import AbstractAgent
from sls.NeuralNetA2C import Network
from sls.EpisodeReplayA2C import EpisodeReplayA2C, SAGTriple
from sls import Env

import numpy as np
from multiprocessing import Process, Pipe


class A2CAgent(AbstractAgent):
    def __init__(
            self,
            train: bool,
            screen_size: int,
            minimap_size: int,
            gamma=0.99,
            batch_size=8,
            n_step_return=5,
            num_worker=8,
    ):
        super(A2CAgent, self).__init__(screen_size)
        self.actions = list(self._DIRECTIONS.keys())
        self.train = train
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.batch_size = batch_size
        self.prev_actions = None
        self.prev_states = None
        self.prev_values = None
        self.loss = 0
        self.network = Network()
        self.episode_replays = [EpisodeReplayA2C(n_step_return, gamma=gamma) for _ in range(num_worker)]
        self.num_worker = num_worker
        self.prev_obs_list = []
        self.counter = 0
        self.new_episode = [True for _ in range(num_worker)]

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_worker)])
        self.ps = [Process(target=self.worker_func, args=work_remote) for (work_remote) in zip(self.work_remotes)]
        for p in self.ps:
            p.start()

    def step(self, obs):

        # EVALUATION
        if not self.train:
            if self._MOVE_SCREEN.id not in obs.observation.available_actions:
                return self._SELECT_ARMY
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)
            current_state = np.expand_dims(np.array(obs[3].feature_screen['unit_density']), axis=-1).astype(np.float32)
            current_action, output_value = self.network.predict_both(x=current_state)
            return self._dir_to_sc2_action(self.actions[current_action[0, 0]], marine_coords)

        # TRAINING
        # reset env and select marine in the first iteration
        if self.counter == 0:
            obs_list = self.reset_worker_env()
            for i in range(self.num_worker):
                self.episode_replays[i].reset()
            self.counter += 1
            current_actions = []
            for i in range(self.num_worker):
                if self._MOVE_SCREEN.id not in obs_list[i].observation.available_actions:
                    current_actions.append(self._SELECT_ARMY)
            self.prev_obs_list = self.step_worker(current_actions, obs_list)
            return 0, 0, False

        obs_list = self.prev_obs_list
        reward_list, is_terminal_list = self.calculate_reward_terminal(obs_list)
        current_states, current_actions, current_values = self.get_state_action_value(obs_list)
        obs_list = self.step_worker(current_actions, obs_list)

        # save state + action + reward
        for i in range(self.num_worker):
            if not self.new_episode[i]:
                SAG = SAGTriple(
                    current_state=self.prev_states[i],
                    current_action=self.prev_actions[i],
                    next_reward=reward_list[i],
                    value=self.prev_values[i]
                )
                self.episode_replays[i].append(transition=SAG)
            self.new_episode[i] = False

        aggregated_states, aggregated_gs, aggregated_action_one_hot, batch_not_empty = self.aggregate_batch()

        if batch_not_empty:
            self.loss = self.network.fit([aggregated_states, aggregated_gs, aggregated_action_one_hot])

        self.prev_states = current_states
        self.prev_actions = current_actions
        self.prev_obs_list = obs_list
        self.prev_values = current_values

        for i in range(self.num_worker):
            if is_terminal_list[i] or obs_list[i].last():
                self.new_episode[i] = True

        all_finished = all([obs.last() for obs in obs_list])
        return self.loss, np.mean(reward_list), all_finished

    def save_model(self, filename):
        self.network.save_model(filename)

    def load_model(self, filename):
        self.network.load_model(filename)

    def worker_func(self, remote):
        env = Env(
            screen_size=self.screen_size,
            minimap_size=self.minimap_size,
            visualize=False
        )
        obs = env.reset()

        while True:
            cmd, action = remote.recv()
            if cmd == "step":
                if not obs.last():
                    obs = env.step(action)
                    remote.send(obs)
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError

    def reset_worker_env(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def step_worker(self, input_actions, obs_list):
        actions = []
        for i, action in enumerate(input_actions):
            if hasattr(action, 'function'):
                actions.append(action)
            else:
                actions.append(self._dir_to_sc2_action(self.actions[action], self._get_unit_pos(self._get_marine(obs_list[i]))))
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def close_worker(self):
        for remote in self.remotes:
            remote.send(("close", None))

    @staticmethod
    def _calc_reward_terminal(reward):
        is_terminal = reward > 0
        if is_terminal:
            reward = 1.0
        else:
            reward = -0.01
        return reward, is_terminal

    def calculate_reward_terminal(self, obs_list):
        reward_list = []
        is_terminal_list = []
        for i in range(self.num_worker):
            current_reward = obs_list[i].reward
            reward, is_terminal = self._calc_reward_terminal(current_reward)
            reward_list.append(reward)
            is_terminal_list.append(is_terminal)
        return reward_list, is_terminal_list

    def get_state_action_value(self, obs_list):
        current_states = []
        current_actions = []
        current_values = []
        for i in range(self.num_worker):
            current_state = np.expand_dims(np.expand_dims(np.array(obs_list[i][3].feature_screen['unit_density']), axis=-1).astype(np.float32), axis=0)
            current_states.append(current_state)
            current_action, current_value = self.network.predict_both(x=current_state)
            current_actions.append(current_action[0])
            current_values.append(current_value[0, 0])
        return current_states, current_actions, current_values

    def aggregate_batch(self):
        aggregated_states = []
        aggregated_gs = []
        aggregated_action_one_hot = []
        batch_not_empty = False

        for i in range(self.num_worker):
            states, gs, action_one_hot, empty_batch = self.episode_replays[i].get_batch(self.batch_size)
            if empty_batch:
                continue
            batch_not_empty = True
            aggregated_states.append(states)
            aggregated_gs.append(gs)
            aggregated_action_one_hot.append(action_one_hot)

        if batch_not_empty:
            aggregated_states = np.concatenate(aggregated_states, axis=0)
            aggregated_gs = np.concatenate(aggregated_gs, axis=0)
            aggregated_action_one_hot = np.concatenate(aggregated_action_one_hot, axis=0)
        return aggregated_states, aggregated_gs, aggregated_action_one_hot, batch_not_empty
