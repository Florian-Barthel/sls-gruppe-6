import datetime
import numpy as np
import tensorflow as tf
from sls.env import Env


class QLRunner:
    def __init__(
            self,
            agent,
            env: Env,
            train: bool,
            load_path: str,
            num_scores_average: int,
            sarsa: bool = False,
            exploration: str = 'epsilon_greedy',
            file_format: str = '.pkl',
            priority_buffer: bool = False
    ):

        self.agent = agent
        self.env = env
        self.train = train
        self.score = 0
        self.episode = 1
        self.score_average_list = []
        self.loss_average_list = []
        self.num_scores_average = num_scores_average
        self.score_average = 0
        self.loss_average = 0
        self.total_episodes = 0
        self.replay_length = 0
        self.exploration = exploration
        self.file_format = file_format
        self.beta = 0
        self.priority_buffer = priority_buffer

        config_description = type(agent).__name__ + '_' + exploration
        if sarsa:
            config_description += '_SARSA'

        self.path = './results/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + config_description
        self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        if not self.train and load_path is not None:
            self.agent.load_model(load_path)

    def summarize(self):
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Average Score per Episode', simple_value=self.score_average)]),
            self.episode
        )
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Epsilon', simple_value=self.agent.epsilon)]),
            self.episode
        )
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Temperature', simple_value=self.agent.temperature)]),
            self.episode
        )
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Loss', simple_value=self.loss_average)]),
            self.episode
        )
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Beta', simple_value=self.get_beta())]),
            self.episode
        )
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Replay Length', simple_value=self.get_replay_length())]),
            self.episode
        )
        if self.train and self.episode % 10 == 0:
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...
            self.agent.save_model(self.path + '/' + str(self.episode) + self.file_format)
        self.episode += 1

    def run(self, episodes):
        self.total_episodes = episodes
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
                self.update_loss()
            self.add_score()
            self.summarize()
            if self.train and self.exploration == 'epsilon_greedy':
                self.update_epsilon()
            elif self.train and self.exploration == 'boltzmann':
                self.update_temperature()

    def add_score(self):
        self.score_average_list.append(self.score)
        self.score_average = np.mean(np.array(self.score_average_list)[-min(len(self.score_average_list), 50):])
        self.score = 0

    def update_loss(self):
        if hasattr(self.agent, 'loss'):
            self.loss_average_list.append(self.agent.loss)
            self.loss_average = np.mean(np.array(self.loss_average_list)[-min(len(self.loss_average_list), 100):])

    def update_epsilon(self):
        w = max(1 - self.episode / 500, 0)
        self.agent.epsilon = 1 * w + 0.05 * (1 - w)

    def update_temperature(self):
        self.agent.temperature = max(self.agent.temperature * 0.98, 0.01)

    def get_replay_length(self):
        if hasattr(self.agent, 'replay_length'):
            self.replay_length = self.agent.replay_length
        return self.replay_length

    def get_beta(self):
        if hasattr(self.agent, 'beta'):
            self.beta = self.agent.beta
        return self.beta