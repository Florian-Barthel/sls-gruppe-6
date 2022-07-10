import datetime
import numpy as np
import tensorflow as tf

from sls.env import Env
from sls.EpisodeReplay import EpisodeReplay
from sls.NeuralNetPG import Network


class PGRunner:
    def __init__(
        self,
        agent,
        env: Env,
        train: bool,
        load_path: str,
        num_scores_average: int,
        gamma: float,
        network: Network,
        file_format: str = '.pkl'
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
        self.file_format = file_format
        self.gamma = gamma
        self.network = network

        config_description = type(agent).__name__

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
            value=[tf.Summary.Value(tag='Loss', simple_value=self.loss_average)]),
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
            episode_replay = EpisodeReplay()
            while True:
                action, episode_transition = self.agent.step(obs)
                if episode_transition is not None:
                    episode_replay.append(episode_transition)
                    # if episode_transition.next_reward > 0:
                    #     break
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            states = episode_replay.get_states()
            actions = np.array(episode_replay.get_actions())
            g = np.array(episode_replay.calculate_g(gamma=self.gamma))
            g_actions = np.stack([g, actions], axis=-1)
            loss = self.network.train_step_train_model(x=states, y=g_actions)
            self.update_loss(loss)
            self.add_score()
            self.summarize()

    def add_score(self):
        self.score_average_list.append(self.score)
        self.score_average = np.mean(np.array(self.score_average_list)[-min(len(self.score_average_list), self.num_scores_average):])
        self.score = 0

    def update_loss(self, loss):
        self.loss_average_list.append(loss)
        self.loss_average = np.mean(np.array(self.loss_average_list)[-min(len(self.loss_average_list), 100):])