import datetime
import os
import numpy as np
import tensorflow as tf
from sls.env import Env
from sls.agents import QLAgent


class QLRunner:
    def __init__(
            self,
            agent: QLAgent,
            env: Env,
            train: bool,
            load_path: str,
            num_scores_average: int
    ):

        self.agent = agent
        self.env = env
        self.train = train
        self.score = 0
        self.episode = 1
        self.score_average_list = []
        self.num_scores_average = num_scores_average
        self.score_average = 0
        self.total_episodes = 0

        self.path = './results/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__
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

        if self.train and self.episode % 20 == 0:
            self.agent.save_model(self.path + '/' + str(self.episode) + '.pkl')
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
            self.add_score()
            self.summarize()
            if self.train:
                self.update_epsilon()

    def add_score(self):
        self.score_average_list.append(self.score)
        self.score_average = np.mean(np.array(self.score_average_list)[-max(len(self.score_average_list), 50):])
        self.score = 0

    def update_epsilon(self):
        self.agent.epsilon = 1 - self.episode / self.total_episodes

