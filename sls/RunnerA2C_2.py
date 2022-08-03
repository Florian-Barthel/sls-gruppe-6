import datetime
import os
import tensorflow as tf
import numpy as np


class RunnerA2C:
    def __init__(self, agent, env, train, load_path, num_scores_average: int, file_format: str = '.pkl'):

        self.agent = agent
        self.env = env
        self.train = train  # run only or train_model model?

        self.num_scores_average = num_scores_average
        self.score_average_list = []
        self.loss_average_list = []
        self.score_average = 0
        self.loss_average = 0
        self.loss = 0

        self.file_format = file_format

        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter

        self.path = './results/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__

        self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        if not self.train and load_path is not None and os.path.isdir(load_path):
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
        self.score = 0
        self.loss = 0

    def run(self, episodes):
        while self.episode <= episodes:

            obs = self.env.reset()
            self.agent.counter = 0
            while True:
                if self.train:
                    self.loss, avg_reward, episode_finished = self.agent.step(obs)
                    self.score += avg_reward
                    if episode_finished:
                        break
                else:
                    action = self.agent.step(obs)
                    self.score += obs.reward
                    if obs.last():
                        break
                    obs = self.env.step(action)

            if self.train:
                self.update_loss(self.loss)
            self.add_score()
            self.summarize()

    def add_score(self):
        self.score_average_list.append(self.score)
        self.score_average = np.mean(np.array(self.score_average_list)[-min(len(self.score_average_list), self.num_scores_average):])
        self.score = 0

    def update_loss(self, loss):
        self.loss_average_list.append(loss)
        self.loss_average = np.mean(np.array(self.loss_average_list)[-min(len(self.loss_average_list), 100):])
