import datetime
import os
import tensorflow as tf


class Runner:
    def __init__(self, agent, env, train, load_path):

        self.agent = agent
        self.env = env
        self.train = train  # run only or train_model model?

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
            value=[tf.Summary.Value(tag='Score per Episode', simple_value=self.score)]),
            self.episode
        )
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...
        self.episode += 1
        self.score = 0

    def run(self, episodes):
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            self.summarize()
