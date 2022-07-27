import datetime
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Pipe, Pool

from sls import worker
from sls.env import Env
from sls.EpisodeReplayA2C import EpisodeReplay
from sls.NeuralNetPG import Network


class RunnerA2C:
    def __init__(
        self,
        agent,
        train: bool,
        load_path: str,
        num_scores_average: int,
        gamma: float,
        network: Network,
        file_format: str = '.pkl',
        num_worker: int = 8,
        screen_size=16,
        minimap_size=16
    ):

        self.agent = agent
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
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.num_worker = num_worker

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

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_worker)])
        self.ps = [Process(target=self.worker_func, args=(work_remote))
                   for (work_remote) in zip(self.work_remotes)]
        for p in self.ps:
            p.start()

        episode_replay = EpisodeReplay()
        self.total_episodes = episodes
        while self.episode <= episodes:
            obs = self.reset_worker_env()
            episode_replay.reset()
            while True:
                action, episode_transition = self.agent.step(obs)
                if episode_transition is not None:
                    episode_replay.append(episode_transition)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            states = episode_replay.get_states()
            g = np.array(episode_replay.calculate_g(gamma=self.gamma))
            if self.train:
                loss = self.network.train_step_train_model(x=states, y=g)
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

    def worker_func(self, remote):
        # Initialisieren des Envs im worker_thread
        env = Env(
            screen_size=self.screen_size,
            minimap_size=self.minimap_size,
            visualize=False
        )
        obs = env.reset()

        while True:
            cmd, action = remote.recv()
            if cmd == "step":
                action = self.agent.step(obs)
                obs = env.step(action)
                remote.send(obs)
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                env.close()
                remote.close()
            else:
                raise NotImplementedError

    def step_worker(self, actions=None):
        actions = actions or [None] * self.num_worker
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def reset_worker_env(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def close_worker(self):
        for remote in self.remotes:
            remote.send(("close", None))
