from sls.env import Env


class Worker:
    def __init__(self, screen_size, minimap_size):
        self.env = Env(
            screen_size=screen_size,
            minimap_size=minimap_size,
            visualize=False
        )

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.env.step(action)

    def close_env(self):
        self.env.close()
