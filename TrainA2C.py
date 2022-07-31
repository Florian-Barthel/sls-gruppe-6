from absl import app
from sls import Env
from sls.agents import *
from sls.NeuralNetA2C import Network
from sls.RunnerA2C_2 import RunnerA2C


_CONFIG = dict(
    episodes=10000,
    screen_size=16,
    minimap_size=16,
    visualize=False,
    train=True,
    agent=A2CAgent,
    load_path='./models/...',
    num_scores_average=50,
    gamma=0.99,
    file_format='.h5'
)

# network = Network()


def main(unused_argv):
    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        gamma=_CONFIG['gamma']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = RunnerA2C(
        agent=agent,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        num_scores_average=_CONFIG['num_scores_average'],
        file_format=_CONFIG['file_format'],
        env=env
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
