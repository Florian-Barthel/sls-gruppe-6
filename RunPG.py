from absl import app
from sls import Env, PGRunner
from sls.agents import *
from sls.NeuralNetPG import Network


_CONFIG = dict(
    episodes=200,
    screen_size=16,
    minimap_size=16,
    visualize=False,
    train=False,
    agent=PGAgent,
    load_path='./results/220710_2015_train_PGAgent/6000.h5',
    num_scores_average=50,
    gamma=0.99,
    file_format='.h5'
)

network = Network()

def main(unused_argv):
    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        network=network
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = PGRunner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        num_scores_average=_CONFIG['num_scores_average'],
        file_format=_CONFIG['file_format'],
        network=network,
        gamma=_CONFIG['gamma']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
