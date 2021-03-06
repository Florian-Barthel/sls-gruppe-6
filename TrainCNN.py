from absl import app
from sls import Env, QLRunner
from sls.agents import *

_CONFIG = dict(
    episodes=1000,
    screen_size=16,
    minimap_size=16,
    visualize=False,
    train=True,
    agent=CNNAgent,
    load_path='./models/...',
    num_scores_average=50,
    discount_factor=0.9, # best 0.9
    sarsa=False,
    exploration='epsilon_greedy',
    file_format='.h5',
    priority_buffer=True
)


def main(unused_argv):
    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        discount_factor=_CONFIG['discount_factor'],
        exploration=_CONFIG['exploration']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = QLRunner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        num_scores_average=_CONFIG['num_scores_average'],
        sarsa=_CONFIG['sarsa'],
        exploration=_CONFIG['exploration'],
        file_format=_CONFIG['file_format'],
        priority_buffer=_CONFIG['priority_buffer']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
