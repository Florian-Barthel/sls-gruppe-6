from absl import app
from sls import Env, QLRunner
from sls.agents import *

_CONFIG = dict(
    episodes=1000,
    screen_size=64,
    minimap_size=16,
    visualize=False,
    train=True,
    agent=QLAgent,
    load_path='./models/...',
    num_scores_average=50,
    num_states_x=32,
    num_states_y=32,
    discount_factor=0.9, # best 0.9
    learning_rate=0.3, # best 0.3
    sarsa=True,
    exploration='boltzmann' # boltzmann or epsilon_greedy
)


def main(unused_argv):
    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        learning_rate=_CONFIG['learning_rate'],
        num_states_x=_CONFIG['num_states_x'],
        num_states_y=_CONFIG['num_states_y'],
        discount_factor=_CONFIG['discount_factor'],
        exploration=_CONFIG['exploration'],
        sarsa=_CONFIG['sarsa']
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
        exploration=_CONFIG['exploration']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
