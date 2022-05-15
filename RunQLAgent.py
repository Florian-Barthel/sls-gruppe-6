from absl import app
from sls import Env, QLRunner
from sls.agents import *

_CONFIG = dict(
    episodes=100,
    screen_size=64,
    minimap_size=16,
    visualize=False,
    train=False,
    agent=QLAgent,
    num_states_x=32,
    num_states_y=32,
    num_scores_average=50,
    load_path='./results/220515_1821_train_QLAgent/1000.pkl'
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        num_states_x=_CONFIG['num_states_x'],
        num_states_y=_CONFIG['num_states_y']
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
        num_scores_average=_CONFIG['num_scores_average']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
