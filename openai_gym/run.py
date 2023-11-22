import numpy as np

from openai_gym.q_learning import QLearning
from openai_gym.envs.cartpole import CartPole
from openai_gym.envs.frozen_lake import FrozenLake
from openai_gym.utils import parse_args

def run_experiment(config):
    # env = CartPole(config, render_mode=None)
    env = FrozenLake(config, render_mode=None)
    config = {
        'num_episodes': 10000,
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon': 1.0,
        'epsilon_decay': 0.99,
        'epsilon_min': 0.01,

    }
    q_learning = QLearning(env, config)
    q_learning.run()
    return np.mean(q_learning.rewards[-100:])

if __name__ == '__main__':
    args = parse_args()
    config = {
        'num_episodes': args.num_episodes,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min,
        'on_policy': args.on_policy
    }
    # fl = FrozenLake(render_mode=render_mode)
    custom_env = CartPole(render_mode=args.render_mode) if args.env == 'cp' else FrozenLake(render_mode=args.render_mode)
    q_learning = QLearning(custom_env, config)
    print(config)
    q_learning.run()
    print(np.mean(q_learning.rewards[-100:]))