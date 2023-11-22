import numpy as np

from openai_gym.q_learning import QLearning
from openai_gym.envs import CartPole, FrozenLake, MountainCar
import openai_gym.utils as utils

def run_experiment(custom_env, config, write_to_file=True):
    print(f"Running experiment on {custom_env} with config {config}")

    q_learning = QLearning(custom_env, config)
    try:
        q_learning.run()
    except KeyboardInterrupt:
        # write the q table to a file
        pass
    if write_to_file:
        np.save(f'q_table_{custom_env}.npy', q_learning.q_table)

def setup_custom_env():
    args = utils.parse_args()
    config = {
        'num_episodes': args.num_episodes,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min,
        'on_policy': args.on_policy
    }
    if args.env == 'cp':
        custom_env = CartPole(render_mode=args.render_mode)
    elif args.env == 'fl':
        custom_env = FrozenLake(render_mode=args.render_mode)
    elif args.env == 'mc':
        custom_env = MountainCar(render_mode=args.render_mode)

    # load q table if it exists
    try:
        custom_env.q_table = np.load(f'q_table_{args.env}.npy')
    except FileNotFoundError:
        pass

    return custom_env, config   

if __name__ == '__main__':
    custom_env, config = setup_custom_env()
    run_experiment(custom_env, config)