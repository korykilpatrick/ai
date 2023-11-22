import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, dest='num_episodes', default=1000000) # num_episodes
    parser.add_argument('-r', dest='render_mode', action='store_const', const='human')
    parser.add_argument('-e', type=str, dest='epsilon', default=1.0)
    parser.add_argument('-a', type=float, dest='alpha', default=0.1)
    parser.add_argument('-g', type=float, dest='gamma', default=0.9)
    parser.add_argument('-ed', type=float, dest='epsilon_decay', default=0.99)
    parser.add_argument('-em', type=float, dest='epsilon_min', default=0.01)
    parser.add_argument('--on-policy', dest='on_policy', action='store_true')
    parser.add_argument('--env', type=str, dest='env', default='cp')
   
    return parser.parse_args()