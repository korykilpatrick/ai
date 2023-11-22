import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    envs = ['cp', 'fl', 'mc']
    parser.add_argument('--env', type=str, dest='env', 
    default='cp', choices=envs)
    
    parser.add_argument('-n', '--num-episodes', type=int, dest='num_episodes', default=1000000)
    parser.add_argument('-r', '--render-mode', dest='render_mode', action='store_const', const='human')
    parser.add_argument('-e', '--epsilon', type=float, dest='epsilon', default=1.0)
    parser.add_argument('-a', '--alpha', type=float, dest='alpha', default=0.1)
    parser.add_argument('-g', '--gamma', type=float, dest='gamma', default=0.9)
    parser.add_argument('-ed', '--epsilon-decay', type=float, dest='epsilon_decay', default=0.99)
    parser.add_argument('-em', '--epsilon-min', type=float, dest='epsilon_min', default=0.01)
    parser.add_argument('-op', '--on-policy', dest='on_policy', action='store_true')
   
    return parser.parse_args()