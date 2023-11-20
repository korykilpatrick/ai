import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000000) # num_episodes
    parser.add_argument('-r', type=str, default=None)
    args = parser.parse_args()
    render_mode = 'human' if args.r is not None else None
    return args.n, render_mode