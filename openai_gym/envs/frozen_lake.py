import gymnasium as gym
import numpy as np

"""
TODO: 
handle slippery vs non-slippery
handle map randomization ?
"""
class FrozenLake:
    def __init__(self, render_mode=None):
        self.env = gym.make('FrozenLake-v1', render_mode=render_mode, is_slippery=False)
        self.is_continuous = False
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
    
    def modify_reward(self, observation, reward, terminated, truncated):
        """ Default setting is to receive reward = 0 for every step taken, except for the last step where reward = 1 if the goal is reached. I'm not sure why this is - it seems like holes should have a negative reward. It's unclear (without experimentation) what the appropriate values should be for the 3 cases of hole, goal, and not terminated.

        This is definitely incorrect and should be modified.
        """
        if terminated and reward == 1.0:
            # reached the goal
            return 10.0
        elif terminated:
            # fell in a hole
            return -1.0
        else:
            # not terminated
            return 0.0

    def __repr__(self):
        return "FrozenLake-v1"
    