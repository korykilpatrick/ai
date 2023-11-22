import gymnasium as gym
import numpy as np

# https://gymnasium.farama.org/environments/classic_control/mountain_car/

# TODO: factor out the discretize_state method into a separate class, allowing for multiple implementations

class MountainCar:
    def __init__(self, bins=[10, 10], render_mode=None):
        self.env = gym.make('MountainCar-v0', render_mode=render_mode)
        self.is_continuous = True
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.bins = bins
        self.q_table = np.zeros(self.bins + [self.env.action_space.n])
    
    def discretize_state(self, observation):
        # bucketing continuous values into discrete bins
        position, velocity = observation

        # np.digitize is 1-indexed, so subtract one to get 0-indexed
        bin_position = np.digitize(position, np.linspace(-1.2, 0.6, self.bins[0])) - 1
        bin_velocity = np.digitize(velocity, np.linspace(-0.07, 0.07, self.bins[1])) - 1

        return (int(bin_position), int(bin_velocity))
    def __repr__(self):
        return "MountainCar-v0"