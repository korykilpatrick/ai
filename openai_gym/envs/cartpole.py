import gymnasium as gym
import numpy as np

class CartPole:
    def __init__(self, bins=[10, 10, 10, 10], render_mode=None):
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.is_continuous = True
        self.bins = bins
        self.q_table = np.zeros(self.bins + [self.env.action_space.n])
    
    def discretize_state(self, observation):
        # bucketing continuous values into discrete bins
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        # np.digitize is 1-indexed, so subtract one to get 0-indexed
        bin_cart_position = np.digitize(cart_position, np.linspace(-2.4, 2.4, self.bins[0])) - 1
        bin_cart_velocity = np.digitize(cart_velocity, np.linspace(-3, 3, self.bins[1])) - 1
        bin_pole_angle = np.digitize(pole_angle, np.linspace(-0.2095, 0.2095, self.bins[2])) - 1
        bin_pole_velocity = np.digitize(pole_velocity, np.linspace(-3, 3, self.bins[3])) - 1

        return (int(bin_cart_position), int(bin_cart_velocity), int(bin_pole_angle), int(bin_pole_velocity))
    
    def __repr__(self):
        return "CartPole-v1"