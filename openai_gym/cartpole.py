import gymnasium as gym
import numpy as np

from utils import parse_args

class CartPole:
    def __init__(self, num_episodes=100, render_mode=None):
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.observation = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = 500
        self.episode_number = 0
        self.num_episodes = num_episodes
        # load q-table if it exists, otherwise initialize to zeros
        try:
            self.q_table = np.load('q_table.npy')
        except FileNotFoundError:
            self.q_table = np.zeros((10, 10, 10, 10, 2))
        self.alpha = 0.1 # learning rate
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.1 # exploration rate
        self.rewards = []

    def discretize_state(self, observation):
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        # np.digitize is 1-indexed, so subtract one to get 0-indexed
        bin_cart_position = np.digitize(cart_position, np.linspace(-2.4, 2.4, 10)) - 1
        bin_cart_velocity = np.digitize(cart_velocity, np.linspace(-3, 3, 10)) - 1
        bin_pole_angle = np.digitize(pole_angle, np.linspace(-0.2095, 0.2095, 10)) - 1
        bin_pole_velocity = np.digitize(pole_velocity, np.linspace(-3, 3, 10)) - 1
        return (int(bin_cart_position), int(bin_cart_velocity), int(bin_pole_angle), int(bin_pole_velocity))
    
    def get_action(self, discretized_state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[discretized_state])

    def update_q_table(self, discretized_state, action, reward, next_discretized_state):
        self.q_table[discretized_state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_discretized_state]) - self.q_table[discretized_state][action])
    
    def run(self):
        for i in range(self.num_episodes):
            self.episode_number += 1
            self.run_episode()
        self.env.close()

    def run_episode(self):
        observation, info = self.env.reset()
        state = self.discretize_state(observation)
        total_reward = 0
        for i in range(self.max_steps):
            action = self.get_action(state)
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            state = self.discretize_state(observation)
            
            prev_state = state
            self.update_q_table(prev_state, action, reward, state)
            if terminated or truncated:
                observation, info = self.env.reset()
                break
        self.rewards.append(total_reward)

if __name__ == '__main__':
    num_episodes, render_mode = parse_args()
    cartpole = CartPole(num_episodes=num_episodes, render_mode=render_mode)
    try:
        cartpole.run()
        print(cartpole.best_performance)
    except KeyboardInterrupt:
        # write q-table to file if user interrupts
        pass
    np.save('q_table.npy', cartpole.q_table)