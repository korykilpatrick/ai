import gymnasium as gym
import numpy as np

class QLearning:
    def __init__(self, custom_env, config):
        self.custom_env = custom_env
        self.env = custom_env.env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = self.env.spec.max_episode_steps
        self.is_continuous = self.custom_env.is_continuous
        
        self.num_episodes = config['num_episodes']
        self.on_policy = config['on_policy']
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.modify_reward = self.custom_env.modify_reward if hasattr(self.custom_env, 'modify_reward') else None
        self.discretize_state = self.custom_env.discretize_state if hasattr(self.custom_env, 'discretize_state') else None
        if self.is_continuous and not self.discretize_state:
            raise NotImplementedError(f"Custom environment {self.custom_env} must implement discretize_state method")
        
        self.episode_number = 0
        self.rewards = []
        self.q_table = self.custom_env.q_table

    def decay_epsilon(self):
        # could take a lot of approaches here
        # TODO: implement a few different approaches and have config specify which one to use
        # if self.episode_number % 100 == 0:
            # self.epsilon = max(self.epsilon_min, self.epsilon - 0.01)
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon - 0.01)

    def discretize_state(self, observation):
        # discretize the state if it's continuous
        raise NotImplementedError
    
    def modify_reward(self, reward, observation, terminated, truncated):
        # optionally have a reward function passed into the config
        raise NotImplementedError
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            # explore
            return self.action_space.sample()
        else:
            # exploit
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        # print('got in here')
        if self.on_policy:
            # SARSA
            self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action])
        else:
            # Q-learning
            self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])
    
    def run(self):
        for i in range(self.num_episodes):
            self.episode_number += 1
            episode_reward = self.run_episode()
            self.rewards.append(episode_reward)
            self.decay_epsilon()
            print(f"Episode {self.episode_number} reward: {episode_reward}")
        self.env.close()

    def run_episode(self):
        observation, info = self.env.reset()
        state = self.discretize_state(observation) if self.is_continuous else observation
        action = self.get_action(state)
        total_reward = 0
        done = False
        while not done:
            observation, reward, terminated, truncated, info = self.env.step(action)
            if self.modify_reward:
                reward = self.modify_reward(reward, observation, terminated, truncated)
            total_reward += reward

            next_state = self.discretize_state(observation) if self.is_continuous else observation
            next_action = self.get_action(next_state)
            self.update_q_table(state, action, reward, next_state, next_action)
            done = terminated or truncated

            state, action = next_state, next_action
        
        return total_reward