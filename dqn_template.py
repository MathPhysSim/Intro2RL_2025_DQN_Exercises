import os
import random
from collections import deque
from datetime import datetime
import time

import gymnasium as gym  # Updated import to gymnasium
import numpy as np
import tensorflow as tf


"""
This template provides a basic structure for implementing a Deep Q-Network (DQN) agent 
to solve reinforcement learning tasks using the OpenAI Gymnasium environment. 

Students are expected to complete the implementation of the neural network architecture, 
the epsilon-greedy action selection policy, the target network update mechanism, and 
the training logic using experience replay. 

The code outlines the main components including experience replay buffer, the DQN model, 
and the agent that interacts with the environment, collects experiences, and learns from them.
"""

lr = 0.005
batch_size = 256
gamma = 0.95
eps = 1
eps_decay = 0.995
eps_min = 0.01
logdir = 'logs'

logdir = os.path.join(logdir, "Classical_DQN_solution_template.py", 'Car_pole', datetime.now().strftime("%Y%m%d-%H%M%S"))
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)

# ReplayBuffer class stores past experiences for training the DQN
class ReplayBuffer:
    """
    Experience Replay Buffer stores past experiences of the agent in the environment.
    This allows the agent to learn from a diverse set of past experiences by sampling
    random batches during training, which helps break correlation between sequential
    data and stabilizes learning.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # Circular buffer of fixed size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])  # Add experience tuple

    def sample(self):
        sample = random.sample(self.buffer, batch_size)  # Randomly sample batch
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)  # Current number of stored experiences


# DQN class defines the neural network and action selection logic
class DQN:
    def __init__(self, state_dim, action_dim):
        """
        Initialize the DQN model.

        epsilon represents the exploration rate used in the epsilon-greedy policy:
        - With probability epsilon, the agent chooses a random action (exploration).
        - With probability (1 - epsilon), the agent chooses the best predicted action (exploitation).
        This balances exploring new actions and exploiting known good actions.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = eps  # Initial exploration rate
        self.model = self.nn_model()  # Initialize neural network model

    def nn_model(self):
        """
        Define the neural network architecture here.

        This neural network takes the state as input and outputs Q-values for each possible action.
        The Q-values represent the expected cumulative future reward of taking each action from the given state.
        """
        # TODO: Implement the neural network architecture
        pass

    def predict(self, state):
        """
        Predict Q-values for the given state using the neural network.

        Q-values estimate how good each action is in the current state.
        """
        state = np.array(state).reshape(1, -1)
        return self.model(state, verbose=0).numpy()[0]  # Predict Q-values for given state

    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon, select a random action to explore.
        Otherwise, select the action with the highest predicted Q-value (exploit).
        """
        # TODO: Implement epsilon-greedy policy to select action
        pass

    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1, verbose=0)  # Train on batch
        loss = history.history['loss'][0]
        return loss


# Agent class manages interaction with environment and training process
class Agent:
    """
    The Agent interacts with the environment by choosing actions based on the DQN's predictions,
    collects experiences (state transitions), and learns by training the DQN with these experiences.

    It maintains two neural networks:
    - The main DQN model which is trained every step.
    - The target network which is updated less frequently to provide stable target Q-values during training.
    """
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]  # State vector dimension
        self.action_dim = self.env.action_space.n  # Number of discrete actions

        self.model = DQN(self.state_dim, self.action_dim)  # Main DQN network
        self.target_model = DQN(self.state_dim, self.action_dim)  # Target network for stability
        self.update_target()  # Initialize target network weights
        self.buffer = ReplayBuffer()  # Experience replay buffer

    def update_target(self):
        """
        Update the target network's weights by copying from the main DQN model.

        This helps stabilize training by keeping target Q-values fixed for a number of steps.
        Students should implement this method to copy weights from self.model to self.target_model.
        """
        # TODO: Implement target network update logic
        pass

    def replay_experience(self):
        """
        Sample a batch of experiences from the replay buffer and train the main DQN model.

        The target Q-values are computed using the target network to provide stable targets.
        Students should implement the training logic here.
        """
        # TODO: Implement training logic using the target network
        pass

    def train(self, max_episodes=1000):
        """
        The main training loop for the agent.

        For each episode:
        - Reset the environment to start a new episode.
        - Select actions using the epsilon-greedy policy.
        - Step the environment and observe next state and reward.
        - Store the experience in the replay buffer.
        - If enough experiences are collected, sample a batch and train the DQN.
        - Periodically update the target network to stabilize training.
        - Decay epsilon to reduce exploration over time.
        """
        start_time = time.time()
        for ep in range(max_episodes):
            done, episode_reward = False, 0
            state, _ = self.env.reset()  # Unpack observation and info from reset
            while not done:
                action = self.model.get_action(state)  # Select action using epsilon-greedy policy
                next_state, reward, terminated, truncated, _ = self.env.step(action)  # Step environment
                done = terminated or truncated  # Determine if episode ended
                self.buffer.add(state, action, reward * 0.01, next_state, done)  # Store experience
                episode_reward += reward
                state = next_state
            if self.buffer.size() >= batch_size:
                loss = self.replay_experience()  # Train from replay buffer
            else:
                loss = 0
            # Update the target network weights to keep training stable
            self.update_target()  # Placeholder call to keep structure

            elapsed = time.time() - start_time
            episodes_left = max_episodes - (ep + 1)
            eta = elapsed / (ep + 1) * episodes_left if ep >= 0 else 0
            print(f"Episoden#{ep} Gesammter Reward:{episode_reward:.2f} Loss:{loss:.4f} Epsilon:{self.model.epsilon:.3f} ETA:{int(eta)}s")


# SimWrapper class wraps environment to render on each step
class SimWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        self.render()  # Render environment for visualization
        return self.env.step(action)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = SimWrapper(env)  # Uncomment to enable rendering during training
    agent = Agent(env)
    agent.train(max_episodes=2000)
