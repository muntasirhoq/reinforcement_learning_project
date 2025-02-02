# reinforce_cartpole.py

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Set up device for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# 2. Function to select an action based on the policy network's output
def select_action(policy_net, state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

# 3. Main training loop for REINFORCE
def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
    num_episodes = 1000
    gamma = 0.99  # Discount factor

    all_rewards = []  # To store total rewards per episode

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            # Choose an action and record its log probability
            action, log_prob = select_action(policy_net, state)
            next_state, reward, done, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        # Calculate discounted rewards (returns)
        returns = []
        G = 0
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate policy loss and update policy network
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Print progress every 50 episodes
        if episode % 50 == 0:
            print(f'Episode {episode}\tTotal Reward: {total_reward}')

    env.close()
    return all_rewards

if __name__ == "__main__":
    rewards = train()

    # Plot rewards over episodes
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE on CartPole-v1')
    plt.show()
