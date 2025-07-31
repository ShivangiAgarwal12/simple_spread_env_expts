# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 23:02:17 2025
Neural network for QNetwork and replay buffer
@author: sande
"""
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# Q-network definition
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
#%%
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, joint_obs, action, reward, joint_next_obs, done):
        self.buffer.append((joint_obs, action, reward, joint_next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32).to(device),
            torch.tensor(a).to(device),
            torch.tensor(r).to(device),
            torch.tensor(np.array(s_), dtype=torch.float32).to(device),
            torch.tensor(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)
#%%
# epsilon greedy action selection
def select_action(q_net, joint_obs, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        q_vals = q_net(torch.tensor(joint_obs, dtype=torch.float32).unsqueeze(0).to(device))
        # pdb.set_trace()
        return torch.argmax(q_vals).item()

# Helper to get joint obs
def get_joint_obs(obs_dict, agent_order):
    return np.concatenate([obs_dict[agent] for agent in agent_order])

