# -*- coding: utf-8 -*-
"""
code to load q networks and train potential phi function
@author: sande
"""
#%%
import torch
import torch.nn as nn
import os
from networks import QNetwork, PotentialNetwork
import torch.optim as optim
import config
from utils import ReplayBuffer, get_joint_obs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%load Q Network
def load_QNetwork(agents, joint_obs_dim,action_dims,joint_action_dim):
    # --- Networks ---
    q_nets = {}
    target_q_nets = {}
    for agent in agents:
        q_list, target_list = [], []
        for i in range(2):
            q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
            target_q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
            ckpt = os.path.join("joint_state_models", f"{agent}_q{i}.pth")
            if os.path.isfile(ckpt):
                q.load_state_dict(torch.load(ckpt, map_location=device))
                target_q.load_state_dict(q.state_dict())
                print("Q loaded")
            q_list.append(q)
            target_list.append(target_q)
        q_nets[agent] = q_list
        target_q_nets[agent] = target_list

    potentials = [PotentialNetwork(joint_obs_dim + joint_action_dim).to(device) for _ in range(2)]
    target_potentials = [PotentialNetwork(joint_obs_dim + joint_action_dim).to(device) for _ in range(2)]

    # Sync targets
    for agent in agents:
        for i in range(2):
            target_q_nets[agent][i].load_state_dict(q_nets[agent][i].state_dict())
    for i in range(2):
        target_potentials[i].load_state_dict(potentials[i].state_dict())

    # --- Optimizers ---
    q_opts = {agent: [optim.Adam(q.parameters(), lr=config.LR_Q) for q in q_nets[agent]] for agent in agents}
    p_opts = [optim.Adam(p.parameters(), lr=config.LR_P) for p in potentials]

    # --- Replay Buffer ---
    buffer = ReplayBuffer(capacity=100000)
    
    return buffer, q_nets, target_q_nets,q_opts, p_opts, potentials, target_potentials