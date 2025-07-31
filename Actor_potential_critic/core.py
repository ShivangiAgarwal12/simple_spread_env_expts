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
import pdb
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%load Q Network
def load_QNetwork(agents, joint_obs_dim,action_dims,joint_action_dim):
    # --- Networks ---
    q_nets = {}
    target_q_nets = {}
    # pdb.set_trace()
    for agent in agents:
        q_list, target_list = [], []
        # for i in range(2):
        q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
        target_q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
        
        os.path.abspath(os.curdir)
        os.chdir("..")
        
        ckpt = os.path.join(os.path.abspath(os.curdir), "joint_state_models", f"{agent}_qnet.pt")
        # ckpt = os.path.join("joint_state_models", f"{agent}_qnet.pt")
        if os.path.isfile(ckpt):
            q.load_state_dict(torch.load(ckpt, map_location=device))
            target_q.load_state_dict(q.state_dict())
            # pdb.set_trace()
            print("Q loaded")
        q_list.append(q)
        target_list.append(target_q)
        q_nets[agent] = q_list
        target_q_nets[agent] = target_list
    pdb.set_trace()
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

def update_q_networks_phi_network(agents, num_actions, state_dim, states,p_opts,q_nets):
    '''
    update here both q and phi functions

    Returns
    -------
    None.

    '''
    potential_net = PotentialNetwork(state_dim, agents)
    for agent in range(agents):
        for _ in range(config.samples_per_agent):
            #sample actions
            a_i = random.randint(0,num_actions - 1)
            
            #action of agent not taken 
            a_i_tilde = random.randint(0, num_actions - 1)
            
            #action for others
            a_minus_i = [random.randint(0, num_actions-1) for _ in range(agents-1)]
            
            #form full joint actions of other agents
            a_joint = list(a_minus_i)
            a_joint.insert(agent, a_i)
            
            a_joint_tilde = list(a_minus_i)
            a_joint_tilde.insert(agent, a_i_tilde)
            
            #convert to tensors
            a_joint_tensor = torch.tensor(a_joint).float().repeat(config.batch_size, 1)
            a_joint_tilde_tensor = torch.tensor(a_joint_tilde).float().repeat(config.batch_size, 1)
            
            #compute potential function
            phi_a = potential_net(states, a_joint_tensor)
            phi_a_tilde = potential_net(states, a_joint_tilde_tensor)
            
            #define here for q function
            q_net = q_nets[agent]
            
            with torch.no_grad():
                q_a = q_net(states, a_joint_tensor)
                q_a_tilde = q_net(states, a_joint_tilde_tensor)
                y_val= q_a - q_a_tilde
            
            # diff = phi_a - phi_a_tilde - y_val
            diff_1 = phi_a - phi_a_tilde
            diff_2 = y_val
            
            loss = nn.MSELoss()(diff_1, diff_2.detach())
            p_opts.zero_grad()
            loss.backward()
            # writer.add_scalar(f"{agent}/loss", loss.item(), episode * MAX_CYCLES + step)
            p_opts.step()
            # loss_accum += (diff ** 2).sum()
            # num_total += batch_size            
            
















