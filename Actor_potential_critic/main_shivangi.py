import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mpe2 import simple_spread_v3
from networks import QNetwork, PotentialNetwork
from utils import ReplayBuffer, get_joint_obs
import config
import core
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# --- Initialize env ---
env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=config.MAX_CYCLES)
obs, _ = env.reset()
agents = env.agents
action_dims = {agent: env.action_space(agent).n for agent in agents}
single_obs_dim = len(next(iter(obs.values())))
joint_obs_dim = single_obs_dim * len(agents)
joint_action_dim = sum(action_dims.values())
#%%load q network from core and train phi function
buffer, q_nets, target_q_nets = \
    core.load_QNetwork(agents,joint_obs_dim,action_dims,joint_action_dim )
#%%load potential function
# potentials = [PotentialNetwork(joint_obs_dim + joint_action_dim).to(device) for _ in range(2)]
# target_potentials = [PotentialNetwork(joint_obs_dim + joint_action_dim).to(device) for _ in range(2)]


phi_net = PotentialNetwork(joint_obs_dim, agents).to(device)
phi_net_optimizer = optim.Adam(phi_net.parameters(), lr=1e-3)
#%%
# --- Training Loop ---
episode_rewards = []
for ep in range(config.NUM_EPISODES):
    loss_accum = 0
    num_total = 0
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}

    for step in range(config.MAX_CYCLES):
        joint_obs = get_joint_obs(obs, agents)
        # sample integer actions
        actions = {agent: env.action_space(agent).sample() for agent in agents}
        next_obs, rewards, dones, truncs, infos = env.step(actions)
        joint_next_obs = get_joint_obs(next_obs, agents)

        # store integer action indices
        joint_action_idxs = np.array([actions[ag] for ag in agents], dtype=np.int64)

        buffer.push(joint_obs,
                    joint_action_idxs,
                    rewards,
                    joint_next_obs,
                    any(dones.values()) or any(truncs.values()))
        for agent in agents:
            total_reward[agent] += rewards[agent]
        obs = next_obs

        if len(buffer) < config.BATCH_SIZE:
            continue

        # --- Sample and immediately cast to long on device ---
        s, a_idxs, r_dict, s_, d = buffer.sample(config.BATCH_SIZE)
        s   = s.to(device)
        s_  = s_.to(device)
        d   = d.to(device).float()
        a_idxs = a_idxs.to(device).long()

        loss = core.update_q_networks_phi_network(agents, joint_action_dim, joint_obs_dim,\
                                                  joint_obs,phi_net_optimizer,q_nets,\
                                          loss_accum, num_total, step)

    avg_reward = np.mean(list(total_reward.values()))
    # writer.add_scalar("avg_reward", avg_reward, ep)
    episode_rewards.append(avg_reward)
    if ep % 100 == 0:
        print(f"Ep {ep}, Avg reward: {avg_reward:.2f}")
env.close()
