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

if config.save_file:
    log_dir = "./apc_logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

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
buffer, q_nets, target_q_nets, q_opts, p_opts, potentials, target_potentials = core.load_QNetwork()
#%%
# --- Training Loop ---
episode_rewards = []
for ep in range(config.NUM_EPISODES):
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

        # --- Q update per agent ---
        ## need to check this parrt, its giving dim issues
        for agent_i, agent in enumerate(agents):
            act_idx = a_idxs[:, agent_i]                # LongTensor [B]
            for i in range(2):
                q_vals = q_nets[agent][i](s)            # [B, A]
                q_pred = q_vals.gather(1, act_idx.unsqueeze(1)).squeeze(1)  # [B]

                with torch.no_grad():
                    next_vals = target_q_nets[agent][1-i](s_)  # [B, A]
                    next_q    = next_vals.gather(1, act_idx.unsqueeze(1)).squeeze(1)
                    target    = r_dict[agent].to(device) + config.GAMMA * next_q * (1 - d)

                loss_q = nn.MSELoss()(q_pred, target)
                q_opts[agent][i].zero_grad()
                loss_q.backward()
                q_opts[agent][i].step()
                # writer.add_scalar(f"loss/{agent}_q{i}", loss_q.item(), ep*config.MAX_CYCLES + step)

        # --- Potential update ---
        # rebuild one‐hot on the (now long) indices
        joint_acts_oh = torch.cat([
            nn.functional.one_hot(a_idxs[:, ai], num_classes=action_dims[ag])
            for ai, ag in enumerate(agents)
        ], dim=1).float().to(device)

        for i in range(2):
            inp_curr = torch.cat([s, joint_acts_oh], dim=1)
            inp_next = torch.cat([s_, joint_acts_oh], dim=1)
            pot_curr = potentials[i](inp_curr).squeeze()
            pot_next = target_potentials[1-i](inp_next).squeeze()

            q_term = sum(
                q_nets[ag][i](s).gather(1, a_idxs[:, j].unsqueeze(1)).squeeze(1)
                - q_nets[ag][i](s_).gather(1, a_idxs[:, j].unsqueeze(1)).squeeze(1)
                for j, ag in enumerate(agents)
            )

            pot_loss = nn.MSELoss()(pot_curr - pot_next, q_term)
            p_opts[i].zero_grad()
            pot_loss.backward()
            p_opts[i].step()
            # writer.add_scalar(f"loss/potential{i}", pot_loss.item(), ep*config.MAX_CYCLES + step)

        # --- Soft updates ---
        if step % config.TARGET_UPDATE_FREQ == 0:
            for agent in agents:
                for i in range(2):
                    for tp, p in zip(target_q_nets[agent][i].parameters(),
                                     q_nets[agent][i].parameters()):
                        tp.data.copy_(config.TAU*p.data + (1-config.TAU)*tp.data)
            for i in range(2):
                for tp, p in zip(target_potentials[i].parameters(),
                                 potentials[i].parameters()):
                    tp.data.copy_(config.TAU*p.data + (1-config.TAU)*tp.data)

    avg_reward = np.mean(list(total_reward.values()))
    # writer.add_scalar("avg_reward", avg_reward, ep)
    episode_rewards.append(avg_reward)
    if ep % 100 == 0:
        print(f"Ep {ep}, Avg reward: {avg_reward:.2f}")

writer.close()
env.close()
