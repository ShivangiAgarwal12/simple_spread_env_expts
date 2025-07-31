#%%
'''
Updates done: added and updated Q function for DQN
'''
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3  
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 100
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 1 #0.999 -- no decay, so epsilon stays as 1 throughout
NUM_EPISODES = 5000
MAX_CYCLES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config.log_data:
# Logging and model directories
    log_dir = "./joint_state_logs"
    os.makedirs(log_dir, exist_ok=True)
    reward_log_path = os.path.join(log_dir, "reward_log.npy")
    model_dir = "./joint_state_models"
    os.makedirs(model_dir, exist_ok=True)
    # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir)

episode_rewards = []

#%%
# Initialize environment
env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=config.MAX_CYCLES)
obs, _ = env.reset()
# pdb.set_trace()
agents = env.agents
action_spaces = {agent: env.action_space(agent).n for agent in agents}

# Observation dimensions
# pdb.set_trace()
single_obs_dim = len(next(iter(obs.values())))
joint_obs_dim = single_obs_dim * len(agents)
#%%initialise networks
# Q-networks, target networks, optimizers, buffers

q_nets = {agent: core.QNetwork(joint_obs_dim, action_spaces[agent]).to(device) for agent in agents}

# pdb.set_trace()
target_nets = {agent: core.QNetwork(joint_obs_dim, action_spaces[agent]).to(device) for agent in agents}
optimizers = {agent: optim.Adam(q_nets[agent].parameters(), lr=config.LR) for agent in agents}
buffers = {agent: core.ReplayBuffer(config.REPLAY_BUFFER_SIZE) for agent in agents}

# Sync target nets
for agent in agents:
    target_nets[agent].load_state_dict(q_nets[agent].state_dict())

epsilon = config.EPS_START

# Training loop
for episode in range(config.NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}

    for step in range(config.MAX_CYCLES):
        joint_obs = core.get_joint_obs(obs, agents)
        actions = {}

        for agent in agents:
            actions[agent] = core.select_action(q_nets[agent], joint_obs, epsilon, env.action_space(agent))
        # pdb.set_trace()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        joint_next_obs = core.get_joint_obs(next_obs, agents)

        for agent in agents:
            buffers[agent].push(
                joint_obs, actions[agent], rewards[agent], joint_next_obs, terminations[agent] or truncations[agent]
            )
            total_reward[agent] += rewards[agent]

        obs = next_obs
        
        for agent in agents:
            if len(buffers[agent]) < config.BATCH_SIZE:
                continue
            pdb.set_trace()
            s, a, r, s_, done = buffers[agent].sample(config.BATCH_SIZE)
            q_vals = q_nets[agent](s).gather(1, a.unsqueeze(1)).squeeze()
            # pdb.set_trace()
            next_actions = torch.argmax(q_nets[agent](s_), dim=1)
            next_q_vals = target_nets[agent](s_).mean(dim=1)
            # next_q_vals = target_nets[agent](s_).gather(1, next_actions.unsqueeze(1)).squeeze()
            target = r + config.GAMMA * next_q_vals * (1 - done.float())

            loss = nn.MSELoss()(q_vals, target.detach())
            optimizers[agent].zero_grad()
            loss.backward()
            # writer.add_scalar(f"{agent}/loss", loss.item(), episode * MAX_CYCLES + step)
            optimizers[agent].step()

        if step % config.TARGET_UPDATE_FREQ == 0:
            for agent in agents:
                target_nets[agent].load_state_dict(q_nets[agent].state_dict())

    # epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)
    avg_reward = np.mean([total_reward[agent] for agent in agents])
    # writer.add_scalar("avg_reward", avg_reward, episode)
    episode_rewards.append(avg_reward)

    if episode % 100 == 0:
        print(f"Episode {episode} | Avg reward: {avg_reward:.2f} | epsilon: {epsilon:.3f}")

writer.close()
#%%
# Save
if config.save_file:
    np.save(reward_log_path, episode_rewards)
    for agent in agents:
        torch.save(q_nets[agent].state_dict(), os.path.join(model_dir, f"{agent}_qnet.pt"))

# Plot
if config.plot_file:
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Multi-Agent DDQN with Joint State")
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "learning_curve.png"))
    plt.close()

env.close()

# after training, to view logs:
# tensorboard --logdir=joint_state_logs
