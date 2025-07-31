# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:09:21 2025

@author: sande
"""

# --- Hyperparameters ---
GAMMA = 0.99
LR_Q = 1e-4
LR_P = 1e-4
BATCH_SIZE = 128
NUM_EPISODES = 5000
MAX_CYCLES = 50
TARGET_UPDATE_FREQ = 100
TAU = 0.05

#%%save file
save_file = False

#samples to calculate actions
samples_per_agent = 50