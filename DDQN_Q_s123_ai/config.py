# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:46:41 2025

@author: sande
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
save_file = False

plot_file = False

log_data = False

#%%
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