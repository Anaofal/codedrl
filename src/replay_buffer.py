# Fichier: src/replay_buffer.py
import numpy as np
import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        experience = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            bool(done)
        )
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones, dtype=np.uint8), dtype=torch.bool).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)