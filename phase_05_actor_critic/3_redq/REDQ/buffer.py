import torch
import numpy as np

class ReplayBuffer:
    """
    It's a Fixed-size cyclic buffer for off policy experience storage
    """
    
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.ptr = 0
        self.full = False
        self.capacity = capacity
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True
    
    def sample(self, batch_size : int, device: torch.device):
        max_idx = self.capacity if self.full else self.ptr
        
        idxs = np.random.randint(0, max_idx, size=batch_size)
        batch = (
            torch.tensor(self.states[idxs],device=device, dtype=torch.float32),
            torch.tensor(self.actions[idxs], device=device, dtype=torch.float32),
            torch.tensor(self.rewards[idxs], device=device, dtype=torch.float32),
            torch.tensor(self.next_states[idxs], device=device, dtype=torch.float32),
            torch.tensor(self.dones[idxs], device=device, dtype=torch.float32),
        )
        
        return batch
    
    def __len__(self):
        return self.capacity if self.full else self.ptr