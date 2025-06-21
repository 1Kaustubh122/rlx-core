import torch
import numpy as np


class ReplayBuffer:
    """
    It's a standard experience replay buffer for off-policy RL algorithms like this one (TD3).
    """
    
    def __init__(self, obs_dim, action_dim, size, device="cuda"):
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, ), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((size, ), dtype=torch.float32, device=device)

        self.device = device
        self.ptr = 0
        self.size = size
        self.full = False
    
    def add(self, obs, action, reward, next_obs, done):
        """
        Storing single transition in the buffer
        """
        
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.from_numpy(next_obs).float().to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
            
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        
        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True
        
    def sample(self, batch_size):
        """
        Returns a batch of (obs, act, rew, next_obs, done)
        """
        max_size = self.size if self.full else self.ptr
        idxs = torch.randint(0, max_size, (batch_size,), device=self.device)

        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs]
        )
    
    def __len__(self):
        """
        Returns the size of the buffer
        """
        return self.size if self.full else self.ptr
    
    def reset(self):
        """
        Resetting the buffer's pointer 
        """
        self.ptr = 0
        self.full = False