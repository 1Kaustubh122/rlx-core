import torch
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_action=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done, next_action)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_actions = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        return states, actions, rewards, next_states, dones, next_actions

    def __len__(self):
        return len(self.buffer)
    
    
class PriortizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment_per_sampling
        self.epsilon = epsilon

        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def push(self, state, action, reward, next_state, done):
        max_prior = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prior
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        
        if len(self.memory) == self.capacity:
            priorties = self.priorities
        else:
            priorties = self.priorities[:self.pos]

        probs = (priorties + self.epsilon) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        samples = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
                
        
        states = torch.stack(states).to(device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        total = len(self.memory)
        weights = (total *probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self,indicies, new_prior):
        for idx, prior in zip(indicies, new_prior):
            self.priorities[idx] = prior

    def __len__(self):
        return len(self.memory)
        
        
class RainbowReplayBuffer:
    def __init__(self, capacity, nstep, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment_per_sampling
        self.epsilon = epsilon

        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.nstep = nstep
        self.nstep_buffer = deque(maxlen=nstep)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def push(self, nstep, state, action, reward, next_state, done):
        self.nstep_buffer.append((state, action, reward, next_state, done))
        
        if len(self.nstep_buffer) < self.nstep:
            return
        
        n_state, n_action = self.nstep_buffer[0][:2]
        n_reward, n_next_state, n_done = self._get_nstep_info()

        max_prior = self.priorities.max() if self.memory else 1.0
        data = (n_state, n_action, n_reward, n_next_state, done)

        if len(self.memory) < self.capacity:
            self.memory.append(data)
        else:
            self.memory[self.pos] = data

        self.priorities[self.pos] = max_prior
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        
        if len(self.memory) == self.capacity:
            priorties = self.priorities
        else:
            priorties = self.priorities[:self.pos]

        probs = (priorties + self.epsilon) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        samples = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
                
        
        states = torch.stack(states).to(device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        total = len(self.memory)
        weights = (total *probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self,indicies, new_prior):
        for idx, prior in zip(indicies, new_prior):
            self.priorities[idx] = prior
            
    def _get_nstep_info(self):
        reward, next_state, done = self.nstep_buffer[-1][-3:]
        
        for transition in reversed(list(self.nstep_buffer)[:-1]):
            r, n_s, d = transition[2:]
            reward = transition[2] + self.alpha * reward * (1 - transition[4])
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return reward, next_state, done
    
    def __len__(self):
        return len(self.memory)
        