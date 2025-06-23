import torch
import torch.nn as nn
from typing import Tuple, List
from __future__ import annotations


LOG_STD_MIN = -20
LOG_STD_MAX = 2

def mlp(in_dim: int, hidden_sizes : Tuple[int, ...], out_dim: int, activation = nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = in_dim
    
    for hl in hidden_sizes:
        layers += [nn.Linear(last_dim, hl), activation]
        last_dim = hl
    
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim:int, action_dim : int, hidden: Tuple[int, ...]):
        super().__init__()
        self.backbone = mlp(state_dim, hidden, hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)
    
    def forward(self, state):
        h = self.backbone(state)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        return mu, std  ## Mean and StD
    
    def sample(self, state):
        mu, std = self(state)
        dist = torch.distributions.Normal(mu, std)
        eps = dist.rsample()
        a = torch.tanh(eps)
        logp = dist.log_prob(eps) - torch.log1p(-a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)
    
    @torch.no_grad()
    def act(self, s, determinstic:bool = False):
        mu, std = self(s)
        if determinstic:
            eps = mu
        else:
            eps = torch.distributions.Normal(mu, std).sample()
        return torch.tanh(eps)
        
