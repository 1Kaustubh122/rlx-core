from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

from .policy import mlp

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden :Tuple[int, ...]):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class QEnsemble(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden : Tuple[int, ...], n: int):
        super().__init__()
        self.qs = nn.ModuleList([QNetwork(state_dim, act_dim, hidden) for _ in range(n)])
        self.n = n
        
    def _one_q(self, idx, state, action):
        return self.qs[idx](state, action) 
        
    def forward_all(self, state, act):
        qs = [q(state, act) for q in self.qs]
        qs = torch.cat(qs, dim=1)
        return qs  
    
    @torch.no_grad()
    def target_min(self, state, action, indx):
        qs = torch.stack([self.qs[i](state, action) for i in indx], dim=1)
        return qs.min(dim=1, keepdim=True)[0]
