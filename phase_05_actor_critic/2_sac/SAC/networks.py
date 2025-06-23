import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2

def mlp(in_dim: int, hidden_sizes : list[int], out_dim: int, activation = nn.ReLU) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    
    for hl in hidden_sizes:
        layers.append(nn.Linear(last_dim, hl))
        layers.append(activation())
        last_dim = hl
    
    layers.append(nn.Linear(last_dim, out_dim))
    
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim:int, action_dim : int, hidden: list[int]):
        super().__init__()
        self.net = mlp(state_dim, hidden, 2 * action_dim)
        self.action_dim = action_dim
    
    def forward(self, state):
        mu_logstd = self.net(state)  ## mean -> where the action should center and log-std -> how wide the bell curve is
        mu, log_std = torch.chunk(mu_logstd, 2, -1)  ## First halg mu (mean) and second half is log (sigma)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) ## force them info defned range
        std = log_std.exp() ## Real standard dev (always +ve)
        dist = Normal(mu, std)
        return dist
    
    def sample(self, state):
        dist = self.forward(state)
        eps = dist.rsample()
        action = torch.tanh(eps)
        log_prob = dist.log_prob(eps) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden :list[int]):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden : list[int]):
        super().__init__()
        self.net = mlp(state_dim, hidden, 1)
    
    def forward(self, state):
        return self.net(state)
