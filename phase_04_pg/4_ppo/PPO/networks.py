import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(256, 256), activation=nn.Tanh):
        super().__init__()
        
        layers = []
        dims = [input_dim] + list(hidden_size)
        
        for i in range(len(hidden_size)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        
        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)
        
        ##DEBUG
        # print(self.net)
        
    def forward(self, x):
        return self.net(x)
        
class GaussianPolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=(256, 256), activation=nn.Tanh):
        super().__init__()
        self.mlp = MLP(input_dim, action_dim, hidden_size, activation)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        mean = self.mlp(obs)
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim, 1, hidden_sizes)
    
    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)
