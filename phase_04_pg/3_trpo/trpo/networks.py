import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128), activation=nn.Tanh):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)

        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers)
        
        ## DEBUG
        # print(self.net)
        
    def forward(self, x):
        return self.net(x)
    

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.mlp = MLP(obs_dim, action_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mean = self.mlp(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp
    
    def get_log_prob(self, obs, action):
        mean, std =  self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action).sum(axis=-1)
        return logp
    
    def kl_divergence(self, obs, other_policy):
        mean0, std0 = other_policy.forward(obs)
        mean1, std1 = self.forward(obs)
        
        var0 = std0 ** 2
        var1 = std1 ** 2
        
        kl = (torch.log(std1/std0) + (var0 + (mean0 - mean1) ** 2) / (2 * var1) - 0.5).sum(axis=-1) 
        return kl
    
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.mlp = MLP(obs_dim, 1, hidden_sizes)
    
    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)
