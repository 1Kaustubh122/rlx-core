import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    This is the Multi-purpose MLP builder for both the Actor and the Critic.
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=(400, 300), activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)
        
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
            
        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    """
    Actor will be a deterministic policy network which outputs actions in between [-1, 1] via Tanh.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(400, 300)):
        super().__init__()
        self.mlp = MLP(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs):
        """
        Returns actions (tanh output).
        """
        return torch.tanh(self.mlp(obs))

class Critic(nn.Module):
    """
    Q(s, a) value network. Takes state and action as input.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(400, 300)):
        super().__init__()
        self.mlp =  MLP((obs_dim + act_dim), 1, hidden_sizes)

    def forward(self, obs, action):
        """
        Returns scalar Q-value.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.mlp(torch.cat([obs, action], dim=-1))
