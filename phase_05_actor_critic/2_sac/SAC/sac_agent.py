import torch
import numpy as np
import torch.nn.functional as F
from .buffer import ReplayBuffer
from .networks import GaussianPolicy, ValueNetwork, QNetwork

class SacAgent:
    """
    Soft Actor-Critic agent with twin critics and value targets
    """
    def __init__(self, state_dim, action_dim, device, cfg):
        self.device = device
        self.gamam = cfg["gamma"]
        self.tau = cfg["tau"]
        self.alpha = cfg["alpha"]
        self.automatic_entropy_tuning = cfg["automatic_entropy_tuning"]
        hidden = cfg["hidden_sizes"]
        
        ## Actor
        self.policy = GaussianPolicy(state_dim, action_dim, hidden).to(device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr = cfg["policy_lr"])
        
        ## Critics
        self.q1 = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr = cfg["q_lr"])
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr = cfg["q_lr"])
        
        ## Value networks
        self.value = ValueNetwork(state_dim, hidden).to(device)
        self.value_target = ValueNetwork(state_dim, hidden).to(device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg["value_lr"])

        ## Auto Entropy as per paper
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["policy_lr"])

    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        s, a, r, s_nxt, d = replay_buffer.sample(batch_size, self.device)
        
        
        ## Value Loss
        with torch.no_grad():
            a_sample, logp_a = self.policy.sample(s)
            q1_pi = self.q1(s, a_sample)
            q2_pi = self.q2(s, a_sample)
            min_q_pi = torch.min(q1_pi, q2_pi)
            v_target = (min_q_pi - self.alpha * logp_a)
        
        v_pred = self.value(s)
        value_loss = F.mse_loss(v_pred, v_target)

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        
        
        ## Q1/Q2  Loss
        with torch.no_grad():
            v_next = self.value_target(s_nxt)
            q_target = r + (1.0 - d) * self.gamam * v_next

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q1_opt.step()

        ## Policy Loss
        a_new, logp_new = self.policy.sample(s)
        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)
        min_q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * logp_new - min_q_new).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for param, targ in zip(self.value.parameters(), self.value_target.parameters()):
            targ.data.mul_(1.0 - self.tau)
            targ.data.add_(self.tau * param.data)

        return {
            "value_loss": value_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
        }