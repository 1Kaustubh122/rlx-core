import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from .q_ensemble import QEnsemble
from dataclasses import dataclass
from __future__ import annotations
from .policy import GaussianPolicy
import math, copy, torch, numpy as np


@dataclass
class REDQConfig:
    tau: float = 0.005
    alpha: float = 0.2
    gamma: float = 0.99
    auto_alpha:bool = True
    target_entropy_mult: float = -1.0

    q_lr: float = 3e-4
    policy_lr: float = 3e-4
    hidden:Tuple[int, ...] = (256, 256)

    batch: int = 256
    min_qs: int = 2      # M
    ensemble: int = 10   # N
    utd_ratio: int = 20  # G
    grad_clip: float = 1.0

class REDQAgent:
    def __init__(self, obs_dim: True, act_dim :int, device: torch.device, cfg: REDQConfig):
        self.cfg, self.device = cfg, device
        self.policy = GaussianPolicy(obs_dim, act_dim, cfg.hidden).to(device)
        self.q_ens = QEnsemble(obs_dim, act_dim, cfg.hidden, cfg.ensemble).to(device)
        self.targ_ens = copy.deepcopy(self.q_ens).eval().to(device)

        for p in self.targ_ens.parameters():
            p.requires_grad_(False)

        self.pol_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.policy_lr)
        self.q_opt = torch.optim.Adam(self.q_ens.parameters(), lr=cfg.q_lr)

        self.log_alpha = torch.tensor(math.log(cfg.alpha), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.policy_lr)

        self.tgt_entropy = cfg.target_entropy_mult * act_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _update_once(self, batch):
        s, a, r, n_s, d = batch

        with torch.no_grad():
            n_a, n_logp = self.policy.sample(n_s)
            idx = torch.randperm(self.cfg.ensemble, device=self.device)[: self.cfg.min_qs]
            q_min = self.targ_ens.target_min(n_s, n_a, idx)
            y = r + (1-d) * self.cfg.gamma * (q_min - self.alpha * n_logp)
        
        q_pred = self.q_ens.forward_all(s, a)
        q_loss = ((q_pred - y).pow(2)).mean()
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_ens.parameters(), self.cfg.grad_clip)
        self.q_opt.step()
        
        new_a, logp = self.policy.sample(s)
        q_avg = self.q_ens.forward_all(s, new_a).mean(1, keepdim=True)
        pol_loss = (self.alpha.detach() * logp - q_avg).mean()
        self.pol_opt.zero_grad()
        pol_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.pol_opt.step()
        
        if self.cfg.auto_alpha:
            a_loss = (-(self.log_alpha) * (logp + self.tgt_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); a_loss.backward(); self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q_ens.parameters(), self.targ_ens.parameters()):
                pt.data.lerp_(p.data, self.cfg.tau)
        return q_loss.item(), pol_loss.item()

    def update(self, batch):
        q_losses, p_losses = [], []
        for _ in range(self.cfg.utd_ratio):
            ql, pl = self._update_once(batch)
            q_losses.append(ql); p_losses.append(pl)
        return {
            "q_loss": float(np.mean(q_losses)),
            "policy_loss": float(np.mean(p_losses)),
            "alpha": float(self.alpha.item()),
        }

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False):
        s = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        return self.policy.act(s, deterministic).cpu().numpy()[0]
    