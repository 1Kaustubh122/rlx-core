from __future__ import annotations
import jax
import jax.numpy as jnp

from replay_buffer import ReplayBuffer
from policy_network import PolicyNetwork
from dynamics_model import EnsembleDynamicModel

class MBPOAgent:
    def __init__(self, cfg, rng, obs_space, act_space):
        self.cfg = cfg
        self.rng = rng
        self.real_buffer = ReplayBuffer(cfg)
        self.model_buffer = ReplayBuffer(cfg)
        self.dynamic_mod = EnsembleDynamicModel(cfg, rng, obs_space, act_space)
        self.policy_net = PolicyNetwork(cfg, rng, obs_space, act_space)
    
    
    def train(self, env) -> dict[str, float]:
        ## TO do: collect real data, update dynamics, model rollouts, policy updates
        pass
    
    def evaluate(self, env, episodes: int = 5):
        ## To do: Eval the policy in real env (no grad)
        pass

    def save(self, path: str):
        ## TO do: check point policy + dynamic params
        pass