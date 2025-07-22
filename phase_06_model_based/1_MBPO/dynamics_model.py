from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

class DynamicsNet(nn.Module):
    hidden_sizes: list[int]
    out_dim: int
    
    @nn.compact
    def __call__(self, x):
        for hid in self.hidden_sizes:
            x = nn.relu(nn.Dense(hid)(x))
            
        mu = nn.Dense(self.out_dim)(x)
        log_var = nn.Dense(self.out_dim)(x)

        return mu, log_var

class EnsembleDynamicModel:
    def __init__(self, cfg, rng, obs_space, act_space):
        ## To do: build cfg.model.dyn_ensemble_size networks + optim
         pass

    
    def predict(self, s, a):
        ## To do: forward pred (mean and var)
        pass
    
    def update(self, batch):
        ## To do: Jit compiled training step
        return {"dynamics/loss_mse": 0.0}