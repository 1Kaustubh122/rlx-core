from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn

class PolicyNetwork(nn.Module):
    hidden_sizes : list[int]
    act_dim: int
    
    @nn.compact
    def __call__(self, x, rng):
        for hid in self.hidden_sizes:
            x = nn.tanh(nn.Dense(hid)(x))

        mu = nn.Dense(self.act_dim)(x)
        log_std = nn.Dense(self.act_dim)(x)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        std = jnp.exp(log_std)
        
        key, sub_k = jax.random.split(rng)

        action = mu + std * jax.random.normal(sub_k, shape=mu.shape)
        
        return jnp.tanh(action), mu, log_std
    
    