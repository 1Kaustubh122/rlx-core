from __future__ import annotations
import isaacsim
import numpy as np
import gymnasium as gym

class IsaacAsemblyEnv(gym.Env):
    metadata = {"render_modes" : ["none"]}
    
    def __init__(self, cfg):
        super().__init__()
        
        ## To load stage and set up physics
        
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (cfg.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (cfg.act_dim,), np.float32)
        
    def reset(self, *, seed:int | None = None, options = None):
        ## To do: Reset sim, and return the first observationspace (np.adarray)
        pass
    
    def step(self, action):
        ## To do: apply action, step sim, return obs, reward, term, trunc, indo
        pass
