import torch
import numpy as np
import gymnasium as gym

class IsaacAnymalNavVecEnv:
    def __init__(self, env_name = "QuadrupedClutteredNavigation-v0", num_envs =64, device="cuda", seed=None, enable_viewport=False):
        self.env = gym.make(
            env_name, 
            num_envs=num_envs,
            sim_device=device, 
            rl_device=device,
            enable_viewport=enable_viewport
        )
        
        self.num_envs = num_envs
        self.device=device
        
        if seed is not None:
            self.env.reset(seed=seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._process_obs(obs), info
    
    def step(self, actions):
        actions_np = actions.detach().cpu().numpy()
        actions_np = np.clip(actions_np, self.action_shape.low, self.action_shape.high)
        obs, reward, terminated, truncated, info = self.env.step(actions_np) 
        done = np.logical_or(terminated or truncated)

        return (
            self._process_obs(obs),
            torch.as_tensor(reward, device=self.device, dtype=torch.float32),
            torch.as_tensor(done, device=self.device, dtype=torch.bool),
            info
        )
    
    def _process_obs(self, obs):
        if isinstance(obs, dict):
            return {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in obs.items()}
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def render(self, mode="human"):
        if hasattr(self.env, "render"):
            self.env.render(mode=mode)
        
    def close(self):
        self.env.close()