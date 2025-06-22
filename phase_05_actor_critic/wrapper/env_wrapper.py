import numpy as np
from dm_control import suite

class DMControlEnv:
    """
    DM Control wrapper -> Flattern obs
    """
    
    def __init__(self, domain: str, task: str):
        self.env = suite.load(domain_name=domain, task_name=task)
        self.specs = self.env.action_spec()
        obs_example = self.env.reset().observation
        self.obs_key = list(obs_example.keys())
        self.obs_dim = sum(v.size for v in obs_example.values())
        self.action_dim = np.prod(self.specs.shape)

    def _flatten_obs(self, obs_dict) -> np.ndarray:
        return np.concatenate([obs_dict[k].ravel() for k in self.obs_key], dtype=np.float32)
    
    def reset(self):
        time_s = self.env.reset()
        return self._flatten_obs(time_s.observation)
    
    def step(self, action: np.ndarray):
        ts = self.env.step(action)
        obs = self._flatten_obs(ts.observation)
        reward = np.array([ts.reward], dtype=np.float32)
        done = np.array([ts.last()], dtype=np.float32)
        return obs, reward, done