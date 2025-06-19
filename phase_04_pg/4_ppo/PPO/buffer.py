import torch

class TrajectoryBuffer:
    obs_buf: torch.Tensor
    act_buf: torch.Tensor
    rew_buf: torch.Tensor
    done_buf: torch.Tensor
    logp_buf: torch.Tensor
    val_buf: torch.Tensor
    adv_buf: torch.Tensor
    ret_buf: torch.Tensor
    
    def __init__(self, obs_dim, action_dim, size, device="cuda", gamma = 0.99, gae_lambda=0.97):

        s_n_o = (size, obs_dim)
        s_n_a = (size, action_dim)
        s_n = (size)

        buffer_specs = {
            'obs_buf' : s_n_o,
            'act_buf' : s_n_a,
            'rew_buf' : s_n,
            'done_buf' : s_n,
            'logp_buf' : s_n,
            'val_buf' : s_n,
            'adv_buf' : s_n,
            'ret_buf' : s_n,            
        }
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.size = size
        self.ptr = 0
     
        for name, shape in buffer_specs.items():
            setattr(self, name, torch.zeros(shape, dtype=torch.float32, device=device))
            
    def store(self, obs, act, rew, done, logp, val):
        assert obs.ndim == 1, f"obs shape: {obs.shape}"
        assert act.ndim == 1, f"act shape: {act.shape}"
        
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.done_buf[idx] = done
        self.logp_buf[idx] = logp
        self.val_buf[idx] = val
        self.ptr += 1
    
    def finish_path(self, last_val):
        path_slice = slice(0, self.ptr) 
        rews = torch.cat((self.rew_buf[path_slice], last_val.unsqueeze(0)), dim=0)
        vals = torch.cat((self.val_buf[path_slice], last_val.unsqueeze(0)), dim=0)
        dones = torch.cat([self.done_buf[path_slice], torch.zeros(1, device=last_val.device)], dim=0)

        adv = torch.zeros_like(self.rew_buf[path_slice])
        last_gae_lam = 0
        for t in reversed(range(self.ptr)):
            nonterminal = 1.0 - dones[t+1] 
            delta = rews[t] + self.gamma * vals[t+1] * nonterminal - vals[t]
            adv[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * nonterminal * last_gae_lam
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + self.val_buf[path_slice]
        
    def get(self):
        adv = self.adv_buf[:self.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (
            self.obs_buf[:self.ptr],
            self.act_buf[:self.ptr],
            self.logp_buf[:self.ptr],
            adv,
            self.ret_buf[:self.ptr]
        )
    
    def reset(self):
        self.ptr = 0