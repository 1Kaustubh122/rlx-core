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
    
    def __init__(self, obs_dim, action_dim, size, num_envs, device="cuda", gamma = 0.99, gae_lambda=0.97):

        s_n_o = (size, num_envs, obs_dim)
        s_n = (size, num_envs)

        buffer_specs = {
            'obs_buf' : s_n_o,
            'act_buf' : s_n_o,
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
        self.num_envs = num_envs
        self.ptr = 0
     
        for name, shape in buffer_specs.items():
            setattr(self, name, torch.zeros(shape, dtype=torch.float32, device=device))
            
    
    def store(self, obs, act, rew, done, logp, val):
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.logp_buf[self.ptr] = logp
        self.val_buf[self.ptr] = val
        self.ptr += 1
        
    def is_full(self):
        return self.ptr >= self.size
    
    def finish_path(self, last_val):
        ptr = self.ptr
        adv = torch.zeros((ptr+1, self.num_envs), device=last_val.device)
        rew = self.rew_buf[:ptr]
        val = self.val_buf[:ptr]
        done = self.done_buf[:ptr]
        adv[-1] = last_val
        
        ## GAE (Generalized Advantage Estimation)

        for t in reversed(range(ptr)):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + self.gamma * val[t+1] * nonterminal - val[t] if t+1 < ptr else \
                rew[t] + self.gamma * last_val * nonterminal - val[t]
            adv[t] = delta + self.gamma * self.gae_lambda * adv[t+1] * nonterminal
        
        self.adv_buf[:ptr] = adv[:ptr]
        self.ret_buf[:ptr] = self.adv_buf[:ptr] + self.val_buf[:ptr]
        self._flatten_for_update()
        
    def _flatten_for_update(self):
        self.obs_flat = self.obs_buf[:self.ptr].reshape(-1, self.obs_buf.size(-1))
        self.act_flat = self.act_buf[:self.ptr].reshape(-1, self.obs_buf.size(-1))
        self.logp_flat = self.logp_buf[:self.ptr].reshape(-1, self.obs_buf.size(-1))
        self.adv_flat = self.adv_buf[:self.ptr].reshape(-1, self.obs_buf.size(-1))
        self.ret_flat = self.ret_buf[:self.ptr].reshape(-1, self.obs_buf.size(-1))
        self.val_flat = self.val_buf[:self.ptr].reshape(-1, self.obs_buf.size(-1))
        
    def get(self):
        adv_mean = self.adv_flat.mean()
        adv_std = self.adv_flat.std() + 1e-8
        adv_norm = (self.adv_flat - adv_mean) / adv_std
        
        return self.obs_flat, self.act_flat, self.logp_flat, adv_norm, self.ret_flat
    
    def reset(self):
        self.ptr = 0