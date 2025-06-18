import torch

from .algo import surrogate_loss, average_kl, flat_grad, conjugate_gradient, hessian_vector_product

class TRPOAgent:
    def __init__(self, policy, value_net, max_kl=0.01, cg_steps=10, ls_max_steps=10, ls_backtrack=0.8, device="cpu"):
        self.policy = policy
        self.value_net = value_net
        self.max_kl = max_kl
        self.cg_steps = cg_steps
        self.ls_max_steps = ls_max_steps
        self.ls_backtrach = ls_backtrack  
        self.device = device
        
    def update(self, obs, act, adv, old_logp, old_policy):
        surr_loss = surrogate_loss(self.policy, obs, act, old_logp, adv)
        policy_params = list(self.policy.parameters())
        grad = flat_grad(surr_loss, policy_params, retain_graph=True).detach()

        def Av_func(v):
            return hessian_vector_product(self.policy, obs, old_policy, v)
        
        step_dir = conjugate_gradient(Av_func, grad, nsteps=self.cg_steps)
        
        shs = 0.5 * (step_dir * Av_func(step_dir)).sum(0, keepdim=True)
        step_size = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_dir * step_size
        
        prev_params = torch.cat([p.data.view(-1) for p in policy_params])
        
        def set_params(vec):
            idx = 0
            for p in policy_params:
                numel = p.numel()
                p.data.copy_(vec[idx:idx+numel].view_as(p))
                idx += numel
                
        # expected_improve = (grad * full_step).sum(0, keepdim=True)
        success = False
        
        for i in range(self.ls_max_steps):
            step_frac = self.ls_backtrach ** i
            new_params = prev_params + step_frac * full_step
            set_params(new_params)
            new_surr_loss = surrogate_loss(self.policy, obs, act, old_logp, adv)
            actual_improve = new_surr_loss - surr_loss
            
            kl = average_kl(self.policy, old_policy, obs)

            if kl <= self.max_kl and actual_improve > 0:  ## Check for improvement, and trust region
                success = True
                break
        
        if not success:
            set_params(prev_params)  ## Revert back to old params
            
        return {
            'surrogate_loss' : surr_loss.item(),
            'kl' : kl.item(),
            'actual_improve' : actual_improve.item() if success else 0.0,
            'step_frac' : step_frac if success else 0.0,
            'success' : success
        }
        