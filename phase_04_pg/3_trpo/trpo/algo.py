import torch

def surrogate_loss(policy, obs, act, old_logp, adv):
    
    logp = policy.get_log_prob(obs, act)
    ratio = torch.exp(logp - old_logp)
    return (ratio * adv).mean(), logp

def average_kl(policy, old_policy, obs):
    with torch.no_grad():
        kl = old_policy.kl_divergence(obs, policy)
    
    return kl.mean()

def flat_grad(y, x, retain_graph, create_graph=False):
    grad = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grad])

def conjugate_gradient(Av_func, b, nsteps=10, residual_tol =1e-10):
    x, r, p = torch.zeros_like(b), b.clone(), b.clone()
    rsold = torch.dot(r, r)
    
    for i in range(nsteps):
        Avp = Av_func(p)
        alpha = rsold / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        rsnew = torch.dot(r, r)

        if rsnew < residual_tol:
            break
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x

def hessian_vector_product(policy, obs, old_policy, v, damping=1e-2):
    mean_kl = old_policy.kl_divergence(obs, policy).mean()
    grads = torch.autograd.grad(mean_kl, policy.parameters(), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    prod = (flat_grad * v).sum()
    hvp = torch.autograd(prod, policy.parameters(), retain_graph=True)
    flat_hvp = torch.cat([g.reshape(-1) for g in hvp]).detach()

    return flat_hvp + damping * v

def trpo_update(policy, old_policy, obs, act, adv, old_logp, max_kl=0.01, cg_steps=10, ls_max_steps=10, ls_backtrach=0.8):
    surr_loss, logp = surrogate_loss(policy, obs, act, old_logp, adv)
    policy_params = list(policy.parameters())
    flat_params = torch.cat([p.data.flattern() for p in policy_params])
    grad = flat_grad(surr_loss, policy_params, retain_graph=True).detach()

    def Av_func(v):
        return hessian_vector_product(policy, obs, old_policy, v)
    
    step_dir = conjugate_gradient(Av_func, grad, nsteps=cg_steps)
    
    shs = 0.5 * (step_dir * Av_func(step_dir)).sum(0, keepdim=True)
    step_size = torch.sqrt(max_kl / (shs + 1e-8))
    full_step = step_dir * step_size
    
    prev_params= torch.cat([p.data.flattern() for p in policy_params])
    def set_params(vec):
        idx = 0
        for p in policy_params:
            numel = p.numel()
            p.data.copy_(vec[idx:idx+numel].view_as(p))
            idx += numel
            
    expected_improve = (grad * full_step).sum(0, keepdim=True)
    success = False
    for i in range(ls_max_steps):
        step_frac = ls_backtrach ** i
        new_params = prev_params + step_dir * full_step
        set_params(new_params)
        new_surr_loss, _ = surrogate_loss(policy, obs, act, old_logp, adv)
        actual_improve = new_surr_loss - surr_loss
        
        kl = average_kl(policy, old_policy, obs)

        if kl <= max_kl and actual_improve > 0:  ## Check for improvement, and trust region
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
    