import torch

def surrogate_loss(policy, obs, act, old_logp, adv):
    
    logp = policy.get_log_prob(obs, act)
    ratio = torch.exp(logp - old_logp)
    return (ratio * adv).mean()

def average_kl(policy, old_policy, obs):
    with torch.no_grad():
        kl = old_policy.kl_divergence(obs, policy)
    
    return kl.mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    grad = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grad])

def conjugate_gradient(Av_func, b, nsteps=10, residual_tol =1e-10):
    x, r, p = torch.zeros_like(b), b.clone(), b.clone()
    rsold = torch.dot(r, r)
    
    for _ in range(nsteps):
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
    hvp = torch.autograd.grad(prod, policy.parameters(), retain_graph=True)
    flat_hvp = torch.cat([g.reshape(-1) for g in hvp]).detach()

    return flat_hvp + damping * v
