import torch

class PPOAgent:                                                                               
    def __init__(
        self, policy_net , value_net , policy_optimizer, value_optimizer, 
        ## +- 20% A/C paper 
        clip_param=0.2, entropy_coef=0.01, gamma = 0.95, device="cuda"
    ):
        self.policy = policy_net
        self.value = value_net
        self.opti_pol = policy_optimizer
        self.opti_val = value_optimizer
        self.clip = clip_param
        self.entropy_coef = entropy_coef
        self.device = device
        self.gamma = gamma
    
    def select_action(self, obs):
        mean, std = self.policy(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp
    
    def get_log_prob(self, obs, action) -> torch.Tensor:
        mean, std =  self.policy(obs)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action).sum(axis=-1)
        return logp
    
    
    def update(self, obs, actions, old_logp, advantages, returns, batch_size=64, epochs=10, max_grad_norm=0.5):
        dataset = torch.utils.data.TensorDataset(obs, actions, old_logp, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

        for _ in range(epochs):
            for mb_obs, mb_act, mb_o_logp, mb_adv, mb_rets in loader:
                mb_obs = mb_obs.to(self.device)
                mb_act = mb_act.to(self.device)
                mb_o_logp = mb_o_logp.to(self.device)
                mb_adv = mb_adv.to(self.device)
                mb_rets = mb_rets.to(self.device)
                
                mean, std = self.policy(mb_obs)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(mb_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(new_logp - mb_o_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * mb_adv  ## Clippng [1-e, 1+e]
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                self.opti_pol.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.opti_pol.step()

                val_pred = self.value(mb_obs)
                value_loss = ((mb_rets - val_pred) ** 2).mean()

                self.opti_val.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_grad_norm)
                self.opti_val.step()

    