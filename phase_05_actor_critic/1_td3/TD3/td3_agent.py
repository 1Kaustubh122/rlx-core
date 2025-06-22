import torch
import numpy as np
import torch.nn as nn
from .buffer import ReplayBuffer
from .networks import Actor, Critic

class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent
    """
    def __init__(
        self,
        obs_sim, act_dim,
        action_low, action_high,
        actor : Actor, critic1: Critic, critic2 : Critic,
        actor_target : Actor, critic1_target:Critic, critic2_target : Critic,
        actor_optimizer, critic1_optimizer, critic2_optimizer,
        gamma=0.99, tau=0.005,
                                          ## 2 Critic update per 1 actor update
        policy_noise=0.2, noise_clip=0.5, policy_delay=2,
        action_limit=1.0, device="cuda" 
    ):
        
        self.obs_dim = obs_sim
        self.act_dim = act_dim
        
        self.action_low = action_low
        self.action_high = action_high
        
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optimizer = actor_optimizer
        
        self.critic1 = critic1
        self.critic1_target = critic1_target
        self.critic1_optimizer = critic1_optimizer
        
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.critic2_optimizer = critic2_optimizer

        self.gamma = gamma
        self.tau = tau

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay
        self.action_limit = action_limit
        
        self.total_it = 0
        
        self.device = device
    

    ## For training
    def select_action(self, obs, noise_std=0.1):
        """
        Returns the action for env step, with noise
        """
        with torch.no_grad():                      
            action = self.actor(obs).cpu()         

        if noise_std > 0:
            action += noise_std * torch.randn_like(action)

        action = torch.clamp(action,
                            self.action_low.cpu(),
                            self.action_high.cpu())

        return action.detach().numpy().astype(np.float32)  

      

    ## For Eval
    def act(self, obs):
        """
        returns the action for the env step
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = self.actor(obs).cpu().data.numpy()
        
        return torch.clamp(action, self.action_low, self.action_high)

    def train(self, replay_buffer : ReplayBuffer, batch_size=100):
        """
        Sample from replay buffer, perform critic(s) and (possibly) delayed actor update.
        """        
        
        ## sampling
        obs, act, rew, next_obs, done = replay_buffer.sample(batch_size)

        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        
        noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_obs)
        next_action = (next_action + noise).clamp(self.action_low, self.action_high)
        
        ## Computing target Q-Vals
        targrt_q1 = self.critic1(next_obs, next_action)
        targrt_q2 = self.critic2(next_obs, next_action)
        
        ## Taking minimum, to avoid overestimation, cause overestimation kills...
        target_Q = torch.min(targrt_q1, targrt_q2)
        target = rew.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * target_Q
        
        ## Critic Update
        current_q1 = self.critic1(obs, act)
        current_q2 = self.critic2(obs, act)
        critic1_loss = nn.MSELoss()(current_q1, target.detach())
        critic2_loss = nn.MSELoss()(current_q2, target.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        self.total_it += 1
        ##Delayed actor policy update
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.sync_target_networks()
            



    def sync_target_networks(self, tau=None):
        """
        Polyak averaging: target <- tau * main + (1-tau) * target
        """
        if tau is None:
            tau = self.tau
        
        for main, target in [
            (self.actor, self.actor_target),
            (self.critic1, self.critic1_target),
            (self.critic2, self.critic2_target)
        ]:
            for param, targ_param in zip(main.parameters(), target.parameters()):
                targ_param.data.copy_(tau * param.data + (1 - tau) * targ_param.data)

    def save(self, path):
        """
        saves state_dict:
        actor
        actor_target
        critic1
        critic1_target
        critic2
        critic2_target
        actor_optimizer
        critic1_optimizer
        critic2_optimizer
        """
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_target" : self.actor_target.state_dict(),
            "critic1" : self.critic1.state_dict(),
            "critic1_target" : self.critic1_target.state_dict(),
            "critic2" : self.critic2.state_dict(),
            "critic2_target" : self.critic2_target.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic1_optimizer" : self.critic1_optimizer.state_dict(),
            "critic2_optimizer" : self.critic2_optimizer.state_dict(),  
        }, path)

    def load(self, path):
        """
        load state_dict:
        actor
        actor_target
        critic1
        critic1_target
        critic2
        critic2_target
        actor_optimizer
        critic1_optimizer
        critic2_optimizer
        """
        chkpt = torch.load(path)
        
        self.actor.load_state_dict(chkpt["actor"])
        self.actor_target.load_state_dict(chkpt["actor_target"])
        self.critic1.load_state_dict(chkpt["critic1"])
        self.critic1_target.load_state_dict(chkpt["critic1_target"])
        self.critic2.load_state_dict(chkpt["critic2"])
        self.critic2_target.load_state_dict(chkpt["critic2_target"])
        self.actor_optimizer.load_state_dict(chkpt["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(chkpt["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(chkpt["critic2_optimizer"])

