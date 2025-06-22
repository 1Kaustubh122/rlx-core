import torch

class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent
    """
    def __init__(
        self,
        obs_dim, act_dim,
        actor, critic1, critic2,
        actor_target, critic1_target, critic2_target,
        actor_optimizer, critic1_optimizer, critic2_optimizer,
        gamma=0.99, tau=0.005,
                                          ## 2 Critic update per 1 actor update
        policy_noise=0.2, noise_clip=0.5, policy_delay=2,
        action_limit=1.0, device="cuda"
    ):
        ...

    def select_action(self, obs, noise_std=0.1):
        """
        Return action for env step. Optionally add exploration noise.
        """
        ...

    def train(self, replay_buffer, batch_size=100):
        """
        Sample from replay buffer, perform critic(s) and (possibly) delayed actor update.
        All ablation toggles must be config-driven.
        """
        ...

    def sync_target_networks(self, tau=None):
        """
        Polyak averaging: target <- tau * main + (1-tau) * target
        """
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
