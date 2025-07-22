from __future__ import annotations
import jax
import hydra
import numpy as np
from omegaconf import DictConfig

from mbpo_agent import MBPOAgent
from logger import init_logger, log_metrics
from envs.isaac_assembly import IsaacAsemblyEnv

## DEBUG
# print(jax.devices())

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rng = jax.random.PRNGKey(cfg.seed)
    np.random.seed(cfg.seed)
    
    env = IsaacAsemblyEnv(cfg.env)
    agent= MBPOAgent(cfg, rng, env.observation_space, env.action_space)
    
    logger = init_logger(cfg)
    
    for epch in range(cfg.train.epochs):
        metrics = agent.train_epoch(env)
        eval_stats = agent.evaluate(env, episodes=5)
        log_metrics(logger, eval_stats, step=epch)
        
    agent.save("output/final_checkpoint")

if __name__ == "__main__":
    main()