import os
import sys
import cv2
import json
import time
import torch
import random
import imageio
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque

os.makedirs("Results", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wrapper.env_wrapper import DMControlEnv
from SAC.sac_agent import SacAgent
from SAC.buffer import ReplayBuffer
SEED = 42
CONFIG = {
    "env_domain": "cheetah",
    "env_task": "run",
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "policy_lr": 3e-4,
    "q_lr": 3e-4,
    "value_lr": 3e-4,
    "hidden_sizes": [256, 256],
    "batch_size": 128,
    "replay_size": 200_000,
    "start_steps": 10_000,
    "updates_per_step": 1,
    "max_steps": 1_000_000,
    "log_interval": 1000,
    "eval_interval": 5000,
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(cfg, render : bool):
    frames = deque(maxlen=25_000)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DMControlEnv(cfg["env_domain"], cfg["env_task"])
    agent = SacAgent(env.obs_dim, env.action_dim, device, cfg)
    buffer = ReplayBuffer(env.obs_dim, env.action_dim, cfg["replay_size"])

    state = env.reset()
    episode_reward, episode_steps, episode = 0, 0, 0
    eps_rews = []
    best_reward_gif = float("-inf")
    best_frames = None
    start_time = time.time()

    for t in trange(1, cfg["max_steps"] + 1):
        if t < cfg["start_steps"]:
            action = np.random.uniform(-1, 1, env.action_dim)
        else:
            with torch.no_grad():
                action, _ = agent.policy.sample(torch.tensor(state, device=device).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
                
        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward.item(); episode_steps += 1
        
        bgr_frame, frames = env.store_best_frame(frames)

        if episode_reward > best_reward_gif:
            best_reward_gif = episode_reward
            best_frames = frames.copy()
        

        # env.render(bgr_frame)
            

        if done :  
            print(f"Episode {episode}  reward {episode_reward:.1f}  steps {episode_steps}")
            eps_rews.append(episode_reward)
            state = env.reset()
            episode_reward, episode_steps = 0, 0
            episode += 1

        if t >= cfg["start_steps"]:
            for _ in range(cfg["updates_per_step"]):
                info = agent.update(buffer, cfg["batch_size"])

        if t % cfg["log_interval"] == 0 and t >= cfg["start_steps"]:
            elapsed = time.time() - start_time
            log_line = {
                "step": t,
                "time": elapsed,
                **{k: float(v) for k, v in info.items()},
            }
            print(json.dumps(log_line))
    
    if best_frames is not None:
        imageio.mimsave("Results/best_episode.gif", best_frames, fps=30)
    
    plt.plot(eps_rews)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.savefig("Results/training_rewards.png")
    plt.close()
    
    if render:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(CONFIG, render=True)
