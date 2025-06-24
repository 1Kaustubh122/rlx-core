from __future__ import annotations
from tqdm import trange
from typing import Dict, Any
import numpy as np, torch
from collections import deque
import matplotlib.pyplot as plt
import os, sys, cv2, argparse, time, json, random, imageio

os.makedirs("Results", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from REDQ.buffer import ReplayBuffer             
from wrapper.env_wrapper import DMControlEnv
from REDQ.agent import REDQAgent, REDQConfig

parser = argparse.ArgumentParser()
parser.add_argument("--domain", default="cheetah", type=str)
parser.add_argument("--task",   default="run",     type=str)
parser.add_argument("--steps",  default=50_000, type=int)
parser.add_argument("--seed",   default=42,        type=int)
args = parser.parse_args()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

REDQ_CFG = REDQConfig(      
    tau=0.005,
    gamma=0.99, 
    alpha=0.2, 
    auto_alpha=True,
    q_lr=3e-4,
    policy_lr=3e-4, 
    hidden=(256, 256),
    min_qs=2, 
    ensemble=6,
    utd_ratio=12, 
    batch=128, 
    grad_clip=1.0,
)

REPLAY_SIZE = 1_000_000
START_STEPS = 5_000         
LOG_INTERVAL = 1_000

def main(cfg: REDQConfig, total_steps: int, render: bool):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DMControlEnv(args.domain, args.task)
    agent = REDQAgent(env.obs_dim, env.action_dim, device, cfg)
    buffer = ReplayBuffer(env.obs_dim, env.action_dim, REPLAY_SIZE)

    state = env.reset()
    ep_reward, ep_len, ep_idx = 0.0, 0, 0
    start_time = time.time(); info: Dict[str, Any] = {}
    eps_rews = []
    best_reward_gif = float("-inf")
    best_frames = None
    frames = deque(maxlen=3_000)
    
    for t in trange(1, total_steps + 1, desc="REDQ"):
        if t < START_STEPS:
            action = np.random.uniform(-1, 1, env.action_dim)
        else:
            action = agent.act(state)

        next_state, reward, done = env.step(action)
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        if isinstance(done, np.ndarray):
            done = bool(done.item())
            
        buffer.add(state, action, reward, next_state, done)
        state, ep_reward, ep_len = next_state, ep_reward + reward, ep_len + 1
        
        # if t % 1000 == 0:
        bgr_frame, frames = env.store_best_frame(frames)
        
        if render:
            env.render(bgr_frame)
        
        if ep_reward > best_reward_gif:
            best_reward_gif = ep_reward
            best_frames = frames.copy()
    
        if done:
            print(f"Episode {ep_idx} | R {ep_reward:.1f} | Len {ep_len}")
            state, ep_reward, ep_len, ep_idx = env.reset(), 0.0, 0, ep_idx + 1

        if t >= START_STEPS and len(buffer) >= cfg.batch:
            batch = buffer.sample(cfg.batch, device)
            info = agent.update(batch)
            
        eps_rews.append(ep_reward)

        if t % LOG_INTERVAL == 0 and info:
            elapsed = time.time() - start_time
            stats = {k: round(v, 5) for k, v in info.items()}
            print(json.dumps({"step": t, "sec": round(elapsed, 1), **stats}))

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
    main(REDQ_CFG, args.steps, render=False)
