import os
import sys
import cv2
import glob
import torch
import imageio
import numpy as np
from tqdm import trange
from dm_control import suite
import matplotlib.pyplot as plt

os.makedirs("Results", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TD3.buffer import ReplayBuffer
from TD3.networks import Actor, Critic
from TD3.td3_agent import TD3Agent

domain, task = "quadruped", "walk"
max_steps = 150_000
start_steps = 25_000 
expl_noise = 0.1 
batch_size = 256
eval_interval = 5_000
lr=3e-4
save_dir = "./checkpoints"
best_model = "./best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"using device {device}")

def flatten_obs(obs):
    flat_parts = []
    for k in obs:
        arr = np.asarray(obs[k])
        flat = arr.flatten()  
        flat_parts.append(flat)
    return np.concatenate(flat_parts)


def train(render : bool):
    env = suite.load(domain, task)
    obs_spec = env.observation_spec()
    obs_dim = int(sum(np.prod(v.shape) for v in obs_spec.values()))
    action_spec = env.action_spec()
    act_dim = action_spec.shape[0]
    print(act_dim, action_spec)
    
    ## For Quadruped
    action_high = action_spec.maximum
    action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
    ## [1.  1.1 0.8 1.  1.1 0.8 1.  1.1 0.8 1.  1.1 0.8]
    
    action_low = action_spec.minimum
    action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    ## [-1.  -1.  -0.8 -1.  -1.  -0.8 -1.  -1.  -0.8 -1.  -1.  -0.8]

    actor = Actor(obs_dim, act_dim).to(device)
    actor_targ = Actor(obs_dim, act_dim).to(device)
    actor_targ.load_state_dict(actor.state_dict())
    actor_optim = torch.optim.Adam(actor.parameters(), lr)
    
    critic1 = Critic(obs_dim, act_dim).to(device)
    critic1_targ = Critic(obs_dim, act_dim).to(device)
    critic1_targ.load_state_dict(critic1.state_dict())
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr)
    
    critic2 = Critic(obs_dim, act_dim).to(device)
    critic2_targ = Critic(obs_dim, act_dim).to(device)
    critic2_targ.load_state_dict(critic2.state_dict())
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr)
    
    
    agent = TD3Agent(
        obs_dim, act_dim,
        action_low, action_high,
        actor, critic1, critic2,
        actor_targ, critic1_targ, critic2_targ,
        actor_optim, critic1_optim, critic2_optim,
        device=device
    )
    
    buffer_size = int(1e6)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, device)
    
    latest_ckpt = None
    ckpt_files = glob.glob(os.path.join(save_dir, "td3_ckpt_epoch*.pt"))
    
    if ckpt_files:
        ckpt_files.sort(key=lambda x: int(x.split("epoch")[-1].split(".pt")[0]), reverse=True)
        latest_ckpt = ckpt_files[0]
        agent.load(latest_ckpt)
        
        print(f"Loaded checkpoint")
    else:
        print("No checkpoint found. Starting from scratch.")

    total_env_steps = 0
    eps_rews = []
    best_reward = float("-inf")
    best_reward_gif = float("-inf")
    best_frames = None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(best_model, exist_ok=True)

    obs_dm = env.reset()
    obs = torch.tensor(flatten_obs(obs_dm.observation), dtype=torch.float32, device=device)
    
    frames = []
    episode_reward = 0
    ep_len = 0

    for step in trange(0, max_steps):
        if step < start_steps:
            action_np = np.random.uniform(action_low.cpu(), action_high.cpu())
        else:
            action_np = agent.select_action(obs ,expl_noise)

        time_step = env.step(action_np)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        episode_reward += reward
        ep_len += 1
        next_obs = torch.tensor(flatten_obs(time_step.observation), dtype=torch.float32, device=device)
        done = bool(time_step.last())
        
        replay_buffer.add(
            obs.cpu(),
            torch.from_numpy(action_np).float(),
            torch.tensor([reward]),
            next_obs.cpu(),
            torch.tensor([done], dtype=torch.float32)
        )

        total_env_steps += 1
        obs = next_obs
        
        if done:
            time_step = env.reset()
            obs = torch.tensor(flatten_obs(time_step.observation), dtype=torch.float32, device=device)
            if episode_reward > best_reward:
                agent.save(os.path.join(best_model, f"best_td3_ckpt_epoch{step+1}.pt"))
                best_reward = episode_reward
                

            episode_reward = 0
            ep_len = 0
            frames = []
        
        eps_rews.append(episode_reward)
        
        if len(replay_buffer) > batch_size:
            agent.train(replay_buffer, batch_size)
        
 
        # if step > max_steps - 10000:
        if render:
        
            rgb_frame = env.physics.render(height=420, width=420, camera_id=0)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            bgr_frame = cv2.resize(
                bgr_frame, 
                (bgr_frame.shape[1] , bgr_frame.shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )
            frames.append(rgb_frame)
            
            if episode_reward > best_reward_gif:
                best_reward_gif = episode_reward
                best_frames = frames.copy()
            
            # if render:
            #     cv2.imshow("training", bgr_frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
                
        if step % 10000 == 0:
            print(f"[Epoch {step}] Steps: {total_env_steps} | AvgRet: {np.mean(eps_rews[-9999:]):.2f} | Best: {best_reward:.2f}")
        
        # if step % 1000 == 0:
            # agent.save(os.path.join(save_dir, f"td3_ckpt_epoch{step+1}.pt"))

  
        
    if best_frames is not None:
        imageio.mimsave("Results/best_episode.gif", best_frames, fps=30)
        plt.plot(eps_rews)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.savefig("Results/training_rewards.png")
        plt.close()

def main():
    render = True
    train(render=render)     
    if render:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()