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

from PPO.ppo_agent import PPOAgent
from PPO.buffer import TrajectoryBuffer
from PPO.networks import GaussianPolicyNet, ValueNet

domain, task = "walker", "walk"
total_epochs = 10000
steps_per_epoch = 1500  
max_ep_len = 1000
gamma = 0.99
gae_lambda = 0.97
buffer_size = steps_per_epoch
save_dir = "./checkpoints"
best_model = "./best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)
print(f"using device {device}")

# def flatten_obs(obs):
#     return np.concatenate([np.atleast_1d(obs[k]) for k in obs])

def flatten_obs(obs):
    flat_parts = []
    for k in obs:
        arr = np.asarray(obs[k])
        flat = arr.flatten()  
        flat_parts.append(flat)
    return np.concatenate(flat_parts)


def train(render: bool):
    env = suite.load(domain, task)
    obs_spec = env.observation_spec()
    obs_dim = int(sum(np.prod(v.shape) for v in obs_spec.values()))
    action_spec = env.action_spec()
    act_dim = action_spec.shape[0]
    print(act_dim, action_spec)

    policy = GaussianPolicyNet(obs_dim, act_dim).to(device)
    value_fn = ValueNet(obs_dim).to(device)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
    
    latest_ckpt = None
    ckpt_files = glob.glob(os.path.join(save_dir, "ppo_ckpt_epoch*.pt"))
    if ckpt_files:
        ckpt_files.sort(key=lambda x: int(x.split("epoch")[-1].split(".pt")[0]), reverse=True)
        latest_ckpt = ckpt_files[0]
        checkpoint = torch.load(latest_ckpt, map_location=device)
        policy.load_state_dict(checkpoint['policy'])
        value_fn.load_state_dict(checkpoint['value_fn'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        total_env_steps = checkpoint.get('steps', 0)
        print(f"Loaded checkpoint: {latest_ckpt} (epoch {start_epoch-1})")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0
        total_env_steps = 0
    
    agent = PPOAgent(
        policy, value_fn, policy_optimizer, value_optimizer, gamma=gamma, device=device
    )

    buffer = TrajectoryBuffer(
        obs_dim, act_dim, buffer_size, device, gamma, gae_lambda
    )
    
    eps_rews = []
    best_reward = float("-inf")
    best_reward_gif = float("-inf")
    best_frames = None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(best_model, exist_ok=True)

    for epoch in trange(start_epoch, total_epochs):
        time_step = env.reset()
        obs = torch.tensor(flatten_obs(time_step.observation), dtype=torch.float32, device=device)
        buffer.reset()
        frames = []
        episode_reward = 0
        ep_len = 0

        for _ in range(steps_per_epoch):
            with torch.no_grad():
                action, logp = agent.select_action(obs.unsqueeze(0))
                value = value_fn(obs.unsqueeze(0))

            action_np = action.squeeze(0).cpu().numpy()
            time_step = env.step(action_np)
            reward = float(time_step.reward) if time_step.reward is not None else 0.0
            episode_reward += reward
            ep_len += 1

            next_obs = torch.tensor(flatten_obs(time_step.observation), dtype=torch.float32, device=device)
            done = bool(time_step.last())

            buffer.store(obs, action.squeeze(0), reward, done, logp.squeeze(0), value.squeeze(0))
            total_env_steps += 1
            
            if epoch > total_epochs - 300:
            # if render:
            
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
                
                if render:
                    cv2.imshow("training", bgr_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if done or ep_len == max_ep_len:
                eps_rews.append(episode_reward)
                if episode_reward > best_reward:
                    torch.save({
                        'policy': policy.state_dict(),
                        'value_fn': value_fn.state_dict(),
                        'epoch': epoch,
                        'steps': total_env_steps
                    }, os.path.join(best_model, f"best_ppo_ckpt_epoch{epoch+1}.pt"))

                    best_reward = episode_reward
                time_step = env.reset()
                obs = torch.tensor(flatten_obs(time_step.observation), dtype=torch.float32, device=device)

                episode_reward = 0
                ep_len = 0
                frames = []
            else:
                obs = next_obs

            if buffer.ptr >= buffer.size:
                break

        with torch.no_grad():
            last_val = value_fn(obs.unsqueeze(0)).squeeze(0)
                
        buffer.finish_path(last_val)
        
        obs_flat, act_flat, logp_old_flat, adv_flat, ret_flat = buffer.get()

        agent.update(obs_flat, act_flat, logp_old_flat, adv_flat, ret_flat)
      

        print(f"[Epoch {epoch}] Steps: {total_env_steps} | AvgRet: {np.mean(eps_rews[-10:]):.2f} | Best: {best_reward:.2f}")
        print(episode_reward)
        if (epoch+1) % 100 == 0 or epoch == total_epochs:
            torch.save({
                'policy': policy.state_dict(),
                'value_fn': value_fn.state_dict(),
                'epoch': epoch,
                'steps': total_env_steps
            }, os.path.join(save_dir, f"ppo_ckpt_epoch{epoch+1}.pt"))
            
            
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
