import os
import sys
import cv2
import torch
import imageio
import numpy as np
from tqdm import trange
from dm_control import suite
import matplotlib.pyplot as plt

os.makedirs("Results", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trpo.agent import TRPOAgent
from trpo.buffer import TrajectoryBuffer
from trpo.networks import GaussianPolicy, ValueNet

domain, task = "quadruped", "walk"
num_envs = 1
total_epochs = 1000         
steps_per_epoch = 1000  
max_ep_len = 1000
gamma = 0.99
gae_lambda = 0.97
buffer_size = steps_per_epoch
max_kl = 0.01
cg_steps = 10
ls_max_steps = 10
ls_backtrack = 0.8
vf_lr = 1e-3
vf_iters = 5
save_dir = "./checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"using device {device}")

def flatten_obs(obs):
    return np.concatenate([np.atleast_1d(obs[k]) for k in obs])

def train(render: bool):
    env = suite.load(domain, task)
    obs_spec = env.observation_spec()
    obs_dim = int(sum(np.prod(v.shape) for v in obs_spec.values()))
    action_spec = env.action_spec()
    act_dim = action_spec.shape[0]
    
    print(act_dim, action_spec)

    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    value_fn = ValueNet(obs_dim).to(device)
    agent = TRPOAgent(
        policy, value_fn,
        max_kl=max_kl, cg_steps=cg_steps,
        ls_max_steps=ls_max_steps, ls_backtrack=ls_backtrack,
        device=device
    )
    vf_optimizer = torch.optim.Adam(value_fn.parameters(), lr=vf_lr)

    buffer = TrajectoryBuffer(
        obs_dim, act_dim, buffer_size, num_envs, device, gamma, gae_lambda
    )
    
    eps_rews = []
    best_reward = float("-inf")
    best_frames = None

    os.makedirs(save_dir, exist_ok=True)
    total_env_steps = 0

    for epoch in trange(total_epochs):
        time_step = env.reset()
        obs = flatten_obs(time_step.observation)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        buffer.reset()
        frames = []
        episode_reward = 0
        ep_len = 0

        for _ in range(steps_per_epoch):
            with torch.no_grad():
                mean, std = policy(obs)
                std = torch.clamp(std, min=1e-3, max=10)  # clamp for numerical stability
                if torch.isnan(mean).any() or torch.isnan(std).any():
                    print("NaN detected in policy output! mean:", mean, "std:", std)
                    print("obs:", obs)
                    exit(1)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum(axis=-1)
                value = value_fn(obs)

            time_step = env.step(action.cpu().numpy()[0])
            reward = float(time_step.reward) if time_step.reward is not None else 0.0
            episode_reward += reward
            ep_len += 1

            next_obs_np = flatten_obs(time_step.observation)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0).to(device)
            done = bool(time_step.last())

            buffer.store(obs, action, reward, done, logp, value)
            total_env_steps += 1

            if epoch > total_epochs - 300:
                rgb_frame = env.physics.render(height=480, width=480, camera_id=0)
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                scale_factor = 1
                bgr_frame = cv2.resize(
                    bgr_frame, 
                    (bgr_frame.shape[1] * scale_factor, bgr_frame.shape[0] * scale_factor), 
                    interpolation=cv2.INTER_LINEAR
                )
                frames.append(rgb_frame)
                cv2.imshow("training", bgr_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if done or ep_len == max_ep_len:
                eps_rews.append(episode_reward)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_frames = frames.copy()
                time_step = env.reset()
                obs = flatten_obs(time_step.observation)
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                episode_reward = 0
                ep_len = 0
                frames = []
            else:
                obs = next_obs

            if buffer.ptr >= buffer.size:
                break

        with torch.no_grad():
            last_val = value_fn(obs)
        buffer.finish_path(last_val)
        obs_flat, act_flat, logp_old_flat, adv_flat, ret_flat = buffer.get()
        old_policy = GaussianPolicy(obs_dim, act_dim).to(device)
        old_policy.load_state_dict(policy.state_dict())
        for param in old_policy.parameters():
            param.requires_grad = False

        agent.update(obs_flat, act_flat, adv_flat, logp_old_flat, old_policy)

        for _ in range(vf_iters):
            v = value_fn(obs_flat)
            vf_loss = ((v - ret_flat) ** 2).mean()
            vf_optimizer.zero_grad()
            vf_loss.backward()
            vf_optimizer.step()

        print(f"[Epoch {epoch}] Steps: {total_env_steps} | AvgRet: {np.mean(eps_rews[-10:]):.2f} | Best: {best_reward:.2f}")
        if (epoch + 1) % 100 == 0 or epoch == 0:
            torch.save({
                'policy': policy.state_dict(),
                'value_fn': value_fn.state_dict(),
                'epoch': epoch,
                'steps': total_env_steps
            }, os.path.join(save_dir, f"trpo_ckpt_epoch{epoch+1}.pt"))

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
