import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register

# --------------------------- Register environment ---------------------------

try:
    register(
        id="LinearSystem-v0",
        entry_point="linear_system_env:LinearDynamicalSystemEnv",
        max_episode_steps=200,
    )
except gym.error.Error:
    pass

# --------------------------- Hyperparameters ---------------------------

SEED = 1
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
TOTAL_TIMESTEPS = 500_000

# DQN-specific
DISCRETE_ACTIONS = [-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10]
BUFFER_SIZE = 10_000
BATCH_SIZE = 128
LEARNING_STARTS = 10_000
TRAIN_FREQUENCY = 4
TARGET_NETWORK_FREQUENCY = 1000
TAU = 1.0  # for soft update (1.0 = hard copy)
START_E = 1.0
END_E = 0.05
EXPLORATION_FRACTION = 0.5  # fraction of TOTAL_TIMESTEPS to anneal epsilon

# --------------------------- Environment helper ---------------------------

def make_env(render_mode=None):
    env = gym.make("LinearSystem-v0", render_mode=render_mode, discrete_actions=DISCRETE_ACTIONS)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

# --------------------------- Utils ---------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# --------------------------- Replay buffer (simple) ---------------------------

class ReplayBuffer:
    def __init__(self, size, observation_space, action_space, device):
        self.size = size
        self.device = device
        obs_shape = observation_space.shape
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *obs_shape), dtype=np.float32)
        # store discrete actions as ints
        self.act_buf = np.zeros((size,), dtype=np.int64)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        self.obs_buf[self.idx] = np.array(obs, dtype=np.float32)
        self.next_obs_buf[self.idx] = np.array(next_obs, dtype=np.float32)
        self.act_buf[self.idx] = int(action)
        self.rew_buf[self.idx] = float(reward)
        self.done_buf[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.size
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.size if self.full else self.idx

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)
        obs = torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device)
        acts = torch.tensor(self.act_buf[idxs], dtype=torch.long, device=self.device).unsqueeze(1)
        rews = torch.tensor(self.rew_buf[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device)
        return obs, next_obs, acts, rews, dones

# --------------------------- Q-network ---------------------------

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.n
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def forward(self, x):
        # expect x shape (..., obs_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# --------------------------- Training loop ---------------------------

def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env()
    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    rb = ReplayBuffer(BUFFER_SIZE, env.observation_space, env.action_space, device)

    os.makedirs("outputs", exist_ok=True)

    start_time = time.time()
    global_step = 0
    episode = 0

    obs, _ = env.reset(seed=SEED)
    done = False
    ep_reward = 0.0

    while global_step < TOTAL_TIMESTEPS:
        global_step += 1

        # epsilon-greedy action
        epsilon = linear_schedule(START_E, END_E, int(EXPLORATION_FRACTION * TOTAL_TIMESTEPS), global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                q_vals = q_network(obs_tensor)
                action = int(torch.argmax(q_vals, dim=-1).cpu().item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # store transition
        rb.add(obs, next_obs, action, reward, float(done))

        obs = next_obs

        # end of episode bookkeeping
        if done:
            episode += 1
            print(f"global_step={global_step}, episodic_return={ep_reward:.2f}")
            # reset
            obs, _ = env.reset(seed=SEED + episode)
            ep_reward = 0.0

        # training step
        if global_step > LEARNING_STARTS and len(rb) >= BATCH_SIZE and (global_step % TRAIN_FREQUENCY == 0):
            obs_b, next_obs_b, acts_b, rews_b, dones_b = rb.sample(BATCH_SIZE)

            # compute TD target
            with torch.no_grad():
                next_q = target_network(next_obs_b)
                next_q_max, _ = next_q.max(dim=1)
                td_target = rews_b + GAMMA * next_q_max * (1.0 - dones_b)

            current_q = q_network(obs_b).gather(1, acts_b).squeeze()
            loss = nn.functional.mse_loss(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # target network update (hard/soft based on TAU)
        if global_step % TARGET_NETWORK_FREQUENCY == 0:
            if TAU == 1.0:
                target_network.load_state_dict(q_network.state_dict())
            else:
                # soft update
                for targ_p, src_p in zip(target_network.parameters(), q_network.parameters()):
                    targ_p.data.copy_(TAU * src_p.data + (1.0 - TAU) * targ_p.data)

    env.close()

    # --------------------------- Evaluation ---------------------------

    print("\nEvaluating with render...")
    env = make_env(render_mode="rgb_array")
    final_frames = []

    for i in range(16):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                q_vals = q_network(obs_tensor)
                action = int(torch.argmax(q_vals, dim=-1).cpu().item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

            # optional: break early if environment indicates success
            if info.get("is_success"):
                break

        print(f"Eval Episode {i+1} Return: {ep_ret:.2f}")
        final_frames.append(env.render())

    env.close()

    # save 4x4 grid
    if len(final_frames) == 16:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(final_frames[idx])
            ax.axis("off")
            ax.set_title(f"Ep {idx+1}")
        plt.tight_layout()
        out_path = "outputs/dqn_eval_grid.png"
        plt.savefig(out_path)
        print(f"Saved 4x4 evaluation grid to {out_path}")
    else:
        print(f"Expected 16 frames, got {len(final_frames)}")

if __name__ == "__main__":
    train()
