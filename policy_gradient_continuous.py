import os
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from gymnasium.envs.registration import register

# --------------------------- Environment Registration ---------------------------

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
LEARNING_RATE = 1e-3
GAMMA = 0.99
TOTAL_TIMESTEPS = 200000

# --------------------------- Environment Helper ---------------------------

def make_env(render_mode=None):
    env = gym.make("LinearSystem-v0", render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

# --------------------------- Utils ---------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# --------------------------- Agent ---------------------------

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.prod(env.observation_space.shape)
        act_dim = np.prod(env.action_space.shape)

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_action(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

# --------------------------- Training Loop ---------------------------

def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env()
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    global_step = 0
    episode = 0

    while global_step < TOTAL_TIMESTEPS:
        episode += 1
        obs, _ = env.reset(seed=SEED + episode)
        done = False

        log_probs = []
        rewards = []

        while not done:
            global_step += 1

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, _ = agent.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.stack(log_probs).squeeze()

        # Normalize advantages
        advantages = returns
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Loss
        policy_loss = -(log_probs * advantages).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"global_step={global_step}, episodic_return={sum(rewards):.2f}")

    env.close()

    # --------------------------- Evaluation ---------------------------

    print("\nEvaluating with render...")
    os.makedirs("outputs", exist_ok=True)

    env = make_env(render_mode="rgb_array")
    final_frames = []

    for i in range(16):
        obs, _ = env.reset()
        done = False
        ep_ret = 0

        while not done:
            with torch.no_grad():
                mean_action = agent.actor_mean(
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                )
                action = mean_action.cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated

        print(f"Eval Episode {i+1} Return: {ep_ret:.2f}")
        final_frames.append(env.render())

    env.close()

    # Grid of images
    if len(final_frames) == 16:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(final_frames[idx])
            ax.axis("off")
            ax.set_title(f"Ep {idx+1}")

        plt.tight_layout()
        plt.savefig("outputs/policy_gradient_continuous_eval_grid.png")
        print("Saved evaluation grid.")
        
if __name__ == "__main__":
    train()