import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from gymnasium.envs.registration import register

# Register the environment
try:
    register(
        id="LinearSystem-v0",
        entry_point="linear_system_env:LinearDynamicalSystemEnv",
        max_episode_steps=300,
    )
except gym.error.Error:
    pass

# Hyperparameters
SEED = 1
LEARNING_RATE = 1e-3
GAMMA = 0.99
TOTAL_TIMESTEPS = 50000
PRINT_INTERVAL = 2000
DISCRETE_ACTIONS = [-10, -1, 0, 1, 10]

def make_env(render_mode=None):
    # Pass discrete_actions to the environment
    env = gym.make("LinearSystem-v0", render_mode=render_mode, discrete_actions=DISCRETE_ACTIONS)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

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
        obs, _ = env.reset(seed=SEED+episode)
        done = False
        
        # Storage for the episode
        log_probs = []
        rewards = []
        
        while not done:
            global_step += 1
            obs_tensor = torch.Tensor(obs).to(device)
            action, log_prob, _ = agent.get_action(obs_tensor)
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            obs = next_obs

        # Calculate returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(device)
        log_probs = torch.stack(log_probs)
        
        # Advantage normalization
        advantages = returns
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy Loss: -log_prob * advantage
        policy_loss = -(log_probs * advantages).mean()
        
        # Total loss
        loss = policy_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if global_step % PRINT_INTERVAL < 200:
            print(f"Step {global_step}, Episode {episode}, Return: {sum(rewards):.2f}, Loss: {loss.item():.4f}")

    env.close()

    # Evaluation
    print("\nEvaluating with render...")
    env = make_env(render_mode="rgb_array")
    for i in range(3):
        obs, _ = env.reset()
        done = False
        ep_ret = 0
        frames = []
        while not done:
            with torch.no_grad():
                logits = agent.actor(torch.Tensor(obs).to(device))
                action = torch.argmax(logits).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            done = terminated or truncated
            frames.append(env.render())
            
            if info.get("is_success"):
                print("Reached the goal!")
                break

        print(f"Eval Episode {i+1} Return: {ep_ret:.2f}")

        if len(frames) > 0:
            plt.imsave(f"pg_discrete_eval_{i+1}.png", frames[-1])
            print(f"Saved final frame to pg_discrete_eval_{i+1}.png")
    env.close()

if __name__ == "__main__":
    train()
