import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
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
TOTAL_TIMESTEPS = 100000
PRINT_INTERVAL = 2000  # Print every n steps

def make_env(render_mode=None):
    env = gym.make("LinearSystem-v0", render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

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
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            action, log_prob, _ = agent.get_action(obs_tensor)
            
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
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
        log_probs = torch.stack(log_probs).squeeze()
        
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
        
        if global_step % PRINT_INTERVAL < 200: # Approximate printing
             print(f"Step {global_step}, Episode {episode}, Return: {sum(rewards):.2f}, Loss: {loss.item():.4f}")

    env.close()

    # Evaluation
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
                # Use mean action for evaluation
                action_mean = agent.actor_mean(torch.Tensor(obs).unsqueeze(0).to(device))
                action = action_mean
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            ep_ret += reward
            done = terminated or truncated
            frames.append(env.render())

            if info.get("is_success"):
                print("Reached the goal!")
                break 

        print(f"Eval Episode {i+1} Return: {ep_ret:.2f}")
        
        # Save the final frame of the rollout
        if len(frames) > 0:
            plt.imsave(f"pg_continuous_eval_{i+1}.png", frames[-1])
            print(f"Saved final frame to pg_continuous_eval_{i+1}.png")
            
    env.close()

if __name__ == "__main__":
    train()
