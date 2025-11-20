import numpy as np
import scipy.linalg
import gymnasium as gym
from gymnasium.envs.registration import register
import time
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple

# Register the environment
# We use a try-except block to avoid re-registration errors if this script is run multiple times
# or if the environment is registered elsewhere.
try:
    register(
        id="LinearSystem-v0",
        entry_point="linear_system_env:LinearDynamicalSystemEnv",
        max_episode_steps=200,
    )
except gym.error.Error:
    pass

def solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the discrete time LQR controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    Returns the feedback gain matrix K such that u[k] = -K x[k]
    """
    # Solve Discrete Algebraic Riccati Equation (DARE)
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    
    # Compute the feedback gain matrix K
    # K = (R + B^T P B)^-1 B^T P A
    
    bt_p_b = B.T @ P @ B
    inv_term = np.linalg.inv(R + bt_p_b)
    bt_p_a = B.T @ P @ A
    
    K = inv_term @ bt_p_a
    
    return K

def main():
    # Define system matrices for LQR design
    # These must match the environment's internal dynamics
    A = np.array([[1.0, 1.0], 
                  [0.0, 1.0]])
    B = np.array([[0.0], 
                  [1.0]])
    
    # LQR weights
    Q = np.eye(2)       # State cost
    R = np.array([[0.1]]) # Control cost (scalar 0.1)
    
    print("Computing LQR gain...")
    K = solve_lqr(A, B, Q, R)
    print(f"LQR Gain K: {K}")
    
    # Create environment
    # Pass Q and R to ensure the environment uses the same cost function for reward calculation
    os.makedirs("outputs", exist_ok=True)
    env = gym.make(
        "LinearSystem-v0", 
        render_mode="rgb_array",
        A=A,
        B=B,
        Q=Q,
        R=R
    )

    num_episodes = 16
    success_count = 0
    episode_rewards = []
    final_frames = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        # print(f"Initial state: {observation}")
        
        done = False
        truncated = False
        step = 0
        total_reward = 0.0
        
        while not (done or truncated):
            # LQR Control Law: u = -K x
            # observation is x (2,)
            # K is (1, 2)
            # u should be (1,)
            
            u = -K @ observation

            # Step the environment
            observation, reward, done, truncated, info = env.step(u)
            
            total_reward += reward

            if info.get("is_success"):
                # print("Reached the goal!")
                pass
            
            step += 1
            
        print(f"Episode finished after {step} steps. Total Reward: {total_reward:.4f}")
        
        episode_rewards.append(total_reward)
        
        final_frames.append(env.render())
        
        if info.get("is_success"):
            # print("SUCCESS: System stabilized!")
            success_count += 1
        elif info.get("out_of_bounds"):
            print("FAILURE: System went out of bounds.")
            
    print(f"\n=== Summary over {num_episodes} episodes ===")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.4f}")
    
    env.close()
    
    # Create 4x4 grid of images
    if len(final_frames) == 16:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(final_frames[idx])
            ax.axis('off')
            ax.set_title(f"Ep {idx+1}")
        
        plt.tight_layout()
        plt.savefig("outputs/lqr_eval_grid.png")
        print("Saved 4x4 evaluation grid to outputs/lqr_eval_grid.png")
    else:
        print(f"Expected 16 frames, got {len(final_frames)}")

if __name__ == "__main__":
    main()
