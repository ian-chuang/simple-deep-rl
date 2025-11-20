import numpy as np
import scipy.linalg
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Register environment (ignore if exists)
try:
    register(
        id="LinearSystem-v0",
        entry_point="linear_system_env:LinearDynamicalSystemEnv",
        max_episode_steps=200,
    )
except gym.error.Error:
    pass

# -------------------------------
# CONFIG
# -------------------------------
NUM_EPISODES = 16
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------
# LQR Solver
# -------------------------------
def solve_lqr(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Setting up LQR evaluation...")

    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    B = np.array([[0.0],
                  [1.0]])

    Q = np.eye(2)
    R = np.array([[0.1]])

    print("Computing LQR gain...")
    K = solve_lqr(A, B, Q, R)
    print("LQR Gain K:", K)

    env = gym.make(
        "LinearSystem-v0",
        render_mode="rgb_array",
        A=A,
        B=B,
        Q=Q,
        R=R,
    )

    episode_rewards = []
    final_frames = []
    success_count = 0

    # ---------------------------
    # Evaluation episodes
    # ---------------------------
    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        print(f"\nEpisode {ep+1}/{NUM_EPISODES}")

        while not (done or truncated):
            u = -K @ obs
            obs, reward, done, truncated, info = env.step(u)
            total_reward += reward
            steps += 1

        print(f" steps={steps}, total_reward={total_reward:.4f}")
        episode_rewards.append(total_reward)
        final_frames.append(env.render())

        if info.get("is_success"):
            success_count += 1

    env.close()

    # ---------------------------
    # Summary
    # ---------------------------
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / NUM_EPISODES * 100.0

    print("\n=== Evaluation Summary ===")
    print(f"Success: {success_count}/{NUM_EPISODES} ({success_rate:.1f}%)")
    print(f"Average Reward: {avg_reward:.4f}")

    # ---------------------------
    # Save 4x4 grid
    # ---------------------------
    grid_path = os.path.join(OUTPUT_DIR, "lqr_eval_grid.png")

    if len(final_frames) == 16:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(final_frames[idx])
            ax.axis("off")
            ax.set_title(f"Ep {idx+1}")

        plt.tight_layout()
        plt.savefig(grid_path)
        print("Saved evaluation grid to:", grid_path)
    else:
        print(f"Expected 16 frames but got {len(final_frames)}.")

    print("Done.")


if __name__ == "__main__":
    main()
