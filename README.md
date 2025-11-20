# simple-deep-rl

Minimal reinforcement learning experiments for a custom linear dynamical system environment. The repository contains policy-gradient, PPO, DQN, and LQR baselines along with plotting utilities.

## Prerequisites
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management (pip-compatible alternative). Install with `pip install uv` or your preferred method.

## Setup
1. **Clone the repository**
	```bash
	git clone https://github.com/ian-chuang/simple-deep-rl.git
	cd simple-deep-rl
	```
2. **Install dependencies** (creates an isolated environment via uv)
	```bash
	uv sync
	```
3. **Run a script** (uv handles interpreter selection and dependencies)
	```bash
	uv run ppo_continuous.py
	```

Generated artifacts (evaluation grids, logs) are written under the `outputs/` and `runs/` directories.
