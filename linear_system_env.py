import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Dict, Any, List

"""
A simple discrete linear dynamical system environment.

Dynamics:
x_{k+1} = A x_k + B u_k + w_k

where:
A = [[1, 1], [0, 1]]
B = [[0], [1]]
w_k ~ N(0, W)

"""
class LinearDynamicalSystemEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, 
        render_mode: Optional[str] = None, 
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[Union[float, np.ndarray]] = None,
        process_noise_cov: Optional[np.ndarray] = None, 
        max_state_bound: float = 500.0, 
        discrete_actions: Optional[np.ndarray] = None,
        max_action: float = 10, # apply only for continuous action space
        init_state_bound: float = 50,
        goal_threshold: float = 3.0,
        success_steps: int = 20,
        render_bound: float = 80.0
    ):
        super().__init__()
        
        # System matrices
        # Default to double integrator
        self.A = A if A is not None else np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        self.B = B if B is not None else np.array([[0.0], [1.0]], dtype=np.float32)
        
        # Cost matrices
        self.Q = Q if Q is not None else np.eye(2, dtype=np.float32)
        self.R = R if R is not None else np.array([[0.1]], dtype=np.float32)
        
        # Process noise covariance W
        if process_noise_cov is None:
            self.W = np.eye(2, dtype=np.float32) * 0.1
        else:
            self.W = np.array(process_noise_cov, dtype=np.float32)

        self.max_state_bound = max_state_bound
        self.max_action = max_action
        self.init_state_bound = init_state_bound
        self.goal_threshold = goal_threshold
        self.success_steps = success_steps
        self.render_bound = render_bound
        self.steps_in_goal = 0

        # Define action and observation space
        if discrete_actions is not None:
            self.discrete_action_values = np.array(discrete_actions, dtype=np.float32)
            self.action_space = spaces.Discrete(len(self.discrete_action_values))
        else:
            self.discrete_action_values = None
            self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32)
        
        # Observation is state x (2D vector)
        # Bounded by max_state_bound
        high = np.array([self.max_state_bound, self.max_state_bound], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.render_mode = render_mode
        self.state = None
        self.trajectory = []
        
        # For visualization
        self.fig = None
        self.ax = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize state randomly or at a specific point
        # Start in a smaller region so it's controllable with limited actions
        self.state = self.np_random.uniform(low=-self.init_state_bound, high=self.init_state_bound, size=(2,)).astype(np.float32)
        
        self.trajectory = [self.state.copy()]
        self.steps_in_goal = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.state, {}

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.discrete_action_values is not None:
            # Map discrete action index to continuous control input
            u = np.array([self.discrete_action_values[action]], dtype=np.float32)
        else:
            u = np.array(action, dtype=np.float32)
            
        # Clip action to valid range
        # u = np.clip(u, -self.max_action, self.max_action)
        
        # Process noise w_k ~ N(0, W)
        w = self.np_random.multivariate_normal(np.zeros(2), self.W).astype(np.float32)
        
        # Dynamics: x_{k+1} = A x_k + B u_k + w_k
        # B is (2,1), u is (1,). B@u gives (2,)
        # If action is scalar from Box(1,), it might be array([u]).
        
        bu = self.B @ u
        # bu is (2,)
        
        self.state = (self.A @ self.state + bu + w).astype(np.float32)
        
        self.trajectory.append(self.state.copy())
        
        # Reward
        state_cost = self.state.T @ self.Q @ self.state
        action_cost = u.T @ self.R @ u
        reward = - (state_cost + action_cost).item()
        
        # Termination condition
        # Stop if state gets too large (unstable) or after fixed steps (handled by TimeLimit wrapper usually)
        out_of_bounds = bool(np.any(np.abs(self.state) > self.max_state_bound))

        # Success condition: stabilized near origin
        in_goal = bool(np.linalg.norm(self.state) < self.goal_threshold)
        if in_goal:
            self.steps_in_goal += 1
        else:
            self.steps_in_goal = 0
        success = self.steps_in_goal >= self.success_steps
        

        # We do NOT terminate on success for stabilization tasks.
        # The agent must learn to maintain the state at the origin.
        terminated = out_of_bounds
        truncated = False # Usually handled by TimeLimit wrapper
        
        info = {
            "is_success": success,
            "out_of_bounds": out_of_bounds
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            # Set limits based on max_state_bound so going off-screen corresponds to termination
            limit = self.render_bound
            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_xlabel("x1")
            self.ax.set_ylabel("x2")
            self.ax.grid(True)
            self.line, = self.ax.plot([], [], 'b-o', alpha=0.5, label='Trajectory')
            self.current_point, = self.ax.plot([], [], 'ro', label='Current State')
            self.ax.legend()

        # Update data
        traj = np.array(self.trajectory)
        self.line.set_data(traj[:, 0], traj[:, 1])
        self.current_point.set_data([self.state[0]], [self.state[1]])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        if self.render_mode == "rgb_array":
            # Convert canvas to image
            try:
                # For newer matplotlib versions
                self.fig.canvas.draw()
                width, height = self.fig.canvas.get_width_height()
                buffer = self.fig.canvas.buffer_rgba()
                image = np.asarray(buffer)
                return image[:, :, :3].copy() # Return RGB, drop Alpha
            except AttributeError:
                # Fallback for older versions or different backends
                self.fig.canvas.draw()
                data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return data.copy()

    def close(self):
        if self.fig:
            plt.close(self.fig)
