import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class CartPoleTrueSwingUp(gym.Env):

    def __init__(self):
        super().__init__()

        # Physical parameters
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        self.tau = 0.02
        self.max_force = 20.0

        # Continuous force action
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: x, x_dot, theta, theta_dot
        high = np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.max_steps = 800
        self.step_count = 0

        # Rendering
        plt.ion()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # Start pole downward
        theta = np.random.uniform(-np.pi, np.pi)
        self.state = np.array([0.0, 0.0, theta, 0.0])

        return self.state.astype(np.float32), {}

    def step(self, action):

        self.step_count += 1

        x, x_dot, theta, theta_dot = self.state

        force = float(action[0]) * self.max_force

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Integrate dynamics
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        # Wrap angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot])

        # ---------------- REWARD ----------------

        upright = np.exp(-4 * theta**2)

        reward = (
         1.5 * np.cos(theta)           # swing objective
         + 4.0 * upright               # strong upright bonus
         - 0.2 * upright * x**2        # center cart ONLY near upright
         - 0.02 * upright * x_dot**2
         - 0.01 * theta_dot**2
        )


        # ---------------------------------------

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def render(self):

        x, _, theta, _ = self.state

        pole_length = self.length * 2

        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)

        plt.clf()
        plt.xlim(-3, 3)
        plt.ylim(-2, 2)

        # Cart
        plt.plot([x - 0.3, x + 0.3], [0, 0], linewidth=8)

        # Pole
        plt.plot([x, pole_x], [0, pole_y], linewidth=4)

        plt.pause(0.001)

