import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from stable_baselines3 import PPO
from env.cartpole_true_swingup import CartPoleTrueSwingUp
import numpy as np

env = CartPoleTrueSwingUp()
model = PPO.load("ppo_true_swingup")

obs, _ = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    env.render()

    x, x_dot, theta, theta_dot = obs
    print(f"Theta: {theta:.3f}, Reward: {reward:.3f}")

    if terminated:
        print("Cart went out of bounds")
        break

