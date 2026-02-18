import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from stable_baselines3 import PPO
from env.cartpole_true_swingup import CartPoleTrueSwingUp
import numpy as np

env = CartPoleTrueSwingUp()
model = PPO.load("models/ppo_true_swingup")


obs, _ = env.reset()

stable_steps = 0
near_upright_steps = 0
max_theta_reached = 0

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    env.render()

    x, x_dot, theta, theta_dot = obs

    max_theta_reached = max(max_theta_reached, abs(theta))

    # Count near-upright
    if abs(theta) < 0.2:
        near_upright_steps += 1

    # Count very stable (tight threshold)
    if abs(theta) < 0.1 and abs(theta_dot) < 0.5:
        stable_steps += 1

    print(f"Step: {step} | Theta: {theta:.3f} | Theta_dot: {theta_dot:.3f} | Reward: {reward:.3f}")

    if terminated or truncated:
        print("Episode ended")
        break

print("\n==== PERFORMANCE SUMMARY ====")
print("Near upright steps (<0.2 rad):", near_upright_steps)
print("Stable steps (<0.1 rad & low velocity):", stable_steps)
print("Max theta reached:", max_theta_reached)

