import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import time
from stable_baselines3 import PPO
from env.cartpole_true_swingup import CartPoleTrueSwingUp

# Load environment
env = CartPoleTrueSwingUp()

# Load trained model
model = PPO.load("ppo_true_swingup")   # adjust path if needed

num_episodes = 10
success_count = 0

ANGLE_THRESHOLD = np.deg2rad(10)
STABLE_STEPS_REQUIRED = 100

for episode in range(num_episodes):

    obs, _ = env.reset()
    stable_steps = 0
    done = False

    print(f"\nEpisode {episode+1}")

    while not done:

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        x, x_dot, theta, theta_dot = obs

        # Render for visualization
        env.render()
        time.sleep(0.01)

        # Check stabilization
        if abs(theta) < ANGLE_THRESHOLD:
            stable_steps += 1
        else:
            stable_steps = 0

        if stable_steps >= STABLE_STEPS_REQUIRED:
            print("Stabilized successfully!")
            success_count += 1
            break

        done = terminated or truncated

print("\n================================")
print(f"Success Rate: {success_count}/{num_episodes}")
print("================================")

