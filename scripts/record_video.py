import sys
import os

import matplotlib
matplotlib.use("Agg") 

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from stable_baselines3 import PPO
from env.cartpole_true_swingup import CartPoleTrueSwingUp

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("results/video", exist_ok=True)

# -----------------------------
# Load trained model
# -----------------------------
model = PPO.load("models/ppo_true_swingup")

# -----------------------------
# Initialize environment
# -----------------------------
env = CartPoleTrueSwingUp()

# Random initial angles each episode
num_episodes = 3  # you can record multiple runs
frames_per_episode = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1

        # Capture frame
        x, _, theta, _ = obs
        pole_length = env.length * 2
        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot([x - 0.3, x + 0.3], [0, 0], linewidth=8, color='black')  # cart base
        ax.plot([x, pole_x], [0, pole_y], linewidth=4, color='red')       # pole
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.axis('off')
        fig.canvas.draw()

        # Convert figure to numpy array
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, :3]
        frames_per_episode.append(frame)
        plt.close(fig)

        done = terminated or truncated

# -----------------------------
# Save frames as video
# -----------------------------
metadata = dict(title='CartPole True Swing Up', artist='Abhinav')
writer = FFMpegWriter(fps=60, metadata=metadata)

video_path = "results/video/ppo_true_swingup.mp4"
fig = plt.figure()
with writer.saving(fig, video_path, dpi=100):
    for frame in frames_per_episode:
        plt.imshow(frame)
        plt.axis('off')
        writer.grab_frame()
        plt.clf()
plt.close(fig)

print(f"Video saved at {video_path}")

