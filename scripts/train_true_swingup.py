import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.cartpole_true_swingup import CartPoleTrueSwingUp

# -----------------------------
# Create folders if they don't exist
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# -----------------------------
# Initialize environment with Monitor for logging
# -----------------------------
env = CartPoleTrueSwingUp()
env = Monitor(env, filename="results/plots/monitor.csv")

# -----------------------------
# Initialize PPO model
# -----------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
    n_steps=2048,
    batch_size=64,
    tensorboard_log="results/plots/"   # enable logging for TensorBoard
)

# -----------------------------
# Train the model
# -----------------------------
model.learn(total_timesteps=1_000_000)

# -----------------------------
# Save the trained model
# -----------------------------
model.save("models/ppo_true_swingup")
print("Training complete!")

# -----------------------------
# Plot and save training reward curve
# -----------------------------
# monitor.csv has a header row we need to skip
df = pd.read_csv("results/plots/monitor.csv", skiprows=1)

plt.figure(figsize=(10,5))
plt.plot(df['r'])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/reward_curve.png")
plt.show()

print("Training results saved in 'results/plots/'")

