import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
from stable_baselines3 import PPO
from env.cartpole_true_swingup import CartPoleTrueSwingUp

# Create folders if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

env = CartPoleTrueSwingUp()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.05,
    n_steps=2048,
    batch_size=64,
    tensorboard_log="results/plots/"   # enable logging
)

model.learn(total_timesteps=800000)

model.save("models/ppo_true_swingup")

print("Training complete!")

