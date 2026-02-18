import pandas as pd
import matplotlib.pyplot as plt

# Load CSV exported from TensorBoard
df = pd.read_csv("results/plots/PPO_3.csv")

# TensorBoard CSV usually has columns: Step, Value
plt.figure(figsize=(10,5))
plt.plot(df["Step"], df["Value"])
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Reward Curve (PPO True Swing-Up)")
plt.grid(True)
plt.tight_layout()

plt.savefig("results/plots/reward_curve_final.png")
plt.show()

print("Saved reward_curve.png successfully.")

