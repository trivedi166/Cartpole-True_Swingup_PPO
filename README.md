# CartPole True Swing-Up using PPO

This project implements a True CartPole Swing-Up task trained using Proximal Policy Optimization (PPO) from Stable-Baselines3.

The objective is to swing the pendulum from random initial angles and stabilize it in the upright position.

---

## Project Overview

- Environment: Custom True CartPole Swing-Up
- Algorithm: PPO (Proximal Policy Optimization)
- Framework: Stable-Baselines3
- Language: Python
- Reward-based reinforcement learning

The trained agent successfully:
- Swings up the pendulum from random initial angles
- Stabilizes it upright
- Maintains balance for long durations


---

## Setup Instructions

### Create Virtual Environment

```bash
python3 -m venv rl_env
source rl_env/bin/activate

---

### Install Dependencies
pip install stable-baselines3 gymnasium numpy pandas matplotlib

---

### Training the Agent 
python scripts/train_true_swingup.py
Training parameters:

Total timesteps: 800,000

Learning rate: 3e-4

Gamma: 0.99

Entropy coefficient: 0.05

n_steps: 2048

batch_size: 64

Training reward curve will be saved in:
results/plots/reward_curve_final.png

---

### Evaluating the Agent
python scripts/evaluate_true_swingup.py

The environment will:

Spawn pendulum at random initial angles

Swing up and stabilize

Render live visualization

---

### Results
Successful swing-up from random angles

Stable balancing achieved

Smooth reward convergence

Deterministic policy used for evaluation

Training curve included in results/.

---

### Demonstration
A screen recording of the trained agent stabilizing the pendulum is present in results/video



