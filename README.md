# Safe Reinforcement Learning with Shielding

Implements a Safety Shield for a reinforcement learning agent to prevent it from taking unsafe actions during training and deployment. Combines Q-learning with a neural-network safety classifier that intervenes when the agent's intended action violates safety constraints.

## Overview

Standard RL agents can explore dangerous states during learning. This project implements the **shielding** paradigm: a safety monitor sits between the agent and the environment, blocking any action the shield classifies as unsafe and substituting a safe alternative. The approach is evaluated on a gridworld-style environment.

## Tech Stack

- **Language:** Python 3
- **Libraries:** TensorFlow/Keras, NumPy, matplotlib, pickle
- **Environment:** Jupyter Notebook

## Key Concepts

- Q-learning and tabular Q-table (`baseline_qtable.pkl`)
- Safety constraint specification and dataset collection
- Neural network safety classifier (`shield_model.keras`)
- Shield intervention: action blocking and safe-action substitution
- Comparison of unsafe vs. shielded agent trajectories
- Visualisation of safety violations and constraint satisfaction

## Project Structure

```
ai-safe-reinforcement-learning-shield/
├── safe_rl_shield.ipynb       # Main notebook
├── baseline_qtable.pkl             # Pre-trained Q-table
├── complete_dataset.pkl            # Full training dataset
├── safety_dataset.pkl              # Safety-labelled transitions
├── shield_model.keras              # Trained safety shield network
├── experiment_2_results.png        # Task 2 results
├── experiment_4_results.png        # Task 4 results
└── experiment_5_results.png        # Task 5 results
```

## How to Run

```bash
pip install tensorflow numpy matplotlib jupyter
jupyter notebook safe_rl_shield.ipynb
```

## Experiments

| Experiment | Description |
|------------|-------------|
| Exp 1 | Implement and train baseline Q-learning agent |
| Exp 2 | Collect safety-labelled transition dataset |
| Exp 3 | Train neural network safety shield classifier |
| Exp 4 | Integrate shield with RL agent — block unsafe actions |
| Exp 5 | Evaluate and compare shielded vs. unshielded performance |
