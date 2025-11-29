# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A reinforcement learning project to train agents to play Snake using Stable-Baselines3 (DQN and PPO). The project includes a custom Snake game simulator with wraparound boundaries (no wall death - snake wraps to opposite side, only self-collision causes death).

## Development Commands

```bash
# Install dependencies
uv sync

# Play game manually (human mode)
uv run python main.py

# Train with DQN
uv run python train.py --algo dqn --timesteps 1000000

# Train with PPO (faster with parallel envs)
uv run python train.py --algo ppo --timesteps 1000000 --n-envs 8

# Play with trained model
uv run python play.py --mode model --model models/<run_name>/best_model.zip --algo ppo

# Play with random actions (for testing)
uv run python play.py --mode random

# Monitor training with TensorBoard
uv run tensorboard --logdir logs
```

## Architecture

### Game Environment (`snake_env.py`)
- Custom Gymnasium environment implementing Snake with toroidal (wraparound) boundaries
- Observation space: 20x20 grid (0=empty, 1=body, 2=head, 3=food)
- Action space: 4 discrete actions (UP=0, RIGHT=1, DOWN=2, LEFT=3)
- Reward: +10 food, -10 death, -0.01 step penalty
- Visual style: dark background (#141A1E), cyan snake (#40E0D0), red food (#DC3C3C)

### Key Game Rules
- Snake wraps around edges (toroidal grid)
- Death only from self-collision
- 2 food items spawn by default
- Timeout after grid_size^2 * 2 steps without eating

### Training (`train.py`)
- PPO: uses SubprocVecEnv for parallel training (default 8 envs)
- DQN: single environment, experience replay buffer
- Saves checkpoints every 50k steps, evaluates every 10k steps
- Models saved to `models/<algo>_<timestamp>/`

### Play (`play.py`)
- `--mode human`: keyboard control (arrows/WASD, Q to quit)
- `--mode model`: watch trained agent play
- `--mode random`: random actions for testing
