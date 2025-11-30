# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A reinforcement learning project to train agents to play Snake using Stable-Baselines3 (DQN and PPO). The project includes a custom Snake game simulator with wraparound boundaries (no wall death - snake wraps to opposite side, only self-collision causes death). Trained agents can be deployed to control real web-based Snake games via screen capture and keyboard simulation.

## Development Commands

```bash
# Install dependencies
uv sync

# Play game manually (human mode)
uv run python main.py

# Train with PPO (recommended, auto-detects hardware)
uv run python train.py --algo ppo --timesteps 1000000

# Train with DQN
uv run python train.py --algo dqn --timesteps 1000000

# Train with GPU (if needed, but CPU is faster for MLP policy)
uv run python train.py --algo ppo --gpu

# Train with visualization during evaluation
uv run python train.py --algo ppo --render

# Play with trained model in simulator
uv run python play.py --mode model --model models/<run_name>/best_model.zip --algo ppo

# Control REAL game with trained agent
uv run python live_play.py --model models/<run_name>/best_model.zip --algo ppo --debug

# Monitor training with TensorBoard
uv run tensorboard --logdir logs
```

## Architecture

### Game Environment (`snake_env.py`)
- Custom Gymnasium environment implementing Snake with toroidal (wraparound) boundaries
- Observation space: 20x20 grid (0=empty, 1=body, 2=head, 3=food)
- Action space: 4 discrete actions (UP=0, RIGHT=1, DOWN=2, LEFT=3)
- Reward: +10 food, -10 death, -0.01 step penalty
- Food: 1-4 items (dynamic), at least 1 always present

### Training (`train.py`)
- Auto-detects CPU/GPU, defaults to CPU (faster for MLP policy)
- PPO: SubprocVecEnv for parallel training (auto-scaled to CPU cores)
- DQN: single environment, experience replay
- `--gpu`: force GPU usage, `--render`: show evaluation games

### Play (`play.py`)
- `--mode human`: keyboard control (arrows/WASD, Q to quit)
- `--mode model`: watch trained agent in simulator
- `--mode random`: random actions for testing

### Live Play (`live_play.py`)
- Controls real Snake games via screen capture + keyboard simulation
- Click-drag region selector to capture game area
- Image processing extracts game state (HSV color detection)
- Sends arrow keys via pyautogui
- `--debug`: shows detection overlay
