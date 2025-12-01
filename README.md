RL Snake (PPO/DQN)
==================

Python 3.12+ project for training a Snake agent with Stable-Baselines3 (PPO or DQN) on a toroidal 20x20 grid. Supports local play, TensorBoard logging, checkpoints, and a live-play script for driving a browser Snake via screen capture + key events.

Quick Start
-----------
1) Install deps (uv or pip):
   - `uv pip install -r pyproject.toml` or `pip install -e .`
2) Train PPO with default settings:
   - `uv run python train.py --algo ppo --timesteps 1_000_000`
3) View logs:
   - `tensorboard --logdir logs`
4) Watch the trained model:
   - `uv run python play.py --model models/<run>/best_model`

Key Scripts
-----------
- `train.py`: main training loop. Flags:
  - `--algo {ppo,dqn}` (default ppo)
  - `--timesteps INT` total steps
  - `--n-envs INT` parallel envs (auto by CPU)
  - `--batch-size INT` (auto by device)
  - `--device {cpu,cuda,mps}` or `--gpu` to force GPU
  - `--grid-size INT` (default 20)
  - `--n-eval-episodes INT` eval episodes per check (default 20)
  - `--render` show eval GUI (slower)
  - `--save-dir`, `--log-dir` for outputs
- `play.py`: human play or watch a trained model. Use `--model PATH` to load.
- `live_play.py`: run a trained policy on the browser Snake via screen capture/keypress emulation (requires mss/pyautogui/OpenCV/Pillow).
- `resume_train.py`: resume from a saved checkpoint.
- `main.py`: minimal entry to start human play.

Environment (snake_env.py)
--------------------------
- Grid: 20x20 (configurable), wraparound edges (toroidal). Only self-collision ends an episode.
- Observation: 16-d vector = 8 directions × [distance to food, distance to body]; distances normalized to [0,1], missing target -> 0. Vision wraps, so the snake “sees” through edges.
- Actions: 4 discrete (UP/RIGHT/DOWN/LEFT). Reverse moves are penalized.
- Reward: +10 on food, -10 on self-collision (termination), -0.01 per step, -5 if wandering too long without food (`max_steps_without_food = grid_size^2 * 2`).
- Rendering: PyGame; tensorboard logs under `logs/`; models/checkpoints under `models/`.

Models
------
- Policies are MLP (`MlpPolicy`) consuming the 16-d observation.
  - PPO: net_arch [128, 128], lr 3e-4, n_steps 2048, n_epochs 4, gamma 0.99, ent_coef 0.005.
  - DQN: net_arch [128, 128], lr 1e-4, target_update_interval 1000, exploration_fraction 0.1 -> final_eps 0.05.
- Parallel envs via `SubprocVecEnv` for PPO; single env for DQN.

Tips
----
- Increase `--n-eval-episodes` to reduce eval noise; use `--render` sparingly.
- If GPU is slower for this small MLP, keep `--device cpu`.
- To debug behavior, run `play.py --model ... --render` and watch for self-collisions vs. food seeking.

Project Structure
-----------------
- `snake_env.py`  Gymnasium env.
- `train.py`      Training entrypoint.
- `play.py`       Human play / watch model.
- `live_play.py`  Browser-control bridge.
- `resume_train.py` Resume from checkpoint.
