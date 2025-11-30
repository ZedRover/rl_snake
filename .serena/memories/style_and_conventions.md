# Style and Conventions
- Language: Python 3.12+ with type hints on public functions/methods; docstrings used on key functions (triple-quoted).
- Naming: snake_case for functions/vars, UPPER_SNAKE for constants; classes in PascalCase.
- CLI: argparse for scripts (`train.py`, `play.py`, `live_play.py`); avoid global side effects besides entrypoint guards.
- Logging/UX: simple `print` statements; rich/tqdm used for progress; keep user messaging concise.
- Formatting/linting: no tools configured in pyproject (no black/ruff config). Default to PEP 8 style and keep code minimal/clear.
- RL specifics: use Gymnasium interfaces (`action_space`, `observation_space`, `reset`, `step`, `render`), Stable-Baselines3 PPO/DQN helpers, PyTorch devices (CPU default, optional CUDA/MPS).