# Task Completion Checklist
- If dependencies changed, run `uv sync` to update the virtual env/lock.
- No automated tests/lint configured; do manual sanity as needed.
- For gameplay or env changes: `uv run python main.py` for human play smoke test; `uv run python play.py --mode random --episodes 1` for quick env check (fast).
- For training changes: run a short smoke train `uv run python train.py --algo ppo --timesteps 1000` (adjust lower if necessary) to ensure script starts and saves.
- For live-play changes: optional manual check `uv run python live_play.py --model models/<run>/best_model.zip --algo ppo --debug` (requires trained model and screen capture setup).
- After changes, ensure scripts still parse CLI args: `uv run python play.py --help`, `uv run python train.py --help`, `uv run python live_play.py --help`.
- Confirm key dirs exist/created as needed: `models/`, `logs/`. Clean up temp artifacts before committing.
