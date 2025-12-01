"""
Resume training from a saved PPO/DQN model.

Example:
    uv run python resume_train.py --model models/ppo_20240101_120000/best_model.zip \\
        --algo ppo --timesteps 500000 --device mps
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from train import SnakeCNN, create_env, detect_hardware, print_hardware_info
from snake_env import SnakeEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Resume training a Snake RL agent")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model zip")
    parser.add_argument(
        "--algo", type=str, choices=["ppo", "dqn"], required=True, help="Algorithm"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Additional timesteps to train (default: 500000)",
    )
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size (default: 20)")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Parallel envs for PPO (default: auto from hardware)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU/MPS if available (overrides auto CPU)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render eval games (slower; uses human render_mode)",
    )
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for outputs (default: <algo>_resume_<timestamp>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    use_gpu = args.gpu or (args.device in ["cuda", "mps"])
    hw = detect_hardware(use_gpu=use_gpu)

    device = args.device if args.device else hw["device"]
    n_envs = args.n_envs if args.n_envs is not None else hw["n_envs"]

    run_name = args.run_name or f"{args.algo}_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(args.save_dir, run_name)
    log_path = os.path.join(args.log_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print_hardware_info({**hw, "device": device, "n_envs": n_envs})
    print(f"\nResuming {args.algo.upper()} from {args.model}")
    print(f"Saving to: {save_path}")
    print(f"Logs: {log_path}\n")

    # Create environments
    if args.algo == "ppo":
        env = make_vec_env(
            create_env(args.grid_size),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )
    else:
        env = SnakeEnv(grid_size=args.grid_size)

    eval_env = SnakeEnv(
        grid_size=args.grid_size,
        render_mode="human" if args.render else None,
    )

    # Callbacks
    checkpoint_freq = 50_000 // n_envs if args.algo == "ppo" else 50_000
    eval_freq = 10_000 // n_envs if args.algo == "ppo" else 10_000

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=args.render,
    )

    # Load model (SnakeCNN is imported so the extractor class is available)
    if args.algo == "ppo":
        model = PPO.load(args.model, env=env, tensorboard_log=log_path, device=device)
    else:
        model = DQN.load(args.model, env=env, tensorboard_log=log_path, device=device)

    # Continue training
    print("Starting resumed training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\nResume complete. Final model saved to: {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
