"""
Training script for Snake RL agents using Stable-Baselines3.
Supports DQN and PPO algorithms with automatic hardware detection.
"""

import argparse
import os
from datetime import datetime

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from snake_env import SnakeEnv


def detect_hardware():
    """Detect available hardware and return optimal settings."""
    # Detect CPU cores
    cpu_count = os.cpu_count() or 4

    # Detect CUDA
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        # More GPU memory = larger batch size
        if gpu_memory >= 8:
            batch_size = 256
        elif gpu_memory >= 4:
            batch_size = 128
        else:
            batch_size = 64
    elif torch.backends.mps.is_available():
        # Apple Silicon MPS
        device = "mps"
        gpu_name = "Apple MPS"
        gpu_memory = None
        batch_size = 128
    else:
        device = "cpu"
        gpu_name = None
        gpu_memory = None
        batch_size = 64

    # Number of parallel envs based on CPU cores (leave some for system)
    n_envs = max(4, min(cpu_count - 2, 16))

    return {
        "device": device,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
        "cpu_count": cpu_count,
        "n_envs": n_envs,
        "batch_size": batch_size,
    }


def print_hardware_info(hw: dict):
    """Print detected hardware information."""
    print("=" * 50)
    print("Hardware Detection")
    print("=" * 50)
    print(f"CPU cores: {hw['cpu_count']}")
    if hw['gpu_name']:
        print(f"GPU: {hw['gpu_name']}")
        if hw['gpu_memory']:
            print(f"GPU Memory: {hw['gpu_memory']:.1f} GB")
    print(f"Device: {hw['device']}")
    print(f"Parallel envs: {hw['n_envs']}")
    print(f"Batch size: {hw['batch_size']}")
    print("=" * 50)


def create_env(grid_size: int = 20):
    """Create a Snake environment."""
    def _init():
        return SnakeEnv(grid_size=grid_size)
    return _init


def main():
    # Detect hardware first
    hw = detect_hardware()

    parser = argparse.ArgumentParser(description="Train Snake RL agent")
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["dqn", "ppo"],
        help="RL algorithm to use (default: ppo)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1000000)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help=f"Number of parallel environments (default: auto={hw['n_envs']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Batch size (default: auto={hw['batch_size']})"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"],
        help=f"Device to use (default: auto={hw['device']})"
    )
    parser.add_argument(
        "--grid-size", type=int, default=20,
        help="Grid size (default: 20)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for TensorBoard logs (default: logs)"
    )
    args = parser.parse_args()

    # Use detected values if not specified
    n_envs = args.n_envs if args.n_envs is not None else hw['n_envs']
    batch_size = args.batch_size if args.batch_size is not None else hw['batch_size']
    device = args.device if args.device and args.device != "auto" else hw['device']

    # Update hw dict for printing
    hw['n_envs'] = n_envs
    hw['batch_size'] = batch_size
    hw['device'] = device

    print_hardware_info(hw)

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_{timestamp}"
    save_path = os.path.join(args.save_dir, run_name)
    log_path = os.path.join(args.log_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print(f"\nTraining {args.algo.upper()} for {args.timesteps:,} timesteps")
    print(f"Grid size: {args.grid_size}, Food: 1-4 (dynamic)")
    print(f"Models will be saved to: {save_path}")
    print(f"TensorBoard logs: {log_path}\n")

    # Create environments
    if args.algo == "ppo":
        env = make_vec_env(
            create_env(args.grid_size),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )
    else:
        # DQN uses single environment
        env = SnakeEnv(grid_size=args.grid_size)

    # Create evaluation environment
    eval_env = SnakeEnv(grid_size=args.grid_size)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs if args.algo == "ppo" else 50_000,
        save_path=save_path,
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=10_000 // n_envs if args.algo == "ppo" else 10_000,
        deterministic=True,
        render=False,
    )

    # Create model with auto-detected settings
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
        )
    else:  # DQN
        # DQN buffer size scales with available memory
        buffer_size = 100_000 if device == "cpu" else 500_000
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=buffer_size,
            learning_starts=10_000,
            batch_size=batch_size,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_path,
            device=device,
        )

    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
