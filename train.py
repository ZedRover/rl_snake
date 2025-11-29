"""
Training script for Snake RL agents using Stable-Baselines3.
Supports DQN and PPO algorithms.
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from snake_env import SnakeEnv


def create_env(grid_size: int = 20):
    """Create a Snake environment."""
    def _init():
        return SnakeEnv(grid_size=grid_size)
    return _init


def main():
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
        "--n-envs", type=int, default=8,
        help="Number of parallel environments (default: 8, only for PPO)"
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

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_{timestamp}"
    save_path = os.path.join(args.save_dir, run_name)
    log_path = os.path.join(args.log_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print(f"Training {args.algo.upper()} for {args.timesteps} timesteps")
    print(f"Grid size: {args.grid_size}, Food: 1-4 (dynamic)")
    print(f"Models will be saved to: {save_path}")
    print(f"TensorBoard logs: {log_path}")

    # Create environments
    if args.algo == "ppo":
        env = make_vec_env(
            create_env(args.grid_size),
            n_envs=args.n_envs,
            vec_env_cls=SubprocVecEnv,
        )
    else:
        # DQN uses single environment
        env = SnakeEnv(grid_size=args.grid_size)

    # Create evaluation environment
    eval_env = SnakeEnv(grid_size=args.grid_size)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=save_path,
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Create model
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_path,
        )
    else:  # DQN
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_path,
        )

    # Train
    print("\nStarting training...")
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
