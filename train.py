"""
Training script for Snake RL agents using Stable-Baselines3.
Supports DQN and PPO algorithms with automatic hardware detection.
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from snake_env import SnakeEnv


class SnakeCNN(BaseFeaturesExtractor):
    """Small CNN for 20x20 grid observations (channel-first)."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def detect_hardware(use_gpu: bool = False):
    """Detect available hardware and return optimal settings.

    For simple MLP policies, CPU is often faster than GPU due to:
    - Small network size (GPU parallelism not utilized)
    - CPU-GPU data transfer overhead per step
    - Environment runs on CPU anyway

    GPU is beneficial for CNN policies with image observations.
    """
    cpu_count = os.cpu_count() or 4

    # Detect available GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_device = "cuda"
    elif torch.backends.mps.is_available():
        gpu_name = "Apple MPS"
        gpu_memory = None
        gpu_device = "mps"
    else:
        gpu_name = None
        gpu_memory = None
        gpu_device = None

    # For MLP policy, CPU is faster; GPU only helps with CNN/large networks
    if use_gpu and gpu_device:
        device = gpu_device
        batch_size = 256 if gpu_memory and gpu_memory >= 8 else 128
    else:
        device = "cpu"
        batch_size = 64

    # Number of parallel envs based on CPU cores
    n_envs = max(4, min(cpu_count - 2, 16))

    return {
        "device": device,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
        "gpu_device": gpu_device,
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
        print(f"GPU available: {hw['gpu_name']}")
        if hw['gpu_memory']:
            print(f"GPU Memory: {hw['gpu_memory']:.1f} GB")
    print(f"Using device: {hw['device']}")
    print(f"Parallel envs: {hw['n_envs']}")
    print(f"Batch size: {hw['batch_size']}")
    print("=" * 50)


def create_env(grid_size: int = 20):
    """Create a Snake environment."""
    def _init():
        return SnakeEnv(grid_size=grid_size)
    return _init


def main():
    # Pre-parse to check for --gpu flag
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--gpu", action="store_true")
    pre_parser.add_argument("--device", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    # Determine if GPU should be used
    use_gpu = pre_args.gpu or (pre_args.device in ["cuda", "mps"])
    hw = detect_hardware(use_gpu=use_gpu)

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
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help=f"Device to use (default: cpu, GPU available: {hw['gpu_device'] or 'none'})"
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available (default: False, CPU is faster for MLP policy)"
    )
    parser.add_argument(
        "--grid-size", type=int, default=20,
        help="Grid size (default: 20)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Show GUI during evaluation (slower but visual)"
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
    device = args.device if args.device else hw['device']

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
    eval_env = SnakeEnv(
        grid_size=args.grid_size,
        render_mode="human" if args.render else None
    )

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
        render=args.render,
    )

    # Create model with auto-detected settings
    policy_kwargs = {
        "features_extractor_class": SnakeCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }

    if args.algo == "ppo":
        model = PPO(
            "CnnPolicy",
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
            policy_kwargs=policy_kwargs,
        )
    else:  # DQN
        # DQN buffer size scales with available memory
        buffer_size = 100_000 if device == "cpu" else 500_000
        model = DQN(
            "CnnPolicy",
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
            policy_kwargs=policy_kwargs,
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
