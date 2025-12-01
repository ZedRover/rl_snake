"""
Play/visualize Snake game with a trained model or human control.
"""

import argparse
import time

import pygame
from stable_baselines3 import DQN, PPO

from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT


def play_with_model(model_path: str, algo: str, episodes: int = 5):
    """Play the game with a trained model."""
    env = SnakeEnv(render_mode="human")

    # Load model
    if algo == "ppo":
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)

    print(f"Playing with {algo.upper()} model: {model_path}")

    for episode in range(episodes):
        obs, info = env.reset()
        env.render()  # Ensure window is initialized
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            env.render()
            total_reward += reward

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        print(f"Episode {episode + 1}: Score = {info['score']}, Total Reward = {total_reward:.2f}")
        time.sleep(1)

    env.close()


def play_human():
    """Play the game with keyboard controls."""
    env = SnakeEnv(render_mode="human")
    obs, info = env.reset()

    # Render first frame to initialize pygame window
    env.render()

    print("Human mode - Use arrow keys to control the snake. Press Q to quit.")

    key_to_action = {
        pygame.K_UP: UP,
        pygame.K_DOWN: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_RIGHT: RIGHT,
        pygame.K_w: UP,
        pygame.K_s: DOWN,
        pygame.K_a: LEFT,
        pygame.K_d: RIGHT,
    }

    current_action = RIGHT  # Initial direction
    running = True
    clock = pygame.time.Clock()

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in key_to_action:
                    current_action = key_to_action[event.key]

        if not running:
            break

        obs, reward, done, truncated, info = env.step(current_action)
        env.render()

        if done or truncated:
            print(f"Game Over! Score: {info['score']}")
            time.sleep(2)
            obs, info = env.reset()
            current_action = RIGHT
            env.render()

        clock.tick(10)  # 10 FPS

    env.close()


def play_random(episodes: int = 3):
    """Play with random actions for testing."""
    env = SnakeEnv(render_mode="human")

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        print(f"Episode {episode + 1}: Score = {info['score']}")
        time.sleep(1)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Play Snake game")
    parser.add_argument(
        "--mode", type=str, default="human", choices=["human", "model", "random"],
        help="Play mode: human (keyboard), model (trained agent), random"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model (required for model mode)"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["dqn", "ppo"],
        help="Algorithm used for the model (default: ppo)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to play (for model/random mode)"
    )
    args = parser.parse_args()

    if args.mode == "model":
        if args.model is None:
            print("Error: --model path required for model mode")
            return
        play_with_model(args.model, args.algo, args.episodes)
    elif args.mode == "human":
        play_human()
    else:
        play_random(args.episodes)


if __name__ == "__main__":
    main()
