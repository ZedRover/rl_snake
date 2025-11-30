"""
Snake Game Environment with Gymnasium interface.
Features wraparound boundaries (toroidal grid) - snake only dies from self-collision.
Visual style matches the target web game: dark background, cyan snake, red food.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Direction vectors (row, col)
DIRECTION_VECTORS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

# Colors matching the target game
COLOR_BACKGROUND = (20, 25, 30)  # Dark background
COLOR_SNAKE = (64, 224, 208)     # Cyan/turquoise snake
COLOR_FOOD = (220, 60, 60)       # Red food
COLOR_GRID_LINE = (30, 35, 40)   # Subtle grid lines


class SnakeEnv(gym.Env):
    """
    Snake game environment with wraparound boundaries.

    The snake wraps around edges (exits one side, appears on opposite side).
    Death only occurs from self-collision.
    Multiple food items can spawn on the board.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 20,
        min_food: int = 1,
        max_food: int = 4,
        render_mode: Optional[str] = None,
        cell_size: int = 30,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.min_food = min_food
        self.max_food = max_food
        self.render_mode = render_mode
        self.cell_size = cell_size

        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)

        # Observation space (channel-first):
        # 0: head mask, 1: body mask, 2: food mask,
        # 3: direction d_row (0,0.5,1 mapped from -1/0/1),
        # 4: direction d_col (0,0.5,1 mapped from -1/0/1),
        # 5: hunger fraction (0-1)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, grid_size, grid_size),
            dtype=np.float32,
        )

        # Pygame setup
        self.window = None
        self.clock = None
        self.window_size = grid_size * cell_size

        # Game state
        self.snake = []  # List of (row, col) positions, head is first
        self.direction = RIGHT
        self.food_positions = set()
        self.score = 0
        self.steps = 0
        self.max_steps_without_food = grid_size * grid_size * 2
        self.steps_since_food = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize snake in center, facing right
        center = self.grid_size // 2
        self.snake = [
            (center, center),      # Head
            (center, center - 1),  # Body
            (center, center - 2),  # Tail
        ]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        # Spawn food
        self.food_positions = set()
        self._spawn_food(initial=True)

        observation = self._get_observation()
        info = {"score": self.score}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        self.steps_since_food += 1

        reward = -0.01  # Small step penalty to encourage efficiency

        # Prevent 180-degree turns (apply penalty if attempted)
        opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action == opposite_directions.get(self.direction):
            reward -= 0.2  # discourage invalid reverse inputs
        else:
            self.direction = action

        # Calculate new head position with wraparound
        head_row, head_col = self.snake[0]
        d_row, d_col = DIRECTION_VECTORS[self.direction]
        new_head = (
            (head_row + d_row) % self.grid_size,
            (head_col + d_col) % self.grid_size,
        )

        # Check for self-collision (excluding tail which will move)
        # We check against all body parts except the last one (tail)
        if new_head in self.snake[:-1]:
            # Game over - self collision
            observation = self._get_observation()
            return observation, -10.0, True, False, {"score": self.score}

        # Move snake
        self.snake.insert(0, new_head)

        # Check if food eaten
        if new_head in self.food_positions:
            self.food_positions.remove(new_head)
            self.score += 10
            self.steps_since_food = 0
            reward += 10.0
            self._spawn_food()
        else:
            self.snake.pop()  # Remove tail if no food eaten

        # Check for timeout (snake wandering too long without eating)
        truncated = self.steps_since_food >= self.max_steps_without_food
        if truncated:
            reward -= 5.0  # explicit penalty for stalling

        observation = self._get_observation()
        info = {"score": self.score}

        return observation, reward, False, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Generate channel-first observation with direction and hunger context."""
        channels = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)

        # Mark snake body/head
        for i, (row, col) in enumerate(self.snake):
            if i == 0:
                channels[0, row, col] = 1.0  # Head mask
            else:
                channels[1, row, col] = 1.0  # Body mask

        # Mark food
        for row, col in self.food_positions:
            channels[2, row, col] = 1.0

        # Direction (broadcast across grid)
        d_row, d_col = DIRECTION_VECTORS[self.direction]
        channels[3] = (d_row + 1.0) / 2.0  # map -1/0/1 -> 0/0.5/1
        channels[4] = (d_col + 1.0) / 2.0  # map -1/0/1 -> 0/0.5/1

        # Hunger fraction since last food (0-1)
        hunger_frac = min(
            1.0, self.steps_since_food / float(self.max_steps_without_food)
        )
        channels[5] = hunger_frac

        return channels

    def _spawn_food(self, initial: bool = False) -> None:
        """Spawn food at random empty positions.

        Args:
            initial: If True, spawn random number of food (1-max_food).
                    If False (after eating), randomly decide whether to spawn new food,
                    but always ensure at least min_food exists.
        """
        empty_positions = []
        snake_set = set(self.snake)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in snake_set and (row, col) not in self.food_positions:
                    empty_positions.append((row, col))

        if initial:
            # Initial spawn: random number between min and max
            target_food = self.np_random.integers(self.min_food, self.max_food + 1)
            while len(self.food_positions) < target_food and empty_positions:
                idx = self.np_random.integers(len(empty_positions))
                pos = empty_positions.pop(idx)
                self.food_positions.add(pos)
        else:
            # After eating: ensure minimum, randomly add more
            # Always ensure at least min_food
            while len(self.food_positions) < self.min_food and empty_positions:
                idx = self.np_random.integers(len(empty_positions))
                pos = empty_positions.pop(idx)
                self.food_positions.add(pos)

            # 50% chance to spawn additional food if below max
            if len(self.food_positions) < self.max_food and empty_positions:
                if self.np_random.random() < 0.5:
                    idx = self.np_random.integers(len(empty_positions))
                    pos = empty_positions.pop(idx)
                    self.food_positions.add(pos)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake RL")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create surface
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(COLOR_BACKGROUND)

        # Draw grid lines (subtle)
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, COLOR_GRID_LINE,
                (0, i * self.cell_size), (self.window_size, i * self.cell_size)
            )
            pygame.draw.line(
                canvas, COLOR_GRID_LINE,
                (i * self.cell_size, 0), (i * self.cell_size, self.window_size)
            )

        # Draw food (red circles)
        for row, col in self.food_positions:
            center = (
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2,
            )
            pygame.draw.circle(canvas, COLOR_FOOD, center, self.cell_size // 3)

        # Draw snake (cyan rectangles with small gaps between segments)
        gap = 2
        for i, (row, col) in enumerate(self.snake):
            rect = pygame.Rect(
                col * self.cell_size + gap,
                row * self.cell_size + gap,
                self.cell_size - 2 * gap,
                self.cell_size - 2 * gap,
            )
            pygame.draw.rect(canvas, COLOR_SNAKE, rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# Register the environment
gym.register(
    id="Snake-v0",
    entry_point="snake_env:SnakeEnv",
)
