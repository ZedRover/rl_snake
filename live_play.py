"""
Live play module - Control real Snake game using trained agent.

Features:
1. Screen region selector (click and drag)
2. Real-time screen capture
3. Image processing to extract game state
4. Keyboard control via arrow keys
"""

import argparse
import time
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image
from stable_baselines3 import DQN, PPO

# Disable pyautogui fail-safe for smoother control
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# Colors from the game (BGR format for OpenCV)
SNAKE_COLOR_BGR = (208, 224, 64)   # Cyan in BGR
FOOD_COLOR_BGR = (60, 60, 220)     # Red in BGR

# Tolerance for color matching
COLOR_TOLERANCE = 40


class RegionSelector:
    """Select a screen region by clicking and dragging."""

    def __init__(self):
        self.region = None
        self.selecting = False
        self.start_pos = None

    def select(self) -> Optional[dict]:
        """
        Open a fullscreen overlay to select a region.
        Returns: dict with keys 'left', 'top', 'width', 'height'
        """
        print("=== Screen Region Selector ===")
        print("1. Move this window aside")
        print("2. Press ENTER when ready to select")
        print("3. Click and drag to select the game area")
        print("4. Press ENTER to confirm or ESC to cancel")
        input("\nPress ENTER to start selection...")

        # Capture full screen
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Full screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Create window
        window_name = "Select Game Region - Drag to select, ENTER to confirm, ESC to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        clone = img.copy()
        rect_start = None
        rect_end = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal rect_start, rect_end, clone

            if event == cv2.EVENT_LBUTTONDOWN:
                rect_start = (x, y)
                rect_end = None

            elif event == cv2.EVENT_MOUSEMOVE and rect_start is not None:
                rect_end = (x, y)
                clone = img.copy()
                cv2.rectangle(clone, rect_start, rect_end, (0, 255, 0), 2)

            elif event == cv2.EVENT_LBUTTONUP:
                rect_end = (x, y)
                clone = img.copy()
                cv2.rectangle(clone, rect_start, rect_end, (0, 255, 0), 2)

        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            cv2.imshow(window_name, clone)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == 13:  # ENTER
                if rect_start and rect_end:
                    break

        cv2.destroyAllWindows()

        # Calculate region
        x1, y1 = rect_start
        x2, y2 = rect_end
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        self.region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        print(f"\nSelected region: {self.region}")
        return self.region


class GameStateExtractor:
    """Extract game state from screenshot."""

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size

    def extract(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Extract game state from screenshot.

        Args:
            screenshot: BGR image of the game area

        Returns:
            channels: (6, grid, grid) float32 observation matching SnakeEnv
                      [head, body, food, d_row, d_col, hunger]
        """
        h, w = screenshot.shape[:2]
        cell_h = h / self.grid_size
        cell_w = w / self.grid_size

        channels = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Cyan/turquoise snake: H=80-100 (in OpenCV H is 0-180)
        snake_lower = np.array([75, 100, 100])
        snake_upper = np.array([100, 255, 255])
        snake_mask = cv2.inRange(hsv, snake_lower, snake_upper)

        # Red food: H=0-10 or 170-180
        food_lower1 = np.array([0, 100, 100])
        food_upper1 = np.array([10, 255, 255])
        food_lower2 = np.array([170, 100, 100])
        food_upper2 = np.array([180, 255, 255])
        food_mask = cv2.inRange(hsv, food_lower1, food_upper1) | cv2.inRange(hsv, food_lower2, food_upper2)

        # Find snake head (the cell with most cyan that's connected to body)
        snake_cells = []
        head_pos = None

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = int(row * cell_h)
                y2 = int((row + 1) * cell_h)
                x1 = int(col * cell_w)
                x2 = int((col + 1) * cell_w)

                cell_snake = snake_mask[y1:y2, x1:x2]
                cell_food = food_mask[y1:y2, x1:x2]

                snake_ratio = np.sum(cell_snake > 0) / cell_snake.size
                food_ratio = np.sum(cell_food > 0) / cell_food.size

                if snake_ratio > 0.1:
                    snake_cells.append((row, col, snake_ratio))
                    channels[1, row, col] = 1.0  # Body mask (head fixed later)
                elif food_ratio > 0.05:
                    channels[2, row, col] = 1.0  # Food

        # Find head: the snake cell that's most "exposed" (at an end)
        # Simple heuristic: the cell with highest color intensity or at edge of snake
        second_pos = None
        if snake_cells:
            # Sort by ratio, highest is likely the head (more filled)
            snake_cells.sort(key=lambda x: x[2], reverse=True)
            head_row, head_col, _ = snake_cells[0]
            head_pos = (head_row, head_col)
            channels[0, head_row, head_col] = 1.0  # Head mask
            channels[1, head_row, head_col] = 0.0  # Remove from body mask

            # Find an adjacent body cell to infer direction
            for row, col, _ in snake_cells[1:]:
                if abs(row - head_row) + abs(col - head_col) == 1:
                    second_pos = (row, col)
                    break

        # Direction channels (broadcast)
        if head_pos and second_pos:
            d_row = head_pos[0] - second_pos[0]
            d_col = head_pos[1] - second_pos[1]
            d_row = max(-1, min(1, d_row))
            d_col = max(-1, min(1, d_col))
        else:
            d_row = 0.0
            d_col = 0.0

        # Map -1/0/1 -> 0/0.5/1 to stay within Box [0,1]
        channels[3] = (d_row + 1.0) / 2.0
        channels[4] = (d_col + 1.0) / 2.0

        # Hunger channel unknown in live play; leave at 0
        return channels

        return grid

    def debug_show(self, screenshot: np.ndarray, grid: np.ndarray):
        """Show debug visualization."""
        h, w = screenshot.shape[:2]
        cell_h = h / self.grid_size
        cell_w = w / self.grid_size

        debug_img = screenshot.copy()

        # Draw grid
        for i in range(self.grid_size + 1):
            y = int(i * cell_h)
            x = int(i * cell_w)
            cv2.line(debug_img, (0, y), (w, y), (50, 50, 50), 1)
            cv2.line(debug_img, (x, 0), (x, h), (50, 50, 50), 1)

        # Draw detected objects
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y = int((row + 0.5) * cell_h)
                x = int((col + 0.5) * cell_w)

                head = grid[0, row, col] > 0.5
                body = grid[1, row, col] > 0.5
                food = grid[2, row, col] > 0.5

                if head:
                    cv2.circle(debug_img, (x, y), 5, (0, 255, 0), -1)
                elif body:
                    cv2.circle(debug_img, (x, y), 3, (255, 255, 0), -1)
                elif food:
                    cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Debug", debug_img)
        cv2.waitKey(1)


class KeyboardController:
    """Send keyboard inputs to control the game."""

    # Action mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    ACTION_KEYS = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    def send_action(self, action: int):
        """Send arrow key for the given action."""
        key = self.ACTION_KEYS.get(action)
        if key:
            pyautogui.press(key)


class LivePlayer:
    """Main class to play real game with trained agent."""

    def __init__(
        self,
        model_path: str,
        algo: str = "ppo",
        grid_size: int = 20,
        fps: float = 10.0,
        debug: bool = False,
    ):
        self.grid_size = grid_size
        self.fps = fps
        self.debug = debug
        self.frame_time = 1.0 / fps

        # Load model
        print(f"Loading {algo.upper()} model from {model_path}...")
        if algo == "ppo":
            self.model = PPO.load(model_path)
        else:
            self.model = DQN.load(model_path)

        self.extractor = GameStateExtractor(grid_size)
        self.controller = KeyboardController()
        self.region = None

    def select_region(self):
        """Select game region."""
        selector = RegionSelector()
        self.region = selector.select()
        return self.region is not None

    def capture_screen(self) -> np.ndarray:
        """Capture the selected region."""
        with mss.mss() as sct:
            screenshot = sct.grab(self.region)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def run(self):
        """Main loop."""
        if not self.region:
            if not self.select_region():
                print("No region selected. Exiting.")
                return

        print("\n=== Starting Live Play ===")
        print(f"FPS: {self.fps}")
        print("Press Ctrl+C to stop\n")

        # Give user time to focus on game window
        print("Starting in 3 seconds... Focus on the game window!")
        time.sleep(3)

        try:
            while True:
                start_time = time.time()

                # Capture screen
                screenshot = self.capture_screen()

                # Extract game state
                grid = self.extractor.extract(screenshot)

                # Debug visualization
                if self.debug:
                    self.extractor.debug_show(screenshot, grid)

                # Get action from model
                action, _ = self.model.predict(grid, deterministic=True)

                # Send keyboard input
                self.controller.send_action(int(action))

                # Maintain FPS
                elapsed = time.time() - start_time
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Play real Snake game with trained agent")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["dqn", "ppo"],
        help="Algorithm used for the model (default: ppo)"
    )
    parser.add_argument(
        "--grid-size", type=int, default=20,
        help="Grid size to use for state extraction (default: 20)"
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Actions per second (default: 10)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show debug visualization"
    )
    args = parser.parse_args()

    player = LivePlayer(
        model_path=args.model,
        algo=args.algo,
        grid_size=args.grid_size,
        fps=args.fps,
        debug=args.debug,
    )
    player.run()


if __name__ == "__main__":
    main()
