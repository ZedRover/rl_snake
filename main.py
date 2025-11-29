"""
RL Snake - Main entry point.
Launch the game in human play mode by default.
"""

from play import play_human


def main():
    print("RL Snake - Press arrow keys to control, Q to quit")
    play_human()


if __name__ == "__main__":
    main()
