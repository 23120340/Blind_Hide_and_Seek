"""
A simple working agent for testing
"""
from environment import Move
import random

class PacmanAgent:
    def __init__(self, pacman_speed: int = 1):
        self.role = 'pacman'
        self.pacman_speed = pacman_speed
    
    def step(self, observation, my_position, enemy_position, current_step):
        """Move randomly."""
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])

class GhostAgent:
    def __init__(self):
        self.role = 'ghost'
    
    def step(self, observation, my_position, enemy_position, current_step):
        """Move randomly."""
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])
