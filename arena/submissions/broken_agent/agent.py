"""
A broken agent for testing error logging
"""
from environment import Move
import random

class Agent:
    def __init__(self, role: str):
        self.role = role
        self.step_count = 0
    
    def step(self, observation, my_position, enemy_position, current_step):
        """
        This agent will throw an error after 5 steps to test error logging
        """
        self.step_count += 1
        
        # Intentionally cause an error after 5 steps
        if self.step_count > 5:
            raise RuntimeError(f"Intentional error for testing at step {current_step}")
        
        # Before the error, just move randomly
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])
