"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """
    Example Pacman agent using a simple greedy strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Breadth First Search"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Memory for limited observation mode
        self.prob_matrix = np.zeros((21, 21))
        self.last_known_enemy_pos = None
        self.current_path = []
        self.current_target = None
    def update_prob_state(self, map_state, enemy_position):
        # 1. Ghost is visible
        if enemy_position is not None:
            self.prob_matrix.fill(0.0)
            self.prob_matrix[enemy_position] = 1.0
            self.last_known_enemy_pos = enemy_position
            return

        
        if self.last_known_enemy_pos is None:
            return

        # 2. Ghost is in the fog 
        
        new_belief = np.zeros_like(self.prob_matrix)

        # Vectorized shifts (UP, DOWN, LEFT, RIGHT)
        new_belief[:-1, :] += self.prob_matrix[1:, :]  # Up
        new_belief[1:, :] += self.prob_matrix[:-1, :]  # Down
        new_belief[:, :-1] += self.prob_matrix[:, 1:]  # Left
        new_belief[:, 1:] += self.prob_matrix[:, :-1]  # Right
        
        # Add the current state as well (assuming the Ghost might use Move.STAY)
        new_belief += self.prob_matrix

        # 3. Environment Masking
        
        walls = (map_state == 1)
        visible_empty_spaces = (map_state == 0)
        
        new_belief[walls] = 0.0
        new_belief[visible_empty_spaces] = 0.0

        # 4. Normalize the probabilities so they sum back to 1.0
        total_probability = np.sum(new_belief)
        if total_probability > 0:
            self.prob_matrix = new_belief / total_probability
        else:
            
            self.prob_matrix.fill(0.0)

    def bfs_search(self, start, goal, map_state):
        """
        Finds the shortest path from start to goal using BFS.
        Returns a list of Move enums.
        """
        # Queue stores tuples of: (current_pos, path_taken_to_get_here)
        queue = deque([(start, [])])
        
        # Track visited nodes to prevent infinite loops
        visited = {start}

        while queue:
            # Pop the oldest item from the left side of the queue
            current_pos, path = queue.popleft()

            # If we reached the target, return the sequence of moves
            if current_pos == goal:
                return path

            # Explore valid neighbors
            for neighbor_pos, move in self._get_neighbors(current_pos, map_state):
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    
                    # Push the neighbor to the right side of the queue
                    queue.append((neighbor_pos, path + [move]))

        # Return empty list if no path is found
        return []
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Simple greedy strategy: move towards the ghost.
        
        When enemy_position is None (limited observation mode),
        uses last known position or explores randomly.
        
        Students should implement better search algorithms like:
        - BFS (Breadth-First Search)
        - DFS (Depth-First Search)
        - A* Search
        - Greedy Best-First Search
        - etc.
        """
        self.update_prob_state(map_state, enemy_position)
        target_position = None  
        # Update memory if enemy is visible
        if enemy_position is not None:
            target_position = enemy_position
        
        elif self.last_known_enemy_pos is not None:
           
            target_position = np.unravel_index(np.argmax(self.prob_matrix), self.prob_matrix.shape)
            
            
            if self.prob_matrix[target_position] == 0.0:
                 target_position = None
        
        if target_position is not None and (not self.current_path or target_position != self.current_target):
            
            self.current_path = self.bfs_search(my_position, target_position, map_state)
            self.current_target = target_position
            
        
        if self.current_path:
            next_move = self.current_path.pop(0)
            
            # Double check that the move is actually valid (in case a ghost moved)
            if self._is_valid_position(self._apply_move(my_position, next_move), map_state):
                 return (next_move, 1) if self.pacman_speed > 1 else next_move
            else:
                 
                 self.current_path = []
        
        
        valid_moves = self._get_neighbors(my_position, map_state)
        if valid_moves:
            import random
            fallback_move = random.choice(valid_moves)[1]
            return (fallback_move, 1) if self.pacman_speed > 1 else fallback_move
            
        return Move.STAY if self.pacman_speed == 1 else (Move.STAY, 1)

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] != 1

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1

    def _apply_move(self, pos, move):
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
class GhostAgent(BaseGhostAgent):
    """
    Example Ghost agent using a simple evasive strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "BFS escape"
        # Memory for limited observation mode
        self.prob_matrix = np.zeros((21, 21))
        self.last_known_enemy_pos = None
        self.current_path = []
    def update_prob_state(self, map_state, enemy_position):
        """Tracks Pacman's probable location when hidden."""
        if enemy_position is not None:
            self.prob_matrix.fill(0.0)
            self.prob_matrix[enemy_position] = 1.0
            self.last_known_enemy_pos = enemy_position
            return

        if self.last_known_enemy_pos is None:
            return

        new_belief = np.zeros_like(self.prob_matrix)
        
        
        new_belief[:-1, :] += self.prob_matrix[1:, :]  # Up
        new_belief[1:, :] += self.prob_matrix[:-1, :]  # Down
        new_belief[:, :-1] += self.prob_matrix[:, 1:]  # Left
        new_belief[:, 1:] += self.prob_matrix[:, :-1]  # Right
        new_belief += self.prob_matrix                 # Stay

        walls = (map_state == 1)
        visible_empty_spaces = (map_state == 0)
        
        new_belief[walls] = 0.0
        new_belief[visible_empty_spaces] = 0.0

        total_prob = np.sum(new_belief)
        if total_prob > 0:
            self.prob_matrix = new_belief / total_prob
        else:
            self.prob_matrix.fill(0.0)
    def find_farthest_target(self, start_pos, threat_pos, map_state):
        
        queue = deque([(start_pos, [])])
        visited = {start_pos}
        
        best_target = start_pos
        max_safety_score = -1
        best_path = []

        
        max_search_depth = 30 

        while queue:
            current_pos, path = queue.popleft()
            
            
            
            dist_from_threat = self._manhattan_distance(current_pos, threat_pos)
            escape_routes = len(self._get_neighbors(current_pos, map_state))
            
            
            safety_score = dist_from_threat + (escape_routes * 0.5)
            
            if safety_score > max_safety_score:
                max_safety_score = safety_score
                best_target = current_pos
                best_path = path
                
            if len(path) >= max_search_depth:
                continue

            for neighbor_pos, move in self._get_neighbors(current_pos, map_state):
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    queue.append((neighbor_pos, path + [move]))

        return best_path
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Simple evasive strategy: move away from Pacman.
        
        When enemy_position is None (limited observation mode),
        uses last known position or moves randomly.
        
        Students should implement better search algorithms like:
        - BFS to find furthest point
        - A* to plan escape route
        - Minimax for adversarial search
        - etc.
        """
        self.update_prob_state(map_state, enemy_position)
        
        threat_pos = None

        
        if enemy_position is not None:
            threat_pos = enemy_position
        elif self.last_known_enemy_pos is not None:
            threat_pos = np.unravel_index(np.argmax(self.prob_matrix), self.prob_matrix.shape)
            if self.prob_matrix[threat_pos] == 0.0:
                 threat_pos = None

        if enemy_position is not None:
            self.current_path = []
                 
        
        if threat_pos is not None and not self.current_path:
            full_escape_path = self.find_farthest_target(my_position, threat_pos, map_state)
            
            
            
            self.current_path = full_escape_path[:3]
            
        # Follow the burst path
        if self.current_path:
            next_move = self.current_path.pop(0)
            
            # Verify the move is still valid 
            if self._is_valid_position(self._apply_move(my_position, next_move), map_state):
                return next_move
            else:
                self.current_path = [] # Path blocked, wipe memory
        
        # --- FALLBACK ---
        valid_moves = self._get_neighbors(my_position, map_state)
        if valid_moves:
            import random
            return random.choice(valid_moves)[1]
            
        return Move.STAY

    def _random_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """Random movement when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] != 1
    def _apply_move(self, pos, move):
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])