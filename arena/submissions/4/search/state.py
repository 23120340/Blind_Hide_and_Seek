"""
Search state representation for Pacman search engine.
"""

from __future__ import annotations
from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
from environment import Move
import utils

@dataclass(frozen=True)
class SearchState:
    """
    Represents a game state for the adversarial search.
    
    Attributes:
        pacman_pos: (row, col) position of Pacman.
        ghost_pos: (row, col) position of the Ghost.
    """
    pacman_pos: Tuple[int, int]
    ghost_pos: Tuple[int, int]

    def __repr__(self) -> str:
        return f"SearchState(P={self.pacman_pos}, G={self.ghost_pos})"

    def to_tuple(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self.pacman_pos, self.ghost_pos)

    def generate_successors(
        self,
        map_array: np.ndarray,
        pacman_speed: int = 2,
    ) -> List[Tuple[utils.PacmanAction, Move, SearchState]]:
        """
        Generate legal joint actions and successor states from this state.
        Now uses generic helpers from utils.py.
        """
        p_actions = utils.legal_pacman_actions(map_array, self.pacman_pos, pacman_speed)
        g_moves = utils.legal_ghost_moves(map_array, self.ghost_pos)

        successors = []
        for p_act in p_actions:
            next_p = utils.apply_pacman_action(map_array, self.pacman_pos, p_act, pacman_speed)
            for g_mov in g_moves:
                next_g = utils.apply_move_once(map_array, self.ghost_pos, g_mov)
                successors.append((p_act, g_mov, SearchState(next_p, next_g)))
        
        return successors
