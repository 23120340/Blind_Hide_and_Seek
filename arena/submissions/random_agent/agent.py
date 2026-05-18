"""Valid random baseline agent."""

import random
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.rng = random.Random(11)

    def step(self, map_state, my_position, enemy_position, step_number):
        moves = MOVES[:]
        self.rng.shuffle(moves)
        for move in moves:
            nxt = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            if 0 <= nxt[0] < map_state.shape[0] and 0 <= nxt[1] < map_state.shape[1] and map_state[nxt] != 1:
                return move, 1
        return Move.STAY, 1


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(22)

    def step(self, map_state, my_position, enemy_position, step_number):
        moves = MOVES[:]
        self.rng.shuffle(moves)
        for move in moves:
            nxt = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            if 0 <= nxt[0] < map_state.shape[0] and 0 <= nxt[1] < map_state.shape[1] and map_state[nxt] != 1:
                return move
        return Move.STAY
