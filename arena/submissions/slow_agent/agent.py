import sys
import time
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        time.sleep(5)
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        time.sleep(5)
        return Move.STAY
