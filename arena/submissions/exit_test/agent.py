import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from agent_interface import PacmanAgent as BasePacmanAgent, GhostAgent as BaseGhostAgent
from environment import Move

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def step(self, map_state, my_position, enemy_position, step_number):
        if step_number >= 2:
            sys.exit(0)   # simulate student calling sys.exit()
        return Move.RIGHT

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY
