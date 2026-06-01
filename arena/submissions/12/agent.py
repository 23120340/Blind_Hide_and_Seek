import sys
from pathlib import Path
from collections import deque
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # State and memory
        self.walkable_tiles = None
        self.seen_tiles = set()
        self.ghost_candidates = set()
        
        self.exploration_state = 0
        self.moves_in_state = 0

    def _find_reachable_tiles(self, start: tuple, map_state: np.ndarray) -> set:
        reachable = {start}
        q = deque([start])
        height, width = map_state.shape
        while q:
            curr = q.popleft()
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                    npos = (nr, nc)
                    if npos not in reachable:
                        reachable.add(npos)
                        q.append(npos)
        return reachable

    def _bfs_path(self, start: tuple, goals: set, map_state: np.ndarray, pref_order=None) -> list:
        if start in goals:
            return [start]
        q = deque([[start]])
        visited = {start}
        height, width = map_state.shape
        
        if pref_order is None:
            pref_order = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            
        while q:
            path = q.popleft()
            curr = path[-1]
            for m in pref_order:
                nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                    npos = (nr, nc)
                    if npos in goals:
                        return path + [npos]
                    if npos not in visited:
                        visited.add(npos)
                        q.append(path + [npos])
        return None

    def _get_region(self, r: int, c: int, mid_r: int, mid_c: int) -> int:
        """
        0: Top-Right
        1: Top-Left
        2: Bottom-Left
        3: Bottom-Right
        """
        if r < mid_r and c >= mid_c: return 0
        if r < mid_r and c < mid_c: return 1
        if r >= mid_r and c < mid_c: return 2
        return 3

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        height, width = map_state.shape
        mid_r, mid_c = height // 2, width // 2

        # 1. Initialize reachable and walkable tiles
        if self.walkable_tiles is None:
            self.walkable_tiles = self._find_reachable_tiles(my_position, map_state)

        # 2. Update seen tiles: Any visible empty cell (0) is considered "seen"
        for r in range(height):
            for c in range(width):
                if map_state[r, c] == 0 or (r, c) == my_position:
                    self.seen_tiles.add((r, c))

        # 3. Update Ghost candidates
        if enemy_position is not None:
            self.ghost_candidates = {enemy_position}
        else:
            if self.ghost_candidates:
                new_candidates = set()
                for cand in self.ghost_candidates:
                    new_candidates.add(cand)
                    for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        nr, nc = cand[0] + m.value[0], cand[1] + m.value[1]
                        if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                            new_candidates.add((nr, nc))
                
                self.ghost_candidates = set()
                for cand in new_candidates:
                    r, c = cand
                    if map_state[r, c] == 0 or cand == my_position:
                        continue
                    self.ghost_candidates.add(cand)

        # 4. Determine Path Target
        path = None
        is_exploration = False
        pref_order = None
        
        if enemy_position is not None:
            # Pursuit Mode
            path = self._bfs_path(my_position, {enemy_position}, map_state)
        elif self.ghost_candidates:
            # Hunt Mode
            path = self._bfs_path(my_position, self.ghost_candidates, map_state)

        if path is None:
            # Exploration
            is_exploration = True
            unseen = self.walkable_tiles - self.seen_tiles
            
            if not unseen:
                # Map is fully explored, reset seen_tiles to start over
                self.seen_tiles = {
                    (r, c) for r in range(height) for c in range(width)
                    if map_state[r, c] == 0 or (r, c) == my_position
                }
                unseen = self.walkable_tiles - self.seen_tiles
                self.exploration_state = 0
                self.moves_in_state = 0

            unseen_TR = {pos for pos in unseen if self._get_region(pos[0], pos[1], mid_r, mid_c) == 0}
            unseen_TL = {pos for pos in unseen if self._get_region(pos[0], pos[1], mid_r, mid_c) == 1}
            unseen_BL = {pos for pos in unseen if self._get_region(pos[0], pos[1], mid_r, mid_c) == 2}
            unseen_BR = {pos for pos in unseen if self._get_region(pos[0], pos[1], mid_r, mid_c) == 3}
            unseen_Bottom = unseen_BL | unseen_BR
            unseen_Top = unseen_TR | unseen_TL

            targets = set()
            
            # State Machine Loop
            while True:
                if self.exploration_state == 0:
                    # Phase 0: Top-Right
                    if not unseen_TR or self.moves_in_state >= 3:
                        self.exploration_state = 1
                        self.moves_in_state = 0
                        continue
                    targets = unseen_TR
                    pref_order = [Move.RIGHT, Move.UP, Move.DOWN, Move.LEFT]
                    break
                    
                elif self.exploration_state == 1:
                    # Phase 1: Top-Left
                    if not unseen_TL:
                        self.exploration_state = 2
                        self.moves_in_state = 0
                        continue
                    targets = unseen_TL
                    pref_order = [Move.LEFT, Move.UP, Move.DOWN, Move.RIGHT]
                    break
                    
                elif self.exploration_state == 2:
                    # Phase 2: Bottom-Left
                    if not unseen_BL:
                        self.exploration_state = 3
                        self.moves_in_state = 0
                        continue
                    targets = unseen_BL
                    pref_order = [Move.LEFT, Move.DOWN, Move.UP, Move.RIGHT]
                    break
                    
                elif self.exploration_state == 3:
                    # Phase 3: Bottom
                    if not unseen_Bottom:
                        self.exploration_state = 4
                        self.moves_in_state = 0
                        continue
                    targets = unseen_Bottom
                    pref_order = [Move.DOWN, Move.RIGHT, Move.LEFT, Move.UP]
                    break
                    
                elif self.exploration_state == 4:
                    # Phase 4: Right then Top
                    if unseen_TR:
                        targets = unseen_TR
                        pref_order = [Move.RIGHT, Move.UP, Move.DOWN, Move.LEFT]
                    elif unseen_TL:
                        targets = unseen_TL
                        pref_order = [Move.LEFT, Move.UP, Move.DOWN, Move.RIGHT]
                    else:
                        targets = unseen
                        if not targets:
                            self.exploration_state = 0
                            self.moves_in_state = 0
                            continue
                    break
                    
                else:
                    self.exploration_state = 0
                    self.moves_in_state = 0
            
            path = self._bfs_path(my_position, targets, map_state, pref_order)

        # 5. Extract move with AGGRESSIVE straight-line speed multiplier
        if path and len(path) >= 2:
            p0 = path[0]
            p1 = path[1]
            d1 = (p1[0] - p0[0], p1[1] - p0[1])
            
            move = None
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if m.value == d1:
                    move = m
                    break
            
            if move is None:
                return (Move.STAY, 1)

            # Maximize steps in direction d1
            steps = 1
            curr_pos = p1
            
            for s in range(2, self.pacman_speed + 1):
                # If BFS path continues and turns, we MUST stop at the turn
                if len(path) > s:
                    next_node = path[s]
                    d_path = (next_node[0] - curr_pos[0], next_node[1] - curr_pos[1])
                    if d_path != d1:
                        break
                
                # Check if we can safely push forward straight
                nr, nc = curr_pos[0] + d1[0], curr_pos[1] + d1[1]
                if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                    curr_pos = (nr, nc)
                    steps = s
                    # Stop exactly on ghost if we hit it
                    if enemy_position is not None and curr_pos == enemy_position:
                        break
                else:
                    break
            
            # Record move for state 0 transition logic
            if is_exploration and self.exploration_state == 0:
                self.moves_in_state += 1
                
            return (move, steps)

        return (Move.STAY, 1)

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        height, width = map_state.shape
        
        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            nr, nc = my_position[0] + m.value[0], my_position[1] + m.value[1]
            if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                valid_moves.append((m, (nr, nc)))
                
        if enemy_position is not None:
            best_move = Move.STAY
            max_dist = -1
            for m, (nr, nc) in valid_moves:
                dist = abs(nr - enemy_position[0]) + abs(nc - enemy_position[1])
                if dist > max_dist:
                    max_dist = dist
                    best_move = m
            return best_move
            
        else:
            target = (height - 2, width - 2)
            best_move = Move.STAY
            min_dist = 9999
            for m, (nr, nc) in valid_moves:
                dist = abs(nr - target[0]) + abs(nc - target[1])
                if dist < min_dist:
                    min_dist = dist
                    best_move = m
            return best_move
