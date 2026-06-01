import sys
from pathlib import Path
from collections import deque
import heapq
import random
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# get_default_walls removed to strictly follow Fog of War rules.

def bfs_path(start, targets, known_spaces):
    """
    Finds the shortest cell path from start to any target in targets.
    Returns: a list of cells [start, cell1, cell2, ..., target]
    """
    if start in targets:
        return [start]
        
    h, w = known_spaces.shape
    visited = {start}
    queue = deque([[start]])
    
    while queue:
        path = queue.popleft()
        curr = path[-1]
        
        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) in targets:
                    return path + [(nr, nc)]
                if known_spaces[nr, nc]:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(path + [(nr, nc)])
                        
    return []

# ============================================================================
# PACMAN AGENT (THE GRANDMASTER SEEKER)
# ============================================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Apex Pacman"
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        
        self.known_spaces = None
        self.known_walls = None
        self.belief_map = None
        self.last_seen_pos = None
        self.last_seen_step = -100
        
        self.prediction_history = []
        self.my_history = deque(maxlen=10)
        
        # We target (5, 12) immediately first, since many ghosts camp there.
        # Then, we follow a patrol route of other corners/spots.
        self.patrol_route = [
            (5, 12), (1, 19), (1, 1), (5, 5), (15, 5), (19, 1), (19, 19), (15, 15)
        ]
        self.patrol_idx = 0
        self.patrol_target = None

    def _get_reachable_frontiers(self, my_position):
        h, w = self.known_spaces.shape
        visited = {my_position}
        queue = deque([my_position])
        frontiers = []
        
        while queue:
            curr = queue.popleft()
            r, c = curr
            
            # Check if curr is a frontier
            is_frontier = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not self.known_spaces[nr, nc] and not self.known_walls[nr, nc]:
                        is_frontier = True
                        break
            
            if is_frontier:
                frontiers.append(curr)
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.known_spaces[nr, nc]:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return frontiers

    def step(self, map_state, my_position, enemy_position, step_number):
        if step_number == 1:
            self.known_spaces = None
            self.known_walls = None
            self.belief_map = None
            self.last_seen_pos = None
            self.last_seen_step = -100
            self.prediction_history = []
            self.my_history.clear()
            self.patrol_idx = 0
            self.patrol_target = None
            
        # Initialize/update maps
        if self.known_spaces is None:
            self.known_spaces = (map_state == 0)
            self.known_walls = (map_state == 1)
        else:
            self.known_spaces = self.known_spaces | (map_state == 0)
            self.known_walls = self.known_walls | (map_state == 1)
            
        self.known_spaces[my_position[0], my_position[1]] = True
        self.my_history.append(my_position)
        
        # Update belief
        self._update_belief(map_state, enemy_position)
        
        # If enemy is visible, chase
        if enemy_position:
            self.last_seen_pos = enemy_position
            self.last_seen_step = step_number
            self.prediction_history.append(enemy_position)
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
                
            target = self._predict_ghost_target(enemy_position, my_position)
            path = bfs_path(my_position, {target}, self.known_spaces)
            if not path or len(path) < 2:
                path = bfs_path(my_position, {enemy_position}, self.known_spaces)
                
            if path and len(path) >= 2:
                next_cell = path[1]
                dr = next_cell[0] - my_position[0]
                dc = next_cell[1] - my_position[1]
                
                # Check consecutive steps in the same direction in the path
                consecutive = 1
                curr_dir = (dr, dc)
                for idx in range(1, len(path) - 1):
                    c_curr = path[idx]
                    c_next = path[idx + 1]
                    d_dir = (c_next[0] - c_curr[0], c_next[1] - c_curr[1])
                    if d_dir == curr_dir:
                        consecutive += 1
                    else:
                        break
                        
                available = self._max_steps(my_position, curr_dir)
                
                # If on straight line of sight to enemy, rush
                if (my_position[0] == enemy_position[0] or my_position[1] == enemy_position[1]) and self._has_line_of_sight(my_position, enemy_position):
                    optimal = min(available, self.pacman_speed)
                else:
                    optimal = min(consecutive, available, self.pacman_speed)
                    
                move = Move.STAY
                if dr == -1: move = Move.UP
                elif dr == 1: move = Move.DOWN
                elif dc == -1: move = Move.LEFT
                elif dc == 1: move = Move.RIGHT
                return (move, max(1, optimal))
                
            return (Move.STAY, 1)
            
        # If not visible:
        # 1. If enemy recently lost (< 8 steps), head towards nearest frontier to last seen position
        if self.last_seen_pos and (step_number - self.last_seen_step < 8):
            frontiers = self._get_reachable_frontiers(my_position)
            if frontiers:
                target = min(frontiers, key=lambda f: manhattan(f, self.last_seen_pos))
                path = bfs_path(my_position, {target}, self.known_spaces)
                if path and len(path) >= 2:
                    return self._move_along_path(my_position, path)
                    
        # 2. Otherwise, use frontier-based exploration to find unexplored areas
        frontiers = self._get_reachable_frontiers(my_position)
        if frontiers:
            target = frontiers[0] # Nearest frontier by path distance
            path = bfs_path(my_position, {target}, self.known_spaces)
            if path and len(path) >= 2:
                return self._move_along_path(my_position, path)
                
        # 3. Otherwise (entire map fully explored), follow patrol route (only to reachable patrol targets)
        reachable_patrols = [p for p in self.patrol_route if self.known_spaces[p[0], p[1]]]
        if reachable_patrols:
            for _ in range(len(self.patrol_route)):
                self.patrol_target = self.patrol_route[self.patrol_idx]
                if my_position != self.patrol_target and self.known_spaces[self.patrol_target[0], self.patrol_target[1]]:
                    break
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_route)
                
            path = bfs_path(my_position, {self.patrol_target}, self.known_spaces)
            if path and len(path) >= 2:
                return self._move_along_path(my_position, path)
            
        # Fallback random exploration using only known spaces
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for move in moves:
            dr, dc = move.value
            nr, nc = my_position[0] + dr, my_position[1] + dc
            if 0 <= nr < self.known_spaces.shape[0] and 0 <= nc < self.known_spaces.shape[1] and self.known_spaces[nr, nc]:
                return (move, 1)
                
        return (Move.STAY, 1)

    def _predict_ghost_target(self, enemy_pos, my_pos):
        dist = manhattan(my_pos, enemy_pos)
        if dist <= 4:
            return enemy_pos
            
        # Estimate velocity
        dr, dc = 0, 0
        if len(self.prediction_history) >= 2:
            last = self.prediction_history[-1]
            prev = self.prediction_history[-2]
            dr, dc = last[0] - prev[0], last[1] - prev[1]
            
        if dr == 0 and dc == 0:
            return enemy_pos
            
        lookahead = min(dist // 2, 4)
        curr_r, curr_c = enemy_pos
        h, w = self.known_spaces.shape
        for _ in range(lookahead):
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < h and 0 <= nc < w and self.known_spaces[nr, nc]:
                curr_r, curr_c = nr, nc
            else:
                break
        return (curr_r, curr_c)

    def _move_along_path(self, my_position, path):
        next_cell = path[1]
        dr = next_cell[0] - my_position[0]
        dc = next_cell[1] - my_position[1]
        
        consecutive = 1
        curr_dir = (dr, dc)
        for idx in range(1, len(path) - 1):
            c_curr = path[idx]
            c_next = path[idx + 1]
            d_dir = (c_next[0] - c_curr[0], c_next[1] - c_curr[1])
            if d_dir == curr_dir:
                consecutive += 1
            else:
                break
                
        available = self._max_steps(my_position, curr_dir)
        optimal = min(consecutive, available, self.pacman_speed)
        
        move = Move.STAY
        if dr == -1: move = Move.UP
        elif dr == 1: move = Move.DOWN
        elif dc == -1: move = Move.LEFT
        elif dc == 1: move = Move.RIGHT
        return (move, max(1, optimal))

    def _max_steps(self, pos, direction):
        dr, dc = direction
        steps = 0
        curr = pos
        h, w = self.known_spaces.shape
        for _ in range(self.pacman_speed):
            nr, nc = curr[0] + dr, curr[1] + dc
            if 0 <= nr < h and 0 <= nc < w and self.known_spaces[nr, nc]:
                steps += 1
                curr = (nr, nc)
            else:
                break
        return steps

    def _has_line_of_sight(self, p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if not self.known_spaces[r1, c]:
                    return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if not self.known_spaces[r, c1]:
                    return False
            return True
        return False

    def _update_belief(self, map_state, enemy_pos):
        h, w = map_state.shape
        if self.belief_map is None:
            self.belief_map = np.zeros((h, w), dtype=float)
            walkable_mask = ~self.known_walls
            self.belief_map[walkable_mask] = 1.0
            self.belief_map /= self.belief_map.sum()
            
        if enemy_pos:
            self.belief_map.fill(0.0)
            self.belief_map[enemy_pos] = 1.0
        else:
            # Propagate / diffuse belief
            new_belief = np.zeros_like(self.belief_map)
            for r in range(h):
                for c in range(w):
                    val = self.belief_map[r, c]
                    if val > 0:
                        neighbors = []
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w and not self.known_walls[nr, nc]:
                                neighbors.append((nr, nc))
                        if neighbors:
                            share = val / len(neighbors)
                            for nr, nc in neighbors:
                                new_belief[nr, nc] += share
                        else:
                            new_belief[r, c] += val
            self.belief_map = new_belief
            
            # Zero out visible region (since enemy is NOT seen here)
            my_pos = self.my_history[-1]
            visible_cells = {my_pos}
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in range(1, 6):
                    nr, nc = my_pos[0] + dr * dist, my_pos[1] + dc * dist
                    if 0 <= nr < h and 0 <= nc < w:
                        if self.known_walls[nr, nc]:
                            break
                        visible_cells.add((nr, nc))
                    else:
                        break
            for r, c in visible_cells:
                self.belief_map[r, c] = 0.0
                
            total = self.belief_map.sum()
            if total > 0:
                self.belief_map /= total
            else:
                # Reset
                walkable_mask = ~self.known_walls
                self.belief_map[walkable_mask] = 1.0
                self.belief_map /= self.belief_map.sum()

    def _get_highest_belief_cell(self, my_pos):
        if self.belief_map is None:
            return None
            
        flat_idx = np.argmax(self.belief_map)
        r, c = np.unravel_index(flat_idx, self.belief_map.shape)
        if self.belief_map[r, c] > 0:
            return (r, c)
            
        # Fallback to any non-wall
        h, w = self.belief_map.shape
        non_walls = []
        for r in range(h):
            for c in range(w):
                if not self.known_walls[r, c]:
                    non_walls.append((r, c))
        if non_walls:
            return min(non_walls, key=lambda p: manhattan(my_pos, p))
        return None

# ============================================================================
# GHOST AGENT (THE SHADOW PARKOUR)
# ============================================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Phantom Ghost"
        self.internal_map = None
        self.last_pos = None
        
        self.camping_target = (5, 12)
        self.is_camping = False
        self.has_spotted_pacman = False

    def _get_reachable_frontiers(self, my_position):
        h, w = self.internal_map.shape
        visited = {my_position}
        queue = deque([my_position])
        frontiers = []
        
        while queue:
            curr = queue.popleft()
            r, c = curr
            
            # Check if curr is a frontier
            is_frontier = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if self.internal_map[nr, nc] == -1:
                        is_frontier = True
                        break
            
            if is_frontier:
                frontiers.append(curr)
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.internal_map[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return frontiers

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_map(map_state)
        
        # Ensure current position is empty
        self.internal_map[my_position[0], my_position[1]] = 0
        
        if step_number <= 1:
            self.has_spotted_pacman = False
            self.is_camping = False
            self.last_pos = None
            
        if enemy_position:
            self.has_spotted_pacman = True
            self.is_camping = False
            
            self.internal_map[enemy_position[0], enemy_position[1]] = 0
            dist_map = self._bfs_distance(enemy_position)
            return self._escape_control(my_position, dist_map)
            
        # 1. Opening Camping Phase
        if not self.has_spotted_pacman:
            if my_position == self.camping_target:
                self.is_camping = True
                return Move.STAY
                
            if not self.is_camping:
                if self._is_valid(self.camping_target[0], self.camping_target[1]):
                    dist_to_camp = self._bfs_distance(self.camping_target)
                    if dist_to_camp[my_position[0], my_position[1]] != np.inf:
                        return self._move_towards_target(my_position, dist_to_camp)
                        
                frontiers = self._get_reachable_frontiers(my_position)
                if frontiers:
                    target_frontier = min(frontiers, key=lambda f: manhattan(f, self.camping_target))
                    dist_to_frontier = self._bfs_distance(target_frontier)
                    if dist_to_frontier[my_position[0], my_position[1]] != np.inf:
                        return self._move_towards_target(my_position, dist_to_frontier)
                        
                return self._junction_survival(my_position)
                
        # 2. Survival & Safe Zone Patrolling Phase
        return self._junction_survival(my_position)

    def _move_towards_target(self, my_pos, dist_map):
        best_move = Move.STAY
        min_dist = float('inf')
        for move in DIRS:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if self._is_valid(nr, nc):
                if dist_map[nr, nc] < min_dist:
                    min_dist = dist_map[nr, nc]
                    best_move = move
        self.last_pos = my_pos
        return best_move

    def _update_map(self, map_state):
        if self.internal_map is None:
            self.internal_map = np.full(map_state.shape, -1)
        self.internal_map[map_state != -1] = map_state[map_state != -1]

    def _is_valid(self, r, c):
        h, w = self.internal_map.shape
        return 0 <= r < h and 0 <= c < w and self.internal_map[r, c] == 0

    def _bfs_distance(self, source):
        h, w = self.internal_map.shape
        dist = np.full((h, w), np.inf)
        if not self._is_valid(source[0], source[1]):
            return dist
            
        queue = deque([source])
        dist[source] = 0
        
        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self._is_valid(nr, nc) and dist[nr, nc] == np.inf:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))
        return dist

    def _count_exits(self, r, c):
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid(r + dr, c + dc):
                count += 1
        return count

    def _escape_control(self, my_pos, dist_map):
        best_move = Move.STAY
        best_score = -float('inf')
        cur_r, cur_c = my_pos
        
        prev_dir = None
        if self.last_pos:
            prev_dir = (cur_r - self.last_pos[0], cur_c - self.last_pos[1])
            
        for move in DIRS:
            dr, dc = move.value
            nr, nc = cur_r + dr, cur_c + dc
            
            if not self._is_valid(nr, nc):
                continue
                
            exits = self._count_exits(nr, nc)
            d = dist_map[nr, nc]
            
            score = 0
            
            if exits <= 1:
                score -= 1000
            
            # BFS distance scoring
            score += (d - 2) * 40
            
            # Exit count scoring
            score += exits * 15
            
            # Turn/zigzag bonus
            if prev_dir and prev_dir != (dr, dc):
                score += 50
                
            # Backtracking penalty
            if self.last_pos == (nr, nc):
                score -= 70
                
            if score > best_score:
                best_score = score
                best_move = move
                
        self.last_pos = my_pos
        return best_move

    def _junction_survival(self, my_pos):
        best_move = Move.STAY
        best_score = -float('inf')
        safe_zone = 5
        
        for move in DIRS:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if not self._is_valid(nr, nc):
                continue
                
            exits = self._count_exits(nr, nc)
            score = 0
            score += exits * 30
            
            vertical = my_pos[0] - nr
            # 1. Out of safe zone: up priority
            if my_pos[0] > safe_zone:
                if exits >= 3:
                    score += vertical * 10
                elif exits == 2:
                    score += vertical * 5
            # 2. In safe zone: horizontal direction or stay
            else:
                if nr > safe_zone:
                    score -= 50
                if vertical == 0:
                    score += 10
                    
            # 3. Dead end penalty
            if exits <= 1:
                score -= 1000
                
            # Backtracking penalty
            if self.last_pos == (nr, nc):
                score -= 50
                
            if score > best_score:
                best_score = score
                best_move = move
                
        self.last_pos = my_pos
        return best_move

