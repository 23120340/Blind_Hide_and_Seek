import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


# =============================================================================
# PACMAN AGENT (Seeker) - Blind Seeker with beam intercept + choke analysis
# =============================================================================

class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Blind Seeker"

        self.global_map = None
        self.visit_count = None
        self.map_version = 0

        self.last_known_enemy_pos = None
        self.last_seen_step = -1

        self.recent_positions = []
        self.recent_limit = 12

        self.frontier_cache = None
        self.frontier_version = -1

        self.bfs_cache = {}
        self.bfs_cache_version = -1

        self.hypotheses = []
        self.hyp_limit = 8

        self.enemy_heat = None
        self.heat_decay = 0.97
        self.last_decay_step = -1

        self.articulation_set = set()
        self.corridor_set = set()
        self.choke_exits = set()
        self.choke_version = -1

        self.pac_dist_cache = None
        self.pac_dist_pos = None
        self.pac_dist_version = -1

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        self._ensure_memory(map_state)
        self._decay_enemy_heat(step_number)
        self._update_global_map(map_state)
        self._update_visit(my_position)
        self._update_choke_cache()

        if enemy_position is not None:
            self._observe_enemy(enemy_position, step_number)
            capture = self._capture_now(map_state, my_position, enemy_position)
            if capture is not None:
                return capture
            intercept = self._intercept_visible(map_state, my_position, enemy_position)
            if intercept is not None:
                return intercept
        else:
            pac_dist_map = self._get_pac_dist_map(my_position)
            self._propagate_hypotheses(map_state, my_position, pac_dist_map)

        target = self._choose_target(step_number)
        if target is not None:
            action = self._aggressive_move_toward_target(map_state, my_position, target)
            if action is not None:
                return action

        action = self._explore_action(map_state, my_position)
        if action is not None:
            return action

        fallback = self._safe_fallback(map_state, my_position)
        return fallback if fallback is not None else (Move.STAY, 1)

    def _ensure_memory(self, map_state: np.ndarray):
        if self.global_map is None:
            self.global_map = np.full(map_state.shape, -1, dtype=np.int8)
            self.visit_count = np.zeros(map_state.shape, dtype=np.int16)
            self.enemy_heat = np.zeros(map_state.shape, dtype=np.int16)
            self.map_version = 0
            self.last_decay_step = -1

    def _update_global_map(self, map_state: np.ndarray):
        seen_mask = map_state != -1
        if np.any(seen_mask):
            prev_unknown = self.global_map == -1
            if np.any(seen_mask & prev_unknown):
                self.map_version += 1
                self.frontier_version = -1
                self.bfs_cache.clear()
                self.bfs_cache_version = self.map_version
                self.choke_version = -1
                self.pac_dist_version = -1
            self.global_map[seen_mask] = map_state[seen_mask]

    def _update_visit(self, pos: tuple):
        r, c = pos
        self.visit_count[r, c] += 1
        self.recent_positions.append(pos)
        if len(self.recent_positions) > self.recent_limit:
            self.recent_positions.pop(0)

    def _observe_enemy(self, enemy_position: tuple, step_number: int):
        self.last_known_enemy_pos = enemy_position
        self.last_seen_step = step_number
        r, c = enemy_position
        self.enemy_heat[r, c] = min(30000, int(self.enemy_heat[r, c]) + 3)
        self.hypotheses = [(enemy_position, 1.0)]

    def _propagate_hypotheses(self, map_state: np.ndarray, my_position: tuple, pac_dist_map: np.ndarray):
        if not self.hypotheses:
            self._seed_hypotheses(map_state)
        visible = map_state != -1
        passable = self._passable_mask_belief(map_state)
        new_hyp = []
        for pos, w in self.hypotheses:
            if visible[pos[0], pos[1]]:
                continue
            for nxt in self._belief_neighbors(pos, passable):
                if not passable[nxt[0], nxt[1]]:
                    continue
                weight = w * self._enemy_motion_weight(nxt, pos, pac_dist_map, visible)
                new_hyp.append((nxt, weight))
        if not new_hyp:
            self._seed_hypotheses(map_state)
            return
        new_hyp.sort(key=lambda x: x[1], reverse=True)
        self.hypotheses = new_hyp[: self.hyp_limit]
        total = sum(h[1] for h in self.hypotheses)
        if total > 0:
            self.hypotheses = [(p, w / total) for (p, w) in self.hypotheses]

    def _seed_hypotheses(self, map_state: np.ndarray):
        passable = self._passable_mask_belief(map_state)
        visible = map_state != -1
        rows, cols = map_state.shape
        candidates = []
        for r in range(rows):
            for c in range(cols):
                if not passable[r, c] or visible[r, c]:
                    continue
                heat = float(self.enemy_heat[r, c])
                score = 1.0 + 0.4 * self._unknown_adjacent((r, c)) + 0.02 * heat
                candidates.append(((r, c), score))
        if not candidates:
            self.hypotheses = []
            return
        candidates.sort(key=lambda x: x[1], reverse=True)
        self.hypotheses = candidates[: self.hyp_limit]
        total = sum(h[1] for h in self.hypotheses)
        if total > 0:
            self.hypotheses = [(p, w / total) for (p, w) in self.hypotheses]

    def _capture_now(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple):
        if self.pacman_speed <= 0:
            return None
        dr = enemy_position[0] - my_position[0]
        dc = enemy_position[1] - my_position[1]
        if dr != 0 and dc != 0:
            return None
        dist = abs(dr) + abs(dc)
        if dist == 0:
            return (Move.STAY, 1)
        if dist > self.pacman_speed:
            return None
        move = None
        if dr < 0:
            move = Move.UP
        elif dr > 0:
            move = Move.DOWN
        elif dc < 0:
            move = Move.LEFT
        elif dc > 0:
            move = Move.RIGHT
        if move is None:
            return None
        steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
        if steps >= dist:
            return (move, min(steps, self.pacman_speed))
        return None

    def _intercept_visible(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple):
        pac_dist_map = self._get_pac_dist_map(my_position)
        beam_paths = self._beam_predict_paths(enemy_position, pac_dist_map, depth=5, width=5)
        target = self._select_intercept_target(beam_paths, pac_dist_map)
        if target is None:
            return self._chase_visible(map_state, my_position, enemy_position)
        return self._aggressive_move_toward_target(map_state, my_position, target)

    def _choose_target(self, step_number: int):
        if self.last_known_enemy_pos is not None and (step_number - self.last_seen_step) <= 6:
            if self._degree(self.last_known_enemy_pos) >= 3:
                return self.last_known_enemy_pos
            best = None
            best_deg = -1
            for nxt in self._planning_neighbors(self.last_known_enemy_pos):
                d = self._degree(nxt)
                if d > best_deg:
                    best_deg = d
                    best = nxt
            return best if best is not None else self.last_known_enemy_pos
        if self.hypotheses:
            return max(self.hypotheses, key=lambda x: x[1])[0]
        return None

    def _aggressive_move_toward_target(self, map_state: np.ndarray, my_position: tuple, target: tuple):
        passable_plan = self._passable_mask_planning()
        dist_map = self._bfs_distances(target, passable_plan)
        best_score = None
        best_action = None
        for move, steps, end_pos, path_cells in self._enumerate_actions(my_position, map_state):
            d = dist_map[end_pos[0], end_pos[1]]
            if d < 0:
                continue
            score = -2.0 * d
            score += 0.6 * self._info_gain_score(path_cells)
            score += 0.6 * self._loop_penalty(end_pos)
            if steps == 2:
                score += 0.8
            if best_score is None or score > best_score:
                best_score = score
                best_action = (move, steps)
        return best_action

    def _explore_action(self, map_state: np.ndarray, my_position: tuple):
        frontier = self._get_frontier_cells()
        if not frontier:
            return None
        passable_plan = self._passable_mask_planning()
        best_score = None
        best_action = None
        for move, steps, end_pos, path_cells in self._enumerate_actions(my_position, map_state):
            dist_map = self._bfs_distances(end_pos, passable_plan)
            best = None
            for f, info in frontier:
                d = dist_map[f[0], f[1]]
                if d < 0:
                    continue
                score = 1.6 * float(info) - 1.0 * float(d)
                if best is None or score > best:
                    best = score
            if best is None:
                continue
            score = best
            score += 0.9 * self._info_gain_score(path_cells)
            score += 0.5 * self._loop_penalty(end_pos)
            if steps == 2:
                score += 0.4
            if best_score is None or score > best_score:
                best_score = score
                best_action = (move, steps)
        return best_action

    def _enemy_motion_weight(self, nxt: tuple, cur: tuple, pac_dist_map: np.ndarray, visible: np.ndarray):
        dist_now = pac_dist_map[cur[0], cur[1]]
        dist_next = pac_dist_map[nxt[0], nxt[1]]
        if dist_now < 0 or dist_next < 0:
            away = 1.0
        else:
            away = 1.15 if dist_next > dist_now else 0.85
        unseen_bias = 1.2 if not visible[nxt[0], nxt[1]] else 0.8
        unknown_bias = 1.0 + 0.2 * self._unknown_adjacent(nxt)
        return away * unseen_bias * unknown_bias

    def _enumerate_actions(self, pos: tuple, map_state: np.ndarray):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            max_steps = self._max_valid_steps(pos, move, map_state, self.pacman_speed)
            for steps in range(1, max_steps + 1):
                end_pos, path_cells = self._advance_path(pos, move, steps)
                actions.append((move, steps, end_pos, path_cells))
        return actions

    def _advance_path(self, pos: tuple, move: Move, steps: int):
        path_cells = []
        r, c = pos
        dr, dc = move.value
        for _ in range(steps):
            r += dr
            c += dc
            path_cells.append((r, c))
        return (r, c), path_cells

    def _info_gain_score(self, path_cells):
        unseen = set()
        rows, cols = self.global_map.shape
        for r, c in path_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and self.global_map[nr, nc] == -1:
                    unseen.add((nr, nc))
        return float(len(unseen))

    def _get_frontier_cells(self):
        if self.frontier_version == self.map_version and self.frontier_cache is not None:
            return self.frontier_cache
        frontier = []
        rows, cols = self.global_map.shape
        for r in range(rows):
            for c in range(cols):
                if self.global_map[r, c] != 0:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and self.global_map[nr, nc] == -1:
                        frontier.append(((r, c), self._unknown_adjacent((r, c))))
                        break
        self.frontier_cache = frontier
        self.frontier_version = self.map_version
        return frontier

    def _loop_penalty(self, pos: tuple):
        r, c = pos
        penalty = 0.3 * float(self.visit_count[r, c])
        if pos in self.recent_positions:
            penalty += 1.0
        if len(self.recent_positions) >= 2 and pos == self.recent_positions[-2]:
            penalty += 0.6
        return -penalty

    def _safe_fallback(self, map_state: np.ndarray, my_position: tuple):
        best = None
        best_score = -1e9
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            steps = self._max_valid_steps(my_position, move, map_state, 1)
            if steps == 0:
                continue
            end_pos, _ = self._advance_path(my_position, move, 1)
            score = -self._loop_penalty(end_pos)
            if score > best_score:
                best_score = score
                best = (move, 1)
        return best

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _direction_to(self, src: tuple, dst: tuple):
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        if dr == -1 and dc == 0:
            return Move.UP
        if dr == 1 and dc == 0:
            return Move.DOWN
        if dr == 0 and dc == -1:
            return Move.LEFT
        if dr == 0 and dc == 1:
            return Move.RIGHT
        return None

    def _passable_mask_planning(self):
        return self.global_map == 0

    def _passable_mask_belief(self, map_state: np.ndarray):
        return map_state != 1

    def _degree(self, pos: tuple):
        r, c = pos
        rows, cols = self.global_map.shape
        deg = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and self.global_map[nr, nc] == 0:
                deg += 1
        return deg

    def _planning_neighbors(self, pos: tuple):
        r, c = pos
        rows, cols = self.global_map.shape
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and self.global_map[nr, nc] == 0:
                out.append((nr, nc))
        return out

    def _belief_neighbors(self, pos: tuple, passable_belief: np.ndarray):
        r, c = pos
        rows, cols = passable_belief.shape
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and passable_belief[nr, nc]:
                out.append((nr, nc))
        return out if out else [pos]

    def _chase_visible(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple):
        passable_plan = self._passable_mask_planning()
        path = self._bfs_path(my_position, enemy_position, passable_plan)
        if not path or len(path) < 2:
            return None
        move = self._direction_to(my_position, path[1])
        if move is None:
            return None
        steps = 1
        if self.pacman_speed > 1 and len(path) > 2:
            move2 = self._direction_to(path[1], path[2])
            if move2 == move and self._max_valid_steps(my_position, move, map_state, 2) >= 2:
                steps = 2
        return (move, steps)

    def _bfs_distances(self, start: tuple, passable_mask: np.ndarray):
        key = (start, self.map_version)
        cached = self.bfs_cache.get(key)
        if cached is not None:
            return cached
        rows, cols = passable_mask.shape
        dist = -np.ones((rows, cols), dtype=np.int16)
        sr, sc = start
        if not passable_mask[sr, sc]:
            return dist
        q = deque()
        q.append((sr, sc))
        dist[sr, sc] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and passable_mask[nr, nc] and dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        self.bfs_cache[key] = dist
        return dist

    def _bfs_path(self, start: tuple, goal: tuple, passable_mask: np.ndarray):
        if start == goal:
            return [start]
        rows, cols = passable_mask.shape
        sr, sc = start
        gr, gc = goal
        if not passable_mask[sr, sc] or not passable_mask[gr, gc]:
            return None
        visited = np.zeros((rows, cols), dtype=np.bool_)
        prev = {}
        q = deque()
        q.append(start)
        visited[sr, sc] = True
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and passable_mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        if goal not in prev and start != goal:
            return None
        path = [goal]
        cur = goal
        while cur != start:
            cur = prev.get(cur)
            if cur is None:
                return None
            path.append(cur)
        path.reverse()
        return path

    def _unknown_adjacent(self, pos: tuple):
        r, c = pos
        rows, cols = self.global_map.shape
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and self.global_map[nr, nc] == -1:
                count += 1
        return count

    def _decay_enemy_heat(self, step_number: int):
        if self.enemy_heat is None:
            return
        if self.last_decay_step < 0:
            self.last_decay_step = step_number
            return
        delta = step_number - self.last_decay_step
        if delta <= 0:
            return
        decay = float(self.heat_decay) ** float(delta)
        self.enemy_heat = (self.enemy_heat.astype(np.float32) * decay).astype(np.int16)
        self.last_decay_step = step_number

    def _get_pac_dist_map(self, my_position: tuple):
        if self.pac_dist_pos == my_position and self.pac_dist_version == self.map_version:
            return self.pac_dist_cache
        dist = self._bfs_distances(my_position, self._passable_mask_planning())
        self.pac_dist_cache = dist
        self.pac_dist_pos = my_position
        self.pac_dist_version = self.map_version
        return dist

    def _update_choke_cache(self):
        if self.choke_version == self.map_version:
            return
        passable = self._passable_mask_planning()
        rows, cols = passable.shape
        degree = np.zeros((rows, cols), dtype=np.int8)
        for r in range(rows):
            for c in range(cols):
                if not passable[r, c]:
                    continue
                deg = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc]:
                        deg += 1
                degree[r, c] = deg

        total = rows * cols
        visited = np.zeros(total, dtype=np.bool_)
        disc = np.full(total, -1, dtype=np.int32)
        low = np.full(total, -1, dtype=np.int32)
        parent = np.full(total, -1, dtype=np.int32)
        time = 0
        articulation = set()

        def idx(rr, cc):
            return rr * cols + cc

        def neighbors(u):
            ur, uc = divmod(u, cols)
            out = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = ur + dr, uc + dc
                if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc]:
                    out.append(idx(nr, nc))
            return out

        def dfs(u):
            nonlocal time
            visited[u] = True
            disc[u] = time
            low[u] = time
            time += 1
            children = 0
            for v in neighbors(u):
                if not visited[v]:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] == -1 and children > 1:
                        articulation.add(u)
                    if parent[u] != -1 and low[v] >= disc[u]:
                        articulation.add(u)
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for r in range(rows):
            for c in range(cols):
                if not passable[r, c]:
                    continue
                u = idx(r, c)
                if not visited[u]:
                    dfs(u)

        self.articulation_set = {(u // cols, u % cols) for u in articulation}
        self.corridor_set = {(r, c) for r in range(rows) for c in range(cols) if passable[r, c] and degree[r, c] == 2}
        exits = set()
        for r in range(rows):
            for c in range(cols):
                if not passable[r, c] or degree[r, c] < 3:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] and degree[nr, nc] == 2:
                        exits.add((nr, nc))
        self.choke_exits = exits
        self.choke_version = self.map_version

    def _beam_predict_paths(self, start: tuple, pac_dist_map: np.ndarray, depth: int, width: int):
        beam = [(start, [start], 0.0)]
        for _ in range(depth):
            new_beam = []
            for pos, path, score in beam:
                options = self._planning_neighbors(pos)
                if not options:
                    options = [pos]
                for nxt in options:
                    dist_now = pac_dist_map[pos[0], pos[1]]
                    dist_next = pac_dist_map[nxt[0], nxt[1]]
                    step_score = score
                    if dist_now >= 0 and dist_next >= 0:
                        if dist_next > dist_now:
                            step_score += 0.6
                        elif dist_next < dist_now:
                            step_score -= 0.4
                    step_score += 0.2 * float(self._degree(nxt))
                    if nxt in self.choke_exits:
                        step_score += 0.4
                    if len(path) >= 2 and nxt == path[-2]:
                        step_score -= 0.3
                    new_beam.append((nxt, path + [nxt], step_score))
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:width]
        return beam

    def _select_intercept_target(self, beam_paths, pac_dist_map: np.ndarray):
        best_target = None
        best_score = None
        for _, path, path_score in beam_paths:
            for t, cell in enumerate(path):
                dist = pac_dist_map[cell[0], cell[1]]
                if dist < 0:
                    continue
                pac_turns = (int(dist) + self.pacman_speed - 1) // self.pacman_speed
                if pac_turns > t:
                    continue
                cut_bonus = 0.0
                if cell in self.choke_exits:
                    cut_bonus += 0.6
                if cell in self.articulation_set:
                    cut_bonus += 0.8
                if cell in self.corridor_set:
                    cut_bonus += 0.3
                score = 5.0 - float(t) + cut_bonus + 0.2 * float(path_score)
                if best_score is None or score > best_score:
                    best_score = score
                    best_target = cell
                break
        return best_target


# =============================================================================
# GHOST AGENT (Hider) - Smart Evader with Voronoi + DFS survival + fog hiding
# =============================================================================

class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.accumulated_map = None
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self.max_depth = 2
        self.noise_scale = 0.05
        self.voronoi_ratio = 2.2

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        self._update_accumulated_map(map_state)

        working_map = self.accumulated_map.copy()
        working_map[working_map == -1] = 1

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat_pos = enemy_position or self.last_known_enemy_pos

        if threat_pos is None:
            return self._explore_fog(my_position, self.accumulated_map, working_map)
        else:
            move = self._smart_run_away(my_position, threat_pos, working_map)
            return move if isinstance(move, Move) else Move.STAY

    def _update_accumulated_map(self, map_state: np.ndarray):
        if self.accumulated_map is None:
            self.accumulated_map = np.full(map_state.shape, -1, dtype=np.int8)
        mask = map_state != -1
        self.accumulated_map[mask] = map_state[mask]

    def _smart_run_away(self, my_pos, threat_pos, map_state):
        best_move = Move.STAY
        max_score = -float('inf')

        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(my_pos, move)
            if self._is_valid_move(next_pos, map_state):
                valid_moves.append((next_pos, move))

        if not valid_moves:
            return Move.STAY

        pacman_reach = self._pacman_worst_positions(threat_pos, map_state)

        for next_pos, move in valid_moves:
            min_cap = min(self._manhattan(next_pos, p) for p in pacman_reach)
            if min_cap < 2:
                continue

            dist = self._bfs_distance(next_pos, threat_pos, map_state)
            worst_p = self._select_worst_pacman(next_pos, pacman_reach, map_state)
            survival = self._dfs_survival(next_pos, worst_p, map_state, depth=1)
            safe_area = self._voronoi_safe_area(next_pos, threat_pos, map_state)

            score = survival * 20 + safe_area * 2 + dist
            score += random.uniform(-self.noise_scale, self.noise_scale)

            if score > max_score:
                max_score = score
                best_move = move
            elif abs(score - max_score) < 1e-9 and random.random() > 0.5:
                best_move = move

        return best_move

    def _explore_fog(self, pos, original_map, working_map):
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_move(next_pos, working_map):
                valid_moves.append((next_pos, move))

        if not valid_moves:
            return Move.STAY

        best_move = None
        best_fog = -1

        for next_pos, move in valid_moves:
            fog_count = self._count_fog_nearby(next_pos, original_map, radius=4)
            if fog_count > best_fog:
                best_fog = fog_count
                best_move = move

        if best_move is None or best_fog == 0:
            return random.choice([m for _, m in valid_moves])

        return best_move

    def _count_fog_nearby(self, pos, map_state, radius=4):
        rows, cols = map_state.shape
        visited = {pos}
        queue = deque([(pos, 0)])
        fog_count = 0

        while queue:
            cur, dist = queue.popleft()
            if dist >= radius:
                continue
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr = cur[0] + move.value[0]
                nc = cur[1] + move.value[1]
                np_ = (nr, nc)
                if np_ in visited or not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                visited.add(np_)
                cell = map_state[nr, nc]
                if cell == -1:
                    fog_count += 1
                elif cell == 0:
                    queue.append((np_, dist + 1))
        return fog_count

    def _bfs_distance(self, start, target, map_state):
        if start == target:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, dist = queue.popleft()
            if current == target:
                return dist
            if dist > 30:
                return dist
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = self._apply_move(current, move)
                if next_pos not in visited and self._is_valid_move(next_pos, map_state):
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))
        return 100

    def _bfs_dist_map(self, start, map_state, max_dist=20):
        rows, cols = map_state.shape
        dist = [[-1] * cols for _ in range(rows)]
        sr, sc = start
        if not (0 <= sr < rows and 0 <= sc < cols):
            return dist
        dist[sr][sc] = 0
        queue = deque([(sr, sc)])
        while queue:
            r, c = queue.popleft()
            if dist[r][c] >= max_dist:
                continue
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = r + move.value[0], c + move.value[1]
                if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == -1:
                    if map_state[nr, nc] == 0:
                        dist[nr][nc] = dist[r][c] + 1
                        queue.append((nr, nc))
        return dist

    def _pacman_worst_positions(self, pac_pos, map_state):
        rows, cols = map_state.shape
        positions = {pac_pos}
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            cur = pac_pos
            for _ in range(self.pacman_speed):
                nr = cur[0] + move.value[0]
                nc = cur[1] + move.value[1]
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if map_state[nr, nc] != 0:
                    break
                cur = (nr, nc)
                positions.add(cur)
        return list(positions)

    def _select_worst_pacman(self, ghost_pos, pac_positions, map_state):
        dist_map = self._bfs_dist_map(ghost_pos, map_state)
        best = pac_positions[0]
        best_dist = 999
        rows, cols = len(dist_map), len(dist_map[0])
        for p in pac_positions:
            r, c = p
            if 0 <= r < rows and 0 <= c < cols:
                d = dist_map[r][c]
                if d != -1 and d < best_dist:
                    best_dist = d
                    best = p
        return best

    def _dfs_survival(self, ghost_pos, pac_pos, map_state, depth):
        if depth >= self.max_depth:
            return self._bfs_distance(ghost_pos, pac_pos, map_state)

        pac_positions = self._pacman_worst_positions(pac_pos, map_state)
        worst_p = self._select_worst_pacman(ghost_pos, pac_positions, map_state)

        best_survival = -1
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(ghost_pos, move)
            if not self._is_valid_move(nxt, map_state):
                continue
            dist = self._bfs_distance(nxt, worst_p, map_state)
            survival = 1 if dist < 2 else 1 + self._dfs_survival(nxt, worst_p, map_state, depth + 1)
            if survival > best_survival:
                best_survival = survival

        return best_survival if best_survival >= 0 else 1

    def _voronoi_safe_area(self, ghost_pos, pac_pos, map_state):
        dist_g = self._bfs_dist_map(ghost_pos, map_state, max_dist=20)
        dist_p = self._bfs_dist_map(pac_pos, map_state, max_dist=20)
        rows, cols = len(dist_g), len(dist_g[0])
        safe = 0
        for r in range(rows):
            for c in range(cols):
                dg = dist_g[r][c]
                dp = dist_p[r][c]
                if dg != -1 and dp != -1 and dg < (dp / self.voronoi_ratio):
                    safe += 1
        return safe

    def _apply_move(self, pos, move):
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def _is_valid_move(self, pos, map_state):
        rows, cols = map_state.shape
        if not (0 <= pos[0] < rows and 0 <= pos[1] < cols):
            return False
        return map_state[pos] == 0

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
