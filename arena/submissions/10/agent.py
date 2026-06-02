"""
Ghost Agent (Hide) - Group 10
BlindArena Lab - Fundamentals of Artificial Intelligence

Core principle: Pacman has speed=2 (2 cells/turn on straight lines), Ghost has speed=1.
Ghost CANNOT outrun Pacman in straight lines.
Winning strategy: exploit MAZE TOPOLOGY — zigzags, corners, and junctions to
slow Pacman down and maximize BFS survival distance.

Constraints:
    - Max 1 second per step
    - Max 16 MB memory (Google Colab CPU-only)
    - Allowed: numpy, pandas, scipy, gurobi + Python built-in
"""

import sys
from pathlib import Path
from collections import deque
import time

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


# ============================================================
# Hardcoded default 21x21 map
# ============================================================
_MAP = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#...................#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#...........#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]

_DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_DELTAS = {m: m.value for m in Move}
_PAC_SPEED = 2


# ============================================================
# GhostAgent — Hide Agent
# ============================================================
class GhostAgent(BaseGhostAgent):
    """
    Ghost agent using precomputed BFS + minimax + maze topology awareness.
    Never STAYs unless completely boxed in.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H, self.W = 21, 21

        # Parse map
        self.empty = set()
        for r, row_str in enumerate(_MAP):
            for c, ch in enumerate(row_str):
                if ch != '#':
                    self.empty.add((r, c))

        # Neighbor lists
        self.nbrs = {}       # cell -> [(neighbor_pos, move_enum), ...]
        self.nbr_list = {}   # cell -> [neighbor_pos, ...]
        for cell in self.empty:
            r, c = cell
            ns, nl = [], []
            for m in _DIRS:
                dr, dc = _DELTAS[m]
                nb = (r + dr, c + dc)
                if nb in self.empty:
                    ns.append((nb, m))
                    nl.append(nb)
            self.nbrs[cell] = ns
            self.nbr_list[cell] = nl

        # Dead-end detection (iterative peeling)
        self.de_depth = {}
        self._find_dead_ends()

        # Junctions (3+ open neighbors)
        self.junctions = frozenset(
            c for c in self.empty if len(self.nbr_list[c]) >= 3
        )

        # All-pairs BFS distances (int8 matrix, ~29 KB)
        cells_sorted = sorted(self.empty)
        self.cid = {c: i for i, c in enumerate(cells_sorted)}
        n = len(cells_sorted)
        self.dist = np.full((n, n), 127, dtype=np.int8)
        for si in range(n):
            s = cells_sorted[si]
            visited = {s: 0}
            q = deque([s])
            self.dist[si, si] = 0
            while q:
                u = q.popleft()
                du = visited[u]
                for v in self.nbr_list[u]:
                    if v not in visited:
                        dv = du + 1
                        visited[v] = dv
                        self.dist[si, self.cid[v]] = dv
                        q.append(v)

        # Precompute: distance to nearest junction
        junc_ids = [self.cid[jc] for jc in self.junctions]
        self.dist_to_junction = {}
        for cell in self.empty:
            ci = self.cid[cell]
            if junc_ids:
                min_d = int(np.min(self.dist[ci, junc_ids]))
            else:
                min_d = 127
            self.dist_to_junction[cell] = min_d

        # Precompute: "pac_maze_cost" approximation
        # Pac speed=2 means: straight steps cost 0.5 turns, corners cost 1 turn.
        # Average maze speed is approx ~1.5 cells/turn.

        # State
        self.last_enemy = None
        self.last_enemy_step = 0
        self.prev_pos = None
        self.prev2_pos = None
        self.recent_positions = deque(maxlen=14)

    # --------------------------------------------------------
    # Precomputation helpers
    # --------------------------------------------------------
    def _find_dead_ends(self):
        eff = {c: len(self.nbr_list[c]) for c in self.empty}
        q = deque()
        for c in self.empty:
            if eff[c] == 1:
                self.de_depth[c] = 1
                q.append(c)
        while q:
            u = q.popleft()
            for v in self.nbr_list[u]:
                if v not in self.de_depth:
                    eff[v] -= 1
                    if eff[v] <= 1:
                        self.de_depth[v] = self.de_depth[u] + 1
                        q.append(v)

    # --------------------------------------------------------
    # Core utilities
    # --------------------------------------------------------
    def _d(self, a, b):
        """O(1) precomputed BFS distance."""
        ia = self.cid.get(a)
        ib = self.cid.get(b)
        if ia is None or ib is None:
            return 127
        return int(self.dist[ia, ib])

    def _pac_reach_1turn(self, pos):
        """All unique positions speed-2 Pacman can reach in exactly 1 turn."""
        res = {pos}
        for m in _DIRS:
            dr, dc = _DELTAS[m]
            p1 = (pos[0] + dr, pos[1] + dc)
            if p1 in self.empty:
                res.add(p1)
                # Step 2 (same direction only for speed bonus)
                p2 = (p1[0] + dr, p1[1] + dc)
                if p2 in self.empty:
                    res.add(p2)
        return res

    def _ghost_options(self, pos):
        """Movement options: directional first, STAY last."""
        opts = [(nb, m) for nb, m in self.nbrs.get(pos, [])]
        opts.append((pos, Move.STAY))
        return opts

    def _movable_options(self, pos):
        """Only non-STAY movement options."""
        return [(nb, m) for nb, m in self.nbrs.get(pos, [])]

    # --------------------------------------------------------
    # Position evaluation
    # --------------------------------------------------------
    def _eval(self, gp, pp):
        """
        Evaluate Ghost position. Higher = better (safer).
        BFS distance is the primary metric.
        Dead-ends are severely penalized.
        """
        manh = abs(gp[0] - pp[0]) + abs(gp[1] - pp[1])
        if manh < 2:
            return -100000

        bd = self._d(gp, pp)
        n_exits = len(self.nbr_list.get(gp, []))
        de = self.de_depth.get(gp, 0)

        # Primary: BFS distance (150 pts/step)
        score = bd * 150

        # Exits: more exits = more flexibility (50 pts each)
        score += n_exits * 50

        # Dead-end penalty
        if de > 0:
            score -= de * 600
            # If Pac can block the exit before Ghost can turn around:
            if bd <= de:
                score -= 25000  # Trapped — avoid at all costs!

        # Junction bonus (reduced — don't over-reward junctions)
        if gp in self.junctions:
            score += 60

        # Distance to nearest junction (prefer being close to an escape junction)
        jd = self.dist_to_junction.get(gp, 127)
        score -= jd * 20

        # 2-step reachability
        reach2 = set()
        for nb in self.nbr_list.get(gp, []):
            reach2.add(nb)
            for nb2 in self.nbr_list.get(nb, []):
                reach2.add(nb2)
        score += len(reach2) * 8

        return score

    # --------------------------------------------------------
    # Anti-oscillation penalty
    # --------------------------------------------------------
    def _osc_penalty(self, ng):
        pen = 0
        if self.prev_pos and ng == self.prev_pos:
            pen += 200   # Strong: never immediately backtrack
        if self.prev2_pos and ng == self.prev2_pos:
            pen += 100   # Medium: avoid 2-step cycle
        pen += sum(1 for p in self.recent_positions if p == ng) * 80
        return pen

    # --------------------------------------------------------
    # Minimax with Alpha-Beta Pruning
    # --------------------------------------------------------
    def _minimax(self, gp, pp, depth, alpha, beta, deadline):
        if time.time() > deadline:
            return self._eval(gp, pp)
        manh = abs(gp[0] - pp[0]) + abs(gp[1] - pp[1])
        if manh < 2:
            return -100000 + depth
        if depth <= 0:
            return self._eval(gp, pp)

        best = -200000
        g_opts = self._ghost_options(gp)
        g_opts.sort(key=lambda x: -self._d(x[0], pp))

        for ng, mv in g_opts:
            stay_pen = -200 if mv == Move.STAY else 0
            p_reach = sorted(self._pac_reach_1turn(pp), key=lambda p: self._d(ng, p))
            worst = 200000
            for np_ in p_reach:
                m2 = abs(ng[0] - np_[0]) + abs(ng[1] - np_[1])
                if m2 < 2:
                    v = -100000 + depth
                else:
                    v = self._minimax(ng, np_, depth - 1,
                                      max(alpha, best), min(worst, beta),
                                      deadline)
                worst = min(worst, v)
                if worst <= alpha:
                    break
            worst += stay_pen
            best = max(best, worst)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _choose_minimax(self, my_pos, threat, g_opts, deadline):
        """Iterative deepening minimax with time management."""
        result = self._greedy_best(my_pos, threat, g_opts)

        for depth in range(1, 7):
            if time.time() > deadline - 0.2:
                break
            cur_best_mv = result
            cur_best_sc = -200000
            completed = True
            g_sorted = sorted(g_opts, key=lambda x: -self._d(x[0], threat))

            for ng, mv in g_sorted:
                if time.time() > deadline - 0.08:
                    completed = False
                    break
                if mv == Move.STAY:
                    continue  # Never start with STAY in minimax

                manh = abs(ng[0] - threat[0]) + abs(ng[1] - threat[1])
                if manh < 2:
                    continue

                p_reach = sorted(self._pac_reach_1turn(threat), key=lambda p: self._d(ng, p))
                worst = 200000
                for np_ in p_reach:
                    m2 = abs(ng[0] - np_[0]) + abs(ng[1] - np_[1])
                    if m2 < 2:
                        v = -100000
                    elif depth <= 1:
                        v = self._eval(ng, np_)
                    else:
                        v = self._minimax(ng, np_, depth - 1, cur_best_sc, worst, deadline)
                    worst = min(worst, v)
                    if worst <= cur_best_sc:
                        break

                # Anti-oscillation penalty
                worst -= self._osc_penalty(ng)

                if worst > cur_best_sc:
                    cur_best_sc = worst
                    cur_best_mv = mv

            if completed:
                result = cur_best_mv

        return result

    # --------------------------------------------------------
    # Greedy best
    # --------------------------------------------------------
    def _greedy_best(self, my_pos, threat, g_opts):
        """Pick best move greedily by eval, never STAY, never backtrack."""
        movable = [(ng, mv) for ng, mv in g_opts if mv != Move.STAY]
        if not movable:
            return Move.STAY

        # Hard ban on backtracking unless it's the only option
        no_backtrack = [(ng, mv) for ng, mv in movable if ng != self.prev_pos]
        candidates = no_backtrack if no_backtrack else movable

        best_sc = -200000
        best_mv = candidates[0][1]
        for ng, mv in candidates:
            sc = self._eval(ng, threat) - self._osc_penalty(ng)
            if sc > best_sc:
                best_sc = sc
                best_mv = mv
        return best_mv

    # --------------------------------------------------------
    # Medium-range flee: path to best escape junction
    # --------------------------------------------------------
    def _medium_flee(self, my_pos, threat, g_opts):
        """
        Find the best junction to flee to: maximize (d_from_pac - d_ghost_to_junc).
        Head toward it with anti-oscillation.
        """
        movable = [(ng, mv) for ng, mv in g_opts if mv != Move.STAY]
        if not movable:
            return Move.STAY

        # Score each junction as escape target
        best_target = None
        best_target_score = -999999
        gi = self.cid.get(my_pos)
        pi = self.cid.get(threat)

        for junc in self.junctions:
            if junc == my_pos:
                continue
            ji = self.cid[junc]
            d_ghost_to_junc = int(self.dist[gi, ji]) if gi is not None else 127
            d_pac_to_junc = int(self.dist[pi, ji]) if pi is not None else 0
            de = self.de_depth.get(junc, 0)
            exits = len(self.nbr_list[junc])

            if de > 0:
                continue  # Skip dead-end junctions

            # Pac turns to arrive (speed ~1.5 average due to maze turns)
            pac_turns = d_pac_to_junc / 1.5
            # Can Ghost arrive first?
            margin = pac_turns - d_ghost_to_junc
            score = margin * 100 + exits * 50 + d_pac_to_junc * 10

            if score > best_target_score:
                best_target_score = score
                best_target = junc

        if best_target is None:
            return self._greedy_best(my_pos, threat, g_opts)

        # Pick the move toward best_target; exclude backtrack
        movable = [(ng, mv) for ng, mv in g_opts if mv != Move.STAY]
        no_backtrack = [(ng, mv) for ng, mv in movable if ng != self.prev_pos]
        candidates = no_backtrack if no_backtrack else movable

        best_sc = -200000
        best_mv = candidates[0][1]
        gi = self.cid.get(my_pos)

        for ng, mv in candidates:
            ni = self.cid.get(ng)
            d_to_target = int(self.dist[ni, self.cid[best_target]]) if ni is not None else 127
            d_to_pac = self._d(ng, threat)
            de = self.de_depth.get(ng, 0)

            sc = d_to_pac * 100 - d_to_target * 80
            sc -= de * 600
            if ng in self.junctions:
                sc += 120
            sc -= self._osc_penalty(ng)

            if sc > best_sc:
                best_sc = sc
                best_mv = mv
        return best_mv

    # --------------------------------------------------------
    # Far flee: maximize eval score
    # --------------------------------------------------------
    def _far_flee(self, my_pos, threat, g_opts):
        """When Pac is far: simply maximize eval with no backtracking."""
        movable = [(ng, mv) for ng, mv in g_opts if mv != Move.STAY]
        if not movable:
            return Move.STAY

        # Hard ban on backtracking unless only option
        no_backtrack = [(ng, mv) for ng, mv in movable if ng != self.prev_pos]
        candidates = no_backtrack if no_backtrack else movable

        best_sc = -200000
        best_mv = candidates[0][1]
        for ng, mv in candidates:
            sc = self._eval(ng, threat) - self._osc_penalty(ng)
            if sc > best_sc:
                best_sc = sc
                best_mv = mv
        return best_mv

    # --------------------------------------------------------
    # Safe Wander (Pacman unknown)
    # --------------------------------------------------------
    def _wander(self, my_pos, g_opts):
        """Avoid dead-ends, prefer junctions, anti-oscillate."""
        movable = [(ng, mv) for ng, mv in g_opts if mv != Move.STAY]
        if not movable:
            return Move.STAY

        best_sc = -200000
        best_mv = movable[0][1]
        for ng, mv in movable:
            de = self.de_depth.get(ng, 0)
            sc = -de * 250
            if ng in self.junctions:
                sc += 150
            sc += len(self.nbr_list.get(ng, [])) * 60
            sc -= self._osc_penalty(ng)
            if sc > best_sc:
                best_sc = sc
                best_mv = mv
        return best_mv

    # --------------------------------------------------------
    # Main step
    # --------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        deadline = t0 + 0.8

        # Update enemy tracking
        if enemy_position is not None:
            self.last_enemy = enemy_position
            self.last_enemy_step = step_number

        g_opts = self._ghost_options(my_position)

        # Boxed in
        if not self.nbrs.get(my_position):
            self._update_history(my_position)
            return Move.STAY

        threat = enemy_position if enemy_position is not None else self.last_enemy

        if threat is not None:
            bd = self._d(my_position, threat)
            # Pac effective turns = BFS / 1.5 (average maze speed accounting for corners)
            pac_eff_turns = bd / 1.5

            if pac_eff_turns <= 6:
                # Close danger: full minimax
                mv = self._choose_minimax(my_position, threat, g_opts, deadline)
            elif pac_eff_turns <= 20:
                # Medium/far: head to best escape junction
                mv = self._medium_flee(my_position, threat, g_opts)
            else:
                # Very far: maximize eval
                mv = self._far_flee(my_position, threat, g_opts)
        else:
            mv = self._wander(my_position, g_opts)

        self._update_history(my_position)
        return mv

    def _update_history(self, pos):
        self.prev2_pos = self.prev_pos
        self.prev_pos = pos
        self.recent_positions.append(pos)


# ============================================================
# PacmanAgent
# ============================================================
UNKNOWN = -1
EMPTY = 0
WALL = 1
INF = 10**9

DIRECTIONS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
DELTA_TO_MOVE = {move.value: move for move in DIRECTIONS}


class PacmanAgent(BasePacmanAgent):
    """Seek agent using memory, belief tracking, BFS, and frontier exploration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Group12 Memory-Belief Seek Agent"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.height = 0
        self.width = 0
        self.known_map = None
        self.visit_count = None
        self.belief = None
        self.last_seen_enemy = None
        self.last_seen_step = -100000
        self.current_my_position = None
        self.recent_positions = deque(maxlen=12)
        self._distance_cache = {}
        self.time_budget = 0.90
        self._step_started_at = 0.0

    def step(self, map_state, my_position, enemy_position, step_number):
        # Assumption: framework positions are in (row, col) order.
        # If a different runner uses (x, y), convert positions before indexing maps.
        self._step_started_at = time.time()
        self._distance_cache = {}
        self.current_my_position = my_position
        self._ensure_state(map_state)
        self._update_memory(map_state, my_position, enemy_position)
        self._mark_visit(my_position)
        self._update_belief(map_state, my_position, enemy_position, step_number)
        self.recent_positions.append(my_position)

        if enemy_position is not None:
            return self._safe_action(
                self._choose_chase_action(my_position, enemy_position)
            )

        if self._has_useful_belief(step_number):
            action = self._choose_hunt_action(my_position)
            if action is not None:
                return self._safe_action(action)

        action = self._choose_explore_action(my_position)
        return self._safe_action(action)

    def _ensure_state(self, map_state):
        height, width = map_state.shape
        if self.known_map is not None and height == self.height and width == self.width:
            return

        self.height = int(height)
        self.width = int(width)
        self.known_map = [[UNKNOWN for _ in range(self.width)] for _ in range(self.height)]
        self.visit_count = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.belief = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        self.last_seen_enemy = None
        self.last_seen_step = -100000
        self.current_my_position = None
        self.recent_positions.clear()

    def _update_memory(self, map_state, my_position, enemy_position):
        for r in range(self.height):
            row = map_state[r]
            known_row = self.known_map[r]
            for c in range(self.width):
                value = int(row[c])
                if value == WALL:
                    known_row[c] = WALL
                elif value == EMPTY:
                    known_row[c] = EMPTY

        mr, mc = my_position
        self.known_map[mr][mc] = EMPTY
        if enemy_position is not None:
            er, ec = enemy_position
            self.known_map[er][ec] = EMPTY

    def _mark_visit(self, pos):
        r, c = pos
        self.visit_count[r][c] += 1

    def _update_belief(self, map_state, my_position, enemy_position, step_number):
        visible_empty = self._visible_empty_cells(map_state, my_position)

        if enemy_position is not None:
            self._clear_belief()
            er, ec = enemy_position
            self.belief[er][ec] = 1.0
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number
            self._clear_belief_on_known_walls()
            return

        total = self._belief_total()
        if total > 1e-12:
            self._diffuse_belief()
        elif self.last_seen_enemy is not None and step_number - self.last_seen_step <= 22:
            self._seed_belief_from_last_seen(step_number, visible_empty)

        for r, c in visible_empty:
            self.belief[r][c] = 0.0

        mr, mc = my_position
        self.belief[mr][mc] = 0.0

        if self._belief_total() <= 1e-12 and self.last_seen_enemy is not None:
            if step_number - self.last_seen_step <= 22:
                self._seed_belief_from_last_seen(step_number, visible_empty)

        self._clear_belief_on_known_walls()
        self._normalize_belief()

    def _visible_empty_cells(self, map_state, my_position):
        visible = {my_position}
        for r in range(self.height):
            row = map_state[r]
            for c in range(self.width):
                if int(row[c]) == EMPTY:
                    visible.add((r, c))
        return visible

    def _clear_belief(self):
        for r in range(self.height):
            row = self.belief[r]
            for c in range(self.width):
                row[c] = 0.0

    def _clear_belief_on_known_walls(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.known_map[r][c] == WALL:
                    self.belief[r][c] = 0.0

    def _belief_total(self):
        total = 0.0
        for row in self.belief:
            total += sum(row)
        return total

    def _diffuse_belief(self):
        new_belief = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for r in range(self.height):
            for c in range(self.width):
                if self.known_map[r][c] == WALL:
                    continue
                probability = self.belief[r][c]
                if probability <= 0.0:
                    continue
                next_cells = self._ghost_next_cells((r, c))
                weighted_cells = []
                weight_total = 0.0
                for nr, nc in next_cells:
                    # Belief may reason through unknown cells; real Pacman moves do not.
                    weight = 1.0
                    if self.current_my_position is not None:
                        weight += 0.35 * self._manhattan((nr, nc), self.current_my_position)
                    weight += 0.7 * max(0, self._degree_not_wall((nr, nc)) - 1)
                    weight += 0.2 * self._visibility_gain((nr, nc))
                    weight -= 0.15 * self.visit_count[nr][nc]
                    weight = max(0.1, weight)
                    weighted_cells.append((nr, nc, weight))
                    weight_total += weight

                if weight_total <= 0.0:
                    continue
                for nr, nc, weight in weighted_cells:
                    new_belief[nr][nc] += probability * weight / weight_total
        self.belief = new_belief

    def _seed_belief_from_last_seen(self, step_number, visible_empty):
        self._clear_belief()
        elapsed = max(1, step_number - self.last_seen_step)
        distances = self._cached_distances(self.last_seen_enemy, allow_unknown=True)
        for (r, c), dist in distances.items():
            if dist > elapsed or (r, c) in visible_empty:
                continue
            if self.known_map[r][c] == WALL:
                continue
            self.belief[r][c] = 1.0 / (1.0 + dist)
        self._normalize_belief()

    def _normalize_belief(self):
        total = self._belief_total()
        if total <= 1e-12:
            return
        inv_total = 1.0 / total
        for r in range(self.height):
            row = self.belief[r]
            for c in range(self.width):
                row[c] *= inv_total

    def _has_useful_belief(self, step_number):
        if self.last_seen_enemy is None:
            return False
        if step_number - self.last_seen_step > 35:
            return False
        return self._belief_total() > 1e-9

    def _choose_chase_action(self, my_position, enemy_position):
        actions = self._candidate_actions(my_position, allow_unknown=False)
        ghost_next_cells = self._ghost_next_cells(enemy_position)
        best_action = None
        best_score = -10**18
        stuck = self._is_stuck()

        for move, steps, new_pos in actions:
            if self._time_low():
                break
            distances_from_new = self._cached_distances(new_pos, allow_unknown=False)
            capture_count = 0
            cutoff_score = 0
            worst_distance = 0
            total_distance = 0

            for ghost_pos in ghost_next_cells:
                if self._manhattan(new_pos, ghost_pos) < 2:
                    distance = 0
                    capture_count += 1
                else:
                    shortest = distances_from_new.get(ghost_pos)
                    distance = shortest if shortest is not None else self._manhattan(new_pos, ghost_pos) + 20
                if self._manhattan(new_pos, ghost_pos) < self._manhattan(my_position, ghost_pos):
                    cutoff_score += 1
                worst_distance = max(worst_distance, distance)
                total_distance += distance

            guaranteed_capture = capture_count == len(ghost_next_cells)
            avg_distance = total_distance / max(1, len(ghost_next_cells))
            direct_distance = distances_from_new.get(enemy_position)
            if direct_distance is None:
                direct_distance = self._manhattan(new_pos, enemy_position) + 20

            escape_count = self._escape_count_after_seek(new_pos, ghost_next_cells)
            visibility_gain = self._visibility_gain(new_pos)
            revisit_penalty = self.visit_count[new_pos[0]][new_pos[1]]
            recent_penalty = 1 if new_pos in self.recent_positions else 0
            trap = self._trap_score(new_pos, enemy_position)

            score = 0.0
            if guaranteed_capture:
                score += 100000.0
            score += 4500.0 * capture_count
            score -= 230.0 * worst_distance
            score -= 70.0 * avg_distance
            score -= 24.0 * direct_distance
            score -= 40.0 * escape_count
            if direct_distance <= 4:
                score -= 65.0 * escape_count
            score += 25.0 * cutoff_score
            score += trap
            if direct_distance <= 5:
                score += 1.5 * trap
            score += 3.5 * visibility_gain
            score += 4.0 * steps
            score -= (7.0 if stuck else 4.0) * revisit_penalty
            score -= (35.0 if stuck else 15.0) * recent_penalty
            if move == Move.STAY:
                score -= 300.0 if stuck else 120.0
            if self._manhattan(my_position, enemy_position) <= 6 and not self._time_low():
                score += 0.25 * self._two_ply_chase_score(
                    my_position,
                    enemy_position,
                    new_pos
                )

            if score > best_score:
                best_score = score
                best_action = (move, steps)

        return best_action or self._best_local_action(my_position)

    def _choose_hunt_action(self, my_position):
        distances = self._cached_distances(my_position, allow_unknown=False)
        best_target = None
        best_score = -10**18
        stuck = self._is_stuck()

        for (r, c), distance in distances.items():
            if self.known_map[r][c] != EMPTY:
                continue
            probability = self.belief[r][c]
            local_mass = self._local_belief_mass((r, c), radius=2)
            if probability <= 0.0 and local_mass <= 0.0:
                continue
            degree = self._degree_not_wall((r, c))
            junction_bonus = max(0, degree - 2)
            unseen_gain = self._visibility_gain((r, c))
            visible_mass = self._visible_belief_mass_from((r, c))
            visits = self.visit_count[r][c]
            score = (
                700.0 * probability
                + 500.0 * local_mass
                + 650.0 * visible_mass
                - 3.5 * distance
                + 5.0 * junction_bonus
                + 1.5 * unseen_gain
                - (3.2 if stuck else 2.0) * visits
            )
            if (r, c) in self.recent_positions:
                score -= 28.0 if stuck else 8.0
            if score > best_score:
                best_score = score
                best_target = (r, c)

        if best_target is None:
            return None

        path = self._bfs_path(my_position, best_target, allow_unknown=False)
        if path and len(path) > 1:
            return self._path_to_action(path)
        return None

    def _choose_explore_action(self, my_position):
        distances = self._cached_distances(my_position, allow_unknown=False)
        best_target = None
        best_score = -10**18
        center = ((self.height - 1) / 2.0, (self.width - 1) / 2.0)
        stuck = self._is_stuck()

        for (r, c), distance in distances.items():
            pos = (r, c)
            if pos == my_position:
                continue
            if self.known_map[r][c] != EMPTY:
                continue

            unseen_gain = self._visibility_gain(pos)
            frontier_bonus = 1 if self._is_frontier(pos) else 0
            if not frontier_bonus:
                continue

            degree = self._degree_not_wall(pos)
            junction_bonus = max(0, degree - 2)
            openness = self._ray_openness(pos)
            centrality = -abs(r - center[0]) - abs(c - center[1])
            visits = self.visit_count[r][c]
            recent_penalty = 1 if pos in self.recent_positions else 0
            distance_score = (-1.0 if stuck else -1.8) * distance

            score = (
                12.0 * unseen_gain
                + 10.0 * frontier_bonus
                + 2.0 * openness
                + distance_score
                + 5.0 * junction_bonus
                + 0.3 * centrality
                - (4.0 if stuck else 2.5) * visits
                - (35.0 if stuck else 12.0) * recent_penalty
            )
            if stuck:
                score += 0.6 * distance
            if score > best_score:
                best_score = score
                best_target = pos

        if best_target is not None:
            path = self._bfs_path(my_position, best_target, allow_unknown=False)
            if path and len(path) > 1:
                return self._path_to_action(path)

        return self._best_local_action(my_position)

    def _best_local_action(self, my_position):
        actions = self._candidate_actions(my_position, allow_unknown=False)
        best_action = (Move.STAY, 1)
        best_score = -10**18
        stuck = self._is_stuck()

        for move, steps, new_pos in actions:
            gain = self._visibility_gain(new_pos)
            degree = self._degree_not_wall(new_pos)
            visits = self.visit_count[new_pos[0]][new_pos[1]]
            recent_penalty = 1 if new_pos in self.recent_positions else 0

            score = (
                12.0 * gain
                + 2.5 * degree
                + 2.0 * steps
                - (5.0 if stuck else 3.0) * visits
                - (35.0 if stuck else 12.0) * recent_penalty
            )
            if move == Move.STAY:
                score -= 500.0 if stuck else 100.0

            if score > best_score:
                best_score = score
                best_action = (move, steps)

        return best_action

    def _local_belief_mass(self, pos, radius=2):
        total = 0.0
        r0, c0 = pos
        for dr in range(-radius, radius + 1):
            remaining = radius - abs(dr)
            r = r0 + dr
            if r < 0 or r >= self.height:
                continue
            for dc in range(-remaining, remaining + 1):
                c = c0 + dc
                if 0 <= c < self.width:
                    total += self.belief[r][c]
        return total

    def _visible_belief_mass_from(self, pos):
        """Belief mass visible from pos using the same cross-shaped sight rays."""
        if not self._in_bounds(pos):
            return 0.0

        total = 0.0
        r, c = pos
        total += self.belief[r][c]

        for move in DIRECTIONS:
            dr, dc = move.value
            for distance in range(1, 6):
                nr, nc = r + dr * distance, c + dc * distance
                if not self._in_bounds((nr, nc)):
                    break
                if self.known_map[nr][nc] == WALL:
                    break
                total += self.belief[nr][nc]

        return total

    def _ray_openness(self, pos):
        """Count how many non-wall cells can be scanned from pos in four rays."""
        if not self._in_bounds(pos):
            return 0

        score = 0
        r, c = pos
        for move in DIRECTIONS:
            dr, dc = move.value
            for distance in range(1, 6):
                nr, nc = r + dr * distance, c + dc * distance
                if not self._in_bounds((nr, nc)):
                    break
                if self.known_map[nr][nc] == WALL:
                    break
                score += 1

        return score

    def _trap_score(self, seek_pos, ghost_pos):
        """Reward chase positions that cover Ghost's next cells and reduce exits."""
        if not self._in_bounds(seek_pos) or not self._in_bounds(ghost_pos):
            return 0.0

        score = 0.0
        ghost_options = self._ghost_next_cells(ghost_pos)
        for g_next in ghost_options:
            if self._manhattan(seek_pos, g_next) < 2:
                score += 25.0
            else:
                degree = self._degree_not_wall(g_next)
                score -= 8.0 * max(0, degree - 1)

        return score

    def _two_ply_chase_score(self, my_position, enemy_position, first_new_pos):
        """Small bounded 2-ply chase; only Manhattan/escape heuristics, no BFS."""
        if self._time_low():
            return 0.0

        ghost_positions = self._ghost_next_cells(enemy_position)
        worst_case_score = 10**9

        for g1 in ghost_positions:
            if self._time_low():
                break

            if self._manhattan(first_new_pos, g1) < 2:
                value = 50000.0
            else:
                best_second_score = -10**9
                second_actions = self._candidate_actions(first_new_pos, allow_unknown=False)

                for _, _, p2 in second_actions:
                    if self._time_low():
                        break

                    if self._manhattan(p2, g1) < 2:
                        second_score = 30000.0
                    else:
                        dist = self._manhattan(p2, g1)
                        escape = self._escape_count_after_seek(p2, self._ghost_next_cells(g1))
                        second_score = -120.0 * dist - 30.0 * escape

                    if second_score > best_second_score:
                        best_second_score = second_score

                value = best_second_score

            if value < worst_case_score:
                worst_case_score = value

        if worst_case_score == 10**9:
            return 0.0

        return worst_case_score

    def _is_stuck(self):
        if len(self.recent_positions) < 8:
            return False
        return len(set(self.recent_positions)) <= 3

    def _candidate_actions(self, pos, allow_unknown):
        actions = [(Move.STAY, 1, pos)]
        max_steps = max(1, self.pacman_speed)
        for move in DIRECTIONS:
            current = pos
            for steps in range(1, max_steps + 1):
                dr, dc = move.value
                nxt = (current[0] + dr, current[1] + dc)
                if not self._is_passable(nxt, allow_unknown=allow_unknown):
                    break
                actions.append((move, steps, nxt))
                current = nxt
        return actions

    def _ghost_next_cells(self, pos):
        cells = [pos]
        for move in DIRECTIONS:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._is_passable(nxt, allow_unknown=True):
                cells.append(nxt)
        return cells

    def _escape_count_after_seek(self, seek_pos, ghost_next_cells):
        count = 0
        for pos in ghost_next_cells:
            if self._manhattan(seek_pos, pos) >= 2:
                count += 1 + max(0, self._degree_not_wall(pos) - 2)
        return count

    def _bfs_path(self, start, goal, allow_unknown):
        if start == goal:
            return [start]
        queue = deque([start])
        parent = {start: None}

        while queue:
            if self._time_low():
                break
            current = queue.popleft()
            for nxt in self._neighbors(current, allow_unknown=allow_unknown):
                if nxt in parent:
                    continue
                parent[nxt] = current
                if nxt == goal:
                    return self._reconstruct_path(parent, start, goal)
                queue.append(nxt)
        return None

    def _cached_distances(self, start, allow_unknown):
        key = (start, bool(allow_unknown))
        if key not in self._distance_cache:
            self._distance_cache[key] = self._bfs_distances(start, allow_unknown)
        return self._distance_cache[key]

    def _bfs_distances(self, start, allow_unknown):
        queue = deque([start])
        distances = {start: 0}

        while queue:
            if self._time_low():
                break
            current = queue.popleft()
            next_distance = distances[current] + 1
            for nxt in self._neighbors(current, allow_unknown=allow_unknown):
                if nxt in distances:
                    continue
                distances[nxt] = next_distance
                queue.append(nxt)
        return distances

    def _shortest_distance(self, start, goal, allow_unknown):
        if start == goal:
            return 0
        queue = deque([start])
        distances = {start: 0}

        while queue:
            if self._time_low():
                return None
            current = queue.popleft()
            next_distance = distances[current] + 1
            for nxt in self._neighbors(current, allow_unknown=allow_unknown):
                if nxt in distances:
                    continue
                if nxt == goal:
                    return next_distance
                distances[nxt] = next_distance
                queue.append(nxt)
        return None

    def _reconstruct_path(self, parent, start, goal):
        path = [goal]
        current = goal
        while current != start:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    def _path_to_action(self, path):
        if not path or len(path) < 2:
            return (Move.STAY, 1)

        first = path[0]
        second = path[1]
        delta = (second[0] - first[0], second[1] - first[1])
        move = DELTA_TO_MOVE.get(delta, Move.STAY)
        if move == Move.STAY:
            return (Move.STAY, 1)

        steps = 1
        limit = min(self.pacman_speed, len(path) - 1)
        for index in range(2, limit + 1):
            prev = path[index - 1]
            current = path[index]
            if (current[0] - prev[0], current[1] - prev[1]) != delta:
                break
            steps += 1
        return (move, steps)

    def _neighbors(self, pos, allow_unknown):
        for move in DIRECTIONS:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._is_passable(nxt, allow_unknown=allow_unknown):
                yield nxt

    def _is_frontier(self, pos):
        if not self._in_bounds(pos):
            return False
        r, c = pos
        if self.known_map[r][c] != EMPTY:
            return False
        for move in DIRECTIONS:
            dr, dc = move.value
            nr, nc = r + dr, c + dc
            if self._in_bounds((nr, nc)) and self.known_map[nr][nc] == UNKNOWN:
                return True
        return False

    def _visibility_gain(self, pos):
        if not self._in_bounds(pos):
            return 0
        gain = 0
        r, c = pos
        if self.known_map[r][c] == UNKNOWN:
            gain += 1

        for move in DIRECTIONS:
            dr, dc = move.value
            for distance in range(1, 6):
                nr, nc = r + dr * distance, c + dc * distance
                if not self._in_bounds((nr, nc)):
                    break
                if self.known_map[nr][nc] == WALL:
                    break
                if self.known_map[nr][nc] == UNKNOWN:
                    gain += 1
        return gain

    def _degree_not_wall(self, pos):
        degree = 0
        for move in DIRECTIONS:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._is_passable(nxt, allow_unknown=True):
                degree += 1
        return degree

    def _is_passable(self, pos, allow_unknown):
        if not self._in_bounds(pos):
            return False
        value = self.known_map[pos[0]][pos[1]]
        if value == WALL:
            return False
        if value == UNKNOWN and not allow_unknown:
            return False
        return True

    def _in_bounds(self, pos):
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _time_low(self):
        return time.time() - self._step_started_at > self.time_budget

    def _safe_action(self, action):
        if not isinstance(action, tuple) or len(action) != 2:
            return (Move.STAY, 1)
        move, steps = action
        if not isinstance(move, Move):
            return (Move.STAY, 1)
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            steps = 1
        steps = max(1, min(self.pacman_speed, steps))
        return (move, steps)


