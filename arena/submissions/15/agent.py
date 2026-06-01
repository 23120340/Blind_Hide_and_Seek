"""
Student submission — Team 252_255_262_264
Blind Hide & Seek Arena — 6-layer hybrid pipeline

Pacman (Seeker):
  Layer 1: Static map + observation accumulator (known_map, wall_map, seen_mask, heatmap)
  Layer 2: Histogram belief tracker — predict (diffuse) + correct (negative evidence) + normalize
  Layer 3: Mode selector — CAPTURE / INTERCEPT / HUNT / EXPLORE / ESCAPE
  Layer 4: Planner — expectimax-1ply (INTERCEPT), risk-aware A* (HUNT/EXPLORE), scored frontier
  Layer 5: Motion — speed-2 safety guard (only when next 2 cells both confirmed empty)
  Layer 6: Safety — 850ms deadline guard, always return valid (Move, steps)

Ghost (Hider):
  Score-based evasion: maximize exits, avoid dead-ends, break axis with Pacman,
  break line-of-sight, tabu anti-loop, corridor commitment, minimax-1ply when close.
"""

import sys
import time
import heapq
import numpy as np
from collections import deque
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


# Static wall map built from environment.py _create_default_map().
# Map layout is fixed; only agent starting positions are randomized.
# Building this once at module level lets both agents precompute graph features
# before their first step() call.
_STATIC_MAP_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....#.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]

_STATIC_WALL_MAP = np.array(
    [[1 if ch == "#" else 0 for ch in row] for row in _STATIC_MAP_LAYOUT],
    dtype=np.uint8,
)

_STATIC_H, _STATIC_W = _STATIC_WALL_MAP.shape

# Precompute all-pairs BFS distance on the static wall map once at import time.
# Map has ~180 passable cells; full cache is ~180*180*4 bytes ≈ 130 KB (well within 100 MB).
# Both agents share this cache for O(1) maze distance lookups.
def _build_dist_cache(wall_map):
    H, W = wall_map.shape
    passable = [(r, c) for r in range(H) for c in range(W) if not wall_map[r, c]]
    moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    cache = {}
    for src in passable:
        dist = {src: 0}
        q = deque([src])
        while q:
            cr, cc = q.popleft()
            for m in moves:
                dr, dc = m.value
                nxt = (cr + dr, cc + dc)
                if nxt not in dist and 0 <= nxt[0] < H and 0 <= nxt[1] < W and not wall_map[nxt]:
                    dist[nxt] = dist[(cr, cc)] + 1
                    q.append(nxt)
        cache[src] = dist
    return cache


# Import Move before building cache — Move is imported from environment at module top.
# We defer construction to first use so that the import chain is complete.
_STATIC_DIST_CACHE = None


def _get_dist_cache():
    global _STATIC_DIST_CACHE
    if _STATIC_DIST_CACHE is None:
        _STATIC_DIST_CACHE = _build_dist_cache(_STATIC_WALL_MAP)
    return _STATIC_DIST_CACHE


class PacmanAgent(BasePacmanAgent):
    """6-layer seeker pipeline. See module docstring for overview."""

    OBS_RADIUS   = 5     # cross-shaped view radius (matches environment.py)
    CAPTURE_DIST = 2     # win condition: Manhattan strictly < 2
    DEADLINE     = 0.9501  # soft-cap seconds; 49ms margin before hard 1s limit
    CLOSE_DIST   = 5       # == OBS_RADIUS: Ghost visible ⇒ maze dist ≤ 5 always
                           # so direct BFS covers ALL INTERCEPT cases; expectimax unreachable
    MED_DIST     = 10      # reserved
    HUNT_ENTROPY = 0.35  # normalized entropy below which HUNT beats EXPLORE
    HUNT_TTL     = 8     # steps since last sighting before switching to EXPLORE
    LAM_RISK     = 3.0
    LAM_INFO     = 1.5
    LAM_CAP      = 5.0
    RHO          = 0.65  # expectimax blend: RHO * worst-case + (1-RHO) * expected

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        # Initialize wall knowledge from the static map so we have full wall info
        # before step 1. Belief and graph features can be computed immediately.
        self.wall_map = _STATIC_WALL_MAP.copy()
        self.H, self.W = self.wall_map.shape

        # known[r,c]: 1=wall confirmed, 0=empty confirmed, -1=unvisited
        self.known = np.where(self.wall_map == 1, np.int8(1), np.int8(-1))
        self.seen_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        self.heatmap   = np.zeros((self.H, self.W), dtype=np.uint16)

        # Belief uniform over all passable cells at start
        self.belief = (self.wall_map == 0).astype(np.float32)
        self.belief /= self.belief.sum()

        # Precompute dead-ends (exits <= 1) and junctions (exits >= 3)
        # from the static wall map so trap_target and choke_point are fast.
        self.dead_ends = set()
        self.junctions = set()
        self._precompute_graph()

        # Shared all-pairs distance cache built from the static wall map.
        # Lookup: self.dist_cache[a][b] = exact maze distance (or absent if unreachable).
        self.dist_cache = _get_dist_cache()

        # Runtime state — reset each round in _observe when step_number == 1
        self.visit_count    = {}
        self.recent         = deque(maxlen=16)
        self.last_enemy     = None   # last known enemy position
        self.prev_enemy     = None   # enemy position one step before last_enemy
        self.last_seen_step = -999
        self.my_pos         = None

        # Single-source BFS cache — recomputed once per step from my_position.
        # All path queries within a step reuse this instead of running new BFS.
        self._pac_parents   = {}     # {pos: (prev_pos, Move)}
        self._pac_dist      = {}     # {pos: int distance from Pacman}

    def _precompute_graph(self):
        """Tag each passable cell as dead-end or junction based on wall_map."""
        for r in range(self.H):
            for c in range(self.W):
                if self.wall_map[r, c]:
                    continue
                exits = self._count_exits_wall((r, c))
                if exits <= 1:
                    self.dead_ends.add((r, c))
                if exits >= 3:
                    self.junctions.add((r, c))

    def _count_exits_wall(self, pos):
        r, c = pos
        return sum(
            1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < self.H and 0 <= c + dc < self.W
            and self.wall_map[r + dr, c + dc] == 0
        )

    def step(self, map_state, my_position, enemy_position, step_number):
        """Main pipeline: observe → belief update → mode select → plan → motion → safety."""
        deadline = time.time() + self.DEADLINE
        self.my_pos = my_position

        # Layer 1: update knowledge base
        self._observe(map_state, my_position, enemy_position, step_number)

        # Layer 2: update belief distribution
        self._belief_update(my_position, enemy_position)

        # Layer 3: choose mode
        mode = self._choose_mode(enemy_position, step_number)

        # Single-source BFS from Pacman — computed once, reused for all path queries.
        # Pass early_stop when Ghost is visible so BFS halts as soon as Ghost found,
        # instead of exploring all 180 reachable cells.
        self._bfs_from_me(my_position, early_stop=enemy_position)

        action = None

        # Layer 4: plan based on mode
        if mode == "CAPTURE":
            # Ghost visible and in capture range — path via cached BFS
            path = self._path_via_cache(my_position, enemy_position)
            if path:
                action = self._path_action(my_position, path, map_state)

        elif mode == "INTERCEPT":
            # Ghost is visible via cross-shaped FOV (radius = OBS_RADIUS = 5).
            # Visible ⇒ same row/col, no walls between ⇒ maze dist ≤ OBS_RADIUS = 5.
            # CLOSE_DIST == OBS_RADIUS ⇒ direct BFS always triggers; expectimax unreachable.
            dist_to_ghost = self._pac_dist.get(enemy_position,
                                self._maze_dist_cached(my_position, enemy_position))

            if dist_to_ghost <= self.CLOSE_DIST:
                # Normal case: BFS straight to Ghost via single-step cache
                path = self._path_via_cache(my_position, enemy_position)
                if path:
                    action = self._path_action(my_position, path, map_state)
            else:
                # Unreachable when CLOSE_DIST >= OBS_RADIUS, kept as safety fallback
                ghost_vel = (0, 0)
                if self.prev_enemy is not None:
                    ghost_vel = (enemy_position[0] - self.prev_enemy[0],
                                 enemy_position[1] - self.prev_enemy[1])
                choke = self._forward_choke(enemy_position, ghost_vel)
                action = self._intercept_action(my_position, enemy_position, deadline, choke)
                if action is None or action[0] == Move.STAY:
                    ring_target = self._capture_ring_nearest(my_position, enemy_position)
                    if ring_target:
                        path = self._path_via_cache(my_position, ring_target)
                        if path:
                            action = self._path_action(my_position, path, map_state)
                if (action is None or action[0] == Move.STAY) and choke:
                    path = self._path_via_cache(my_position, choke)
                    if path:
                        action = self._path_action(my_position, path, map_state)

        elif mode == "HUNT":
            # Ghost recently lost — head to belief peak via risk-aware A*
            if time.time() < deadline:
                peak = tuple(int(x) for x in
                             np.unravel_index(self.belief.argmax(), self.belief.shape))
                path = self._risk_aware_astar(my_position, {peak}, "HUNT", deadline) or \
                       self._path_via_cache(my_position, peak)
                if path:
                    action = self._path_action(my_position, path, map_state)

        elif mode == "EXPLORE":
            # Ghost long-lost or belief too diffuse — scored frontier exploration
            if time.time() < deadline:
                target = self._info_target(my_position)
                if target and target != my_position:
                    path = self._risk_aware_astar(my_position, {target}, "EXPLORE", deadline) or \
                           self._path_via_cache(my_position, target)
                    if path:
                        action = self._path_action(my_position, path, map_state)

        elif mode == "ESCAPE":
            # ABAB loop detected — diversify to low-visit neighbor
            action = self._fallback(my_position, map_state)

        # Layer 4 fallback: BFS to belief peak using cache
        if action is None and time.time() < deadline:
            peak = tuple(int(x) for x in
                         np.unravel_index(self.belief.argmax(), self.belief.shape))
            path = self._path_via_cache(my_position, peak)
            if path:
                action = self._path_action(my_position, path, map_state)

        # Layer 6: safety — guarantee non-None valid return
        if action is None:
            action = self._fallback(my_position, map_state)

        move, steps = action if isinstance(action, tuple) and len(action) == 2 \
                      else (Move.STAY, 1)
        if not isinstance(move, Move):
            move = Move.STAY
        steps = max(1, min(int(steps), self.pacman_speed))
        return (move, steps)

    # Layer 1: observation accumulator

    def _observe(self, map_state, my_pos, enemy_pos, step_number):
        """Merge new observation into accumulated knowledge base.

        On step 1 we also handle map size mismatch (tournament may use a
        different arena) by reinitializing all arrays from scratch.
        """
        if step_number == 1:
            if map_state.shape != (self.H, self.W):
                # Different arena — rebuild everything from observation
                self.H, self.W = map_state.shape
                self.wall_map  = np.zeros(map_state.shape, dtype=np.uint8)
                self.known     = np.full(map_state.shape, -1, dtype=np.int8)
                self.seen_mask = np.zeros(map_state.shape, dtype=np.uint8)
                self.heatmap   = np.zeros(map_state.shape, dtype=np.uint16)
                self.belief    = None
                self.dead_ends.clear()
                self.junctions.clear()

            # Walls are always visible in map_state regardless of position,
            # so merging here confirms (or first-builds) the wall layer.
            self.wall_map[map_state == 1] = 1
            self.known[map_state == 1]    = 1

            if self.belief is None:
                self.belief = (self.wall_map == 0).astype(np.float32)
                s = self.belief.sum()
                if s > 0:
                    self.belief /= s

            if not self.dead_ends:
                self._precompute_graph()

            # Reset per-round trackers
            self.visit_count    = {}
            self.recent.clear()
            self.last_enemy     = None
            self.prev_enemy     = None
            self.last_seen_step = -999

        # Merge visible region into knowledge base
        vis = map_state != -1
        self.known[vis]     = map_state[vis]
        self.wall_map[vis]  = (map_state[vis] == 1).astype(np.uint8)
        self.seen_mask[vis] = 1

        # Age heatmap (all cells +1 each step, visible cells reset to 0)
        np.add(self.heatmap, 1, out=self.heatmap)
        self.heatmap[vis] = 0

        self.visit_count[my_pos] = self.visit_count.get(my_pos, 0) + 1
        self.recent.append(my_pos)

        # Track enemy position and velocity for velocity prediction
        if enemy_pos is not None:
            self.prev_enemy     = self.last_enemy
            self.last_enemy     = enemy_pos
            self.last_seen_step = step_number
        else:
            # Don't overwrite prev_enemy when not visible
            pass

    # Layer 2: histogram belief tracker

    def _belief_update(self, my_pos, enemy_pos):
        """Bayes filter: predict by diffusion, correct by observation, normalize.

        Predict: spread each cell's belief mass uniformly to its legal neighbors
        (simple ghost motion model — uniform over passable moves + stay).

        Correct:
          - If Ghost visible: collapse to delta function at observed position.
            Skip diffusion entirely — no point computing O(H*W) spread when
            we already know exact position.
          - If not visible: diffuse then zero-out cells in our line-of-sight.
        """
        if self.belief is None:
            return

        if enemy_pos is not None:
            # Ghost visible: collapse belief immediately, skip O(H*W) diffusion
            self.belief[:] = 0.0
            self.belief[enemy_pos] = 1.0
            return

        H, W = self.H, self.W

        pred = np.zeros((H, W), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                b = self.belief[r, c]
                if self.wall_map[r, c] or b < 1e-9:
                    continue
                for (nr, nc), p in self._ghost_successors((r, c)):
                    pred[nr, nc] += b * p

        for vr, vc in self._cells_visible_from(my_pos):
            pred[vr, vc] = 0.0
        pred[self.wall_map == 1] = 0.0


        total = pred.sum()
        if total > 1e-9:
            self.belief = pred / total
        else:
            legal = (self.wall_map == 0).astype(np.float32)
            s = legal.sum()
            self.belief = legal / s if s > 0 else legal

    def _ghost_successors(self, pos):
        """Uniform transition model: Ghost moves to any legal neighbor or stays."""
        r, c = pos
        candidates = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and not self.wall_map[nr, nc]:
                candidates.append((nr, nc))
        candidates.append(pos)  # stay option
        p = 1.0 / len(candidates)
        return [(s, p) for s in candidates]

    def _cells_visible_from(self, pos, radius=None):
        """Cross-shaped raycasting matching environment.py get_visible_cells_cross().
        Wall stops the ray and is not included in the visible set."""
        if radius is None:
            radius = self.OBS_RADIUS
        r, c = pos
        visible = {pos}
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for d in range(1, radius + 1):
                nr, nc = r + dr * d, c + dc * d
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    break
                if self.wall_map[nr, nc]:
                    break
                visible.add((nr, nc))
        return visible

    def _belief_entropy(self):
        """Normalized entropy in [0, 1]. 0 = certain, 1 = uniform."""
        if self.belief is None:
            return 1.0
        flat = self.belief[self.belief > 1e-9]
        entropy  = float(-np.sum(flat * np.log(flat + 1e-12)))
        n_legal  = int(np.sum(self.wall_map == 0))
        max_ent  = float(np.log(n_legal + 1e-12))
        return entropy / max_ent if max_ent > 1e-9 else 1.0

    # Layer 3: mode selector

    def _choose_mode(self, enemy_pos, step_number):
        """Priority: CAPTURE > INTERCEPT > ESCAPE > HUNT > EXPLORE."""
        if enemy_pos is not None:
            man = self._manhattan(self.my_pos, enemy_pos)
            if man < self.CAPTURE_DIST:  # strictly < 2 per PDF
                return "CAPTURE"
            return "INTERCEPT"

        if self._detect_loop():
            return "ESCAPE"

        steps_lost = step_number - self.last_seen_step
        if self._belief_entropy() < self.HUNT_ENTROPY and steps_lost <= self.HUNT_TTL:
            return "HUNT"
        return "EXPLORE"

    def _detect_loop(self):
        """Detect ABAB oscillation or repeated stays."""
        r = self.recent
        if len(r) >= 4 and r[-4] == r[-2] and r[-3] == r[-1]:
            return True
        if len(r) >= 2 and r[-1] == r[-2]:
            return True
        return False

    # Layer 4: planners

    def _intercept_action(self, my_pos, enemy_pos, deadline, choke=None):
        """Shallow expectimax 1-ply joint-action evaluation.

        Ghost model: uniform over legal moves + stay.
        Score = RHO * worst-case + (1-RHO) * expected value.
        Velocity prediction: if Ghost was visible last step, extrapolate one step.
        Choke bonus: if moving toward the forward choke point, add extra score.
        """
        predicted_enemy = None
        if self.prev_enemy is not None:
            dr = enemy_pos[0] - self.prev_enemy[0]
            dc = enemy_pos[1] - self.prev_enemy[1]
            pe = (enemy_pos[0] + dr, enemy_pos[1] + dc)
            if self._in_bounds_hw(pe) and self.known[pe] != 1:
                predicted_enemy = pe

        trap = self._trap_target(enemy_pos)

        ghost_nexts = [enemy_pos]
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(enemy_pos, m)
            if self._is_passable_known(nxt):
                ghost_nexts.append(nxt)
        g_prob = 1.0 / len(ghost_nexts)

        pac_candidates = []
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            n1 = self._apply_move(my_pos, m)
            if self._is_passable_known(n1):
                pac_candidates.append((n1, m, 1))
                if self.pacman_speed >= 2:
                    n2 = self._apply_move(n1, m)
                    if self._is_passable_known(n2):
                        pac_candidates.append((n2, m, 2))

        if not pac_candidates:
            return (Move.STAY, 1)

        best_action, best_score = None, -1e12
        for pac_next, move, steps in pac_candidates:
            if time.time() > deadline:
                break
            worst_v, exp_v = 1e12, 0.0
            for g_next in ghost_nexts:
                v = self._leaf_eval(pac_next, g_next, trap)
                worst_v = min(worst_v, v)
                exp_v  += g_prob * v

            # Velocity prediction bonus: landing near where Ghost is heading
            if predicted_enemy is not None:
                if self._manhattan(pac_next, predicted_enemy) < self.CAPTURE_DIST:
                    exp_v += 500.0

            # Choke bonus: prefer moves that bring us closer to the forward choke point
            if choke is not None:
                choke_d_now  = self._maze_dist_cached(my_pos, choke)
                choke_d_next = self._maze_dist_cached(pac_next, choke)
                if choke_d_next < choke_d_now:
                    exp_v += 40.0

            score = self.RHO * worst_v + (1.0 - self.RHO) * exp_v
            if score > best_score:
                best_score = score
                best_action = (move, steps)

        return best_action if best_action else (Move.STAY, 1)

    def _forward_choke(self, ghost_pos, ghost_vel):
        """Find the next junction Ghost will reach by raycasting in its velocity direction.

        If Ghost is moving (vel != 0,0), follow that direction until hitting a wall or
        a junction (exit count >= 3). The junction is the point where Ghost must decide
        to turn — arriving there before Ghost cuts off its escape options.

        Returns None if velocity is (0,0) or no junction found within 8 cells.
        """
        dr, dc = ghost_vel
        if dr == 0 and dc == 0:
            return None

        cur = ghost_pos
        for _ in range(8):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not self._in_bounds_hw(nxt) or self.wall_map[nxt]:
                # Ghost will hit a wall — the current cell is where it must turn
                return cur if cur != ghost_pos else None
            if nxt in self.junctions:
                return nxt
            cur = nxt

        return cur if cur != ghost_pos else None

    def _capture_ring_nearest(self, my_pos, enemy_pos):
        """Return the capture-ring cell (enemy or its passable neighbors)
        that has the shortest BFS distance from my_pos.

        Using the capture-ring instead of targeting just enemy_pos gives
        multiple attack angles and shorter average paths."""
        ring = [enemy_pos]
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(enemy_pos, m)
            if self._is_passable_known(nxt):
                ring.append(nxt)

        best, best_d = None, 1e9
        for cell in ring:
            if cell == my_pos:
                return cell
            d = self._bfs_distance(my_pos, cell)
            if d < best_d:
                best_d = d
                best = cell
        return best

    def _leaf_eval(self, my_pos, ghost_pos, trap_pos):
        """Leaf evaluation for joint expectimax."""
        man = self._manhattan(my_pos, ghost_pos)
        if man < self.CAPTURE_DIST:
            return 10000.0

        maze_d  = self._bfs_distance(my_pos, ghost_pos)
        escapes = sum(
            1 for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
            if self._is_passable_known(self._apply_move(ghost_pos, m))
        )
        score = -8.0 * maze_d - 5.0 * escapes
        if escapes <= 1:
            score += 50.0
        if ghost_pos in self.dead_ends:
            score += 30.0
        score -= self._manhattan(my_pos, trap_pos) * 2.0
        return score

    def _trap_target(self, enemy_pos):
        """Return the cell (enemy or neighbor) with the fewest exits.
        Prefers confirmed dead-ends from the static graph."""
        options = [enemy_pos]
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(enemy_pos, m)
            if self._is_passable_known(nxt):
                options.append(nxt)

        best, best_score = enemy_pos, -1e12
        for pos in options:
            exits   = self._count_exits_known(pos)
            unknown = self._unknown_adjacent(pos)
            de_bonus = 20.0 if pos in self.dead_ends else 0.0
            score = (-exits * 12.0
                     - unknown * 4.0
                     - self._manhattan(pos, enemy_pos) * 0.5
                     + de_bonus)
            if score > best_score:
                best, best_score = pos, score
        return best

    def _risk_aware_astar(self, start, goals, mode, deadline, limit=1200):
        """A* on known map with cost adjustments for unknown cells, turn penalty,
        belief risk/attraction, LOS information gain, and capture-ring bonus.

        mode='HUNT'    — negative risk_pen (attracted to belief peak)
        mode='EXPLORE' — positive risk_pen (repelled from known-belief regions)
        Falls back to plain BFS if timeout or no path found.
        """
        if not goals or start in goals:
            return []

        tie = 0
        heap = []
        h0 = min(self._manhattan(start, g) for g in goals)
        heapq.heappush(heap, (h0, 0.0, start, None, tie))
        best_g = {start: 0.0}
        parent = {}
        exp = 0

        while heap and exp < limit:
            if time.time() > deadline:
                break
            f, g, pos, last_dir, _ = heapq.heappop(heap)
            exp += 1

            if pos in goals:
                return self._reconstruct_path(parent, start, pos)

            for nxt, move in self._walkable_neighbors_known(pos):
                unknown_pen = 1.5 if self.known[nxt] == -1 else 0.0
                turn_pen    = 0.5 if last_dir is not None and move != last_dir else 0.0
                bv = float(self.belief[nxt]) if self.belief is not None else 0.0
                risk_pen = (-self.LAM_RISK if mode == "HUNT" else self.LAM_RISK) * bv
                info_b   = self.LAM_INFO * self._los_gain(nxt)
                cap_b    = self.LAM_CAP  * self._capture_ring_bonus(nxt)

                new_g = g + max(0.1, 1.0 + unknown_pen + turn_pen + risk_pen - info_b - cap_b)
                if new_g >= best_g.get(nxt, 1e18):
                    continue

                best_g[nxt] = new_g
                parent[nxt] = (pos, move)
                h = min(self._manhattan(nxt, gp) for gp in goals)
                tie += 1
                heapq.heappush(heap, (new_g + h, new_g, nxt, move, tie))

        target = next(iter(goals))
        return self._bfs_path(start, target, self.known)

    def _info_target(self, start):
        """BFS over known map scoring each reachable cell.

        Score = unknown_adj * 8 + heat * 2 + belief * 10 - dist - visits * 3.
        Balances information gain (unexplored neighbors), staleness (heatmap),
        belief mass (likely ghost region), travel cost, and revisit penalty.
        """
        q = deque([(start, 0)])
        visited = {start}
        best, best_score = None, -1e12

        while q:
            pos, dist = q.popleft()
            if dist > 35:
                continue

            unknown = self._unknown_adjacent(pos)
            heat    = float(self.heatmap[pos])
            bv      = float(self.belief[pos]) if self.belief is not None else 0.0
            vis     = self.visit_count.get(pos, 0)

            score = unknown * 8.0 + heat * 2.0 + bv * 10.0 - dist * 1.0 - vis * 3.0
            if score > best_score and pos != start:
                best, best_score = pos, score

            for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                nxt = self._apply_move(pos, m)
                if nxt not in visited and self._is_passable_known(nxt):
                    visited.add(nxt)
                    q.append((nxt, dist + 1))

        return best if best is not None else start

    def _fallback(self, my_pos, map_state):
        """Pick the legal neighbor with fewest visits and most exits (anti-loop)."""
        best_move, best_score = Move.STAY, -1e12
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(my_pos, m)
            if not self._is_passable(nxt, map_state):
                continue
            vis     = self.visit_count.get(nxt, 0)
            recent  = 10.0 if nxt in self.recent else 0.0
            unknown = 5.0  if self.known[nxt] == -1 else 0.0
            exits   = self._count_exits_known(nxt)
            score   = exits * 2.0 + unknown - vis * 3.0 - recent
            if score > best_score:
                best_score = score
                best_move  = m
        return (best_move, 1)

    # Layer 5: motion — speed-2 guard

    def _path_action(self, my_pos, path, map_state):
        """Convert path to (Move, steps).

        Speed 2 is only used when:
          1. pacman_speed >= 2
          2. Next two steps are the same direction (no turn required)
          3. Both intermediate cells are CONFIRMED empty (known == 0, not -1 or 1)
          4. Both cells are not walls in the current observation either

        Using known == 0 (not just != 1) avoids stepping into unknown cells at
        speed 2 where we cannot be sure there is no wall.
        """
        if not path:
            return (Move.STAY, 1)

        first_move = path[0]

        if (self.pacman_speed >= 2
                and len(path) >= 2
                and path[1] == first_move):
            n1 = self._apply_move(my_pos, first_move)
            n2 = self._apply_move(n1, first_move)
            if (self._in_bounds_hw(n1) and self._in_bounds_hw(n2)
                    and self.wall_map[n1] == 0
                    and self.wall_map[n2] == 0
                    and map_state[n1] != 1
                    and map_state[n2] != 1):
                return (first_move, 2)

        return (first_move, 1)

    # Utility helpers

    def _bfs_from_me(self, start, early_stop=None):
        """Single-source BFS from Pacman position on known map.

        Populates self._pac_parents and self._pac_dist for the current step.
        All subsequent path queries call _path_via_cache() instead of running
        a new BFS — O(n_cells) once vs O(n_cells) × N queries per step.

        early_stop: if given, BFS halts as soon as this cell is dequeued.
        When Ghost is visible, passing enemy_position avoids exploring the
        full 180-cell map and only reaches the cells needed for the path.
        """
        parents = {start: (None, None)}
        dist    = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            # Early stop: we have enough info to reconstruct path to early_stop
            if early_stop is not None and cur == early_stop:
                break
            cd  = dist[cur]
            for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                nxt = self._apply_move(cur, m)
                if nxt in parents:
                    continue
                if not self._in_bounds_hw(nxt) or self.known[nxt] == 1:
                    continue
                parents[nxt] = (cur, m)
                dist[nxt]    = cd + 1
                q.append(nxt)
        self._pac_parents = parents
        self._pac_dist    = dist

    def _path_via_cache(self, start, target):
        """Reconstruct path from cached _pac_parents in O(path_len).

        Returns [] if target unreachable or start==target.
        Falls back to _bfs_path if cache is empty (should not happen after _bfs_from_me).
        """
        if start == target:
            return []
        if target not in self._pac_parents:
            return self._bfs_path(start, target, self.known)
        path = []
        cur = target
        while cur != start:
            prev, move = self._pac_parents[cur]
            path.append(move)
            cur = prev
        path.reverse()
        return path

    def _bfs_path(self, start, target, grid):
        """Standard BFS returning list of Move directions. Used as fallback
        when _path_via_cache cannot serve the request (cache miss or different grid)."""
        if start == target:
            return []
        q = deque([start])
        parent = {start: (None, None)}
        while q:
            cur = q.popleft()
            for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                nxt = self._apply_move(cur, m)
                if nxt in parent:
                    continue
                r, c = nxt
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    continue
                if grid[r, c] == 1:
                    continue
                parent[nxt] = (cur, m)
                if nxt == target:
                    return self._reconstruct_path(parent, start, target)
                q.append(nxt)
        return []

    def _maze_dist_cached(self, a, b):
        """O(1) maze distance lookup from the precomputed static wall map cache.
        Falls back to Manhattan if a cell is not in the cache (e.g., after map resize)."""
        if a == b:
            return 0
        row = self.dist_cache.get(a)
        if row is not None:
            d = row.get(b)
            if d is not None:
                return d
        return self._manhattan(a, b)

    def _bfs_distance(self, a, b):
        """Maze distance: tries cache first, then BFS on known map as fallback.
        The cache covers all static-wall passable cells; BFS covers cells revealed
        after a map-size mismatch rebuild."""
        cached = self._maze_dist_cached(a, b)
        if cached != self._manhattan(a, b):
            return cached
        # Cache miss or Manhattan happened to equal maze dist — run BFS to confirm
        if a == b:
            return 0
        q = deque([(a, 0)])
        seen = {a}
        while q:
            cur, d = q.popleft()
            for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                nxt = self._apply_move(cur, m)
                if nxt in seen or not self._in_bounds_hw(nxt):
                    continue
                if self.known[nxt] == 1:
                    continue
                if nxt == b:
                    return d + 1
                seen.add(nxt)
                q.append((nxt, d + 1))
        return self._manhattan(a, b)

    def _reconstruct_path(self, parent, start, target):
        path, cur = [], target
        while cur != start:
            prev, move = parent[cur]
            path.append(move)
            cur = prev
        path.reverse()
        return path

    def _walkable_neighbors_known(self, pos):
        result = []
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(pos, m)
            if self._in_bounds_hw(nxt) and self.known[nxt] != 1:
                result.append((nxt, m))
        return result

    def _is_passable_known(self, pos):
        return self._in_bounds_hw(pos) and self.known[pos] != 1

    def _is_passable(self, pos, grid):
        r, c = pos
        return (0 <= r < grid.shape[0]
                and 0 <= c < grid.shape[1]
                and grid[r, c] != 1)

    def _in_bounds_hw(self, pos):
        r, c = pos
        return 0 <= r < self.H and 0 <= c < self.W

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _unknown_adjacent(self, pos):
        count = 0
        for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(pos, m)
            if self._in_bounds_hw(nxt) and self.known[nxt] == -1:
                count += 1
        return count

    def _count_exits_known(self, pos):
        return sum(
            1 for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
            if self._is_passable_known(self._apply_move(pos, m))
        )

    def _los_gain(self, pos):
        """Estimate new cells revealed if we move to pos (cross-view raycasting)."""
        if self.seen_mask is None:
            return 0.0
        count = 0.0
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for d in range(1, self.OBS_RADIUS + 1):
                nr, nc = r + dr * d, c + dc * d
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    break
                if self.wall_map[nr, nc]:
                    break
                if self.seen_mask[nr, nc] == 0:
                    count += 1.0
                    if self.belief is not None:
                        count += 5.0 * float(self.belief[nr, nc])
        return count

    def _capture_ring_bonus(self, pos):
        """Bonus score for cells within capture range of the belief peak."""
        if self.belief is None:
            return 0.0
        peak = tuple(int(x) for x in
                     np.unravel_index(self.belief.argmax(), self.belief.shape))
        return 1.0 if self._manhattan(pos, peak) < self.CAPTURE_DIST else 0.0



class GhostAgent(BaseGhostAgent):
    """
    Ghost agent with a 3-phase hiding strategy under partial observability.

    The agent operates in three sequential phases:

    Phase 1 — OPENING:
        At game start, Ghost BFS-pathfinds to a pre-selected ambush spot (5, 12)
        on the upper half of the map. This spot is chosen because:
        - It is surrounded by walls on top and bottom, limiting Pacman's approach
          to horizontal directions only.
        - The path to it goes RIGHT from Ghost's spawn (9, 10), avoiding Pacman's
          typical upward route through the left-center corridor.
        - FULL_MAP_ARRAY is used (not map_state) to ensure reliable pathfinding
          even when most of the map is still marked as unknown (-1).

    Phase 2 — CAMPING:
        Ghost stays still at the ambush spot. Since the spot has walls blocking
        Pacman's line of sight from most directions, Ghost remains undetected
        until Pacman physically enters the adjacent cell. The moment Pacman is
        visible, Ghost immediately switches to evasion.

    Phase 3 — EVASION:
        Ghost scores all neighboring cells and picks the best escape move.
        Scoring criteria (in order of weight):
          1. Panic mode: if ALL neighbors are within BFS distance 3 of Pacman,
             skip all penalties and simply maximize distance (×100 weight).
          2. Otherwise: apply panic penalty (−2000) for dist ≤ 3, warning
             penalty (−500) for dist ≤ 5, and distance bonus (+dist × 20).
          3. Line-of-sight break: +1000 if moving to this cell breaks Pacman's
             straight-line visibility — rewards rounding corners to lose Pacman.
          4. Junction/corridor awareness: +100 for junctions (≥3 exits),
             −500 for dead ends (1 exit), −100 for straight corridors (2
             collinear exits), +50 for L-shaped 2-exit cells.
          5. Wall hugging: +20 per adjacent wall — staying near walls reduces
             the angles from which Pacman can spot Ghost.
          6. Long-range drift: −5 × Manhattan distance to the globally farthest
             cell from Pacman, giving Ghost a consistent long-term direction.
          7. Loop avoidance: −200 if the cell was recently visited (history
             deque of last 4 positions).

        Anti-stuck fallback: if scoring produces STAY (e.g. all moves score
        equally poorly), Ghost picks a random unvisited neighbor to break ties.

    Key design decisions:
        - FULL_MAP_ARRAY is a hardcoded static copy of the fixed 21×21 map used
          for all BFS calls. This avoids errors from -1 (unknown) cells in
          map_state being mistakenly treated as passable corridors.
        - last_known_pacman persists after Pacman leaves vision, so evasion
          continues to run away from the last known threat position.
        - Evasion uses map_state (not FULL_MAP_ARRAY) for neighbor lookups and
          LOS checks so Ghost correctly avoids cells it can currently see are
          walls, while BFS distances use FULL_MAP_ARRAY for global accuracy.
    """

    # Hardcoded full 21×21 map (1 = wall, 0 = open path).
    # Used for all BFS pathfinding to avoid -1 (unknown) cells in map_state
    # being incorrectly treated as walkable during partial observation.
    FULL_MAP_ARRAY = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1],
        [1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1],
        [1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
        [1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1],
        [1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
        [1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1],
        [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ], dtype=np.int8)
    MAX_THINK_SECONDS = 0.89

    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.

        Attributes:
            history (deque): Last 4 visited positions, used to penalize
                revisiting cells and prevent short oscillation loops.
            last_known_pacman (tuple | None): Most recently seen Pacman
                position. Used as evasion target even after losing sight.
            turns_since_seen (int): Steps elapsed since Pacman was last
                visible. Resets to 0 whenever enemy_position is not None.
            opening_target (tuple | None): Ambush destination for Phase 1.
                Set to (5, 12) on first step if that cell is passable.
            opening_moves (list[Move]): Precomputed BFS path to opening_target,
                consumed one move per step during Phase 1.
            sectors (dict): Map quadrant reference points, computed once on
                the first step from the map dimensions.
            in_opening_phase (bool): True while Ghost is travelling to the
                ambush spot. Becomes False on arrival.
            in_camping_phase (bool): True while Ghost is stationary at the
                ambush spot. Becomes False the moment Pacman is visible.
        """
        super().__init__(**kwargs)
        self.name = "Hiep_Ghost"

        # Recent position history for loop-avoidance penalty in evasion
        self.history = deque(maxlen=4)

        # Last confirmed Pacman position (persists after losing sight)
        self.last_known_pacman = None

        # Counter incremented each step Pacman is not visible
        self.turns_since_seen = 0

        # Opening phase: destination and precomputed path
        self.opening_target = None
        self.opening_moves = []

        # Map quadrant reference points (computed lazily on first step)
        self.sectors = {}

        # Phase flags
        self.in_opening_phase = True
        self.in_camping_phase = False

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        """
        Decide the next move for Ghost given the current partial observation.

        On the first call, lazily initialises sector reference points and
        selects the opening ambush target. On every subsequent call, the
        method updates enemy tracking state and dispatches to whichever
        phase is currently active.

        Args:
            map_state (np.ndarray): 21×21 partial observation grid.
                Cell values: 0 = visible empty, 1 = wall, -1 = unknown.
            my_position (tuple): Ghost's current (row, col).
            enemy_position (tuple | None): Pacman's (row, col) if visible,
                else None.
            step_number (int): Current step count (max 200).

        Returns:
            Move: One of Move.UP / DOWN / LEFT / RIGHT / STAY.
        """
        decision_deadline = time.perf_counter() + self.MAX_THINK_SECONDS
        height, width = map_state.shape

        # --- Lazy one-time initialisation ---
        if not self.sectors:
            self.sectors = {
                "TR": (height // 4, width * 3 // 4),
                "TL": (height // 4, width // 4),
                "BL": (height * 3 // 4, width // 4),
                "BR": (height * 3 // 4, width * 3 // 4),
            }
            # Ambush spot on upper-right area; path goes RIGHT from spawn,
            # away from Pacman's typical upward left-corridor approach.
            ambush_spot = (5, 12)
            if (
                ambush_spot[0] < height
                and ambush_spot[1] < width
                and map_state[ambush_spot[0], ambush_spot[1]] != 1
            ):
                self.opening_target = ambush_spot
            else:
                # Fallback if ambush cell is a wall (shouldn't happen on
                # the fixed map, but guards against map variations)
                self.opening_target = my_position
                self.in_opening_phase = False
                self.in_camping_phase = True

        # --- Enemy tracking ---
        if enemy_position is not None:
            # Pacman is visible: interrupt any passive phase immediately
            if self.in_opening_phase or self.in_camping_phase:
                self.in_opening_phase = False
                self.in_camping_phase = False
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1

        # Fallback: if Pacman was never seen, assume map centre
        if self.last_known_pacman is None:
            self.last_known_pacman = (height // 2, width // 2)

        if self._is_timeout(decision_deadline):
            return self._timeout_fallback_move(my_position, self.last_known_pacman, map_state)

        # --- Phase 1: OPENING — travel to ambush spot ---
        if self.in_opening_phase:
            if my_position == self.opening_target:
                # Arrived; switch to camping
                self.in_opening_phase = False
                self.in_camping_phase = True
                return Move.STAY

            # Compute path on first step or if it was exhausted prematurely
            if not self.opening_moves:
                self.opening_moves = self._bfs_path(
                    my_position, self.opening_target, self.FULL_MAP_ARRAY, decision_deadline
                )

            if self.opening_moves:
                next_move = self.opening_moves.pop(0)
                if next_move:
                    dr, dc = next_move.value
                    self.history.append((my_position[0] + dr, my_position[1] + dc))
                    return next_move

            # Path exhausted without reaching target (edge case): fall through
            self.in_opening_phase = False

            if self._is_timeout(decision_deadline):
                return self._timeout_fallback_move(my_position, self.last_known_pacman, map_state)

        # --- Phase 2: CAMPING — stay hidden at ambush spot ---
        if self.in_camping_phase:
            return Move.STAY

        if self._is_timeout(decision_deadline):
            return self._timeout_fallback_move(my_position, self.last_known_pacman, map_state)

        # --- Phase 3: EVASION — run from last known Pacman position ---

        # BFS distances from Pacman's last known position to every reachable
        # cell. Uses FULL_MAP_ARRAY (not map_state) for global accuracy,
        # avoiding -1 unknown cells being treated as shortcuts.
        pacman_dist_map = self._get_bfs_distance_map(
            self.last_known_pacman, self.FULL_MAP_ARRAY, decision_deadline
        )

        if self._is_timeout(decision_deadline):
            return self._timeout_fallback_move(my_position, self.last_known_pacman, map_state)

        # Long-range drift target: the cell farthest from Pacman by BFS.
        # Ghost drifts toward this globally safe region step by step.
        target_pos = my_position
        max_dist = -1
        for pos, dist in pacman_dist_map.items():
            if self._is_timeout(decision_deadline):
                return self._timeout_fallback_move(my_position, self.last_known_pacman, map_state)
            if dist > max_dist:
                max_dist = dist
                target_pos = pos

        best_move = self._evaluate_best_move(
            my_position,
            self.last_known_pacman,
            target_pos,
            map_state,
            pacman_dist_map,
            decision_deadline,
        )
        if best_move is None:
            best_move = Move.STAY

        # Anti-stuck fallback: if scoring chose STAY but Pacman was seen
        # recently, pick any unvisited neighbour at random to keep moving
        if best_move == Move.STAY and self.turns_since_seen < 5:
            valid = self._get_valid_neighbors(my_position, map_state)
            candidates = [n for n in valid if n not in self.history]
            if candidates:
                next_pos = random.choice(candidates)
                dr, dc = next_pos[0] - my_position[0], next_pos[1] - my_position[1]
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if move.value == (dr, dc):
                        return move

        dr, dc = best_move.value
        self.history.append((my_position[0] + dr, my_position[1] + dc))
        return best_move

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------

    def _get_bfs_distance_map(
        self,
        start_pos: tuple,
        map_state: np.ndarray,
        deadline: float = None,
    ) -> dict:
        """
        Compute shortest-path distances from start_pos to every reachable cell.

        Args:
            start_pos (tuple): (row, col) of the BFS source.
            map_state (np.ndarray): Map to use for wall detection.
                Pass FULL_MAP_ARRAY for global distances, or the live
                map_state for distances restricted to visible cells.

        Returns:
            dict: Mapping (row, col) → BFS distance from start_pos.
                  Cells unreachable from start_pos are absent from the dict.
        """
        distances = {start_pos: 0}
        queue = deque([start_pos])
        while queue:
            if self._is_timeout(deadline):
                break
            curr = queue.popleft()
            dist = distances[curr]
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in distances:
                    distances[(nr, nc)] = dist + 1
                    queue.append((nr, nc))
        return distances

    def _get_valid_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """
        Return all walkable neighbours of pos (no walls, within bounds).

        A cell is walkable if its map_state value is not 1 (wall).
        Unknown cells (-1) are treated as walkable when map_state is the
        live observation, and correctly as walls when FULL_MAP_ARRAY is used.

        Args:
            pos (tuple): (row, col) to expand.
            map_state (np.ndarray): Map to check against.

        Returns:
            list[tuple]: List of (row, col) neighbours that are walkable.
        """
        r, c = pos
        height, width = map_state.shape
        valid = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = r + move.value[0], c + move.value[1]
            if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                valid.append((nr, nc))
        return valid

    def _has_line_of_sight(self, p1: tuple, p2: tuple, map_state: np.ndarray) -> bool:
        """
        Check whether p1 and p2 share an unobstructed straight-line view.

        Line of sight exists only along the same row or the same column.
        Any wall (value 1) between the two points breaks it. If p1 and p2
        are on neither the same row nor the same column, returns False.

        Args:
            p1 (tuple): First position (row, col).
            p2 (tuple): Second position (row, col).
            map_state (np.ndarray): Map used for wall detection.

        Returns:
            bool: True if there is an unobstructed straight-line view,
                  False otherwise.
        """
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if map_state[r1, c] == 1:
                    return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if map_state[r, c1] == 1:
                    return False
            return True
        return False  # Not aligned — no LOS possible

    def _bfs_path(
        self,
        start: tuple,
        target: tuple,
        map_state: np.ndarray,
        deadline: float = None,
    ) -> list:
        """
        Find the shortest path from start to target using BFS.

        Args:
            start (tuple): Starting (row, col).
            target (tuple): Destination (row, col).
            map_state (np.ndarray): Map for wall detection. Use
                FULL_MAP_ARRAY for reliable opening/return paths.

        Returns:
            list[Move]: Sequence of Move enums to follow from start to
                        target. Returns an empty list if start == target
                        or if no path exists.
        """
        if start == target:
            return []
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            if self._is_timeout(deadline):
                return []
            curr, path = queue.popleft()
            if curr == target:
                return path
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dr, dc = nr - curr[0], nc - curr[1]
                    move_enum = None
                    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        if move.value == (dr, dc):
                            move_enum = move
                            break
                    if move_enum:
                        queue.append(((nr, nc), path + [move_enum]))
        return []

    # ------------------------------------------------------------------
    # Evasion scoring
    # ------------------------------------------------------------------

    def _evaluate_best_move(
        self,
        my_pos: tuple,
        enemy_pos: tuple,
        target_pos: tuple,
        map_state: np.ndarray,
        pacman_dist_map: dict,
        deadline: float = None,
    ) -> Move:
        """
        Score every valid neighbour and return the Move leading to the best one.

        Scoring breakdown (applied to each candidate cell (nr, nc)):

        1. Distance-based safety:
           - Full-panic mode (all neighbours within BFS dist 3):
             score += dist × 100  (pure distance maximisation, no penalties)
           - Normal mode:
             dist ≤ 3  → score −2000  (danger zone, strongly avoid)
             dist ≤ 5  → score −500   (warning zone, prefer to escape)
             score += dist × 20        (general distance reward)

        2. Line-of-sight break:
           score +1000 if moving to (nr, nc) removes straight-line visibility
           to enemy_pos. Rewards turning corners to lose Pacman immediately.

        3. Topology (number of exits from the candidate cell):
           ≥ 3 exits (junction) → +100  (multiple escape routes available)
           1 exit  (dead end)   → −500  (trap: Pacman can block the exit)
           2 exits, collinear   → −100  (straight corridor, easy to follow)
           2 exits, L-shaped    → +50   (corner provides a turning option)

        4. Wall hugging:
           score += 20 × (number of walls adjacent to (nr, nc))
           Staying near walls narrows the angles from which Pacman can spot
           Ghost and limits Pacman's LOS reach toward Ghost's position.

        5. Long-range drift:
           score −= 5 × Manhattan distance to target_pos, where target_pos
           is the globally farthest cell from Pacman by BFS. This gently
           steers Ghost away from Pacman's side of the map over time.

        6. Loop avoidance:
           score −200 if (nr, nc) is in self.history (last 4 positions).
           Prevents Ghost from oscillating between two adjacent cells.

        Args:
            my_pos (tuple): Ghost's current (row, col).
            enemy_pos (tuple): Pacman's last known (row, col).
            target_pos (tuple): Globally farthest cell from Pacman (BFS).
            map_state (np.ndarray): Live partial map for neighbour/LOS checks.
            pacman_dist_map (dict): BFS distances from enemy_pos (computed
                from FULL_MAP_ARRAY for global accuracy).

        Returns:
            Move: The Move enum leading to the highest-scoring neighbour,
                  or Move.STAY if no valid neighbours exist.
        """
        valid_neighbors = self._get_valid_neighbors(my_pos, map_state)
        moves_score = []
        height, width = map_state.shape

        # Detect full-panic: all reachable neighbours are within danger range
        all_dists = [pacman_dist_map.get((nr, nc), 0) for nr, nc in valid_neighbors]
        full_panic = valid_neighbors and all(d <= 3 for d in all_dists)

        for nr, nc in valid_neighbors:
            if self._is_timeout(deadline):
                break
            score = 0
            real_dist_to_enemy = pacman_dist_map.get((nr, nc), 0)

            # 1) Distance-based safety
            if full_panic:
                # Cornered — ignore zone penalties, just flee as far as possible
                score += real_dist_to_enemy * 100
            else:
                if real_dist_to_enemy <= 3:
                    score -= 2000   # Danger zone: strongly avoid
                elif real_dist_to_enemy <= 5:
                    score -= 500    # Warning zone: prefer to escape
                score += real_dist_to_enemy * 20  # General distance reward

            # 2) Line-of-sight break: strongly reward breaking Pacman's view
            if not self._has_line_of_sight((nr, nc), enemy_pos, map_state):
                score += 1000

            # 3) Topology: prefer junctions, penalise dead ends and corridors
            next_valid_moves = self._get_valid_neighbors((nr, nc), map_state)
            num_exits = len(next_valid_moves)

            is_corridor = False
            if num_exits == 2:
                r1, c1 = next_valid_moves[0]
                r2, c2 = next_valid_moves[1]
                if r1 == r2 or c1 == c2:
                    is_corridor = True  # Two exits are collinear → straight corridor

            if num_exits >= 3:
                score += 100    # Junction: multiple escape routes
            elif num_exits == 1:
                score -= 500    # Dead end: Pacman can trap Ghost here
            elif is_corridor:
                score -= 100    # Straight corridor: easy for Pacman to follow
            else:
                score += 50     # L-shaped 2-exit: corner provides a turning option

            # 4) Wall hugging: prefer cells surrounded by walls
            adjacent_walls = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                check_r, check_c = nr + dr, nc + dc
                if 0 <= check_r < height and 0 <= check_c < width:
                    if map_state[check_r, check_c] == 1:
                        adjacent_walls += 1
            if adjacent_walls > 0:
                score += 20 * adjacent_walls

            # 5) Long-range drift: gently steer toward the globally safe region
            dist_to_target = abs(nr - target_pos[0]) + abs(nc - target_pos[1])
            score -= dist_to_target * 5

            # 6) Loop avoidance: penalise recently visited cells
            if (nr, nc) in self.history:
                score -= 200

            # Resolve direction enum for this neighbour
            dr, dc = nr - my_pos[0], nc - my_pos[1]
            move_enum = Move.STAY
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if move.value == (dr, dc):
                    move_enum = move
                    break
            moves_score.append((score, move_enum))

        if not moves_score:
            return Move.STAY
        moves_score.sort(key=lambda x: x[0], reverse=True)
        return moves_score[0][1]

    def _is_timeout(self, deadline: float) -> bool:
        """Return True when the current decision already reached time budget."""
        return deadline is not None and time.perf_counter() >= deadline

    def _timeout_fallback_move(
        self,
        my_position: tuple,
        enemy_position: tuple,
        map_state: np.ndarray,
    ) -> Move:
        """
        Immediate fallback used when think-time budget is exhausted.
        Chooses a valid neighbour that increases Manhattan distance to enemy.
        """
        valid_neighbors = self._get_valid_neighbors(my_position, map_state)
        if not valid_neighbors:
            return Move.STAY

        best_next_pos = max(
            valid_neighbors,
            key=lambda pos: abs(pos[0] - enemy_position[0]) + abs(pos[1] - enemy_position[1]),
        )
        dr, dc = best_next_pos[0] - my_position[0], best_next_pos[1] - my_position[1]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if move.value == (dr, dc):
                return move
        return Move.STAY
