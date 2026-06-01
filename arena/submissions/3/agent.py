"""
Lab 2 – Blind Adversary: Agent Implementation

Both agents operate under PARTIAL OBSERVABILITY:
  - Cross-shaped FOV, radius 5, blocked by walls
  - enemy_position is None when out of line-of-sight
  - map_state cells: 0=visible-empty, 1=wall, -1=unseen

PacmanAgent (Seek):
  A* chase when enemy visible + intelligent map-sweep exploration
  when enemy is hidden.  Builds a persistent memory map from every
  observation to enable pathfinding through previously-seen terrain.

GhostAgent (Hide):
  Risk-aware evasion with Pacman-position probability tracking.
  Maintains a belief distribution over Pacman's possible locations,
  updated each step via Bayesian filtering.  Uses minimax when Pacman
  is visible and probability-weighted flee when Pacman is hidden.

Allowed: builtins, numpy, pandas, scipy, gurobi.
Budget: <1 s per step, <16 MB.
"""

import sys
from pathlib import Path
from collections import deque
import heapq
import random

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
MOVE_DELTAS = {m: m.value for m in ALL_MOVES}
OPPOSITE = {
    Move.UP: Move.DOWN, Move.DOWN: Move.UP,
    Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT,
    Move.STAY: Move.STAY,
}
MAP_H, MAP_W = 21, 21

# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def is_valid(pos, known_map):
    """True if pos is in-bounds and known-empty (== 0)."""
    r, c = pos
    return 0 <= r < MAP_H and 0 <= c < MAP_W and known_map[r, c] == 0


def is_passable(pos, known_map):
    """True if pos is in-bounds and NOT a wall (could be unseen)."""
    r, c = pos
    return 0 <= r < MAP_H and 0 <= c < MAP_W and known_map[r, c] != 1


def apply_move(pos, move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos, known_map, allow_unseen=False):
    """Yield (next_pos, move) for valid neighbors.
    If allow_unseen, treat -1 cells as passable (optimistic)."""
    checker = is_passable if allow_unseen else is_valid
    for m in ALL_MOVES:
        nxt = apply_move(pos, m)
        if checker(nxt, known_map):
            yield nxt, m


def a_star(start, goal, known_map, allow_unseen=False):
    """A* returning list[Move]. Uses optimistic traversal if allow_unseen."""
    if start == goal:
        return []
    cnt = 0
    heap = [(manhattan(start, goal), cnt, start, [])]
    g = {start: 0}
    while heap:
        _, _, cur, path = heapq.heappop(heap)
        if cur == goal:
            return path
        cg = g[cur]
        for nxt, mv in get_neighbors(cur, known_map, allow_unseen):
            ng = cg + 1
            if ng < g.get(nxt, 999999):
                g[nxt] = ng
                cnt += 1
                heapq.heappush(heap, (ng + manhattan(nxt, goal), cnt, nxt, path + [mv]))
    return []


def pacman_next_positions(pac_pos, known_map):
    """All positions Pacman can reach in one turn (speed 1 or 2)."""
    positions = {pac_pos}
    for m in ALL_MOVES:
        p1 = apply_move(pac_pos, m)
        if is_valid(p1, known_map):
            positions.add(p1)
            p2 = apply_move(p1, m)
            if is_valid(p2, known_map):
                positions.add(p2)
    return positions


def pacman_turn_distances(origin, known_map):
    """BFS modelling Pacman's speed-2: 1 BFS level = 1 game turn."""
    dist = {origin: 0}
    queue = deque([origin])
    while queue:
        pos = queue.popleft()
        d = dist[pos]
        for m in ALL_MOVES:
            p1 = apply_move(pos, m)
            if is_valid(p1, known_map) and p1 not in dist:
                dist[p1] = d + 1
                queue.append(p1)
            if is_valid(p1, known_map):
                p2 = apply_move(p1, m)
                if is_valid(p2, known_map) and p2 not in dist:
                    dist[p2] = d + 1
                    queue.append(p2)
    return dist


# ──────────────────────────────────────────────────────────────────────
# Persistent memory map
# ──────────────────────────────────────────────────────────────────────

class MemoryMap:
    """
    Accumulates observations across steps to build a persistent map.
    Walls (1) are always visible.  Empty cells (0) are revealed when
    within the FOV.  Cells never observed remain -1 (unknown).
    """

    def __init__(self):
        # Start with walls known (they're always in map_state)
        # but empty cells unknown
        self.grid = np.full((MAP_H, MAP_W), -1, dtype=np.int8)
        self._initialized = False

    def update(self, map_state):
        """Merge a new partial observation into the persistent map."""
        if not self._initialized:
            # First observation: walls are always visible everywhere
            wall_mask = map_state == 1
            self.grid[wall_mask] = 1
            self._initialized = True
        # Reveal newly visible empty cells
        visible_empty = map_state == 0
        self.grid[visible_empty] = 0

    def get(self):
        """Return the current best-known map."""
        return self.grid


# ══════════════════════════════════════════════════════════════════════
#  PACMAN AGENT – Blind Seek
# ══════════════════════════════════════════════════════════════════════

class PacmanAgent(BasePacmanAgent):
    """
    Seek agent for partial-observability arena.

    Strategy
    --------
    When enemy IS visible:
      A* chase with speed-2 optimisation (same as Lab 1).

    When enemy is NOT visible:
      1. Memory map: accumulate all observations.
      2. If we have a last-known position, go there first.
      3. Otherwise, run a systematic sweep: BFS toward the nearest
         UNSEEN (-1) cell to maximise area coverage.  This is the
         fastest way to "scan" the maze and find the ghost.
      4. Frontier-based exploration: prefer unseen cells adjacent
         to known-empty cells (more likely to reveal corridors).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.memory = MemoryMap()
        self.last_known_enemy_pos = None
        self.last_known_step = 0
        self.exploration_target = None
        self.last_position = None
        self.stuck_count = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        # Update memory map
        self.memory.update(map_state)
        known = self.memory.get()

        # Detect being stuck
        if my_position == self.last_position:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_position = my_position

        # Enemy visible → chase
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_known_step = step_number
            self.exploration_target = None
            return self._chase(my_position, enemy_position, known)

        # Enemy not visible → search
        return self._search(my_position, known, step_number)

    # ── Chase (enemy visible) ─────────────────────────────────────────

    def _chase(self, my_pos, enemy_pos, known):
        path = a_star(my_pos, enemy_pos, known, allow_unseen=False)
        if not path:
            path = a_star(my_pos, enemy_pos, known, allow_unseen=True)
        if path:
            return self._speed_action(my_pos, path, known)
        return self._greedy_toward(my_pos, enemy_pos, known)

    # ── Search (enemy invisible) ──────────────────────────────────────

    def _search(self, my_pos, known, step_number):
        # 1. If we have a stale last-known position, go there first
        #    (but only if it's recent – within 15 steps)
        if (self.last_known_enemy_pos is not None
                and step_number - self.last_known_step <= 15):
            path = a_star(my_pos, self.last_known_enemy_pos, known, allow_unseen=True)
            if path:
                # Reached it or close → clear
                if manhattan(my_pos, self.last_known_enemy_pos) <= 1:
                    self.last_known_enemy_pos = None
                else:
                    return self._speed_action(my_pos, path, known)

        # 2. Frontier-based exploration: find nearest unseen cell
        #    adjacent to a known-empty cell (= exploration frontier)
        if (self.exploration_target is None
                or self.exploration_target == my_pos
                or self.stuck_count > 3):
            self.exploration_target = self._find_frontier_target(my_pos, known)
            self.stuck_count = 0

        if self.exploration_target:
            path = a_star(my_pos, self.exploration_target, known, allow_unseen=True)
            if path:
                return self._speed_action(my_pos, path, known)

        # 3. Fallback: random walk
        return self._random_move(my_pos, known)

    def _find_frontier_target(self, my_pos, known):
        """
        BFS from my_pos over known-empty cells.  Return the nearest
        known-empty cell that has at least one unseen (-1) neighbor.
        This cell is the 'frontier' – going there will reveal new area.
        """
        queue = deque([(my_pos, 0)])
        seen = {my_pos}
        best = None
        best_dist = float("inf")

        while queue:
            pos, d = queue.popleft()
            if d > best_dist:
                break
            # Check if this cell is adjacent to unseen territory
            has_unseen_neighbor = False
            for m in ALL_MOVES:
                nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
                if 0 <= nr < MAP_H and 0 <= nc < MAP_W and known[nr, nc] == -1:
                    has_unseen_neighbor = True
                    break
            if has_unseen_neighbor and d > 0:
                if d < best_dist:
                    best = pos
                    best_dist = d
                continue  # Don't need to search further from this branch

            for nxt, _ in get_neighbors(pos, known, allow_unseen=False):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, d + 1))

        if best is not None:
            return best

        # No frontier found – all map explored.  Pick a far-away cell
        # for random patrol.
        farthest = None
        fd = -1
        for pos in seen:
            d_val = manhattan(pos, my_pos)
            if d_val > fd:
                fd = d_val
                farthest = pos
        return farthest

    # ── Speed-2 action builder ────────────────────────────────────────

    def _speed_action(self, pos, path, known):
        move = path[0]
        steps = 1
        if self.pacman_speed >= 2 and len(path) >= 2 and path[1] == move:
            p1 = apply_move(pos, move)
            p2 = apply_move(p1, move)
            if is_valid(p1, known) and is_valid(p2, known):
                steps = 2
        return (move, steps)

    def _greedy_toward(self, pos, target, known):
        best_move, best_d = Move.STAY, manhattan(pos, target)
        for nxt, m in get_neighbors(pos, known, allow_unseen=True):
            d = manhattan(nxt, target)
            if d < best_d:
                best_d, best_move = d, m
        return (best_move, 1)

    def _random_move(self, pos, known):
        moves = list(ALL_MOVES)
        random.shuffle(moves)
        for m in moves:
            if is_valid(apply_move(pos, m), known):
                return (m, 1)
        return (Move.STAY, 1)


# ══════════════════════════════════════════════════════════════════════
#  GHOST AGENT – Blind Hide with Bayesian belief tracking
# ══════════════════════════════════════════════════════════════════════

class GhostAgent(BaseGhostAgent):
    """
    Hide agent for partial-observability arena.

    Key idea: maintain a BELIEF MAP – a probability distribution over
    where Pacman might be.  Each step:

      1. PREDICT: spread the belief according to Pacman's possible
         moves (speed-2 straight or speed-1 turn).
      2. UPDATE: if Pacman is visible, collapse belief to a point.
         If cells are visible and Pacman is NOT there, zero out
         those cells (negative evidence).
      3. ACT: use the belief centroid (expected Pacman position) as
         the threat direction for evasion.

    Evasion strategy mirrors Lab 1's risk-aware approach but uses the
    belief-estimated threat instead of true position.
    """

    PANIC_THRESHOLD = 6
    CAUTION_THRESHOLD = 12

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory = MemoryMap()
        self.last_move = None
        self._haven_target = None
        # Belief: probability that Pacman is at each cell
        # Initialised uniformly over non-wall cells on first step
        self.belief = None
        self._belief_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 999

    # ── main entry ────────────────────────────────────────────────────

    def step(self, map_state, my_position, enemy_position, step_number):
        self.memory.update(map_state)
        known = self.memory.get()

        # Initialise belief on first step
        if not self._belief_initialized:
            self._init_belief(known)
            self._belief_initialized = True

        # ── Belief update ─────────────────────────────────────────────
        # 1. Predict: spread belief via Pacman's movement model
        if step_number > 1:
            self._predict_belief(known)

        # 2. Update with observation
        if enemy_position is not None:
            # Positive sighting: collapse belief
            self.belief[:] = 0.0
            self.belief[enemy_position[0], enemy_position[1]] = 1.0
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
            # Negative evidence: zero out all visible empty cells
            visible_mask = (map_state == 0)
            self.belief[visible_mask] = 0.0
            # Also zero out our own position
            self.belief[my_position[0], my_position[1]] = 0.0

        # Normalise belief
        total = self.belief.sum()
        if total > 0:
            self.belief /= total
        else:
            # Lost track entirely – reinit uniform
            self._init_belief(known)

        # ── Decide action ─────────────────────────────────────────────
        if enemy_position is not None:
            # Pacman visible → use precise evasion
            threat = enemy_position
        else:
            # Use belief centroid as estimated threat
            threat = self._belief_centroid()

        if threat is None:
            return self._wander(my_position, known)

        # Compute Pacman-turn distances on the KNOWN map from threat
        ptd = pacman_turn_distances(threat, known)
        my_td = ptd.get(my_position, 999)

        # Invalidate stale haven
        if self._haven_target and ptd.get(self._haven_target, 999) < my_td:
            self._haven_target = None

        if my_td <= self.PANIC_THRESHOLD:
            move = self._panic(my_position, threat, known, ptd)
        elif my_td <= self.CAUTION_THRESHOLD:
            move = self._caution(my_position, threat, known, ptd)
        else:
            move = self._calm(my_position, threat, known, ptd)

        self.last_move = move
        return move

    # ── Belief management ─────────────────────────────────────────────

    def _init_belief(self, known):
        """Uniform belief over all non-wall cells."""
        self.belief = np.zeros((MAP_H, MAP_W), dtype=np.float32)
        passable = (known != 1)  # walls always known
        self.belief[passable] = 1.0
        total = self.belief.sum()
        if total > 0:
            self.belief /= total

    def _predict_belief(self, known):
        """Spread belief: each cell's probability is distributed among
        the cell itself (STAY) and its Pacman-reachable neighbors
        (speed 1 or 2 in a straight line)."""
        new_belief = np.zeros_like(self.belief)
        # For efficiency, iterate only cells with non-zero belief
        nonzero = np.argwhere(self.belief > 1e-9)
        for idx in nonzero:
            r, c = int(idx[0]), int(idx[1])
            prob = self.belief[r, c]
            reachable = pacman_next_positions((r, c), known)
            share = prob / len(reachable)
            for (nr, nc) in reachable:
                new_belief[nr, nc] += share
        self.belief = new_belief

    def _belief_centroid(self):
        """Return the expected (weighted mean) position of Pacman."""
        total = self.belief.sum()
        if total < 1e-12:
            return self.last_known_enemy_pos
        rows, cols = np.indices((MAP_H, MAP_W))
        er = int(round((rows * self.belief).sum() / total))
        ec = int(round((cols * self.belief).sum() / total))
        er = max(0, min(MAP_H - 1, er))
        ec = max(0, min(MAP_W - 1, ec))
        return (er, ec)

    # ── PANIC: 2-ply minimax ──────────────────────────────────────────

    def _panic(self, my_pos, pac_pos, known, ptd):
        pac_nexts = pacman_next_positions(pac_pos, known)

        g_candidates = [(my_pos, Move.STAY)]
        for nxt, m in get_neighbors(my_pos, known):
            g_candidates.append((nxt, m))

        best_move = Move.STAY
        best_score = -float("inf")

        for g1_pos, g1_move in g_candidates:
            min_over_pac = float("inf")
            for p1_pos in pac_nexts:
                if manhattan(p1_pos, g1_pos) < 2:
                    min_over_pac = -100
                    break
                g2_cands = [g1_pos]
                for g2nxt, _ in get_neighbors(g1_pos, known):
                    g2_cands.append(g2nxt)
                p2_nexts = pacman_next_positions(p1_pos, known)
                best_g2 = -float("inf")
                for g2_pos in g2_cands:
                    worst_p2 = min(manhattan(p2, g2_pos) for p2 in p2_nexts)
                    if worst_p2 > best_g2:
                        best_g2 = worst_p2
                if best_g2 < min_over_pac:
                    min_over_pac = best_g2

            escapes = sum(1 for _ in get_neighbors(g1_pos, known))
            escape_bonus = escapes * 0.15
            osc_pen = -0.3 if (self.last_move and g1_move == OPPOSITE.get(self.last_move)) else 0
            dead_pen = -2.0 if escapes <= 1 else 0
            ptd_bonus = ptd.get(g1_pos, 0) * 0.2

            score = min_over_pac + escape_bonus + osc_pen + dead_pen + ptd_bonus
            if score > best_score:
                best_score = score
                best_move = g1_move

        return best_move

    # ── CAUTION: weighted multi-factor ────────────────────────────────

    def _caution(self, my_pos, pac_pos, known, ptd):
        best_move = Move.STAY
        best_score = -float("inf")

        candidates = [(my_pos, Move.STAY)]
        for nxt, m in get_neighbors(my_pos, known):
            candidates.append((nxt, m))

        for pos, move in candidates:
            td = ptd.get(pos, 0)
            escapes = sum(1 for _ in get_neighbors(pos, known))
            osc = -1.5 if (self.last_move and move == OPPOSITE.get(self.last_move)) else 0
            dead = -5 if escapes <= 1 else 0
            stay_pen = -0.5 if move == Move.STAY else 0
            score = td * 2.5 + escapes * 1.2 + osc + dead + stay_pen
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # ── CALM: head toward safest haven ────────────────────────────────

    def _calm(self, my_pos, pac_pos, known, ptd):
        if self._haven_target is None or self._haven_target == my_pos:
            self._haven_target = self._find_haven(my_pos, known, ptd)
        if self._haven_target and self._haven_target != my_pos:
            path = a_star(my_pos, self._haven_target, known)
            if path:
                return path[0]
        return self._caution(my_pos, pac_pos, known, ptd)

    def _find_haven(self, my_pos, known, ptd):
        best, best_td = None, -1
        queue = deque([(my_pos, 0)])
        seen = {my_pos}
        while queue:
            pos, d = queue.popleft()
            exits = sum(1 for _ in get_neighbors(pos, known))
            td = ptd.get(pos, 0)
            if exits >= 2 and td > best_td:
                best, best_td = pos, td
            for nxt, _ in get_neighbors(pos, known):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, d + 1))
        return best

    # ── Wander (no threat estimate) ───────────────────────────────────

    def _wander(self, pos, known):
        moves = list(ALL_MOVES)
        random.shuffle(moves)
        for m in moves:
            if is_valid(apply_move(pos, m), known):
                return m
        return Move.STAY
