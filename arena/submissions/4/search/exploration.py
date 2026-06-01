"""
search/exploration.py
Systematic fog-sweep and BFS-guided exploration for PacmanAgent.

Algorithm overview
------------------
When the Ghost's position is unknown (no active target), Pacman runs a
three-tier sweep:

  Tier 1 — Zone of Control sweep
      Target the nearest walkable unexplored cell that lies within the
      Ghost's physically reachable zone (graph distance ≤ steps_since_sighting
      from last_known_enemy_pos). This prioritises the area the Ghost can
      actually be in, ignoring the far side of the map.

  Tier 2 — Full-map sweep
      If tier 1 yields nothing (zone is fully explored or no last sighting),
      target the nearest walkable unexplored cell on the whole map.

  Tier 3 — Reset sweep
      If the entire map is explored, clear explored_cells and re-seed it
      with the currently visible cells, then fall back to a center-bias
      move for this one step. This handles long games where Pacman has
      already swept everything once.

In all tiers, the "best action" is the legal Pacman action (Move, steps)
that moves closest (by graph distance) to the chosen target cell. Using
infra.distance() makes this O(1) per candidate, so the total cost per
step is O(|actions| × 1) = O(5) — negligible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import heapq

import numpy as np

from environment import Move
from infrastructure import Infrastructure, INF_DIST
from utils import Position, PacmanAction, apply_pacman_action, manhattan
from collections import defaultdict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_explored(
    map_state: np.ndarray,
    explored_cells: Dict[Position, int],
    step_number: int,
) -> None:
    """
    Mark every cell that is currently visible (not -1) as explored.
    Call this at the start of every step, before choosing an action.

    Parameters
    ----------
    map_state      : raw FOW map (0=path, 1=wall, -1=hidden).
    explored_cells : map of cell -> last-seen step (mutated in place).
    step_number    : current environment step number.
    """
    rows, cols = map_state.shape
    for r in range(rows):
        for c in range(cols):
            if map_state[r, c] != -1:
                explored_cells[(r, c)] = step_number



def exploration_action(
    actions: List[PacmanAction],
    map_state: np.ndarray,
    my_position: Position,
    explored_cells: Dict[Position, int],
    infra: Infrastructure,
    pacman_speed: int,
    last_known_enemy_pos: Optional[Position],
    steps_since_sighting: int,
    step_number: int,
    decay_steps: int,
) -> PacmanAction:
    """
    Choose the best exploration action for Pacman when the Ghost's
    exact position is unknown.

    Parameters
    ----------
    actions               : list of legal (Move, steps) for Pacman this step.
    map_state             : raw FOW map for this step.
    my_position           : Pacman's current (row, col).
    explored_cells        : map of cells -> last-seen step (mutated in place
                            by update_explored each step).
    infra                 : the fully-built Infrastructure object.
    pacman_speed          : maximum straight-line steps per turn.
    last_known_enemy_pos  : last known Ghost position, or None.
    steps_since_sighting  : how many steps have elapsed since last sighting.
    decay_steps           : number of steps before a seen cell is treated
                            as stale for exploration targeting.

    Returns
    -------
    The best (Move, steps) action, guaranteed to be in `actions`.
    """
    # Defensive guard: infra must be ready for any calls to its API
    # (all_walkable_cells, distance, etc.). When infra is missing or
    # not yet preprocessed, fall back to a simple safe action instead
    # of dereferencing None and crashing the agent process.
    if infra is None or not getattr(infra, "is_preprocessed", False):
        # Infra not ready: prefer the first legal non-STAY action.
        for action in actions:
            if action[0] != Move.STAY:
                return action
        return actions[0] if actions else (Move.STAY, 1)
    # ── Tier 1: Zone of Control sweep ──────────────────────────────────────
    # The Ghost moves at speed 1, so after S unsighted steps it can be at
    # most graph distance S from its last known position.
    target = _find_target_in_zone(
        my_position=my_position,
        explored_cells=explored_cells,
        infra=infra,
        last_known_enemy_pos=last_known_enemy_pos,
        steps_since_sighting=steps_since_sighting,
        step_number=step_number,
        decay_steps=decay_steps,
        prefer_never_seen=True,
    )
    if target is None:
        target = _find_target_in_zone(
            my_position=my_position,
            explored_cells=explored_cells,
            infra=infra,
            last_known_enemy_pos=last_known_enemy_pos,
            steps_since_sighting=steps_since_sighting,
            step_number=step_number,
            decay_steps=decay_steps,
            prefer_never_seen=False,
        )

    # ── Tier 2: Full-map sweep ──────────────────────────────────────────────
    # Build a single sweep-heap for this step when there's no recent
    # enemy sighting. This avoids repeated linear scans when selecting
    # the nearest unexplored cell (amortised O(n log n) vs O(n^2)).
    sweep_heap = None
    if target is None:
        if last_known_enemy_pos is None:
            sweep_heap = _build_sweep_queue(my_position, infra)
        target = _find_nearest_unexplored(
            my_position=my_position,
            explored_cells=explored_cells,
            infra=infra,
            step_number=step_number,
            decay_steps=decay_steps,
            prefer_never_seen=True,
            sweep_heap=sweep_heap,
        )
        if target is None:
            target = _find_nearest_unexplored(
                my_position=my_position,
                explored_cells=explored_cells,
                infra=infra,
                step_number=step_number,
                decay_steps=decay_steps,
                prefer_never_seen=False,
                sweep_heap=sweep_heap,
            )

    # ── Tier 3: Reset ──────────────────────────────────────────────────────
    if target is None:
        _reset_explored(map_state, explored_cells, step_number, decay_steps)
        return _center_bias_action(actions, map_state, my_position, pacman_speed, infra)

    # ── Convert target to action ─────────────────────────────────────────
    return _action_toward(actions, my_position, target, infra, map_state, pacman_speed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_target_in_zone(
    my_position: Position,
    explored_cells: Dict[Position, int],
    infra: Infrastructure,
    last_known_enemy_pos: Optional[Position],
    steps_since_sighting: int,
    step_number: int,
    decay_steps: int,
    prefer_never_seen: bool,
) -> Optional[Position]:
    """
    Return the nearest unexplored walkable cell within the Ghost's ZoC,
    or None if the zone is already fully explored (or no sighting exists).

    Zone of Control: cells whose graph distance from last_known_enemy_pos
    is ≤ steps_since_sighting (the Ghost cannot be further than this).
    """
    if last_known_enemy_pos is None:
        return None

    best_target: Optional[Position] = None
    best_dist = float("inf")

    for cell in infra.all_walkable_cells():
        if not _is_unexplored(cell, explored_cells, step_number, decay_steps, prefer_never_seen):
            continue
        # ZoC constraint: Ghost at speed-1 cannot have reached beyond this radius
        ghost_reach = infra.distance(last_known_enemy_pos, cell)
        if ghost_reach > steps_since_sighting or ghost_reach >= INF_DIST:
            continue
        d = infra.distance(my_position, cell)
        if d < INF_DIST and d < best_dist:
            best_dist = d
            best_target = cell

    return best_target


def _find_nearest_unexplored(
    my_position: Position,
    explored_cells: Dict[Position, int],
    infra: Infrastructure,
    step_number: int,
    decay_steps: int,
    prefer_never_seen: bool,
    sweep_heap: Optional[List[Tuple[int, Position]]] = None,
) -> Optional[Position]:
    """
    Return the nearest unexplored walkable cell on the whole map,
    or None if the entire map has been visited.
    """
    # If a pre-built heap is provided, pop until we find an unexplored
    # cell (skipping entries that have since become explored). This is
    # much more efficient when selecting multiple targets from the same
    # snapshot of the world during one step.
    if sweep_heap is not None:
        while sweep_heap:
            _, cell = heapq.heappop(sweep_heap)
            if not _is_unexplored(cell, explored_cells, step_number, decay_steps, prefer_never_seen):
                continue
            return cell
        return None

    # Fallback: linear scan (original behaviour)
    best_target: Optional[Position] = None
    best_dist = float("inf")

    for cell in infra.all_walkable_cells():
        if not _is_unexplored(cell, explored_cells, step_number, decay_steps, prefer_never_seen):
            continue
        d = infra.distance(my_position, cell)
        if d < INF_DIST and d < best_dist:
            best_dist = d
            best_target = cell

    return best_target


def _reset_explored(
    map_state: np.ndarray,
    explored_cells: Dict[Position, int],
    step_number: int,
    decay_steps: int,
) -> None:
    """
    Reset explored memory and re-seed currently visible cells.
    Called when the entire map has already been swept.
    """
    explored_cells.clear()
    rows, cols = map_state.shape
    for r in range(rows):
        for c in range(cols):
            if map_state[r, c] != -1:
                explored_cells[(r, c)] = step_number


def _action_toward(
    actions: List[PacmanAction],
    my_position: Position,
    target: Position,
    infra: Infrastructure,
    map_state: np.ndarray,
    pacman_speed: int,
) -> PacmanAction:
    """
    From the legal actions, pick the one whose resulting position is
    closest (graph distance) to target.
    Uses infra.distance() for O(1) lookups per action.
    """
    # Defensive guard: infra must be preprocessed for distance lookups.
    # If it's not ready, fall back to the first legal action (or STAY).
    if infra is None or not getattr(infra, "is_preprocessed", False):
        return actions[0] if actions else (Move.STAY, 1)

    best_action = actions[0]
    best_dist = float("inf")

    for action in actions:
        move, steps = action
        if move == Move.STAY:
            continue
        nxt = apply_pacman_action(map_state, my_position, action, pacman_speed)
        d = infra.distance(nxt, target)
        if d < best_dist:
            best_dist = d
            best_action = action

    return best_action


def _center_bias_action(
    actions: List[PacmanAction],
    map_state: np.ndarray,
    my_position: Position,
    pacman_speed: int,
    infra: Infrastructure,
) -> PacmanAction:
    """
    Fallback for when the entire map is explored and no target exists.
    Move toward the map center (high connectivity area) to maximise
    the chance of re-sighting the Ghost quickly after a reset.
    """
    height, width = map_state.shape
    center = (height // 2, width // 2)

    # Find the walkable cell closest to the center of the map
    walkable = infra.all_walkable_cells()
    if walkable:
        target_center = min(walkable, key=lambda cell: abs(cell[0] - center[0]) + abs(cell[1] - center[1]))
    else:
        target_center = center

    best_action = actions[0]
    best_score = float("inf")

    # Use the raw `map_state` via `apply_pacman_action` so predicted
    # landing positions match the game engine's computations.
    for action in actions:
        move, _ = action
        if move == Move.STAY:
            continue
        nxt = apply_pacman_action(map_state, my_position, action, pacman_speed)
        d = infra.distance(nxt, target_center)
        if d < best_score:
            best_score = d
            best_action = action

    return best_action


def _build_sweep_queue(
    my_position: Position,
    infra: Infrastructure,
) -> List[Tuple[int, Position]]:
    """
    Build a min-heap (list) of (dist_to_pacman, cell) for all walkable cells.

    This helper is an optional enhancement (STEP 7) to avoid scanning the full
    list each time. The returned list is a heap suitable for `heapq.heappop`.
    """
    heap: List[Tuple[int, Position]] = []
    for cell in infra.all_walkable_cells():
        d = infra.distance(my_position, cell)
        if d < INF_DIST:
            heapq.heappush(heap, (d, cell))
    return heap


def _is_unexplored(
    cell: Position,
    explored_cells: Dict[Position, int],
    step_number: int,
    decay_steps: int,
    prefer_never_seen: bool,
) -> bool:
    last_seen = explored_cells.get(cell)
    if last_seen is None:
        return True
    if prefer_never_seen:
        return False
    if decay_steps <= 0:
        return False
    return (step_number - last_seen) > decay_steps

# ---------------------------------------------------------------------------
# Ghost policy estimation (Expectiminimax support)
# ---------------------------------------------------------------------------

class GhostPolicyEstimator:
    """
    Builds an empirical probability distribution over the Ghost's moves
    for every position it has been observed at.

    How it works
    ------------
    Each time the Ghost is visible on two consecutive steps, we can infer
    which Move it took (by comparing its previous position to the current one).
    We count these observations per position using Laplace smoothing so that
    unseen moves always have a small nonzero probability.

    After enough sightings the distribution sharpens from uniform toward the
    Ghost's real policy, letting Expectiminimax intercept probable moves
    instead of worst-case ones.

    Memory: 216 cells × 5 moves × ~70 B per entry ≈ 75 KB. Negligible.
    """

    def __init__(self, uniform_weight: float = 2.0):
        # uniform_weight is a pseudo-count added to every legal move.
        # At 2.0, you need ~10 real observations before the empirical
        # signal dominates the uniform prior. Lower = sharper faster,
        # but noisier on small samples.
        self._counts: dict = defaultdict(lambda: defaultdict(float))
        self._uniform_weight = float(uniform_weight)

    def observe(self, position: Position, move: "Move") -> None:
        """
        Record that the Ghost was at `position` and chose `move`.
        Call this every step where the Ghost was visible on the previous
        step AND the current step.
        """
        if position is not None and move is not None:
            self._counts[position][move] += 1.0

    def move_probs(self, position: Position, legal_moves: list) -> dict:
        """
        Return a probability distribution (dict mapping Move → float)
        over `legal_moves` for the Ghost at `position`.

        Uses Laplace smoothing: every move gets uniform_weight pseudo-counts
        on top of the real observed counts, so P(move) is never 0.
        """
        counts = self._counts.get(position, {})
        total = sum(
            counts.get(m, 0.0) + self._uniform_weight
            for m in legal_moves
        )
        if total == 0.0:
            n = max(len(legal_moves), 1)
            return {m: 1.0 / n for m in legal_moves}
        return {
            m: (counts.get(m, 0.0) + self._uniform_weight) / total
            for m in legal_moves
        }

def intercept_expectiminimax_action(
    actions: List[PacmanAction],
    my_position: Position,
    ghost_position: Position,
    infra: Infrastructure,
    estimator: GhostPolicyEstimator,
    depth: int = 3,
) -> PacmanAction:
    """
    Multi-ply Expectiminimax fallback for when the main adversarial search
    times out or returns no result.

    Tree structure
    --------------
    - Pacman nodes  → MAX  (pick the action with highest expected value)
    - Ghost nodes   → CHANCE (weighted average over the Ghost's move
                      distribution from `estimator`)

    Evaluation leaf: negative graph distance between Pacman and Ghost.
    Closer = better for Pacman, so higher score = closer.

    Why depth=3
    -------------------------------------
    Branching: Pacman ~4 moves × Ghost ~4 moves = 16 per ply.
    At depth 3: 4 × (4 × (4 × 4)) = 1 024 leaf evaluations.
    Each evaluation = 1 O(1) infra.distance() call.
    Total: ~1 024 dict lookups ≈ 2 ms.

    Parameters
    ----------
    actions        : legal (Move, steps) list for Pacman this step.
    my_position    : Pacman's current (row, col).
    ghost_position : Ghost's current (row, col) — must be visible.
    infra          : preprocessed Infrastructure for O(1) distances.
    estimator      : GhostPolicyEstimator holding observed move counts.
    depth          : search depth (3 = safe default, 4 = slightly better).
    """

    def _pacman_node(p_pos: Position, g_pos: Position, d: int) -> float:
        """Pacman's turn: pick the max over all Pacman actions."""
        if d == 0:
            # Leaf: higher score = Ghost is closer
            return -float(infra.distance(p_pos, g_pos))

        best = -float("inf")
        for action in infra.get_pacman_actions(p_pos, emit_all_steps=False):
            nxt_p = infra.apply_pacman_action(p_pos, action)
            val = _ghost_node(nxt_p, g_pos, d - 1)
            if val > best:
                best = val
        return best

    def _ghost_node(p_pos: Position, g_pos: Position, d: int) -> float:
        """Ghost's turn: chance node — weighted average over move distribution."""
        legal = infra.get_ghost_actions(g_pos, include_stay=True)
        probs = estimator.move_probs(g_pos, legal)
        expected = 0.0
        for move in legal:
            nxt_g = infra.apply_ghost_action(g_pos, move)
            val = _pacman_node(p_pos, nxt_g, d)
            expected += probs[move] * val
        return expected

    # Root: Pacman picks the action with the best expected outcome
    best_action = actions[0]
    best_val = -float("inf")

    for action in actions:
        move, _ = action
        if move == Move.STAY:
            continue
        nxt_p = infra.apply_pacman_action(my_position, action)
        # After Pacman moves, Ghost responds as a chance node
        val = _ghost_node(nxt_p, ghost_position, depth - 1)
        if val > best_val:
            best_val = val
            best_action = action

    return best_action
