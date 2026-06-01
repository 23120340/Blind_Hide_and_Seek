"""Heuristic evaluation and action ordering for hide-and-seek search."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from environment import Move
# Generic helpers from utils.py
from utils import (
    PacmanAction,
    Position,
    apply_move_once,
    apply_pacman_action,
    manhattan,
)
# Optimized map data from infrastructure.py
from infrastructure import Infrastructure
from .state import SearchState

FeatureDict = Dict[str, float]


class HeuristicEvaluator:
    """
    Evaluator that computes features, normalization, and scores with caching.
    Delegates expensive graph operations to the Infrastructure provider.
    """

    def __init__(
        self,
        pacman_speed: int,
        capture_distance_threshold: int,
        w1: float,
        w2: float,
        w3: float,
        w_turn: Optional[float] = None,
        w_dead_end: Optional[float] = None,
        w_safe_space: Optional[float] = None,
        w_los: Optional[float] = None,
    ):
        self.pacman_speed = max(1, int(pacman_speed))
        self.capture_distance_threshold = max(1, int(capture_distance_threshold))
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)

        # Store explicit overrides for FOW updates
        self._explicit_w_safe_space = w_safe_space
        self._explicit_w_los = w_los

        self.w_turn = float(w_turn) if w_turn is not None else (0.6 * self.w1)
        self.w_dead_end = float(w_dead_end) if w_dead_end is not None else (0.5 * self.w3)

        self.is_fow = False
        self._update_weights()

        self._infra: Optional[Infrastructure] = None

        # Caches for expensive feature components
        self._pacman_reach_cache: Dict[Position, Set[Position]] = {}
        self._ghost_transition_cache: Dict[Position, List[Position]] = {}
        self._adjacent_cache: Dict[Position, Tuple[Position, ...]] = {}
        self._distance_to_branch: Dict[Position, int] = {}

    def _update_weights(self) -> None:
        if self.is_fow:
            # Under FOW, we raise default weights for safe space and line-of-sight
            self.w_safe_space = float(self._explicit_w_safe_space) if self._explicit_w_safe_space is not None else (1.6 * self.w2)
            self.w_los = float(self._explicit_w_los) if self._explicit_w_los is not None else (0.7 * self.w1)
        else:
            self.w_safe_space = float(self._explicit_w_safe_space) if self._explicit_w_safe_space is not None else (0.8 * self.w2)
            self.w_los = float(self._explicit_w_los) if self._explicit_w_los is not None else (0.35 * self.w1)

    def set_fow(self, is_fow: bool) -> None:
        """Enable or disable Fog of War mode, adjusting default weights accordingly."""
        self.is_fow = is_fow
        self._update_weights()

    def bind_infra(self, infra: Infrastructure) -> None:
        """Bind an Infrastructure provider and reset local caches if the map changed."""
        if self._infra is not infra:
            self._infra = infra
            self._pacman_reach_cache.clear()
            self._ghost_transition_cache.clear()
            self._adjacent_cache.clear()
            self._distance_to_branch.clear()
            self._build_topology_cache(infra)

    def _build_topology_cache(self, infra: Infrastructure) -> None:
        """Build branch-distance topology using the infrastructure's precomputed graph."""
        walkable_positions = infra.all_walkable_cells()
        if not walkable_positions:
            return

        for pos in walkable_positions:
            self._adjacent_cache[pos] = tuple(infra.neighbors(pos))

        branch_cells = [
            pos
            for pos in walkable_positions
            if len(self._adjacent_cache[pos]) >= 3
        ]

        if not branch_cells:
            branch_cells = [
                pos
                for pos in walkable_positions
                if len(self._adjacent_cache[pos]) >= 2
            ]

        if not branch_cells:
            self._distance_to_branch = {pos: 0 for pos in walkable_positions}
            return

        distance_to_branch: Dict[Position, int] = {pos: -1 for pos in walkable_positions}
        queue: deque = deque()

        for source in branch_cells:
            distance_to_branch[source] = 0
            queue.append(source)

        while queue:
            current = queue.popleft()
            current_dist = distance_to_branch[current]

            for nxt in self._adjacent_cache.get(current, ()):
                if distance_to_branch.get(nxt, -1) != -1:
                    continue
                distance_to_branch[nxt] = current_dist + 1
                queue.append(nxt)

        self._distance_to_branch = {
            pos: max(0, dist)
            for pos, dist in distance_to_branch.items()
            if dist >= 0
        }

    def _require_infra(self) -> Infrastructure:
        if self._infra is None:
            raise RuntimeError("HeuristicEvaluator requires bind_infra() before evaluation.")
        return self._infra

    def maze_distance(self, a: Position, b: Position) -> float:
        """Graph distance between two cells. Delegates to Infrastructure."""
        infra = self._require_infra()
        if a == b:
            return 0.0
        return float(infra.distance(a, b))

    def _adjacent_walkable_positions(self, pos: Position) -> Tuple[Position, ...]:
        cached = self._adjacent_cache.get(pos)
        if cached is not None:
            return cached

        infra = self._require_infra()
        cached = tuple(infra.neighbors(pos))
        self._adjacent_cache[pos] = cached
        return cached

    def _pacman_reachability(self, pacman_pos: Position) -> Set[Position]:
        """Set of unique positions Pacman can reach in one turn."""
        cached = self._pacman_reach_cache.get(pacman_pos)
        if cached is not None:
            return cached

        infra = self._require_infra()
        reach: Set[Position] = set()
        for action in infra.get_pacman_actions(pacman_pos, emit_all_steps=True):
            reach.add(infra.apply_pacman_action(pacman_pos, action))

        self._pacman_reach_cache[pacman_pos] = reach
        return reach

    def _ghost_transitions(self, ghost_pos: Position) -> List[Position]:
        """List of positions the Ghost can be in after its next move."""
        cached = self._ghost_transition_cache.get(ghost_pos)
        if cached is not None:
            return cached

        infra = self._require_infra()
        # Use high-performance precomputed transitions from infra
        transitions = infra.get_ghost_transitions(ghost_pos)
        next_positions = [pos for _, pos in transitions]

        self._ghost_transition_cache[ghost_pos] = next_positions
        return next_positions

    def _is_in_pacman_los(self, my_pos: Position, pacman_pos: Position) -> bool:
        infra = self._require_infra()
        full_map = infra._map
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = pacman_pos
            for _ in range(5):  # Check up to 5 cells in each direction (5 is the LOS range from the requirements)
                r += dr
                c += dc
                if not (0 <= r < full_map.shape[0] and 0 <= c < full_map.shape[1]):
                    break
                if full_map[r, c] == 1:
                    break
                if (r, c) == my_pos:
                    return True
        return False

    def _local_safe_space_ratio(
        self,
        pacman_pos: Position,
        ghost_pos: Position,
        max_ghost_steps: int = 3,
    ) -> float:
        visited: Set[Position] = {ghost_pos}
        queue = deque([(ghost_pos, 0)])

        safe_cells = 0
        total_cells = 0

        while queue:
            pos, ghost_steps = queue.popleft()
            total_cells += 1

            pacman_dist = self.maze_distance(pacman_pos, pos)
            pacman_turns = pacman_dist / float(self.pacman_speed)
            if pacman_turns > float(ghost_steps) + 0.5:
                safe_cells += 1

            if ghost_steps >= max_ghost_steps:
                continue

            for nxt in self._adjacent_walkable_positions(pos):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, ghost_steps + 1))

        if total_cells <= 0:
            return 0.0
        return float(safe_cells) / float(total_cells)

    def extract_features(self, pacman_pos: Position, ghost_pos: Position) -> FeatureDict:
        post_sync_dist = self.maze_distance(pacman_pos, ghost_pos)
        capture_dist = float(manhattan(pacman_pos, ghost_pos))
        pacman_turn_dist = post_sync_dist / float(self.pacman_speed)

        ghost_next_positions = self._ghost_transitions(ghost_pos)
        hider_mobility = float(len(ghost_next_positions))
        ghost_degree = float(max(0, len(ghost_next_positions) - 1))

        pacman_reach = self._pacman_reachability(pacman_pos)
        intercept_coverage = float(sum(1 for pos in ghost_next_positions if pos in pacman_reach))

        corridor_depth = float(self._distance_to_branch.get(ghost_pos, 0))
        dead_end_multiplier = max(0.0, (3.0 - min(3.0, ghost_degree)) / 3.0)
        dead_end_pressure = corridor_depth * dead_end_multiplier

        safe_space = self._local_safe_space_ratio(pacman_pos, ghost_pos)
        is_in_pacman_los = float(self._is_in_pacman_los(ghost_pos, pacman_pos))

        return {
            "post_sync_dist": post_sync_dist,
            "capture_dist": capture_dist,
            "pacman_turn_dist": pacman_turn_dist,
            "hider_mobility": hider_mobility,
            "ghost_degree": ghost_degree,
            "intercept_coverage": intercept_coverage,
            "dead_end_pressure": dead_end_pressure,
            "safe_space": safe_space,
            "is_in_pacman_los": is_in_pacman_los,
            "ghost_next_cells": float(max(1, len(ghost_next_positions))),
        }

    def normalize_features(self, features: FeatureDict) -> FeatureDict:
        # Get dimensions from bound infrastructure
        infra = self._require_infra()
        distance_scale = float(max(1, infra.height + infra.width))
        mobility_scale = 5.0  # (UP, DOWN, LEFT, RIGHT, STAY)
        # Normalize pacman_turn_dist (which is post_sync_dist / pacman_speed) by
        # distance_scale rather than distance_scale / pacman_speed, so that
        # turn_distance = post_sync_dist / (pacman_speed * distance_scale)
        # carries distinct information from distance = post_sync_dist / distance_scale.
        dead_end_scale = distance_scale

        return {
            "distance": features["post_sync_dist"] / distance_scale,
            "turn_distance": features["pacman_turn_dist"] / max(1.0, distance_scale),
            "mobility": features["hider_mobility"] / mobility_scale,
            "coverage": features["intercept_coverage"] / features["ghost_next_cells"],
            "dead_end_pressure": features["dead_end_pressure"] / max(1.0, dead_end_scale),
            "safe_space": min(1.0, max(0.0, features["safe_space"])),
            "los_pressure": min(1.0, max(0.0, features["is_in_pacman_los"])),
        }

    def score_features(self, features: FeatureDict) -> float:
        if features["capture_dist"] < float(self.capture_distance_threshold):
            return 1000.0 + features["intercept_coverage"]

        normalized = self.normalize_features(features)
        return (
            -(self.w1 * normalized["distance"])
            - (self.w_turn * normalized["turn_distance"])
            - (self.w2 * normalized["mobility"])
            + (self.w3 * normalized["coverage"])
            + (self.w_dead_end * normalized["dead_end_pressure"])
            - (self.w_safe_space * normalized["safe_space"])
            + (self.w_los * normalized["los_pressure"]) # Ghosts don't want to be in Pacman's LOS, so this is a penalty for the Ghost and a reward for Pacman
        )

    def evaluate_state(self, state: SearchState, map_state: Optional[np.ndarray] = None) -> float:
        del map_state
        features = self.extract_features(state.pacman_pos, state.ghost_pos)
        return self.score_features(features)

    def order_pacman_actions(
        self,
        actions: List[PacmanAction],
        state: SearchState,
        map_state: Optional[np.ndarray] = None,
    ) -> List[PacmanAction]:
        del map_state
        pacman_pos, ghost_pos = state.pacman_pos, state.ghost_pos
        infra = self._require_infra()

        def action_key(action: PacmanAction):
            next_pacman_pos = infra.apply_pacman_action(pacman_pos, action)
            features = self.extract_features(next_pacman_pos, ghost_pos)
            normalized = self.normalize_features(features)
            capture_now = int(features["capture_dist"] < float(self.capture_distance_threshold))
            return (
                -capture_now,
                normalized["distance"],
                normalized["turn_distance"],
                normalized["mobility"],
                normalized["safe_space"],
                -normalized["coverage"],
                -normalized["dead_end_pressure"],
                -normalized["los_pressure"],
                -action[1],
            )

        return sorted(actions, key=action_key)

    def order_ghost_moves_for_pacman(
        self,
        moves: List[Move],
        state: SearchState,
        map_state: Optional[np.ndarray] = None,
    ) -> List[Move]:
        del map_state
        pacman_pos, ghost_pos = state.pacman_pos, state.ghost_pos
        infra = self._require_infra()

        def move_key(move: Move):
            next_ghost_pos = infra.apply_ghost_action(ghost_pos, move)
            return self.evaluate_state(SearchState(pacman_pos, next_ghost_pos))

        return sorted(moves, key=move_key)

    def order_ghost_moves_for_ghost(
        self,
        moves: List[Move],
        state: SearchState,
        map_state: Optional[np.ndarray] = None,
    ) -> List[Move]:
        del map_state
        pacman_pos, ghost_pos = state.pacman_pos, state.ghost_pos
        infra = self._require_infra()

        def move_key(move: Move):
            next_ghost_pos = infra.apply_ghost_action(ghost_pos, move)
            features = self.extract_features(pacman_pos, next_ghost_pos)
            normalized = self.normalize_features(features)
            return (
                -normalized["distance"],
                -normalized["turn_distance"],
                -normalized["mobility"],
                -normalized["safe_space"],
                normalized["coverage"],
                normalized["dead_end_pressure"],
                normalized["los_pressure"],
            )

        return sorted(moves, key=move_key)

    def bind_map(self, map_state: np.ndarray) -> None:
        """
        Legacy compatibility method. If a raw map is bound without an infra,
        we create a transient one. Recommended: use bind_infra().
        """
        if self._infra is None:
            new_infra = Infrastructure(map_state, pacman_speed=self.pacman_speed)
            new_infra.preprocess()
            self.bind_infra(new_infra)
