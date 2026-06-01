"""Simultaneous maximin with iterative deepening and pluggable evaluation."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np

from environment import Move
# Generic helpers from utils.py
import utils
# Optimized map data from infrastructure.py
from .state import SearchState

logger = logging.getLogger(__name__)

# Entry types for transposition table.
ENTRY_EXACT = 0
ENTRY_LOWERBOUND = 1
ENTRY_UPPERBOUND = 2

PacmanAction = utils.PacmanAction
EvaluateFn = Callable[[SearchState, np.ndarray], float]
PacmanOrderFn = Callable[[List[PacmanAction], SearchState, np.ndarray], List[PacmanAction]]
GhostOrderFn = Callable[[List[Move], SearchState, np.ndarray], List[Move]]
TTEntry = Dict[str, Any]


class SearchDiagnostics(TypedDict):
    nodes: int
    p_cutoffs: int
    g_cutoffs: int
    tt_hits: int
    tt_cuts: int
    completed_depth: int
    depth_stats: Dict[int, float]


def _default_evaluate(
    state: SearchState,
    capture_distance_threshold: int,
) -> float:
    pr, pc = state.pacman_pos
    gr, gc = state.ghost_pos
    # Use Manhattan from utils
    dist = utils.manhattan(state.pacman_pos, state.ghost_pos)
    if dist < capture_distance_threshold:
        return 10000.0
    return -float(dist)


def _order_candidates(
    original: List[Any],
    proposed: Optional[List[Any]],
) -> List[Any]:
    if not proposed:
        return list(original)

    original_set = set(original)
    ordered: List[Any] = []
    seen: set = set()

    for item in proposed:
        if item in original_set and item not in seen:
            ordered.append(item)
            seen.add(item)

    for item in original:
        if item not in seen:
            ordered.append(item)

    return ordered


def _next_state(
    state: SearchState,
    map_array: np.ndarray,
    pacman_action: PacmanAction,
    ghost_action: Move,
    pacman_speed: int,
) -> SearchState:
    next_pacman_pos = utils.apply_pacman_action(
        map_array,
        state.pacman_pos,
        pacman_action,
        pacman_speed,
    )
    next_ghost_pos = utils.apply_move_once(map_array, state.ghost_pos, ghost_action)
    return SearchState(pacman_pos=next_pacman_pos, ghost_pos=next_ghost_pos)


def _maximin(
    state: SearchState,
    depth: int,
    alpha: float,
    beta: float,
    pacman_speed: int,
    map_array: np.ndarray,
    end_time: float,
    diagnostics: SearchDiagnostics,
    transposition_table: Dict[SearchState, TTEntry],
    capture_distance_threshold: int,
    evaluate_state_fn: EvaluateFn,
    order_pacman_actions_fn: Optional[PacmanOrderFn],
    order_ghost_actions_fn: Optional[GhostOrderFn],
) -> Tuple[float, Optional[PacmanAction], Optional[Move]]:
    if time.perf_counter() >= end_time:
        raise TimeoutError()

    entry = transposition_table.get(state)
    if entry and entry["depth"] >= depth:
        diagnostics["tt_hits"] += 1
        flag = entry["flag"]
        value = entry["value"]
        best_p_act = entry["best_p_act"]
        best_g_act = entry["best_g_act"]

        if flag == ENTRY_EXACT:
            diagnostics["tt_cuts"] += 1
            return value, best_p_act, best_g_act

        if flag == ENTRY_LOWERBOUND:
            alpha = max(alpha, value)
        elif flag == ENTRY_UPPERBOUND:
            beta = min(beta, value)

        if alpha >= beta:
            diagnostics["tt_cuts"] += 1
            return value, best_p_act, best_g_act

    alpha_orig = alpha
    beta_orig = beta

    if utils.manhattan(state.pacman_pos, state.ghost_pos) < capture_distance_threshold:
        return 10000.0 + depth, None, None

    if depth <= 0:
        return evaluate_state_fn(state, map_array), None, None

    # Use utils directly for legal actions
    pacman_actions = utils.legal_pacman_actions(map_array, state.pacman_pos, pacman_speed)
    ghost_actions = utils.legal_ghost_moves(map_array, state.ghost_pos)
    
    if not pacman_actions or not ghost_actions:
        return evaluate_state_fn(state, map_array), None, None

    diagnostics["nodes"] += 1

    if order_pacman_actions_fn is not None:
        sorted_p_actions = _order_candidates(
            pacman_actions,
            order_pacman_actions_fn(list(pacman_actions), state, map_array),
        )
    else:
        def optimistic_key(p_action: PacmanAction) -> float:
            optimistic = -float("inf")
            for g_action in ghost_actions:
                next_state = _next_state(
                    state,
                    map_array,
                    p_action,
                    g_action,
                    pacman_speed,
                )
                optimistic = max(optimistic, evaluate_state_fn(next_state, map_array))
            return optimistic

        sorted_p_actions = sorted(pacman_actions, key=optimistic_key, reverse=True)

    if entry and entry.get("best_p_act") in sorted_p_actions:
        favored = entry["best_p_act"]
        sorted_p_actions = [favored] + [act for act in sorted_p_actions if act != favored]

    best_value = -float("inf")
    best_p_act = sorted_p_actions[0]
    best_g_act = ghost_actions[0]

    for p_action in sorted_p_actions:
        projected_pacman = utils.apply_pacman_action(
            map_array,
            state.pacman_pos,
            p_action,
            pacman_speed,
        )

        if order_ghost_actions_fn is not None:
            ordering_state = SearchState(
                pacman_pos=projected_pacman,
                ghost_pos=state.ghost_pos,
            )
            sorted_g_actions = _order_candidates(
                ghost_actions,
                order_ghost_actions_fn(list(ghost_actions), ordering_state, map_array),
            )
        else:
            sorted_g_actions = sorted(
                ghost_actions,
                key=lambda g_action: evaluate_state_fn(
                    SearchState(
                        pacman_pos=projected_pacman,
                        ghost_pos=utils.apply_move_once(map_array, state.ghost_pos, g_action),
                    ),
                    map_array,
                ),
            )

        if entry and entry.get("best_g_act") in sorted_g_actions:
            favored_g = entry["best_g_act"]
            sorted_g_actions = [favored_g] + [g for g in sorted_g_actions if g != favored_g]

        min_value = float("inf")
        min_g_action = sorted_g_actions[0]
        current_beta = beta

        for g_action in sorted_g_actions:
            next_state = SearchState(
                pacman_pos=projected_pacman,
                ghost_pos=utils.apply_move_once(map_array, state.ghost_pos, g_action),
            )
            value, _, _ = _maximin(
                state=next_state,
                depth=depth - 1,
                alpha=alpha,
                beta=current_beta,
                pacman_speed=pacman_speed,
                map_array=map_array,
                end_time=end_time,
                diagnostics=diagnostics,
                transposition_table=transposition_table,
                capture_distance_threshold=capture_distance_threshold,
                evaluate_state_fn=evaluate_state_fn,
                order_pacman_actions_fn=order_pacman_actions_fn,
                order_ghost_actions_fn=order_ghost_actions_fn,
            )

            if value < min_value:
                min_value = value
                min_g_action = g_action

            current_beta = min(current_beta, min_value)
            if min_value <= alpha:
                diagnostics["g_cutoffs"] += 1
                break

        if min_value > best_value:
            best_value = min_value
            best_p_act = p_action
            best_g_act = min_g_action

        alpha = max(alpha, best_value)
        if alpha >= beta:
            diagnostics["p_cutoffs"] += 1
            break

    if best_value <= alpha_orig:
        flag = ENTRY_UPPERBOUND
    elif best_value >= beta_orig:
        flag = ENTRY_LOWERBOUND
    else:
        flag = ENTRY_EXACT

    transposition_table[state] = {
        "value": best_value,
        "depth": depth,
        "flag": flag,
        "best_p_act": best_p_act,
        "best_g_act": best_g_act,
    }

    return best_value, best_p_act, best_g_act


def simultaneous_maximin_search(
    state: SearchState,
    map_array: np.ndarray,
    pacman_speed: int = 2,
    time_limit: float = 1.0,
    capture_distance_threshold: int = 1,
    evaluate_state_fn: Optional[EvaluateFn] = None,
    order_pacman_actions_fn: Optional[PacmanOrderFn] = None,
    order_ghost_actions_fn: Optional[GhostOrderFn] = None,
    profiling_enabled: bool = False,
) -> Tuple[PacmanAction, Move, SearchDiagnostics]:
    """Run iterative deepening simultaneous maximin and return best joint action and diagnostics."""
    start_time = time.perf_counter()
    time_limit = max(0.05, float(time_limit))
    safety_margin = 0.05
    end_time = start_time + max(0.001, time_limit - safety_margin)

    if profiling_enabled:
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False

    diagnostics = {
        "nodes": 0,
        "p_cutoffs": 0,
        "g_cutoffs": 0,
        "tt_hits": 0,
        "tt_cuts": 0,
        "completed_depth": 0,
        "depth_stats": {},
    }
    transposition_table: Dict[SearchState, TTEntry] = {}
    capture_distance_threshold = max(1, int(capture_distance_threshold))

    if evaluate_state_fn is None:
        evaluate_state_fn = lambda node, _map: _default_evaluate(
            node,
            capture_distance_threshold,
        )

    pacman_actions = utils.legal_pacman_actions(map_array, state.pacman_pos, pacman_speed)
    ghost_actions = utils.legal_ghost_moves(map_array, state.ghost_pos)
    
    if not pacman_actions or not ghost_actions:
        return (Move.STAY, 1), Move.STAY, diagnostics

    best_p_act: PacmanAction = pacman_actions[0]
    best_g_act: Move = ghost_actions[0]

    depth = 1
    completed_depth = 0

    try:
        while depth <= 100 and time.perf_counter() < end_time:
            d_start = time.perf_counter()
            _, p_action, g_action = _maximin(
                state=state,
                depth=depth,
                alpha=-float("inf"),
                beta=float("inf"),
                pacman_speed=pacman_speed,
                map_array=map_array,
                end_time=end_time,
                diagnostics=diagnostics,
                transposition_table=transposition_table,
                capture_distance_threshold=capture_distance_threshold,
                evaluate_state_fn=evaluate_state_fn,
                order_pacman_actions_fn=order_pacman_actions_fn,
                order_ghost_actions_fn=order_ghost_actions_fn,
            )

            if p_action is not None:
                best_p_act = p_action
            if g_action is not None:
                best_g_act = g_action

            completed_depth = depth
            diagnostics["completed_depth"] = depth
            diagnostics["depth_stats"][depth] = time.perf_counter() - d_start
            depth += 1
    except TimeoutError:
        pass

    elapsed = time.perf_counter() - start_time
    _report_search_diagnostics(diagnostics, elapsed, completed_depth, profiling_enabled)

    return best_p_act, best_g_act, diagnostics


def _report_search_diagnostics(
    diagnostics: SearchDiagnostics,
    elapsed: float,
    completed_depth: int,
    profiling_enabled: bool,
) -> None:
    """Calculate and log search statistics and profiling data."""
    total_nodes = diagnostics["nodes"]
    
    if profiling_enabled:
        total_cuts = diagnostics["p_cutoffs"] + diagnostics["g_cutoffs"]
        tt_hits = diagnostics["tt_hits"]
        tt_cuts = diagnostics["tt_cuts"]

        # Calculate branching factor estimate (EBF)
        ebf = (total_nodes ** (1 / completed_depth)) if completed_depth > 0 else 0
        tt_efficiency = (tt_hits / max(1, total_nodes)) * 100
        cutoff_efficiency = (total_cuts / max(1, total_nodes)) * 100

        logger.info(
            "Search Profiling: final_depth=%d nodes=%d T=%0.3fs (%.0f n/s)",
            completed_depth, total_nodes, elapsed, total_nodes / max(0.001, elapsed)
        )
        logger.info(
            "Pruning: p_cuts=%d g_cuts=%d tt_hits=%d tt_cuts=%d",
            diagnostics["p_cutoffs"], diagnostics["g_cutoffs"], tt_hits, tt_cuts
        )
        logger.info(
            "Efficiency: EBF=%.2f TT_eff=%.1f%% Cut_eff=%.1f%%",
            ebf, tt_efficiency, cutoff_efficiency
        )
        for d, t in diagnostics["depth_stats"].items():
            logger.debug("  depth=%d time=%.4fs", d, t)
    else:
        logger.info(
            "Simultaneous maximin depth=%d nodes=%d T=%.3fs",
            completed_depth, total_nodes, elapsed
        )
