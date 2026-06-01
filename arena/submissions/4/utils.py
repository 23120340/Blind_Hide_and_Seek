"""Shared movement and geometry utilities for student agents."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from environment import Move

Position = Tuple[int, int]
PacmanAction = Tuple[Move, int]

CARDINAL_MOVES: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def in_bounds(map_state: np.ndarray, pos: Position) -> bool:
    """True iff pos is within the map dimensions."""
    row, col = pos
    height, width = map_state.shape
    return 0 <= row < height and 0 <= col < width


def is_walkable(map_state: np.ndarray, pos: Position) -> bool:
    """True iff pos is in-bounds and not a wall (1)."""
    if not in_bounds(map_state, pos):
        return False
    row, col = pos
    # Walls are encoded as 1. 0 is walkable. -1 (fog) is treated as traversable.
    return map_state[row, col] != 1


def apply_move_once(map_state: np.ndarray, pos: Position, move: Move) -> Position:
    """
    Apply a single step in 'move' direction. 
    Returns new position if walkable, else returns 'pos' unchanged.
    """
    if move == Move.STAY:
        return pos
    dr, dc = move.value
    candidate = (pos[0] + dr, pos[1] + dc)
    if is_walkable(map_state, candidate):
        return candidate
    return pos


def max_valid_steps(
    map_state: np.ndarray,
    pos: Position,
    move: Move,
    max_steps_allowed: int,
) -> int:
    """
    Count how many straight-line steps (up to max_steps_allowed) can be 
    taken from 'pos' in 'move' direction before hitting a wall.
    """
    if move == Move.STAY:
        return 0
    steps = 0
    current = pos
    dr, dc = move.value
    for _ in range(max_steps_allowed):
        nxt = (current[0] + dr, current[1] + dc)
        if not is_walkable(map_state, nxt):
            break
        current = nxt
        steps += 1
    return steps


def apply_pacman_action(
    map_state: np.ndarray,
    pos: Position,
    action: PacmanAction,
    pacman_speed: int,
) -> Position:
    """
    Apply a Pacman (Move, steps) action. 
    Steps are capped by pacman_speed and wall collisions.
    """
    move, steps = action
    if move == Move.STAY:
        return pos
    
    steps = max(1, min(int(steps), pacman_speed))
    dr, dc = move.value
    current = pos
    for _ in range(steps):
        nxt = (current[0] + dr, current[1] + dc)
        if not is_walkable(map_state, nxt):
            break
        current = nxt
    return current


def legal_pacman_actions(
    map_state: np.ndarray,
    pos: Position,
    pacman_speed: int,
) -> List[PacmanAction]:
    """Return all legal (Move, steps) actions for Pacman at 'pos'."""
    actions: List[PacmanAction] = [(Move.STAY, 1)]
    for move in CARDINAL_MOVES:
        max_s = max_valid_steps(map_state, pos, move, pacman_speed)
        for s in range(1, max_s + 1):
            actions.append((move, s))
    return actions


def legal_ghost_moves(map_state: np.ndarray, pos: Position) -> List[Move]:
    """Return all legal Move values for a Ghost at 'pos'."""
    moves: List[Move] = [Move.STAY]
    for move in CARDINAL_MOVES:
        if apply_move_once(map_state, pos, move) != pos:
            moves.append(move)
    return moves


def manhattan(a: Position, b: Position) -> int:
    """Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
