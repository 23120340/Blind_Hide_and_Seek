"""Navigation and map utilities for POMCP."""

from __future__ import annotations

from collections import deque
import time
from typing import List, Optional, Sequence, Set

import numpy as np
from environment import Move

from .structures import Action, Position

DIRS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def update_known_map(known_map: Optional[np.ndarray], map_state: np.ndarray) -> np.ndarray:
    if known_map is None:
        known_map = np.full(map_state.shape, -1, dtype=np.int8)
    visible = map_state != -1
    known_map[visible] = map_state[visible]
    return known_map


def visible_cells(map_state: np.ndarray) -> Set[Position]:
    visible = set()
    height, width = map_state.shape
    for row in range(height):
        for col in range(width):
            if map_state[row, col] != -1:
                visible.add((row, col))
    return visible


def possible_cells(known_map: np.ndarray) -> List[Position]:
    cells = []
    height, width = known_map.shape
    for row in range(height):
        for col in range(width):
            if known_map[row, col] != 1:
                cells.append((row, col))
    return cells


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_quadrant(known_map: Optional[np.ndarray], pos: Position) -> int:
    if known_map is None:
        return 0
    r, c = pos
    mid_r = known_map.shape[0] // 2
    mid_c = known_map.shape[1] // 2
    if r < mid_r:
        return 0 if c < mid_c else 1
    return 2 if c < mid_c else 3


def is_walkable(known_map: Optional[np.ndarray], pos: Position) -> bool:
    if known_map is None:
        return False
    row, col = pos
    height, width = known_map.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return known_map[row, col] != 1


def compute_frontier(known_map: np.ndarray) -> List[Position]:
    frontier = []
    height, width = known_map.shape
    for row in range(height):
        for col in range(width):
            if known_map[row, col] != 0:
                continue
            for move in DIRS:
                nr, nc = row + move.value[0], col + move.value[1]
                if 0 <= nr < height and 0 <= nc < width and known_map[nr, nc] == -1:
                    frontier.append((row, col))
                    break
    return frontier


def compute_bfs_grid(
    known_map: np.ndarray,
    sources: Sequence[Position],
    deadline: Optional[float] = None,
) -> np.ndarray:
    grid = np.full(known_map.shape, 999, dtype=np.float32)
    if not sources:
        return grid

    q = deque()
    for s in sources:
        grid[s] = 0.0
        q.append(s)

    height, width = known_map.shape
    while q:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        r, c = q.popleft()
        d = grid[r, c]
        for m in DIRS:
            nr, nc = r + m.value[0], c + m.value[1]
            if 0 <= nr < height and 0 <= nc < width and known_map[nr, nc] != 1:
                if grid[nr, nc] > d + 1.0:
                    grid[nr, nc] = d + 1.0
                    q.append((nr, nc))
    return grid


def compute_smart_target_grid(
    known_map: np.ndarray,
    targets: List[Position],
    deadline: Optional[float] = None,
) -> np.ndarray:
    bfs_grid = compute_bfs_grid(known_map, targets, deadline=deadline)

    if deadline is not None and time.perf_counter() >= deadline:
        return bfs_grid

    density_grid = np.zeros(known_map.shape, dtype=np.float32)
    for r, c in targets:
        for rr in range(known_map.shape[0]):
            if deadline is not None and time.perf_counter() >= deadline:
                return bfs_grid
            for cc in range(known_map.shape[1]):
                if known_map[rr, cc] != 1:
                    dist = abs(r - rr) + abs(c - cc)
                    density_grid[rr, cc] += 20.0 / (dist + 1)

    return bfs_grid * 2.0 - density_grid


def legal_actions(known_map: Optional[np.ndarray], pacman_pos: Position, pacman_speed: int) -> List[Action]:
    actions: List[Action] = []
    if known_map is None:
        return actions

    for move in DIRS:
        row, col = pacman_pos
        for steps in range(1, pacman_speed + 1):
            row += move.value[0]
            col += move.value[1]
            if not is_walkable(known_map, (row, col)):
                break
            actions.append((move, steps))
    return actions


def apply_pacman_action(known_map: Optional[np.ndarray], pacman_pos: Position, action: Action) -> Position:
    move, steps = action
    row, col = pacman_pos
    for _ in range(steps):
        next_pos = (row + move.value[0], col + move.value[1])
        if not is_walkable(known_map, next_pos):
            break
        row, col = next_pos
    return (row, col)


def is_intersection(known_map: Optional[np.ndarray], pos: Position, current_move: Move) -> bool:
    if current_move in (Move.UP, Move.DOWN):
        ortho = [Move.LEFT, Move.RIGHT]
    elif current_move in (Move.LEFT, Move.RIGHT):
        ortho = [Move.UP, Move.DOWN]
    else:
        return False

    for move in ortho:
        next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
        if is_walkable(known_map, next_pos):
            return True
    return False
