"""Particle filter utilities for POMCP."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .structures import Position

Rng = Union[np.random.RandomState, np.random.Generator]
from .navigation import DIRS, is_walkable, manhattan, possible_cells, visible_cells


def sample_ghost_move(
    known_map: np.ndarray,
    ghost_pos: Position,
    pacman_pos: Position,
    rng: Rng = np.random,
) -> Position:
    candidates = []
    for move in DIRS:
        next_pos = (ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1])
        if is_walkable(known_map, next_pos):
            candidates.append(next_pos)
    if not candidates:
        return ghost_pos

    if rng.random() < 0.8:
        best_cands = []
        max_dist = -1
        for c in candidates:
            d = manhattan(c, pacman_pos)
            if d > max_dist:
                max_dist = d
                best_cands = [c]
            elif d == max_dist:
                best_cands.append(c)
        return best_cands[int(rng.randint(0, len(best_cands)))]

    return candidates[int(rng.randint(0, len(candidates)))]


def update_possible_ghost_positions(
    possible_positions: Set[Position],
    known_map: np.ndarray,
    visible: Set[Position],
) -> Set[Position]:
    if not possible_positions:
        return possible_positions

    new_possible = set()
    for pos in possible_positions:
        new_possible.add(pos)
        for move in DIRS:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if is_walkable(known_map, next_pos):
                new_possible.add(next_pos)

    return {p for p in new_possible if p not in visible}


def resample(
    particles: List[Position],
    count: int,
    rng: Rng = np.random,
) -> List[Position]:
    num_particles = len(particles)
    if num_particles == 0:
        return []

    new_particles = []
    r = rng.uniform(0, 1.0 / count)
    for i in range(count):
        u = r + i / count
        idx = int(u * num_particles)
        new_particles.append(particles[idx])
    return new_particles


def rejuvenate(
    particles: List[Position],
    possible_positions: Set[Position],
    rate: float,
    rng: Rng = np.random,
) -> None:
    if rate <= 0.0:
        return
    count = int(len(particles) * rate)
    if count <= 0:
        return

    candidates = list(possible_positions)
    if not candidates:
        return

    for _ in range(count):
        idx = int(rng.randint(0, len(particles)))
        particles[idx] = candidates[int(rng.randint(0, len(candidates)))]


def update_particles(
    particles: List[Position],
    possible_positions: Set[Position],
    known_map: np.ndarray,
    map_state: np.ndarray,
    pacman_pos: Position,
    enemy_position: Optional[Position],
    particle_count: int,
    rejuvenation_rate: float,
    ghost_start_pos: Position,
    rng: Rng = np.random,
) -> Tuple[List[Position], Set[Position]]:
    visible = visible_cells(map_state)

    if enemy_position is not None:
        new_particles = [enemy_position for _ in range(particle_count)]
        return new_particles, {enemy_position}

    possible_positions = update_possible_ghost_positions(possible_positions, known_map, visible)
    if not possible_positions:
        possible_positions = {p for p in possible_cells(known_map) if p not in visible}

    new_particles: List[Position] = []
    for pos in particles:
        next_pos = sample_ghost_move(known_map, pos, pacman_pos, rng)
        if next_pos in visible:
            continue
        if next_pos not in possible_positions:
            continue
        new_particles.append(next_pos)

    if not new_particles:
        candidates = list(possible_positions)
        if not candidates:
            candidates = possible_cells(known_map)

        if candidates:
            new_particles = [
                candidates[int(rng.randint(0, len(candidates)))]
                for _ in range(particle_count)
            ]
        else:
            new_particles = [ghost_start_pos for _ in range(particle_count)]

    new_particles = resample(new_particles, particle_count, rng)
    rejuvenate(new_particles, possible_positions, rejuvenation_rate, rng)
    return new_particles, possible_positions
