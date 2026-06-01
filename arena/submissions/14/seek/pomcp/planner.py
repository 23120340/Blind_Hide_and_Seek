"""POMCP planner orchestrator for fog-seek."""

from __future__ import annotations

import random
import sys
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple

import numpy as np
from environment import Move

try:
    import resource
except ImportError:
    resource = None

from .belief import update_particles
from .config import (
    CAPTURE_REWARD,
    DISTANCE_WEIGHT,
    DEBUG,
    FRONTIER_WEIGHT,
    HISTORY_LENGTH,
    MEM_BUDGET_MB,
    MCTS_SIMULATIONS,
    PARTICLE_COUNT,
    REJUVENATION_RATE,
    ROLLOUT_DEPTH,
    STEP_PENALTY,
    TIME_BUDGET,
    UCB_EXPLORATION_C,
)
from .navigation import (
    compute_bfs_grid,
    compute_smart_target_grid,
    get_quadrant,
    is_intersection,
    is_walkable,
    legal_actions,
    manhattan,
    update_known_map,
    visible_cells,
)
from .simulation import SimulationContext
from .structures import Action, Node, Position


class POMCPPlanner:
    def __init__(self, pacman_speed: int, ghost_start_pos: Position):
        self.pacman_speed = max(1, int(pacman_speed))
        self.ghost_start_pos = ghost_start_pos
        self.known_map = None
        self.possible_ghost_positions: set[Position] = {ghost_start_pos}
        self.particles: List[Position] = [ghost_start_pos for _ in range(PARTICLE_COUNT)]
        self.history: List[Position] = []

        # Region Focus & Fatigue
        self.focus_quadrant = -1
        self.quadrant_steps = 0
        self.ignored_quadrants = set()

        # Global Patrol State
        self.steps_since_seen = 0
        self.quadrant_last_visited = {0: 0, 1: 0, 2: 0, 3: 0}
        self.global_steps = 0
        self.global_visited = set()
        self.is_ghost_seen = False
        self._mem_budget_bytes = int(MEM_BUDGET_MB * 1024 * 1024)
        self._trace_started = False
        self._rss_baseline = None

    def plan(
        self,
        map_state: np.ndarray,
        pacman_pos: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> Action:
        deadline = time.perf_counter() + TIME_BUDGET
        self.known_map = update_known_map(self.known_map, map_state)

        self.particles, self.possible_ghost_positions = update_particles(
            particles=self.particles,
            possible_positions=self.possible_ghost_positions,
            known_map=self.known_map,
            map_state=map_state,
            pacman_pos=pacman_pos,
            enemy_position=enemy_position,
            particle_count=PARTICLE_COUNT,
            rejuvenation_rate=REJUVENATION_RATE,
            ghost_start_pos=self.ghost_start_pos,
        )

        if time.perf_counter() >= deadline:
            return self._fallback_action(pacman_pos)

        self._maybe_set_rss_baseline()
        if self._over_memory_budget():
            return self._fallback_action(pacman_pos)

        self.global_steps += 1
        self.global_visited.add(pacman_pos)

        visible = visible_cells(map_state)
        self.global_visited.update(visible)

        current_q = get_quadrant(self.known_map, pacman_pos)
        self.quadrant_last_visited[current_q] = self.global_steps

        target_grid = None
        frontier_grid = None

        self.is_ghost_seen = enemy_position is not None
        if self.is_ghost_seen:
            self.steps_since_seen = 0
            target_grid = compute_bfs_grid(self.known_map, [enemy_position], deadline=deadline)
            self.ignored_quadrants.clear()
            self.focus_quadrant = -1
            self.quadrant_steps = 0
        else:
            self.steps_since_seen += 1

            if self.steps_since_seen <= 40 and self.possible_ghost_positions:
                active_quads = {get_quadrant(self.known_map, p) for p in self.possible_ghost_positions}

                if not (active_quads - self.ignored_quadrants):
                    self.ignored_quadrants.clear()

                valid_quads = active_quads - self.ignored_quadrants

                if self.focus_quadrant not in valid_quads:
                    if valid_quads:
                        best_q = -1
                        best_dist = float("inf")
                        for p in self.possible_ghost_positions:
                            if get_quadrant(self.known_map, p) in valid_quads:
                                d = manhattan(pacman_pos, p)
                                if d < best_dist:
                                    best_dist = d
                                    best_q = get_quadrant(self.known_map, p)
                        self.focus_quadrant = best_q
                    else:
                        self.focus_quadrant = -1
                    self.quadrant_steps = 0

                active_seeds = [
                    p for p in self.possible_ghost_positions
                    if get_quadrant(self.known_map, p) == self.focus_quadrant
                ]
                if not active_seeds:
                    active_seeds = list(self.possible_ghost_positions)

                self.particles = [random.choice(active_seeds) for _ in range(PARTICLE_COUNT)]

                target_grid = compute_bfs_grid(self.known_map, active_seeds, deadline=deadline)
            else:
                patrol_targets = []
                if self.known_map is not None:
                    for r in range(self.known_map.shape[0]):
                        for c in range(self.known_map.shape[1]):
                            if is_walkable(self.known_map, (r, c)):
                                if (r, c) not in self.global_visited:
                                    if self.known_map.shape == (21, 21) and r in (7, 11):
                                        continue
                                    patrol_targets.append((r, c))

                if not patrol_targets:
                    self.global_visited.clear()
                    for r in range(self.known_map.shape[0]):
                        for c in range(self.known_map.shape[1]):
                            if is_walkable(self.known_map, (r, c)):
                                if self.known_map.shape == (21, 21) and r in (7, 11):
                                    continue
                                patrol_targets.append((r, c))

                target_grid = compute_smart_target_grid(
                    self.known_map,
                    patrol_targets,
                    deadline=deadline,
                )

        if time.perf_counter() >= deadline:
            return self._fallback_action(pacman_pos)

        if self.focus_quadrant != -1:
            if get_quadrant(self.known_map, pacman_pos) == self.focus_quadrant:
                self.quadrant_steps += 1
            if self.quadrant_steps > 15:
                self.ignored_quadrants.add(self.focus_quadrant)
                self.quadrant_last_visited[self.focus_quadrant] = self.global_steps
                self.quadrant_steps = 0
                self.focus_quadrant = -1

        self.history.append(pacman_pos)
        if len(self.history) > HISTORY_LENGTH:
            self.history.pop(0)

        root = Node()
        root.untried = legal_actions(self.known_map, pacman_pos, self.pacman_speed)

        unique_particles = len(set(self.particles))
        if unique_particles <= 3:
            current_rollout_depth = ROLLOUT_DEPTH + 4
        elif unique_particles <= 10:
            current_rollout_depth = ROLLOUT_DEPTH + 2
        elif unique_particles > 30:
            current_rollout_depth = max(2, ROLLOUT_DEPTH - 2)
        else:
            current_rollout_depth = ROLLOUT_DEPTH

        context = SimulationContext(
            known_map=self.known_map,
            pacman_speed=self.pacman_speed,
            target_grid=target_grid,
            frontier_grid=frontier_grid,
            history_penalty_fn=self._get_history_penalty,
            deadline=deadline,
            capture_reward=CAPTURE_REWARD,
            step_penalty=STEP_PENALTY,
            distance_weight=DISTANCE_WEIGHT,
            frontier_weight=FRONTIER_WEIGHT,
            ucb_exploration_c=UCB_EXPLORATION_C,
        )

        simulations = 0
        while time.perf_counter() < deadline and simulations < MCTS_SIMULATIONS:
            if self._over_memory_budget():
                break
            ghost_pos = self._sample_particle()
            context.simulate(pacman_pos, ghost_pos, root, depth=current_rollout_depth)
            simulations += 1

        if not root.children:
            return self._fallback_action(pacman_pos)

        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]

        move, dist = best_action
        if dist == 1 and self.pacman_speed == 2:
            step1_pos = (pacman_pos[0] + move.value[0], pacman_pos[1] + move.value[1])
            step2_pos = (pacman_pos[0] + move.value[0] * 2, pacman_pos[1] + move.value[1] * 2)
            if is_walkable(self.known_map, step2_pos) and not is_intersection(
                self.known_map, step1_pos, move
            ):
                best_action = (move, 2)

        if DEBUG:
            if enemy_position is None:
                if self.steps_since_seen <= 5:
                    print("\033[91m[POMCP DEBUG] HOT PURSUIT! Đang mất dấu cự ly gần, cấm đầu đuổi theo!\033[0m")
                elif self.steps_since_seen <= 40:
                    print(
                        "\033[93m[POMCP DEBUG] Đang bám theo "
                        f"{len(self.possible_ghost_positions)} hạt seed ẩn (Ghosts).\033[0m"
                    )
                else:
                    print("\033[91m[POMCP DEBUG] SEED OVERLOAD! Chuyển sang chế độ Patrol check phòng.\033[0m")

                if self.focus_quadrant != -1:
                    q_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                    print(
                        "\033[95m[POMCP DEBUG] Đang tập trung dọn sạch vùng: "
                        f"{q_names[self.focus_quadrant]}\033[0m"
                    )
                if self.ignored_quadrants:
                    print(
                        "\033[91m[POMCP DEBUG] Đã mệt, bỏ qua các vùng: "
                        f"{self.ignored_quadrants}\033[0m"
                    )
                if self.steps_since_seen <= 40 and self.possible_ghost_positions and target_grid is not None:
                    sorted_seeds = sorted(
                        list(self.possible_ghost_positions),
                        key=lambda p: manhattan(pacman_pos, p),
                    )
                    print(
                        "\033[93m[POMCP DEBUG] Các hạt seed gần nhất: "
                        f"{sorted_seeds[:3]}\033[0m"
                    )
            print(
                "\033[96m[POMCP DEBUG] Quyết định chốt: "
                f"{best_action[0].name} x{best_action[1]}\033[0m"
            )

        return best_action

    def _get_history_penalty(self, pos: Position) -> float:
        if self.steps_since_seen <= 5:
            return 0.0

        try:
            idx = self.history.index(pos)
            return (idx + 1) * 20.0
        except ValueError:
            return 0.0

    def _sample_particle(self) -> Position:
        if not self.particles:
            if self.possible_ghost_positions:
                return random.choice(list(self.possible_ghost_positions))
            return self.ghost_start_pos
        return self.particles[np.random.randint(0, len(self.particles))]

    def _fallback_action(self, pacman_pos: Position) -> Action:
        actions = legal_actions(self.known_map, pacman_pos, self.pacman_speed)
        return actions[0] if actions else (Move.STAY, 1)

    def _get_rss_bytes(self) -> Optional[int]:
        if resource is None:
            return self._get_tracemalloc_bytes()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024

    def _get_tracemalloc_bytes(self) -> Optional[int]:
        if not self._trace_started:
            tracemalloc.start()
            self._trace_started = True
        current, peak = tracemalloc.get_traced_memory()
        return max(current, peak)

    def _maybe_set_rss_baseline(self) -> None:
        if self._rss_baseline is not None:
            return
        rss = self._get_rss_bytes()
        if rss is not None:
            self._rss_baseline = rss

    def _over_memory_budget(self) -> bool:
        rss = self._get_rss_bytes()
        if rss is None:
            return False
        if self._rss_baseline is None:
            return False
        return (rss - self._rss_baseline) > self._mem_budget_bytes
