"""MCTS simulation and rollout logic for POMCP."""

from __future__ import annotations

import math
import random
import time
from typing import Optional, Set

from environment import Move

from .belief import sample_ghost_move
from .navigation import apply_pacman_action, legal_actions, manhattan
from .structures import Action, Node, Position


class SimulationContext:
    def __init__(
        self,
        known_map,
        pacman_speed: int,
        target_grid,
        frontier_grid,
        history_penalty_fn,
        deadline: float,
        capture_reward: float,
        step_penalty: float,
        distance_weight: float,
        frontier_weight: float,
        ucb_exploration_c: float,
    ):
        self.known_map = known_map
        self.pacman_speed = pacman_speed
        self.target_grid = target_grid
        self.frontier_grid = frontier_grid
        self.history_penalty_fn = history_penalty_fn
        self.deadline = deadline
        self.capture_reward = capture_reward
        self.step_penalty = step_penalty
        self.distance_weight = distance_weight
        self.frontier_weight = frontier_weight
        self.ucb_exploration_c = ucb_exploration_c

    def simulate(self, pacman_pos: Position, ghost_pos: Position, node: Node, depth: int) -> float:
        if time.perf_counter() >= self.deadline:
            return -self.step_penalty
        if depth <= 0:
            return -self.step_penalty

        if not node.untried and not node.children:
            node.untried = legal_actions(self.known_map, pacman_pos, self.pacman_speed)

        if node.untried:
            action = node.untried.pop()
            next_pac = apply_pacman_action(self.known_map, pacman_pos, action)
            next_ghost = sample_ghost_move(self.known_map, ghost_pos, next_pac)
            if self.is_capture(pacman_pos, next_pac, ghost_pos, next_ghost):
                reward = self.capture_reward
            else:
                reward = self.rollout(next_pac, next_ghost, depth - 1)

            dist = self._distance_term(next_pac, next_ghost)
            reward += -self.step_penalty - self.distance_weight * dist
            reward -= self.history_penalty_fn(next_pac)
            child = Node(visits=1, value=reward)
            node.children[action] = child
            node.visits += 1
            node.value += reward
            return reward

        action = self.ucb_select(node)
        child = node.children[action]
        next_pac = apply_pacman_action(self.known_map, pacman_pos, action)
        next_ghost = sample_ghost_move(self.known_map, ghost_pos, next_pac)
        if self.is_capture(pacman_pos, next_pac, ghost_pos, next_ghost):
            reward = self.capture_reward
        else:
            reward = self.simulate(next_pac, next_ghost, child, depth - 1)

        dist = self._distance_term(next_pac, next_ghost)
        reward += -self.step_penalty - self.distance_weight * dist
        reward -= self.history_penalty_fn(next_pac)
        child.visits += 1
        child.value += reward
        node.visits += 1
        node.value += reward
        return reward

    def rollout(
        self,
        pacman_pos: Position,
        ghost_pos: Position,
        depth: int,
        visited: Optional[Set[Position]] = None,
    ) -> float:
        if time.perf_counter() >= self.deadline:
            return -self.step_penalty
        if depth <= 0:
            return -self.step_penalty

        if visited is None:
            visited = set()

        action = self.rollout_policy(pacman_pos, ghost_pos, visited)
        next_pac = apply_pacman_action(self.known_map, pacman_pos, action)
        next_ghost = sample_ghost_move(self.known_map, ghost_pos, next_pac)

        if self.is_capture(pacman_pos, next_pac, ghost_pos, next_ghost):
            return self.capture_reward

        distance = self._distance_term(next_pac, next_ghost)
        reward = -self.step_penalty - self.distance_weight * distance
        reward -= self.history_penalty_fn(next_pac) * 0.5

        if next_pac == pacman_pos:
            reward -= 100.0

        new_visited = visited.copy()
        new_visited.add(pacman_pos)

        return reward + self.rollout(next_pac, next_ghost, depth - 1, new_visited)

    def rollout_policy(
        self,
        pacman_pos: Position,
        ghost_pos: Position,
        visited: Set[Position],
    ) -> Action:
        actions = legal_actions(self.known_map, pacman_pos, self.pacman_speed)
        if not actions:
            return (Move.STAY, 1)

        best_actions = []
        best_score = math.inf

        for action in actions:
            next_pac = apply_pacman_action(self.known_map, pacman_pos, action)

            if self.target_grid is not None:
                score = float(self.target_grid[next_pac])
                if next_pac == pacman_pos:
                    score += 100.0
            elif self.frontier_grid is not None:
                score = self.frontier_weight * float(self.frontier_grid[next_pac])
                score += float(manhattan(next_pac, ghost_pos))
            else:
                score = float(manhattan(next_pac, ghost_pos))

            score += self.history_penalty_fn(next_pac)

            if next_pac in visited:
                score += 1000.0

            if score < best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions) if best_actions else random.choice(actions)

    def ucb_select(self, node: Node) -> Action:
        best_action = None
        best_value = -math.inf
        for action, child in node.children.items():
            if child.visits <= 0:
                return action
            exploit = child.value / float(child.visits)
            explore = self.ucb_exploration_c * math.sqrt(
                math.log(node.visits + 1) / float(child.visits)
            )
            value = exploit + explore
            if value > best_value:
                best_value = value
                best_action = action
        return best_action if best_action else next(iter(node.children))

    @staticmethod
    def is_capture(
        pac_start: Position,
        pac_end: Position,
        ghost_start: Position,
        ghost_end: Position,
    ) -> bool:
        if pac_end == ghost_end:
            return True

        pac_path = [pac_start]
        if pac_start != pac_end:
            dx = math.copysign(1, pac_end[0] - pac_start[0]) if pac_end[0] != pac_start[0] else 0
            dy = math.copysign(1, pac_end[1] - pac_start[1]) if pac_end[1] != pac_start[1] else 0
            curr = pac_start
            while curr != pac_end:
                curr = (curr[0] + int(dx), curr[1] + int(dy))
                pac_path.append(curr)

        ghost_path = [ghost_start, ghost_end]

        for p in pac_path:
            for g in ghost_path:
                if p == g:
                    return True
        return False

    def _distance_term(self, next_pac: Position, next_ghost: Position) -> float:
        if self.target_grid is not None:
            return float(self.target_grid[next_pac])
        if self.frontier_grid is not None:
            return self.frontier_weight * float(self.frontier_grid[next_pac]) + float(
                manhattan(next_pac, next_ghost)
            )
        return float(manhattan(next_pac, next_ghost))
