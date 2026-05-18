"""
Alpha-beta minimax agent for Blind Hide and Seek.

When the opponent is visible we run a depth-bounded alpha-beta search on
the simultaneous-move game tree (treated as alternating moves: Pacman then
Ghost). Search depth is small (2 plies of each agent) to stay within the
step time budget, but the evaluation function is rich:

    eval = capture_bonus           # +inf if pacman has captured ghost
         - escape_bonus            # -inf if ghost is "captured" by walls
         + maze_distance(P, G)     # core spacing
         + ghost_exit_count        # ghost likes branching
         + ghost_voronoi_area      # ghost-first reachable cells
         - pacman_voronoi_area_lost
         - corridor_trap_penalty

For Pacman we negate the evaluation; we always express scores from Ghost's
perspective, so Pacman is the minimiser. This gives consistent semantics
regardless of who is searching.

When the opponent is not visible, the agent falls back to:
- Pacman: frontier exploration biased toward last-known enemy.
- Ghost: greedy safe-cell selection (same evaluation, depth 1).
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
MOVES_STAY = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

INF = 10**12


class Core:
    def _init_state(self):
        self.known = None
        self.visits = {}
        self.last_enemy = None
        self.recent = deque(maxlen=10)

    def _observe(self, grid, pos, enemy):
        if self.known is None or self.known.shape != grid.shape:
            self.known = np.full(grid.shape, -1, dtype=int)
        visible = grid != -1
        self.known[visible] = grid[visible]
        self.known[pos] = 0
        if enemy is not None:
            self.last_enemy = enemy
            self.known[enemy] = 0
        self.visits[pos] = self.visits.get(pos, 0) + 1
        self.recent.append(pos)

    def _inside(self, pos, grid):
        return 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]

    def _open(self, pos, grid):
        return self._inside(pos, grid) and grid[pos] != 1

    def _move(self, pos, move):
        return pos[0] + move.value[0], pos[1] + move.value[1]

    def _neighbors(self, pos, grid, stay=False):
        out = []
        for move in (MOVES_STAY if stay else MOVES):
            nxt = self._move(pos, move)
            if move == Move.STAY or self._open(nxt, grid):
                out.append((nxt, move))
        return out

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_dist_map(self, start, grid, max_depth=60):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            d = dist[pos]
            if d >= max_depth:
                continue
            for nxt, _ in self._neighbors(pos, grid):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)
        return dist

    def _bfs_path(self, start, goal, grid, limit=100):
        if start == goal:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            if len(path) >= limit:
                continue
            for nxt, move in self._neighbors(pos, grid):
                if nxt in seen:
                    continue
                new_path = path + [move]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                queue.append((nxt, new_path))
        return []

    def _maze_distance(self, start, goal, grid, max_depth=40):
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nxt, _ in self._neighbors(pos, grid):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return max_depth + self._manhattan(start, goal)

    def _unknown_adjacent(self, pos):
        total = 0
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._inside(nxt, self.known) and self.known[nxt] == -1:
                total += 1
        return total

    def _pacman_candidates(self, pos, speed, grid):
        """Pacman action set: each direction with 1..speed strides, plus STAY."""
        actions = []
        for move in MOVES:
            cur = pos
            for stride in range(1, speed + 1):
                nxt = self._move(cur, move)
                if not self._open(nxt, grid):
                    break
                cur = nxt
                actions.append((move, stride, cur))
        actions.append((Move.STAY, 1, pos))
        return actions

    def _ghost_candidates(self, pos, grid):
        return [(move, nxt) for nxt, move in self._neighbors(pos, grid, stay=True)]

    def _voronoi_count(self, pacman_pos, ghost_pos, pacman_speed):
        pac_d = self._bfs_dist_map(pacman_pos, self.known, max_depth=45)
        gho_d = self._bfs_dist_map(ghost_pos, self.known, max_depth=45)
        ghost_terr = 0
        pac_terr = 0
        for cell, gc in gho_d.items():
            pc = pac_d.get(cell, 999)
            pt = (pc + pacman_speed - 1) // pacman_speed
            if gc < pt:
                ghost_terr += 1
            elif gc > pt:
                pac_terr += 1
        return ghost_terr, pac_terr

    def _evaluate(self, pacman, ghost, pacman_speed):
        """Ghost-perspective evaluation. Higher is better for ghost."""
        if self._manhattan(pacman, ghost) < 2:
            return -INF

        ghost_terr, pac_terr = self._voronoi_count(pacman, ghost, pacman_speed)
        maze = self._maze_distance(pacman, ghost, self.known, max_depth=40)
        exits = len(self._neighbors(ghost, self.known))

        score = 0.0
        score += min(maze, 35) * 12.0
        score += ghost_terr * 6.0
        score -= pac_terr * 0.3
        score += exits * 10.0

        if exits <= 1:
            score -= 200
        elif exits == 2 and self._corridor_without_junction(ghost):
            score -= 35

        return score

    def _corridor_without_junction(self, start):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            if dist > 4:
                continue
            if len(self._neighbors(pos, self.known)) >= 3:
                return False
            for nxt, _ in self._neighbors(pos, self.known):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return True


class MinimaxSearch(Core):
    """Provides alpha-beta minimax over the simultaneous move game."""

    def _alphabeta_pacman_root(self, pacman, ghost, depth, speed):
        """Pacman acts first. Returns (best action tuple, score from ghost view)."""
        best_action = (Move.STAY, 1)
        best_score = INF  # pacman minimises ghost-view score
        alpha, beta = -INF, INF
        for move, stride, pac_next in self._pacman_candidates(pacman, speed, self.known):
            if self._manhattan(pac_next, ghost) < 2:
                return (move, stride), -INF  # immediate capture, score = -INF for ghost
            score = self._minimax_ghost(pac_next, ghost, depth - 1, alpha, beta, speed)
            if score < best_score:
                best_score = score
                best_action = (move, stride)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return best_action, best_score

    def _alphabeta_ghost_root(self, pacman, ghost, depth, speed):
        """Ghost acts first. Returns (best move, score)."""
        best_move = Move.STAY
        best_score = -INF
        alpha, beta = -INF, INF
        for move, gho_next in self._ghost_candidates(ghost, self.known):
            if self._manhattan(pacman, gho_next) < 2:
                continue  # do not move into capture range
            score = self._minimax_pacman(pacman, gho_next, depth - 1, alpha, beta, speed)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_move, best_score

    def _minimax_pacman(self, pacman, ghost, depth, alpha, beta, speed):
        if depth <= 0:
            return self._evaluate(pacman, ghost, speed)
        if self._manhattan(pacman, ghost) < 2:
            return -INF
        best = INF
        for move, stride, pac_next in self._pacman_candidates(pacman, speed, self.known):
            if self._manhattan(pac_next, ghost) < 2:
                return -INF
            v = self._minimax_ghost(pac_next, ghost, depth - 1, alpha, beta, speed)
            if v < best:
                best = v
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def _minimax_ghost(self, pacman, ghost, depth, alpha, beta, speed):
        if depth <= 0:
            return self._evaluate(pacman, ghost, speed)
        if self._manhattan(pacman, ghost) < 2:
            return -INF
        best = -INF
        for move, gho_next in self._ghost_candidates(ghost, self.known):
            if self._manhattan(pacman, gho_next) < 2:
                continue
            v = self._minimax_pacman(pacman, gho_next, depth - 1, alpha, beta, speed)
            if v > best:
                best = v
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        if best == -INF:
            # ghost has no safe move - evaluate static
            return self._evaluate(pacman, ghost, speed)
        return best


class PacmanAgent(BasePacmanAgent, MinimaxSearch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self._init_state()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)

        if enemy_position is not None:
            depth = 3  # 3 plies total (P-G-P) - keeps within ~1s budget
            action, _ = self._alphabeta_pacman_root(my_position, enemy_position, depth, self.pacman_speed)
            return action

        # No enemy visible: head toward last known position or explore.
        target = self.last_enemy if self.last_enemy and self.last_enemy != my_position else self._frontier_target(my_position)
        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)
        return self._fallback(my_position, map_state)

    def _frontier_target(self, start):
        dist = self._bfs_dist_map(start, self.known, max_depth=45)
        best = None
        best_score = -10**9
        for pos, d in dist.items():
            if pos == start:
                continue
            score = self._unknown_adjacent(pos) * 12 - d * 0.9 - self.visits.get(pos, 0) * 4
            if score > best_score:
                best_score = score
                best = pos
        return best

    def _path_action(self, pos, path, grid):
        first = path[0]
        steps = 0
        cur = pos
        for move in path[: self.pacman_speed]:
            if move != first:
                break
            nxt = self._move(cur, move)
            if not self._open(nxt, grid):
                break
            cur = nxt
            steps += 1
        return first, max(1, steps)

    def _fallback(self, pos, grid):
        best_score = -10**9
        best_move = Move.STAY
        for nxt, move in self._neighbors(pos, grid):
            score = self._unknown_adjacent(nxt) * 5 - self.visits.get(nxt, 0) * 2
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, 1


class GhostAgent(BaseGhostAgent, MinimaxSearch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax Ghost"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self._init_state()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)
        threat = enemy_position or self.last_enemy

        if threat is None:
            return self._wander(my_position, map_state)

        depth = 3
        move, _ = self._alphabeta_ghost_root(threat, my_position, depth, self.pacman_speed)
        if move is None:
            return Move.STAY
        return move

    def _wander(self, pos, grid):
        best_score = -10**9
        best_move = Move.STAY
        for nxt, move in self._neighbors(pos, grid):
            exits = len(self._neighbors(nxt, self.known))
            score = exits * 8 + self._unknown_adjacent(nxt) * 3
            score -= self.visits.get(nxt, 0) * 2
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
