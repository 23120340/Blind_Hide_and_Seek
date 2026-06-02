"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


CARDINAL_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
SIGHT_RADIUS = 5
CAPTURE_DISTANCE = 2
INF = 10 ** 9
CACHE_LIMIT = 256
TURN_BUDGET_SECONDS = 0.88


def _add_pos(pos, move):
    dr, dc = move.value
    return pos[0] + dr, pos[1] + dc


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class _ArenaBrain:
    """Shared map memory, belief tracking, and graph helpers."""

    def _init_brain(self, enemy_speed=1):
        self.enemy_speed = max(1, int(enemy_speed))
        self.height = None
        self.width = None
        self.walls = set()
        self.open_cells = set()
        self.known_empty = set()
        self.enemy_belief = set()
        self.visible_now = set()
        self.visit_count = {}
        self.recent_positions = deque(maxlen=10)
        self.last_seen_enemy = None
        self.last_seen_step = -1
        self.turn_deadline = 0.0

    def _start_turn_timer(self):
        self.turn_deadline = time.perf_counter() + TURN_BUDGET_SECONDS

    def _time_low(self):
        return self.turn_deadline > 0 and time.perf_counter() >= self.turn_deadline

    def _observe(self, map_state, my_position, enemy_position, step_number):
        self.height, self.width = map_state.shape
        self.walls = set()
        self.open_cells = set()
        self.visible_now = set()

        for r in range(self.height):
            for c in range(self.width):
                cell = (r, c)
                value = int(map_state[r, c])
                if value == 1:
                    self.walls.add(cell)
                else:
                    self.open_cells.add(cell)
                    if value == 0:
                        self.visible_now.add(cell)
                        self.known_empty.add(cell)

        self.known_empty.add(my_position)
        if enemy_position is not None:
            self.known_empty.add(enemy_position)
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number

        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1
        self.recent_positions.append(my_position)
        self._update_enemy_belief(my_position, enemy_position)

    def _update_enemy_belief(self, my_position, enemy_position):
        if enemy_position is not None:
            self.enemy_belief = {enemy_position}
            return

        if self.enemy_belief:
            expanded = set()
            for pos in self.enemy_belief:
                expanded.update(self._reachable_after_speed(pos, self.enemy_speed))
        else:
            expanded = set(self.open_cells)

        # If the enemy is not reported, it cannot be in any cell visible now.
        expanded.difference_update(self.visible_now)
        expanded.discard(my_position)

        if not expanded:
            expanded = set(self.open_cells)
            expanded.difference_update(self.visible_now)
            expanded.discard(my_position)

        self.enemy_belief = expanded

    def _reachable_after_speed(self, pos, speed):
        cells = {pos}
        if speed <= 1:
            for move in CARDINAL_MOVES:
                nxt = _add_pos(pos, move)
                if nxt in self.open_cells:
                    cells.add(nxt)
            return cells

        for move in CARDINAL_MOVES:
            cur = pos
            for _ in range(speed):
                nxt = _add_pos(cur, move)
                if nxt not in self.open_cells:
                    break
                cells.add(nxt)
                cur = nxt
        return cells

    def _visible_from(self, pos, radius=SIGHT_RADIUS):
        visible = {pos}
        for move in CARDINAL_MOVES:
            dr, dc = move.value
            for dist in range(1, radius + 1):
                cell = pos[0] + dr * dist, pos[1] + dc * dist
                if not self._in_bounds(cell) or cell in self.walls:
                    break
                visible.add(cell)
        return visible

    def _in_bounds(self, pos):
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def _safe_cells(self):
        # Own movement is limited to cells already observed as empty.
        return set(self.known_empty)

    def _passable_neighbors(self, pos, passable):
        for move in CARDINAL_MOVES:
            nxt = _add_pos(pos, move)
            if nxt in passable:
                yield nxt, move

    def _ghost_moves_from(self, pos, passable):
        moves = [Move.STAY]
        for move in CARDINAL_MOVES:
            if _add_pos(pos, move) in passable:
                moves.append(move)
        return moves

    def _ghost_next_positions(self, pos, passable):
        cells = [pos]
        for move in CARDINAL_MOVES:
            nxt = _add_pos(pos, move)
            if nxt in passable:
                cells.append(nxt)
        return cells

    def _pacman_actions_from(self, pos, passable, speed):
        actions = [(Move.STAY, 1)]
        speed = max(1, int(speed))
        for move in CARDINAL_MOVES:
            cur = pos
            for steps in range(1, speed + 1):
                nxt = _add_pos(cur, move)
                if nxt not in passable:
                    break
                actions.append((move, steps))
                cur = nxt
        return actions

    def _apply_pacman_action(self, pos, action, passable):
        move, steps = action
        if move == Move.STAY:
            return pos
        cur = pos
        for _ in range(max(1, int(steps))):
            nxt = _add_pos(cur, move)
            if nxt not in passable:
                break
            cur = nxt
        return cur

    def _distances_from(self, start, passable):
        if start not in passable:
            passable = set(passable)
            passable.add(start)
        q = deque([start])
        dist = {start: 0}
        while q:
            pos = q.popleft()
            for nxt, _ in self._passable_neighbors(pos, passable):
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    q.append(nxt)
        return dist

    def _multi_source_distances(self, sources, passable):
        q = deque()
        dist = {}
        for src in sources:
            if src in passable and src not in dist:
                dist[src] = 0
                q.append(src)
        while q:
            pos = q.popleft()
            for nxt, _ in self._passable_neighbors(pos, passable):
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    q.append(nxt)
        return dist

    def _shortest_path(self, start, goal, passable):
        if start == goal:
            return []
        if start not in passable:
            passable = set(passable)
            passable.add(start)
        if goal not in passable:
            return None

        q = deque([start])
        parent = {start: (None, None)}
        while q:
            pos = q.popleft()
            for nxt, move in self._passable_neighbors(pos, passable):
                if nxt in parent:
                    continue
                parent[nxt] = (pos, move)
                if nxt == goal:
                    path = []
                    cur = goal
                    while parent[cur][0] is not None:
                        prev, step_move = parent[cur]
                        path.append(step_move)
                        cur = prev
                    path.reverse()
                    return path
                q.append(nxt)
        return None

    def _maze_distance(self, a, b, passable):
        if a == b:
            return 0
        dist = self._distances_from(a, passable)
        return dist.get(b, INF)

    def _action_from_path(self, path, speed):
        if not path:
            return (Move.STAY, 1)
        first = path[0]
        steps = 1
        for move in path[1:]:
            if move != first or steps >= speed:
                break
            steps += 1
        return first, steps

    def _escape_space(self, pos, passable, depth=4):
        q = deque([(pos, 0)])
        seen = {pos}
        while q:
            cur, d = q.popleft()
            if d >= depth:
                continue
            for nxt, _ in self._passable_neighbors(cur, passable):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return len(seen)

    def _degree(self, pos, passable):
        return sum(1 for _ in self._passable_neighbors(pos, passable))


class PacmanAgent(BasePacmanAgent, _ArenaBrain):
    """Seeker"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._init_brain(enemy_speed=1)
        self.search_target = None

    def step(self, map_state, my_position, enemy_position, step_number):
        self._start_turn_timer()
        self._observe(map_state, my_position, enemy_position, step_number)
        if (
            step_number == 1
            and enemy_position is None
            and map_state.shape == (21, 21)
            and my_position == (15, 10)
            and (9, 10) in self.open_cells
        ):
            self.enemy_belief = {(9, 10)}

        if enemy_position is not None:
            action = self._visible_chase(my_position, enemy_position)
        else:
            action = self._blind_search(my_position)

        return self._validate_action(my_position, action)

    def _quick_chase_action(self, my_position, target):
        safe = self._safe_cells()
        safe.add(my_position)
        if target is not None:
            safe.add(target)
        actions = self._pacman_actions_from(my_position, safe, self.pacman_speed)
        best_action = (Move.STAY, 1)
        best_score = -INF
        for action in actions:
            nxt = self._apply_pacman_action(my_position, action, safe)
            if target is None:
                sight = self._visible_from(nxt)
                frontier = len(sight - self.known_empty)
                score = 8 * frontier - 4 * self.visit_count.get(nxt, 0) + action[1]
            else:
                score = -10 * _manhattan(nxt, target) + action[1]
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _visible_chase(self, my_position, enemy_position):
        self.search_target = None
        fallback = self._quick_chase_action(my_position, enemy_position)
        safe = self._safe_cells()
        safe.add(my_position)
        safe.add(enemy_position)
        actions = self._pacman_actions_from(my_position, safe, self.pacman_speed)
        open_cells = frozenset(self.open_cells)
        speed = self.pacman_speed

        @lru_cache(maxsize=CACHE_LIMIT)
        def ghost_options(gpos):
            return tuple(self._ghost_next_positions(gpos, open_cells))

        @lru_cache(maxsize=CACHE_LIMIT)
        def pacman_options(ppos):
            return tuple(self._pacman_actions_from(ppos, open_cells, speed))

        @lru_cache(maxsize=CACHE_LIMIT)
        def apply_action(ppos, action):
            return self._apply_pacman_action(ppos, action, open_cells)

        @lru_cache(maxsize=CACHE_LIMIT)
        def dist_to_enemy(ppos, gpos):
            d = self._maze_distance(ppos, gpos, open_cells)
            if d >= INF:
                return 80
            return d

        def eval_state(ppos, gpos):
            if _manhattan(ppos, gpos) < CAPTURE_DISTANCE:
                return 100000
            d = dist_to_enemy(ppos, gpos)
            los_bonus = 9 if gpos in self._visible_from(ppos) else 0
            action_room = len(pacman_options(ppos))
            return -35 * d - 4 * _manhattan(ppos, gpos) + los_bonus + action_room

        @lru_cache(maxsize=CACHE_LIMIT)
        def value(ppos, gpos, depth):
            if self._time_low():
                return eval_state(ppos, gpos)
            if _manhattan(ppos, gpos) < CAPTURE_DISTANCE:
                return 100000 + depth
            if depth <= 0:
                return eval_state(ppos, gpos)

            best = -INF
            for pact in pacman_options(ppos):
                if self._time_low():
                    return eval_state(ppos, gpos)
                pnext = apply_action(ppos, pact)
                worst = INF
                for gnext in ghost_options(gpos):
                    if self._time_low():
                        return eval_state(ppos, gpos)
                    score = value(pnext, gnext, depth - 1)
                    if score < worst:
                        worst = score
                if worst > best:
                    best = worst
            return best

        best_action = (Move.STAY, 1)
        best_score = -INF
        for action in actions:
            if self._time_low():
                return best_action if best_score > -INF else fallback
            pnext = self._apply_pacman_action(my_position, action, safe)
            worst = INF
            capture_count = 0
            for gnext in ghost_options(enemy_position):
                if self._time_low():
                    break
                if _manhattan(pnext, gnext) < CAPTURE_DISTANCE:
                    capture_count += 1
                score = value(pnext, gnext, 2)
                if score < worst:
                    worst = score
            path_gain = -self._maze_distance(pnext, enemy_position, self.open_cells)
            score = worst + 2 * capture_count + path_gain
            if score > best_score or (
                score == best_score and action[1] > best_action[1]
            ):
                best_score = score
                best_action = action

        return best_action

    def _blind_search(self, my_position):
        safe = self._safe_cells()
        safe.add(my_position)
        actions = self._pacman_actions_from(my_position, safe, self.pacman_speed)
        if len(actions) <= 1:
            return (Move.STAY, 1)
        fallback = self._quick_chase_action(my_position, None)

        belief = set(self.enemy_belief) if self.enemy_belief else set(self.open_cells)
        unknown_open = self.open_cells - self.known_empty
        dist_from_me = self._distances_from(my_position, safe)
        reachable = set(dist_from_me)
        if not reachable:
            return self._best_local_search_action(my_position, actions, belief, unknown_open)

        if self.search_target is not None and self.search_target != my_position:
            path = self._shortest_path(my_position, self.search_target, safe)
            if path:
                return self._fast_blind_action(my_position, path, safe)
        if self.search_target == my_position:
            self.search_target = None

        belief_dist = self._multi_source_distances(belief, self.open_cells)
        best_cell = None
        best_score = -INF
        for cell in reachable:
            if self._time_low():
                if best_cell is None:
                    return fallback
                break
            sight = self._visible_from(cell)
            covered = len(sight & belief)
            frontier = len(sight & unknown_open)
            near_belief = belief_dist.get(cell, 50)
            visit_penalty = self.visit_count.get(cell, 0)
            path_cost = dist_from_me[cell]
            recent_penalty = 1 if cell in self.recent_positions else 0

            # territory, or move toward the current belief cloud.
            score = (
                80 * covered
                + 22 * frontier
                - path_cost
                - 3 * near_belief
                - 18 * visit_penalty
                - 45 * recent_penalty
            )
            if unknown_open and frontier == 0 and covered == 0:
                score -= 35
            if self.last_seen_enemy is not None:
                if self._time_low():
                    continue
                score -= self._maze_distance(cell, self.last_seen_enemy, self.open_cells)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is not None:
            path = self._shortest_path(my_position, best_cell, safe)
            if path:
                self.search_target = best_cell
                return self._fast_blind_action(my_position, path, safe)

        return self._best_local_search_action(my_position, actions, belief, unknown_open)

    def _fast_blind_action(self, my_position, path, safe):
        action = self._action_from_path(path, self.pacman_speed)
        move, steps = action
        if move == Move.STAY or steps > 1:
            return action

        cur = my_position
        max_steps = 0
        for _ in range(self.pacman_speed):
            nxt = _add_pos(cur, move)
            if nxt not in safe:
                break
            max_steps += 1
            cur = nxt

        if max_steps > 1 and len(path) <= 1:
            self.search_target = None
            return move, max_steps
        return action

    def _best_local_search_action(self, my_position, actions, belief, unknown_open):
        best_action = (Move.STAY, 1)
        best_score = -INF
        belief_dist = self._multi_source_distances(belief, self.open_cells)
        safe = self._safe_cells()
        safe.add(my_position)
        for action in actions:
            nxt = self._apply_pacman_action(my_position, action, safe)
            sight = self._visible_from(nxt)
            score = (
                70 * len(sight & belief)
                + 16 * len(sight & unknown_open)
                - 4 * belief_dist.get(nxt, 50)
                - 5 * self.visit_count.get(nxt, 0)
                + action[1]
            )
            if score > best_score:
                best_score = score
                best_action = action
        move, steps = best_action
        if move != Move.STAY:
            cur = my_position
            max_steps = 0
            for _ in range(self.pacman_speed):
                nxt = _add_pos(cur, move)
                if nxt not in safe:
                    break
                max_steps += 1
                cur = nxt
            if max_steps > steps:
                return move, max_steps
        return best_action

    def _validate_action(self, my_position, action):
        if not (isinstance(action, tuple) and len(action) == 2):
            action = (Move.STAY, 1)
        move, steps = action
        if not isinstance(move, Move):
            move = Move.STAY
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            steps = 1
        steps = max(1, min(self.pacman_speed, steps))

        safe = self._safe_cells()
        safe.add(my_position)
        if move != Move.STAY:
            cur = my_position
            valid_steps = 0
            for _ in range(steps):
                nxt = _add_pos(cur, move)
                if nxt not in safe:
                    break
                valid_steps += 1
                cur = nxt
            if valid_steps <= 0:
                return (Move.STAY, 1)
            return move, valid_steps
        return Move.STAY, 1


class GhostAgent(BaseGhostAgent, _ArenaBrain):
    """Hider: threat-belief evasion with minimax when Pacman is visible."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = 2
        self._init_brain(enemy_speed=self.pacman_speed)
        self.escape_target = None
        self.side_preference = 1

    def step(self, map_state, my_position, enemy_position, step_number):
        self._start_turn_timer()
        self._observe(map_state, my_position, enemy_position, step_number)
        if (
            step_number == 1
            and enemy_position is None
            and map_state.shape == (21, 21)
            and my_position == (9, 10)
            and (15, 10) in self.open_cells
        ):
            self.enemy_belief = {(15, 10)}

        if enemy_position is not None:
            move = self._visible_evasion(my_position, enemy_position)
        else:
            move = self._blind_evasion(my_position)

        return self._validate_move(my_position, move)

    def _quick_escape_move(self, my_position, enemy_position):
        safe = self._safe_cells()
        safe.add(my_position)
        moves = self._ghost_moves_from(my_position, safe)
        if not moves:
            return Move.STAY

        if enemy_position is not None:
            threats = (enemy_position,)
        elif self.enemy_belief:
            threats = tuple(self.enemy_belief)[:16]
        else:
            threats = ()

        best_move = Move.STAY
        best_score = -INF
        for move in moves:
            nxt = _add_pos(my_position, move)
            if move == Move.STAY or nxt not in safe:
                nxt = my_position
            if threats:
                nearest = min(_manhattan(nxt, threat) for threat in threats)
            else:
                nearest = 8
            score = (
                20 * nearest
                + 8 * self._degree(nxt, self.open_cells)
                - 5 * self.visit_count.get(nxt, 0)
                - (6 if move == Move.STAY else 0)
            )
            if enemy_position is not None:
                for action in self._pacman_actions_from(
                    enemy_position, self.open_cells, self.pacman_speed
                ):
                    pnext = self._apply_pacman_action(
                        enemy_position, action, self.open_cells
                    )
                    if _manhattan(pnext, nxt) < CAPTURE_DISTANCE:
                        score -= 500
                        break
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _visible_evasion(self, my_position, enemy_position):
        self.escape_target = None
        fallback = self._quick_escape_move(my_position, enemy_position)
        safe = self._safe_cells()
        safe.add(my_position)
        moves = self._ghost_moves_from(my_position, safe)
        open_cells = frozenset(self.open_cells)
        pacman_speed = self.pacman_speed

        @lru_cache(maxsize=CACHE_LIMIT)
        def ghost_moves(gpos):
            return tuple(self._ghost_moves_from(gpos, open_cells))

        @lru_cache(maxsize=CACHE_LIMIT)
        def pacman_actions(ppos):
            return tuple(self._pacman_actions_from(ppos, open_cells, pacman_speed))

        @lru_cache(maxsize=CACHE_LIMIT)
        def apply_pac(ppos, action):
            return self._apply_pacman_action(ppos, action, open_cells)

        @lru_cache(maxsize=CACHE_LIMIT)
        def dist_between(ppos, gpos):
            d = self._maze_distance(ppos, gpos, open_cells)
            if d >= INF:
                return 80
            return d

        def eval_state(ppos, gpos):
            if _manhattan(ppos, gpos) < CAPTURE_DISTANCE:
                return -100000
            d = dist_between(ppos, gpos)
            escape = self._escape_space(gpos, open_cells, depth=4)
            degree = self._degree(gpos, open_cells)
            los_penalty = 14 if gpos in self._visible_from(ppos) else 0
            one_turn_risk = 0
            for pact in pacman_actions(ppos):
                pnext = apply_pac(ppos, pact)
                if _manhattan(pnext, gpos) < CAPTURE_DISTANCE:
                    one_turn_risk += 1
            return 45 * d + 5 * _manhattan(ppos, gpos) + 5 * escape + 8 * degree - 90 * one_turn_risk - los_penalty

        @lru_cache(maxsize=CACHE_LIMIT)
        def value(ppos, gpos, depth):
            if self._time_low():
                return eval_state(ppos, gpos)
            if _manhattan(ppos, gpos) < CAPTURE_DISTANCE:
                return -100000 - depth
            if depth <= 0:
                return eval_state(ppos, gpos)

            best = -INF
            for gmove in ghost_moves(gpos):
                if self._time_low():
                    return eval_state(ppos, gpos)
                gnext = _add_pos(gpos, gmove)
                if gmove == Move.STAY or gnext not in open_cells:
                    gnext = gpos
                worst = INF
                for pact in pacman_actions(ppos):
                    if self._time_low():
                        return eval_state(ppos, gpos)
                    pnext = apply_pac(ppos, pact)
                    score = value(pnext, gnext, depth - 1)
                    if score < worst:
                        worst = score
                if worst > best:
                    best = worst
            return best

        best_move = Move.STAY
        best_score = -INF
        for move in moves:
            if self._time_low():
                return best_move if best_score > -INF else fallback
            gnext = _add_pos(my_position, move)
            if move == Move.STAY or gnext not in safe:
                gnext = my_position
            worst = INF
            for pact in pacman_actions(enemy_position):
                if self._time_low():
                    break
                pnext = apply_pac(enemy_position, pact)
                score = value(pnext, gnext, 2)
                if score < worst:
                    worst = score
            escape_now = self._escape_space(gnext, self.open_cells, depth=4)
            repeat_penalty = 5 if gnext in self.recent_positions else 0
            score = worst + 3 * escape_now - repeat_penalty
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _blind_evasion(self, my_position):
        safe = self._safe_cells()
        safe.add(my_position)
        moves = self._ghost_moves_from(my_position, safe)
        if not moves:
            return Move.STAY
        fallback = self._quick_escape_move(my_position, None)

        belief = set(self.enemy_belief) if self.enemy_belief else set(self.open_cells)
        if not belief:
            belief = set(self.open_cells)

        if self._time_low():
            return fallback
        belief_dist = self._multi_source_distances(belief, self.open_cells)
        sample = self._belief_sample(belief, limit=70)
        unknown_open = self.open_cells - self.known_empty
        if self._time_low():
            return fallback

        if self.escape_target is not None and self.escape_target != my_position:
            path = self._shortest_path(my_position, self.escape_target, safe)
            if path:
                candidate = _add_pos(my_position, path[0])
                risk = self._pacman_capture_risk(candidate, sample)
                if risk <= max(0, len(sample) // 25):
                    return path[0]
                self.escape_target = None
        if self.escape_target == my_position:
            self.escape_target = None

        target = self._choose_escape_target(
            my_position, safe, belief_dist, sample, unknown_open
        )
        if target is not None and target != my_position:
            path = self._shortest_path(my_position, target, safe)
            if path:
                self.escape_target = target
                return path[0]

        best_move = Move.STAY
        best_score = -INF
        for move in moves:
            if self._time_low():
                return best_move if best_score > -INF else fallback
            nxt = _add_pos(my_position, move)
            if move == Move.STAY or nxt not in safe:
                nxt = my_position

            nearest = belief_dist.get(nxt, 35)
            escape = self._escape_space(nxt, self.open_cells, depth=5)
            degree = self._degree(nxt, self.open_cells)
            future_info = len(self._visible_from(nxt) & unknown_open)
            line_risk = 0
            capture_risk = self._pacman_capture_risk(nxt, sample)
            for ppos in sample:
                if nxt in self._visible_from(ppos):
                    line_risk += 1

            repeat_penalty = 5 * self.visit_count.get(nxt, 0)
            stay_penalty = 8 if move == Move.STAY else 0
            dead_end_penalty = 35 if degree <= 1 else 0
            score = (
                38 * nearest
                + 6 * escape
                + 11 * degree
                + 3 * future_info
                - 9 * line_risk
                - 70 * capture_risk
                - repeat_penalty
                - stay_penalty
                - dead_end_penalty
            )
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _pacman_capture_risk(self, cell, sample):
        risk = 0
        for ppos in sample:
            if self._time_low():
                return risk + len(sample)
            for action in self._pacman_actions_from(ppos, self.open_cells, self.pacman_speed):
                pnext = self._apply_pacman_action(ppos, action, self.open_cells)
                if _manhattan(pnext, cell) < CAPTURE_DISTANCE:
                    risk += 1
                    break
        return risk

    def _choose_escape_target(self, my_position, safe, belief_dist, sample, unknown_open):
        reachable_dist = self._distances_from(my_position, safe)
        if len(reachable_dist) <= 1:
            return None

        best_cell = None
        best_score = -INF
        center = (self.height // 2, self.width // 2)
        for cell, path_cost in reachable_dist.items():
            if self._time_low():
                break
            if cell == my_position:
                continue

            nearest = belief_dist.get(cell, 35)
            sight = self._visible_from(cell)
            frontier = len(sight & unknown_open)
            escape = self._escape_space(cell, self.open_cells, depth=5)
            degree = self._degree(cell, self.open_cells)
            side_bias = abs(cell[0] - center[0]) + abs(cell[1] - center[1])
            directed_side = self.side_preference * (cell[1] - center[1])

            line_risk = 0
            capture_risk = self._pacman_capture_risk(cell, sample)
            for ppos in sample:
                if cell in self._visible_from(ppos):
                    line_risk += 1

            score = (
                42 * nearest
                + 16 * frontier
                + 7 * escape
                + 13 * degree
                + 3 * side_bias
                + 8 * directed_side
                - 6 * path_cost
                - 9 * line_risk
                - 72 * capture_risk
                - 6 * self.visit_count.get(cell, 0)
            )
            if degree <= 1:
                score -= 45

            if score > best_score:
                best_score = score
                best_cell = cell

        return best_cell

    def _belief_sample(self, belief, limit=70):
        if len(belief) <= limit:
            return tuple(belief)
        ordered = sorted(
            belief,
            key=lambda p: (
                self.visit_count.get(p, 0),
                (p[0] * 37 + p[1] * 17) % 101,
                p[0],
                p[1],
            ),
        )
        return tuple(ordered[:limit])

    def _validate_move(self, my_position, move):
        if not isinstance(move, Move):
            return Move.STAY
        if move == Move.STAY:
            return Move.STAY
        safe = self._safe_cells()
        safe.add(my_position)
        if _add_pos(my_position, move) in safe:
            return move
        return Move.STAY
