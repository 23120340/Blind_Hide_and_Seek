import sys
import time
import math
from collections import deque, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List, Set

import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

from seek import POMCPPlanner

# Known ghost start (row, col). Update if the start changes.
GHOST_START_POS = (9, 10)


class PacmanAgent(BasePacmanAgent):
    """POMCP-based Pacman agent (fog-seek)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "FogSeek POMCP v1"
        self.planner = POMCPPlanner(
            pacman_speed=self.pacman_speed,
            ghost_start_pos=GHOST_START_POS,
        )

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ):
        return self.planner.plan(map_state, my_position, enemy_position, step_number)
# ================================================================
# GHOST AGENT - Group 14
# Strategy: Junction-seeking opening + BFS flee + heuristic scoring
# ================================================================

_G_DELTAS = {
    Move.UP: (-1, 0), Move.DOWN: (1, 0),
    Move.LEFT: (0, -1), Move.RIGHT: (0, 1),
    Move.STAY: (0, 0),
}
_G_DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


def _gnp(pos, mv):
    """Next position after move."""
    dr, dc = _G_DELTAS[mv]
    return (pos[0] + dr, pos[1] + dc)


def _gmd(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _gok(pos, ms):
    """Is cell walkable? (not wall, may be fog)"""
    h, w = ms.shape
    r, c = pos
    return 0 <= r < h and 0 <= c < w and ms[r, c] != 1


def _gexits(pos, ms):
    """Count walkable neighbors."""
    return sum(1 for m in _G_DIRS if _gok(_gnp(pos, m), ms))


def _gwalls(pos, ms):
    """Count wall/boundary neighbors."""
    h, w = ms.shape
    c = 0
    for m in _G_DIRS:
        r2, c2 = _gnp(pos, m)
        if r2 < 0 or r2 >= h or c2 < 0 or c2 >= w or ms[r2, c2] == 1:
            c += 1
    return c


def _glos(p1, p2, ms):
    """Line-of-sight check (same row or col, no walls between)."""
    r1, c1 = p1; r2, c2 = p2
    if r1 == r2:
        lo, hi = min(c1, c2), max(c1, c2)
        return all(ms[r1, c] != 1 for c in range(lo + 1, hi))
    if c1 == c2:
        lo, hi = min(r1, r2), max(r1, r2)
        return all(ms[r, c1] != 1 for r in range(lo + 1, hi))
    return False


def _glegal(pos, ms):
    """Legal moves including STAY."""
    moves = [Move.STAY]
    for mv in _G_DIRS:
        if _gok(_gnp(pos, mv), ms):
            moves.append(mv)
    return moves


def _gbfs_dist(start, ms, max_d=25):
    """BFS distance map from start position."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= max_d:
            continue
        for mv in _G_DIRS:
            nb = _gnp(cur, mv)
            if nb not in dist and _gok(nb, ms):
                dist[nb] = d + 1
                q.append(nb)
    return dist


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent nhóm 14
    - Thay thế Manhattan bằng khoảng cách BFS thực tế để tránh ngõ cụt.
    - Tìm điểm xa nhất (target_pos) để hướng tới.
    - Tự động dùng 'Fake Pacman' ép góc ở vòng khai cuộc để đi đa dạng.
    - Bẻ góc (Perpendicular) khi bị đuổi sát trên cùng hàng/cột.
    """

    CAPTURE_DIST = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost14_v2_Parkour"
        self.pac_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.merged_map = None

        self.last_pac = None
        self.last_pac_step = -99
        self.pos_hist = deque(maxlen=10)
        self.step_num = 0
        self.opening_phase = True

    def _merge(self, ms):
        if self.merged_map is None:
            self.merged_map = np.full_like(ms, -1)
        known = ms != -1
        self.merged_map[known] = ms[known]
        return self.merged_map

    def step(self, map_state, my_position, enemy_position, step_number):
        gpos = (int(my_position[0]), int(my_position[1]))
        self.step_num = step_number
        ms = self._merge(map_state)
        h, w = ms.shape

        pac = None
        if enemy_position is not None:
            pac = (int(enemy_position[0]), int(enemy_position[1]))
            self.last_pac = pac
            self.last_pac_step = step_number
            self.opening_phase = False

        est_pac = self.last_pac
        
        # OPENING: Chạy tới góc phục kích (Ambush) thay vì chạy loăng ngoăng
        ambush_target = None
        if self.opening_phase and step_number <= 12:
            # Chọn điểm xa trung tâm để né quét mặc định của Pacman
            ambush_target = (3, w - 4)  # Góc trên phải
            if not _gok(ambush_target, ms):
                ambush_target = (h - 4, w - 4)
            
            if gpos == ambush_target:
                self.opening_phase = False
            else:
                # Đặt est_pac ở giữa bản đồ để kích hoạt tránh né khỏi tâm
                est_pac = (h // 2, w // 2)

        if est_pac is None:
            est_pac = (h // 2, w // 2)

        # Tính BFS map từ Pacman (hoặc est_pac)
        pac_bfs_map = _gbfs_dist(est_pac, ms, max_d=60)

        # Tìm TARGET (điểm an toàn xa nhất), ưu tiên ambush_target nếu đang opening
        if self.opening_phase and ambush_target:
            target_pos = ambush_target
        else:
            target_pos = gpos
            max_dist = -1
            for p, d in pac_bfs_map.items():
                if d > max_dist:
                    max_dist = d
                    target_pos = p

        best_move = self._evaluate_best_move(gpos, est_pac, target_pos, ms, pac_bfs_map)
        
        # Anti-stuck nếu bị kẹt
        if best_move == Move.STAY and pac is None:
            candidates = []
            for mv in _G_DIRS:
                nb = _gnp(gpos, mv)
                if _gok(nb, ms) and nb not in self.pos_hist:
                    candidates.append(mv)
            if candidates:
                # Ưu tiên các hướng không phải quay lại
                best_move = candidates[0]

        g_next = _gnp(gpos, best_move) if best_move != Move.STAY else gpos
        self.pos_hist.append(g_next)
        return best_move

    def _evaluate_best_move(self, gpos, pac, target_pos, ms, pac_bfs_map):
        moves_score = []
        gr, gc = gpos
        pr, pc = pac
        
        # Xác định corridor để phạt
        valid_neighbors = [_gnp(gpos, m) for m in _G_DIRS if _gok(_gnp(gpos, m), ms)]
        num_exits_curr = len(valid_neighbors)

        for mv in _glegal(gpos, ms):
            g1 = _gnp(gpos, mv) if mv != Move.STAY else gpos
            if _gmd(g1, pac) < self.CAPTURE_DIST:
                continue

            score = 0.0
            
            # 1. TRUE BFS SAFETY
            real_dist = pac_bfs_map.get(g1, 0)
            if real_dist <= 3:
                score -= 2000.0  # Tử địa
            elif real_dist <= 5:
                score -= 500.0   # Cực kỳ nguy hiểm
            
            score += real_dist * 25.0

            # 2. LOS BREAKING
            if not _glos(g1, pac, ms):
                score += 800.0

            # 3. PERPENDICULAR ESCAPE (Chạy vuông góc nếu bị đuổi dọc theo hàng/cột)
            if real_dist <= 7:
                dr, dc = _G_DELTAS.get(mv, (0, 0))
                if gr == pr and dr != 0:
                    score += 700.0  # Đang trên cùng hàng ngang -> bẻ dọc
                elif gc == pc and dc != 0:
                    score += 700.0  # Đang trên cùng cột dọc -> bẻ ngang

            # 4. STRUCTURE & ANTI-CORRIDOR
            exits = _gexits(g1, ms)
            is_corridor = False
            if exits == 2:
                # Kiểm tra xem có phải hành lang thẳng không
                nbrs = [_gnp(g1, m) for m in _G_DIRS if _gok(_gnp(g1, m), ms)]
                if len(nbrs) == 2:
                    r1, c1 = nbrs[0]
                    r2, c2 = nbrs[1]
                    if r1 == r2 or c1 == c2:
                        is_corridor = True

            if exits >= 3:
                score += 200.0
            elif exits == 1:
                score -= 1500.0
            elif is_corridor:
                score -= 150.0
            else:
                score += 100.0

            # 5. TARGET SEEKING (Parkour)
            dist_to_target = _gmd(g1, target_pos)
            score -= dist_to_target * 10.0

            # 6. ANTI-LOOP
            if g1 in list(self.pos_hist):
                score -= 300.0 * list(self.pos_hist).count(g1)

            # 7. WALL HUGGING
            score += 15.0 * _gwalls(g1, ms)
            
            if mv == Move.STAY:
                score -= 100.0

            moves_score.append((score, mv))

        if not moves_score:
            # Fallback
            for mv in _G_DIRS:
                g1 = _gnp(gpos, mv)
                if _gok(g1, ms):
                    moves_score.append((_gmd(g1, pac) * 100, mv))
            if not moves_score:
                return Move.STAY

        moves_score.sort(key=lambda x: x[0], reverse=True)
        return moves_score[0][1]
