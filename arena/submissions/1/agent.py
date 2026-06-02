import sys
from pathlib import Path
from collections import deque
import time
import random
import numpy as np


src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent

from environment import Move
#----------------------Pacman----------------#
class PacmanAgent(BasePacmanAgent):
    """Seeker (Pacman) sử dụng Flat Monte Carlo Search."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_time = 0.9  # Giới hạn an toàn dưới 1 giây
        self.max_speed = 2   # Pacman có thể đi tối đa 2 bước trên đường thẳng
        
    def _is_valid_position(self, pos, map_state):
        """Kiểm tra vị trí hợp lệ (không phải tường hoặc sương mù)."""
        row, col = pos
        height, width = map_state.shape
        
        # Kiểm tra giới hạn bản đồ
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        # Chỉ đi vào các ô trống (0), tuyệt đối không đi vào tường (1) hay vùng chưa biết (-1)
        return map_state[row, col] == 0

    def _manhattan_distance(self, pos1, pos2):
        """Tính khoảng cách Manhattan."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_legal_actions(self, pos, map_state):
        """Lấy tất cả các hành động hợp lệ bao gồm cả hệ số tốc độ (1 hoặc 2 bước)."""
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            
            # Pacman có thể đi từ 1 đến max_speed bước trên cùng một đường thẳng
            for steps in range(1, self.max_speed + 1):
                path_valid = True
                
                # Phải đảm bảo toàn bộ đường đi đến ô đích không bị chặn
                for s in range(1, steps + 1):
                    check_pos = (pos[0] + delta_row * s, pos[1] + delta_col * s)
                    if not self._is_valid_position(check_pos, map_state):
                        path_valid = False
                        break
                        
                if path_valid:
                    actions.append((move, steps))
                    
        # Nếu bị kẹt hoàn toàn, đứng yên
        if not actions:
            actions.append((Move.STAY, 1))
            
        return actions

    def simulate_random_walk(self, start_pos, enemy_pos, map_state, depth=5):
        """Thực hiện một lượt rollout Monte Carlo."""
        current_pos = start_pos
        score = 0
        
        for _ in range(depth):
            actions = self.get_legal_actions(current_pos, map_state)
            move, steps = random.choice(actions)
            
            if move == Move.STAY:
                break
                
            delta_row, delta_col = move.value
            current_pos = (current_pos[0] + delta_row * steps, current_pos[1] + delta_col * steps)
            
            if enemy_pos is not None:
                # Trạng thái thấy địch: Thưởng điểm âm dựa trên khoảng cách (càng gần càng tốt)
                dist = self._manhattan_distance(current_pos, enemy_pos)
                if dist < 2:
                    score += 1000  # Điều kiện thắng của Seek agent
                    break
                else:
                    score -= dist
            else:
                # Trạng thái mất dấu (Fog of War): Khuyến khích di chuyển để mở rộng tầm nhìn
                unseen_cells = np.argwhere(map_state == -1)
                if len(unseen_cells) > 0:
                    distances = [self._manhattan_distance(current_pos, (ur, uc)) for ur, uc in unseen_cells]
                    score -= min(distances) # Trừ điểm dựa trên khoảng cách tới sương mù gần nhất

        return score

    def step(self, map_state, my_position, enemy_position, step_number):
        """Hàm xử lý chính gọi mỗi lượt đi."""
        start_time = time.time()
        
        # Lấy danh sách nước đi hợp lệ
        legal_actions = self.get_legal_actions(my_position, map_state)
        
        if len(legal_actions) == 1:
            return legal_actions[0]

        action_scores = {action: 0 for action in legal_actions}
        action_counts = {action: 0 for action in legal_actions}
        
        # Vòng lặp Monte Carlo
        while time.time() - start_time < self.max_time:
            for action in legal_actions:
                move, steps = action
                if move == Move.STAY:
                    continue
                
                # Tính vị trí dự kiến sau khi đi nước này
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row * steps, my_position[1] + delta_col * steps)
                
                # Thực hiện Rollout từ vị trí dự kiến
                # Độ sâu depth có thể điều chỉnh tùy theo khả năng xử lý của CPU
                reward = self.simulate_random_walk(next_pos, enemy_position, map_state, depth=7)
                
                action_scores[action] += reward
                action_counts[action] += 1
                
        # Tìm hành động có điểm trung bình cao nhất
        best_action = (Move.STAY, 1)
        best_avg_score = -float('inf')
        
        for action in legal_actions:
            if action_counts[action] > 0:
                avg_score = action_scores[action] / action_counts[action]
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_action = action
                    
        return best_action
        
#---------------------Ghost----------------------#
class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Optimized Hide Agent"
        self.last_known_enemy_pos = None
        self.visited_history = deque(maxlen=15)
        self.escape_target = None
        self.target_update_frequency = 5  # Tăng tần suất cập nhật để phản ứng nhạy bén hơn
        self.last_target_update = 0
        self.stuck_counter = 0
        self.last_position = None
        
        # Quy ước: -1 = Chưa biết, 0 = Đường đi trống, 1 = Tường chắn
        self.global_map = np.full((21, 21), -1)

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Hàm xử lý và ra quyết định di chuyển tại mỗi bước log.

        """
        # Cập nhật Bản đồ toàn cục dựa trên thông tin vừa quan sát được ở bước hiện tại
        visible_mask = (map_state != -1)
        self.global_map[visible_mask] = map_state[visible_mask]

        # Lưu lịch sử tọa độ thực tế để phục vụ thuật toán nhận diện vòng lặp di chuyển
        self.visited_history.append(my_position)

        if my_position == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = my_position

        # Phân nhánh theo trạng thái xuất hiện của kẻ địch
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            return self._defend_against_threat(my_position, enemy_position, step_number)
        else:
            return self._explore_and_prepare(my_position, step_number)

    def _defend_against_threat(self, my_pos, enemy_pos, step_number):
        """Chiến thuật phòng thủ phản xạ và thiết lập lộ trình bỏ chạy khi đối mặt hiểm họa."""
        if self.stuck_counter > 2:
            return self._try_escape_trap(my_pos)

        should_update = (step_number - self.last_target_update) >= self.target_update_frequency

        if should_update or self.escape_target is None or self.escape_target == my_pos:
            self.escape_target = self._find_best_escape_position(my_pos, enemy_pos)
            self.last_target_update = step_number

        if self.escape_target and self.escape_target != my_pos:
            move = self._move_towards_target(my_pos, self.escape_target)
            if move != Move.STAY:
                return move

        return self._immediate_evasion(my_pos, enemy_pos)

    def _find_best_escape_position(self, my_pos, enemy_pos):
        
        # Duyệt BFS trên Bản đồ bộ nhớ toàn cục 
        
        queue = deque([(my_pos, 0)])
        visited = {my_pos}
        best_pos = my_pos
        best_score = self._position_score(my_pos, enemy_pos)

        max_depth = 15  # Tăng độ sâu tìm kiếm 

        while queue:
            pos, depth = queue.popleft()

            if depth > max_depth:
                continue

            score = self._position_score(pos, enemy_pos)
            if score > best_score:
                best_score = score
                best_pos = pos

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (pos[0] + delta_row, pos[1] + delta_col)

                # Chỉ cho phép duyệt qua những ô đã biết chắc chắn là đường đi trống (0)
                if next_pos not in visited and self._is_walkable_global(next_pos):
                    visited.add(next_pos)
                    queue.append((next_pos, depth + 1))

        return best_pos

    def _position_score(self, pos, enemy_pos):
        
        # Hàm lượng hóa chất lượng vị trí dựa trên cấu trúc hình học của bản đồ toàn cục
        
        # Khoảng cách hình học Manhattan tới hiểm họa
        dist = abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1])

        # Phân tích các ô lân cận di chuyển được xung quanh tọa độ đang xét
        walkable_neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            np_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_walkable_global(np_pos):
                walkable_neighbors.append(move)

        num_neighbors = len(walkable_neighbors)
        flexibility_bonus = 0

        if num_neighbors == 1:
            # Ngõ cụt: Phạt nặng vì nếu Seeker nhìn thấy corridor này, bạn sẽ bị bắt mà không có lối thoát
            flexibility_bonus = -15
        elif num_neighbors >= 3:
            # Giao lộ (Ngã 3, Ngã 4): Thưởng điểm lớn vì cung cấp nhiều hướng rẽ để cắt đuôi
            flexibility_bonus = 12
        elif num_neighbors == 2:
            # Phân tích dạng hành lang thẳng hay góc cua chữ L
            m1, m2 = walkable_neighbors[0], walkable_neighbors[1]
            is_straight = (
                (m1 == Move.UP and m2 == Move.DOWN) or (m1 == Move.DOWN and m2 == Move.UP) or
                (m1 == Move.LEFT and m2 == Move.RIGHT) or (m1 == Move.RIGHT and m2 == Move.LEFT)
            )
            if is_straight:
                # Phạt đường thẳng vì Seeker di chuyển thẳng với tốc độ 2 ô/bước
                flexibility_bonus = -6
            else:
                # Góc cua chữ L: Thưởng nhẹ vì ép Seeker phải dừng bứt tốc và đổi hướng
                flexibility_bonus = 4

        # Phạt các ô vừa đi qua trong lịch sử gần nhất để chống hiện tượng đi giật lùi vô nghĩa
        history_penalty = -5 if pos in self.visited_history else 0

        return dist * 5 + flexibility_bonus + history_penalty

    def _move_towards_target(self, my_pos, target):
        """Tìm đường đi ngắn nhất thực tế dựa trên BFS thu nhỏ tới điểm đích thoát hiểm."""
        if my_pos == target:
            return Move.STAY

        queue = deque([(my_pos, [])])
        visited = {my_pos}

        while queue:
            pos, path = queue.popleft()
            if pos == target:
                return path[0] if path else Move.STAY

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (pos[0] + dr, pos[1] + dc)

                if next_pos not in visited and self._is_walkable_global(next_pos):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))

        return Move.STAY

    def _immediate_evasion(self, my_pos, enemy_pos):
        """Di chuyển sang các ô xung quanh có điểm số an toàn cao nhất."""
        best_move = None
        best_score = float('-inf')

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if self._is_walkable_global(next_pos):
                score = self._position_score(next_pos, enemy_pos)
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move or Move.STAY

    def _try_escape_trap(self, my_pos):
        """Cơ chế thoát khi phát hiện Agent bị kẹt trạng thái di chuyển lặp vòng tròn."""
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)

        for move in moves:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            # Chọn ô đi được và không nằm trong 3 vị trí vừa đứng ở các lượt trước
            if self._is_walkable_global(next_pos) and next_pos not in list(self.visited_history)[-3:]:
                return move

        return Move.STAY

    def _explore_and_prepare(self, my_pos, step_number):
        """
        Đánh giá điểm số các ô xung quanh dựa vào số lượng vùng sương mù (-1) mà ô đó tiếp giáp.

        """
        valid_moves = []
        exploration_scores = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if self._is_walkable_global(next_pos):
                valid_moves.append(move)
                
                # Quét xem ô dự định bước tới có thể mở rộng tầm nhìn sang bao nhiêu ô chưa biết (-1)
                unknown_count = 0
                for m_adj in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    adj_pos = (next_pos[0] + m_adj.value[0], next_pos[1] + m_adj.value[1])
                    if 0 <= adj_pos[0] < 21 and 0 <= adj_pos[1] < 21:
                        if self.global_map[adj_pos[0], adj_pos[1]] == -1:
                            unknown_count += 1
                
                # Phạt điểm nếu quay lại các ô cũ trong lịch sử để định hướng đi thẳng ra vùng biên mới
                history_penalty = -4 if next_pos in self.visited_history else 0
                exploration_scores.append(unknown_count * 3 + history_penalty)

        if valid_moves:
            best_idx = np.argmax(exploration_scores)
            return valid_moves[best_idx]

        return Move.STAY

    def _is_walkable_global(self, pos):
        """Kiểm tra tính hợp lệ của ô di chuyển dựa hoàn toàn trên Bản đồ toàn cục."""
        row, col = pos
        if row < 0 or row >= 21 or col < 0 or col >= 21:
            return False
        # không giả định ô -1 là an toàn, chỉ đi vào ô đã xác nhận là đường đi trống (0)
        return self.global_map[row, col] == 0


class HideAgent(GhostAgent):
    """Lớp chính thức để hệ thống kiểm thử import."""
    pass
