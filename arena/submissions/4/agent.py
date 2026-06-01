"""Hide-and-seek agents using adversarial search and heuristic evaluation."""

import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

# Keep local student modules importable before adding src path.
agent_root = Path(__file__).parent
if str(agent_root) not in sys.path:
    sys.path.insert(0, str(agent_root))

src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

# Generic helpers from utils.py
from utils import (
    PacmanAction,
    Position,
    apply_move_once,
    apply_pacman_action,
    legal_ghost_moves,
    legal_pacman_actions,
    manhattan,
)

# Optimized map data from infrastructure.py
from infrastructure import Infrastructure

from search.heuristic import HeuristicEvaluator
from search.maximin import simultaneous_maximin_search
from search.state import SearchState
from search.exploration import (
    update_explored,
    exploration_action,
    GhostPolicyEstimator,
    intercept_expectiminimax_action,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_optional(name: str, kwarg_val: Optional[float]) -> Optional[float]:
    """Return a float if the env var or kwarg is explicitly set, otherwise None.

    Returning None lets HeuristicEvaluator compute a consistent derived default
    rather than having a second set of hard-coded fallbacks here.
    """
    raw = os.getenv(name)
    if raw is not None and raw.strip() != "":
        try:
            return float(raw)
        except ValueError:
            pass
    if kwarg_val is not None:
        return float(kwarg_val)
    return None


def _env_flag(names: List[str], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return default


class PacmanAgent(BasePacmanAgent):
    """Seeker agent using adversarial search with heuristic move ordering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Adversarial Seeker"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.capture_distance_threshold = max(
            1,
            int(kwargs.get("capture_distance_threshold", 2)),
        )
        self.search_time_limit = _env_float("SEARCH_TIME_LIMIT", float(kwargs.get("time_limit", 0.95)))
        self.debug_mode = _env_flag(["DEBUG_MODE"], False)

        self.w1 = _env_float("PACMAN_W1", float(kwargs.get("w1", 3.0)))
        self.w2 = _env_float("PACMAN_W2", float(kwargs.get("w2", 0.25)))
        self.w3 = _env_float("PACMAN_W3", float(kwargs.get("w3", 1.8)))
        self.w_turn = _env_float_optional("PACMAN_W_TURN", kwargs.get("w_turn"))
        self.w_dead_end = _env_float_optional("PACMAN_W_DEAD_END", kwargs.get("w_dead_end"))
        self.w_safe_space = _env_float_optional("PACMAN_W_SAFE_SPACE", kwargs.get("w_safe_space"))
        self.w_los = _env_float_optional("PACMAN_W_LOS", kwargs.get("w_los"))

        self.last_known_enemy_pos: Optional[Position] = None
        # Memory for systematic fog sweep
        self.explored_cells: Dict[Position, int] = {}
        self.explored_decay_steps = max(1, int(_env_float("PACMAN_EXPLORE_DECAY", 40)))
        self._prev_pos: Optional[Position] = None
        self._recent_positions: Deque[Position] = deque(maxlen=3)
        # Staleness counter for last sighting
        self.steps_since_sighting: int = 0
        # Expectiminimax: track previous ghost position to infer its move
        self._prev_ghost_pos: Optional[Position] = None
        # Expectiminimax: empirical Ghost policy estimator
        self.ghost_policy_estimator = GhostPolicyEstimator(uniform_weight=2.0)
        
        # Shared optimized map data
        self.infra: Optional[Infrastructure] = None
        
        self.evaluator = HeuristicEvaluator(
            pacman_speed=self.pacman_speed,
            capture_distance_threshold=self.capture_distance_threshold,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w_turn=self.w_turn,
            w_dead_end=self.w_dead_end,
            w_safe_space=self.w_safe_space,
            w_los=self.w_los,
        )

    def _ensure_infra(self, map_state: np.ndarray) -> Infrastructure:
        if self.infra is None:
            # KEY FIX: reconstruct the full wall layout from the first FOW observation.
            # The spec guarantees walls are ALWAYS visible (marked 1).
            # Therefore any non-1 cell (0 or -1) is definitively a walkable path.
            # We pass this full_map to Infrastructure so the distance matrix is correct
            # from step 1. The raw map_state (with -1) is still passed to the search
            # engine separately so it can treat fog cells as optimistically traversable.
            full_map = np.where(map_state == 1, np.int8(1), np.int8(0))
            self.infra = Infrastructure(full_map, pacman_speed=self.pacman_speed)
            self.infra.preprocess()
            self.evaluator.bind_infra(self.infra)
            # Detect FOW and update evaluator weights
            is_fow = bool(np.any(map_state == -1))
            self.evaluator.set_fow(is_fow)
        return self.infra

    def step(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> PacmanAction:
        # Build (or reuse) the full-map Infrastructure
        self._ensure_infra(map_state)

        # Update fog memory each step (before any decision)
        update_explored(
            map_state,
            self.explored_cells,
            step_number,
        )

        # Update last known position and staleness counter
        if enemy_position is not None:
            # Observe Ghost move for policy estimation:
            # if we saw the Ghost last step AND this step, we can infer
            # which Move it took by comparing the two positions.
            if self._prev_ghost_pos is not None:
                dr = enemy_position[0] - self._prev_ghost_pos[0]
                dc = enemy_position[1] - self._prev_ghost_pos[1]
                observed_move = None
                for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if mv.value == (dr, dc):
                        observed_move = mv
                        break
                if observed_move is None and dr == 0 and dc == 0:
                    observed_move = Move.STAY
                if observed_move is not None:
                    self.ghost_policy_estimator.observe(
                        self._prev_ghost_pos, observed_move
                    )
            self._prev_ghost_pos = enemy_position
            self.last_known_enemy_pos = enemy_position
            self.steps_since_sighting = 0
        else:
            self._prev_ghost_pos = None   # lost sight — reset chain
            self.steps_since_sighting += 1

        # Instant-reach check: we arrived at last-known cell but ghost isn't here
        if my_position == self.last_known_enemy_pos and enemy_position is None:
            self.last_known_enemy_pos = None

        target = enemy_position or self.last_known_enemy_pos

        # Staleness gate (threshold agreed with Person D)
        _STALE_THRESHOLD = 6
        if self.steps_since_sighting > _STALE_THRESHOLD:
            target = None

        actions = legal_pacman_actions(map_state, my_position, self.pacman_speed)

        # ── Ghost visible or recently sighted → adversarial search ────────────
        if target is not None:
            search_action = self._run_adversarial_search(map_state, my_position, target)
            if search_action is not None:
                return self._commit_action(my_position, search_action)
            # Main search timed out — use Expectiminimax intercept fallback
            return self._commit_action(
                my_position,
                intercept_expectiminimax_action(
                actions=actions,
                my_position=my_position,
                ghost_position=target,
                infra=self.infra,
                estimator=self.ghost_policy_estimator,
                depth=3,
                ),
            )

        # ── Ghost unknown → systematic fog sweep ──────────────────────────────
        exp_act = exploration_action(
            actions=actions,
            map_state=map_state,
            my_position=my_position,
            explored_cells=self.explored_cells,
            infra=self.infra,
            pacman_speed=self.pacman_speed,
            last_known_enemy_pos=self.last_known_enemy_pos,
            steps_since_sighting=self.steps_since_sighting,
            step_number=step_number,
            decay_steps=self.explored_decay_steps,
        )
        return self._commit_action(my_position, exp_act)

    def _run_adversarial_search(
        self,
        map_state: np.ndarray,
        pacman_pos: Position,
        ghost_pos: Position,
    ) -> Optional[PacmanAction]:
        try:
            state = SearchState(pacman_pos=pacman_pos, ghost_pos=ghost_pos)
            best_action, _, diags = simultaneous_maximin_search(
                state=state,
                map_array=map_state,
                pacman_speed=self.pacman_speed,
                time_limit=self.search_time_limit,
                capture_distance_threshold=self.capture_distance_threshold,
                evaluate_state_fn=self.evaluator.evaluate_state,
                order_pacman_actions_fn=self.evaluator.order_pacman_actions,
                order_ghost_actions_fn=self.evaluator.order_ghost_moves_for_pacman,
                profiling_enabled=self.debug_mode,
            )
            if self.debug_mode:
                self._nodes_evaluated = diags.get("nodes", 0)
                self._nodes_pruned = diags.get("p_cutoffs", 0) + diags.get("g_cutoffs", 0)
                self._max_depth = diags.get("completed_depth", 0)
            return best_action
        except TimeoutError:
            logger.debug("PacmanAgent adversarial search timed out")
            return None
        except Exception:
            logger.exception("Unexpected error in PacmanAgent adversarial search")
            return None

    def _commit_action(self, my_position: Position, action: PacmanAction) -> PacmanAction:
        self._prev_pos = my_position
        self._recent_positions.append(my_position)
        return action


    def _exploration_action(
        self,
        actions: List[PacmanAction],
        map_state: np.ndarray,
        my_position: Position,
    ) -> PacmanAction:
        height, width = map_state.shape
        center = (height // 2, width // 2)

        best_action = (Move.STAY, 1)
        best_score = -float("inf")

        for action in actions:
            next_pos = apply_pacman_action(
                map_state,
                my_position,
                action,
                self.pacman_speed,
            )
            local_mobility = len(legal_ghost_moves(map_state, next_pos))
            center_bias = -0.05 * float(manhattan(next_pos, center))
            step_bonus = 0.1 * float(action[1])
            score = float(local_mobility) + center_bias + step_bonus

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class GhostAgent(BaseGhostAgent):
    """Hider agent using adversarial search with defensive ordering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Adversarial Hider"

        # The loader does not pass pacman_speed to Ghost by default.
        self.estimated_pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_distance_threshold = max(
            1,
            int(kwargs.get("capture_distance_threshold", 2)),
        )
        self.search_time_limit = _env_float("SEARCH_TIME_LIMIT", float(kwargs.get("time_limit", 0.95)))
        self.debug_mode = _env_flag(["DEBUG_MODE"], False)

        self.w1 = _env_float("GHOST_W1", float(kwargs.get("w1", 3.0)))
        self.w2 = _env_float("GHOST_W2", float(kwargs.get("w2", 0.25)))
        self.w3 = _env_float("GHOST_W3", float(kwargs.get("w3", 1.8)))
        self.w_turn = _env_float_optional("GHOST_W_TURN", kwargs.get("w_turn"))
        self.w_dead_end = _env_float_optional("GHOST_W_DEAD_END", kwargs.get("w_dead_end"))
        self.w_safe_space = _env_float_optional("GHOST_W_SAFE_SPACE", kwargs.get("w_safe_space"))
        self.w_los = 100000 # Staying in LOS of Pacman is the single worst possible thing for the Ghost, so we set this very high to prioritize LOS avoidance above all else.
        
        self._early_bias_weight: float = 10 # Early-game upper-half bias, decays to 0 over _early_bias_steps
        self._early_bias_steps: int = 8

        self.last_known_enemy_pos: Optional[Position] = None
        self.step_since_sighting: int = 0 # Step since last seen Pacman, used for fog heuristic
        self.stale_threshold: int = 6 # How many steps of not seeing Pacman before we consider the info stale
        self._prev_pos: Optional[Position] = None
        self._recent_positions: Deque[Position] = deque(maxlen=3)
        
        # Shared optimized map data
        self.infra: Optional[Infrastructure] = None
        
        self.evaluator = HeuristicEvaluator(
            pacman_speed=self.estimated_pacman_speed,
            capture_distance_threshold=self.capture_distance_threshold,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w_turn=self.w_turn,
            w_dead_end=self.w_dead_end,
            w_safe_space=self.w_safe_space,
            w_los=self.w_los,
        )

    def _ensure_infra(self, map_state: np.ndarray) -> Infrastructure:
        if self.infra is None:
            full_map = np.where(map_state == 1, 1, 0) # Turn empty space and fog to 0 (walkable) and walls to 1 (non-walkable)
            self.infra = Infrastructure(full_map, pacman_speed=self.estimated_pacman_speed)
            self.infra.preprocess()
            self.evaluator.bind_infra(self.infra)
            # Detect FOW and update evaluator weights
            is_fow = bool(np.any(map_state == -1))
            self.evaluator.set_fow(is_fow)
        return self.infra

    def step(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> Move:
        self._ensure_infra(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.step_since_sighting = 0
        else:
            self.step_since_sighting += 1

        # Instant reset last_known_enemy_pos if we're standing on it
        # "If we reach the last-known cell and the enemy is not there, clear it instantly"
        if self.last_known_enemy_pos is not None and my_position == self.last_known_enemy_pos:
            self.last_known_enemy_pos = None

        # Clear the last known enemy position if it's become stale
        if self.step_since_sighting >= self.stale_threshold:
            self.last_known_enemy_pos = None
    
        prev_pos = self._prev_pos
        pacman_ref = enemy_position or self.last_known_enemy_pos
        legal_moves = legal_ghost_moves(map_state, my_position)

        if enemy_position is None:
            half_bias_weight = 0.0
            if step_number <= self._early_bias_steps:
                decay = max(0.0, 1.0 - (float(step_number) / float(self._early_bias_steps)))
                half_bias_weight = self._early_bias_weight * decay
            return self._commit_move(
                my_position,
                self._unseen_patrol_move(
                    map_state,
                    my_position,
                    legal_moves,
                    pacman_ref=self.last_known_enemy_pos,
                    prev_pos=prev_pos,
                    recent_positions=list(self._recent_positions),
                    half_bias_weight=half_bias_weight,
                    mid_row=map_state.shape[0] // 2,
                ),
            )

        search_move = self._run_adversarial_search(map_state, pacman_ref, my_position)
        if search_move is not None:
            return self._commit_move(my_position, search_move)

        # Final fallback if search fails
        return self._commit_move(
            my_position,
            self._unseen_patrol_move(
                map_state,
                my_position,
                legal_moves,
                pacman_ref=pacman_ref,
                prev_pos=prev_pos,
                recent_positions=list(self._recent_positions),
                half_bias_weight=0.0,
                mid_row=map_state.shape[0] // 2,
            ),
        )

    def _run_adversarial_search(
        self,
        map_state: np.ndarray,
        pacman_pos: Position,
        ghost_pos: Position,
    ) -> Optional[Move]:
        try:
            state = SearchState(pacman_pos=pacman_pos, ghost_pos=ghost_pos)
            _, best_move, diags = simultaneous_maximin_search(
                state=state,
                map_array=map_state,
                pacman_speed=self.estimated_pacman_speed,
                time_limit=self.search_time_limit,
                capture_distance_threshold=self.capture_distance_threshold,
                evaluate_state_fn=self.evaluator.evaluate_state,
                order_pacman_actions_fn=self.evaluator.order_pacman_actions,
                order_ghost_actions_fn=self.evaluator.order_ghost_moves_for_ghost,
                profiling_enabled=self.debug_mode,
            )
            if self.debug_mode:
                self._nodes_evaluated = diags.get("nodes", 0)
                self._nodes_pruned = diags.get("p_cutoffs", 0) + diags.get("g_cutoffs", 0)
                self._max_depth = diags.get("completed_depth", 0)
            return best_move
        except TimeoutError:
            logger.debug("GhostAgent adversarial search timed out")
            return None
        except Exception:
            logger.exception("Unexpected error in GhostAgent adversarial search")
            return None


    def _unseen_patrol_move(
        self,
        map_state: np.ndarray,
        my_position: Position,
        legal_moves: List[Move],
        pacman_ref: Optional[Position],
        prev_pos: Optional[Position],
        recent_positions: List[Position],
        half_bias_weight: float,
        mid_row: int,
    ) -> Move:
        if len(legal_moves) > 1:
            legal_moves = [m for m in legal_moves if m != Move.STAY]

        best_score = -float("inf")
        best_moves: List[Move] = []

        for move in legal_moves:
            next_pos = apply_move_once(map_state, my_position, move)
            mobility = len(legal_ghost_moves(map_state, next_pos))

            score = float(mobility)
            if pacman_ref is not None and self.infra is not None:
                dist = self.infra.distance(next_pos, pacman_ref)
                score += 0.35 * float(dist)

            if mobility >= 3:
                score += 0.2

            if prev_pos is not None and next_pos == prev_pos:
                score -= 0.9

            if next_pos in recent_positions:
                age = len(recent_positions) - 1 - recent_positions.index(next_pos)
                score -= 0.5 + 0.2 * float(age)

            if half_bias_weight > 0.0 and next_pos[0] < mid_row:
                score += half_bias_weight

            if move == Move.STAY:
                score -= 0.75

            if score > best_score + 1e-6:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) <= 1e-6:
                best_moves.append(move)

        if not best_moves:
            return Move.STAY
        if len(best_moves) == 1:
            return best_moves[0]
        return best_moves[int(np.random.randint(0, len(best_moves)))]

    def _commit_move(self, my_position: Position, move: Move) -> Move:
        self._prev_pos = my_position
        self._recent_positions.append(my_position)
        return move

    def _mobility_first_move(
        self,
        map_state: np.ndarray,
        my_position: Position,
        legal_moves: List[Move],
    ) -> Move:
        best_move = Move.STAY
        best_score = -float("inf")

        for move in legal_moves:
            next_pos = apply_move_once(map_state, my_position, move)
            score = float(len(legal_ghost_moves(map_state, next_pos)))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
