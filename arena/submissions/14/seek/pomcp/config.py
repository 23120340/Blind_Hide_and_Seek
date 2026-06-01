"""POMCP configuration for fog-seek.

Tuning goals:
- Keep decision time < 1s.
- Maintain memory usage < 16MB.
"""

import os

# TIME_BUDGET:
# - Max wall-clock time per step.
TIME_BUDGET = 0.9

def _env_flag(name: str, default: bool = False) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# DEBUG:
# - Enable verbose planner logging (disable for grading).
DEBUG = _env_flag("POMCP_DEBUG", False)

# MEM_BUDGET_MB:
# - Soft guard for peak resident memory usage.
MEM_BUDGET_MB = 16

# PARTICLE_COUNT:
# - Number of belief particles.
PARTICLE_COUNT = 200

# MCTS_SIMULATIONS:
# - Maximum simulations per step (bounded by TIME_BUDGET).
MCTS_SIMULATIONS = 3000

# HISTORY_LENGTH:
# - Số bước lưu lại trong lịch sử để chống đi lặp vòng tròn
HISTORY_LENGTH = 10

# ROLLOUT_DEPTH:
# - Depth for rollout evaluation (plys).
ROLLOUT_DEPTH = 10

# UCB_EXPLORATION_C:
# - Exploration constant for UCB1.
UCB_EXPLORATION_C = 30.0

# REJUVENATION_RATE:
# - Fraction of particles replaced by random positions each update.
REJUVENATION_RATE = 0.05

# CAPTURE_REWARD:
# - Reward when Pacman captures Ghost.
CAPTURE_REWARD = 100.0

# STEP_PENALTY:
# - Small cost per step to favor faster capture.
STEP_PENALTY = 0.2

# DISTANCE_WEIGHT:
# - Weight for distance penalty in rollout evaluation.
DISTANCE_WEIGHT = 1.0

# FRONTIER_WEIGHT:
# - Bonus for moving toward frontier when ghost unseen.
FRONTIER_WEIGHT = 0.8
