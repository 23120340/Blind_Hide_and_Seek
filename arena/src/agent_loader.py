"""
Module for loading student agents dynamically.

Loads agent classes directly in-process using importlib. This is the
student-facing loader used for local testing. For tournament evaluation
with subprocess sandboxing, run arena.py with --sandbox (which swaps in
the instructor version of this module from instructors/agent_loader.py).
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Move


class AgentLoadError(Exception):
    """Exception raised when agent loading fails."""
    pass


class AgentLoader:
    """Loads student agents from the submissions directory."""

    def __init__(self, submissions_dir: str = "submissions"):
        self.submissions_dir = Path(submissions_dir)
        if not self.submissions_dir.exists():
            self.submissions_dir.mkdir(parents=True)

    def load_agent(self, student_id: str, agent_type: str, init_kwargs: Optional[dict] = None):
        """
        Load a student agent in-process and return the instantiated object.

        Args:
            student_id: Student ID (folder name in submissions/)
            agent_type: 'pacman' or 'ghost'
            init_kwargs: Optional dict; 'pacman_speed' key forwarded for Pacman.

        Returns:
            An instantiated PacmanAgent or GhostAgent.

        Raises:
            AgentLoadError: If the agent cannot be loaded or fails validation.
        """
        from agent_interface import PacmanAgent as BasePacmanAgent
        from agent_interface import GhostAgent as BaseGhostAgent

        agent_dir = self.submissions_dir / student_id
        agent_file = agent_dir / "agent.py"

        if not agent_file.exists():
            raise AgentLoadError(
                f"Agent file not found for student {student_id} at {agent_file}"
            )

        # Allow the student to import helper modules from their folder
        student_dir_str = str(agent_dir.resolve())
        if student_dir_str not in sys.path:
            sys.path.insert(0, student_dir_str)

        try:
            spec = importlib.util.spec_from_file_location(
                f"agent_{student_id}", agent_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            raise AgentLoadError(
                f"Failed to load agent module for {student_id}: "
                f"{type(e).__name__}: {e}"
            )

        # Determine expected class
        if agent_type.lower() == "pacman":
            class_name = "PacmanAgent"
            expected_parent = BasePacmanAgent
            init_kwargs_final = {"pacman_speed": (init_kwargs or {}).get("pacman_speed", 1)}
        elif agent_type.lower() == "ghost":
            class_name = "GhostAgent"
            expected_parent = BaseGhostAgent
            init_kwargs_final = {}
        else:
            raise AgentLoadError(f"Invalid agent type: {agent_type!r}")

        if not hasattr(module, class_name):
            raise AgentLoadError(
                f"agent.py for {student_id} must define a '{class_name}' class."
            )

        agent_class = getattr(module, class_name)

        if not issubclass(agent_class, expected_parent):
            raise AgentLoadError(
                f"{class_name} in {student_id} must inherit from "
                f"{expected_parent.__name__} (from agent_interface)."
            )

        try:
            agent = agent_class(**init_kwargs_final)
        except Exception as e:
            raise AgentLoadError(
                f"Failed to instantiate {class_name} for {student_id}: "
                f"{type(e).__name__}: {e}"
            )

        return agent

    def validate_agent_move(self, move, agent_type: str, student_id: str, pacman_speed: Optional[int] = None):
        """Validate that an agent's move is legal."""
        from environment import Move

        if agent_type.lower() == 'pacman':
            return self._validate_pacman_action(move, student_id, pacman_speed)

        if not isinstance(move, Move):
            raise AgentLoadError(
                f"Agent {student_id} ({agent_type}) returned invalid move type: {type(move)}. "
                f"Must return a Move enum value."
            )
        return move

    def _validate_pacman_action(self, action, student_id: str, pacman_speed: Optional[int]) -> Tuple['Move', int]:
        from environment import Move

        if isinstance(action, Move):
            move = action
            steps = 1
        elif isinstance(action, tuple) and len(action) == 2:
            move, steps = action
        else:
            raise AgentLoadError(
                f"Agent {student_id} (pacman) must return a Move or a (Move, steps) tuple. "
                f"Got {action!r}."
            )

        if not isinstance(move, Move):
            raise AgentLoadError(
                f"Agent {student_id} (pacman) returned invalid move component: {move!r}."
            )

        try:
            steps = int(steps)
        except (TypeError, ValueError):
            raise AgentLoadError(
                f"Agent {student_id} (pacman) provided non-integer steps value: {steps!r}."
            )

        if steps < 1:
            raise AgentLoadError(
                f"Agent {student_id} (pacman) must request at least 1 step."
            )

        if pacman_speed is not None and steps > pacman_speed:
            raise AgentLoadError(
                f"Agent {student_id} (pacman) requested {steps} steps which exceeds the maximum speed {pacman_speed}."
            )

        return move, steps
