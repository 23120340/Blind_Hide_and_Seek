"""Shared data structures for POMCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from environment import Move

Position = Tuple[int, int]
Action = Tuple[Move, int]


@dataclass
class Node:
    visits: int = 0
    value: float = 0.0
    children: Dict[Action, "Node"] = field(default_factory=dict)
    untried: List[Action] = field(default_factory=list)
