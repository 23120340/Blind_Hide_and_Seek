# Pacman vs Ghost Arena - Student Guide

Welcome to the Pacman vs Ghost Arena! This guide will help you implement your own AI agents using search algorithms.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Understanding the Game](#understanding-the-game)
4. [Creating Your Agent](#creating-your-agent)
5. [Implementing Search Algorithms](#implementing-search-algorithms)
6. [Testing Your Agent](#testing-your-agent)
7. [Debugging Tips](#debugging-tips)
8. [Common Errors](#common-errors)
9. [Advanced Strategies](#advanced-strategies)

---

## Quick Start

### 1. Create Your Submission Folder

```bash
cd submissions
mkdir <your_student_id>
```

Replace `<your_student_id>` with your actual student ID (e.g., `student_001`, `alice`, `john_doe`)

### 2. Copy the Template

```bash
cp TEMPLATE_agent.py <your_student_id>/agent.py
```

### 3. Edit Your Agent

Open `submissions/<your_student_id>/agent.py` in your favorite editor and implement your search algorithm.

### 4. Test Your Agent

```bash
cd ../src
python arena.py --seek <your_student_id> --hide example_student
```

---

## Installation

### Prerequisites

- **Python 3.7+** (Python 3.11 recommended)
- **Conda** environment manager
- **NumPy** library

### Setup Steps

```bash
# 1. Activate conda environment
conda activate ml

# 2. Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` contains:
```text
numpy>=1.20.0
```

### Verify Installation

Test that everything works:

```bash
cd src
python arena.py --seek example_student --hide example_student
```

You should see a colorful visualization of Pacman (blue) chasing Ghost (red) in a maze!

---

## Understanding the Game

### Objective

- **Pacman (Seeker)**: Catch the Ghost by moving to within the capture distance (reaches the same position by default, but this threshold can be configured).
- **Ghost (Hider)**: Evade Pacman for as long as possible (survive until max steps).

### Win Conditions

- **Pacman wins**: Catches Ghost before the maximum steps are reached.
- **Ghost wins**: Survives for max steps (default: 200) without being caught.

### The Map

The game is played on a grid maze, and you receive the current layout as a 2D numpy array:

- `0` = Empty space (you can move here)
- `1` = Wall (you cannot move here)
- `-1` = Unseen/Fog of war (you cannot see what is here due to limited observation radius)

### Movement

You can move in 5 directions by returning a `Move` enum:

```python
Move.UP      # Move up    (row - 1, col)
Move.DOWN    # Move down  (row + 1, col)
Move.LEFT    # Move left  (row, col - 1)
Move.RIGHT   # Move right (row, col + 1)
Move.STAY    # Don't move (row, col)
```

**Advanced Pacman Movement:**
While Ghost agents must always return a single `Move` enum, Pacman agents have access to a straight-path speed multiplier. If the aren is configured with `pacman_speed > 1`, Pacman can choose to move multiple steps in the same direction in a single turn! To do this, a Pacman agent can instead return a tuple `(Move, steps)` where `steps` is an integer between 1 and the maximum allowed speed.

### Important: Synchronous Execution

**Both agents move at the SAME time!**

- Both receive the state simultaneously
- Both decide their moves at the same time
- Both positions update at once

This means you cannot react to your opponent's move instantly - you must predict it!

### Game Information You Receive

Every step, your `step()` method receives:

1. **`map_state`**: 2D numpy array of the maze (`0`=empty, `1`=wall, `-1`=unseen)
2. **`my_position`**: Your current position as `(row, col)`
3. **`enemy_position`**: Enemy's current position as `(row, col)` if they are visible, or **`None`** if they are outside your observation radius (when fog of war is enabled)
4. **`step_number`**: Current step number in the game (starts at 1)

---

## Creating Your Agent

### Required Code Structure

Your `agent.py` must define a `PacmanAgent` and/or `GhostAgent` class:

```python
import sys
from pathlib import Path

# Add src to path so you can import framework classes
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """Your Pacman (Seeker) implementation."""
    
    def __init__(self, **kwargs):
        """Initialize your agent. Called once at game start."""
        super().__init__(**kwargs)
        # Initialize your data structures here
        # Example: self.path_cache = {}
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Called every step. Return your move decision.
        
        Args:
            map_state: numpy array (height x width), 0=empty, 1=wall, -1=unseen
            my_position: (row, col) tuple of Pacman's position
            enemy_position: (row, col) tuple of Ghost's position or None if unseen
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or tuple (Move, int): One of the Move enums OR a tuple of
            the Move enum and number of steps (if pacman_speed > 1).
        """
        # Handle fog of war gracefully (when enemy is not visible)
        if enemy_position is None:
            # Implement your exploration/searching logic here
            return Move.STAY
        
        # Example: Simple greedy move toward enemy
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        if abs(row_diff) > abs(col_diff):
            if row_diff > 0:
                return Move.DOWN
            else:
                return Move.UP
        else:
            if col_diff > 0:
                return Move.RIGHT
            else:
                return Move.LEFT


class GhostAgent(BaseGhostAgent):
    """Your Ghost (Hider) implementation."""
    
    def __init__(self, **kwargs):
        """Initialize your agent. Called once at game start."""
        super().__init__(**kwargs)
        # Initialize your data structures here
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Called every step. Return your move decision.
        
        Args:
            map_state: numpy array, 0=empty, 1=wall, -1=unseen
            my_position: (row, col) tuple of Ghost's position
            enemy_position: (row, col) tuple of Pacman's position or None
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of the Move enums (UP, DOWN, LEFT, RIGHT, STAY)
        """
        # Handle fog of war gracefully 
        if enemy_position is None:
            # Ghost should keep moving unpredictably when Pacman is out of sight!
            return Move.STAY
        
        # Example: Simple greedy move away from enemy
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        if abs(row_diff) > abs(col_diff):
            if row_diff > 0:
                return Move.UP  # Move away
            else:
                return Move.DOWN
        else:
            if col_diff > 0:
                return Move.LEFT  # Move away
            else:
                return Move.RIGHT
```

### Essential Helper Functions

Add these helper methods to your agent class to quickly parse the grid layout:

```python
def _is_valid_position(self, pos, map_state):
    """Check if position is valid (not wall or unseen boundaries)."""
    row, col = pos
    height, width = map_state.shape
    
    # Check bounds
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    
    # Check not a wall 
    # (Optional: treat unseen (-1) carefully based on your strategy!)
    return map_state[row, col] == 0


def _apply_move(self, pos, move):
    """Apply a move to a position, return new position."""
    delta_row, delta_col = move.value
    return (pos[0] + delta_row, pos[1] + delta_col)


def _get_neighbors(self, pos, map_state):
    """Get all valid neighboring positions and their moves."""
    neighbors = []
    
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        next_pos = self._apply_move(pos, move)
        if self._is_valid_position(next_pos, map_state):
            neighbors.append((next_pos, move))
    
    return neighbors


def _manhattan_distance(self, pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

---

## Implementing Search Algorithms

### Breadth-First Search (BFS)

**Best for Pacman** - Finds the shortest path to the Ghost. You can implement a standard BFS using a queue to track visited nodes and find the optimal sequence of moves.

### A* Search

**Best for Pacman** - Optimal path with heuristic guidance. Use a priority queue with a heuristic (like Manhattan distance) to estimate the cost to the goal, making the search significantly faster than BFS.

### Greedy Best-First Search

**Faster than A*** but not always optimal. Explores the map based entirely on the heuristic distance.

### Evasion Strategy for Ghost

**Maximize distance from Pacman** - Since the ghost wants to stay alive, simple strategies might include finding the immediately adjacent move that maximizes the Manhattan distance. 

**Find furthest reachable position** - Advanced strategies involve running a BFS to map out reachable distances and heading toward the furthest accessible tile.

---

## Testing Your Agent

### Basic Testing

```bash
# From src directory
cd src

# Test your Pacman against example Ghost
python arena.py --seek <your_id> --hide example_student

# Test your Ghost against example Pacman
python arena.py --seek example_student --hide <your_id>

# Test both your agents against each other
python arena.py --seek <your_id> --hide <your_id>
```

### Advanced Testing Options

You can specify additional game mechanics such as the capture distance threshold, speeds, and the fog-of-war observation conditions directly from the command line:

```bash
# Faster testing (no visualization)
python arena.py --seek <your_id> --hide example_student --no-viz

# Slower visualization for debugging (1 second delays)
python arena.py --seek <your_id> --hide example_student --delay 1.0

# Adjust max steps (longer game: 300, shorter game: 50)
python arena.py --seek <your_id> --hide example_student --max-steps 300

# Stochastic starting mode (random starting positions instead of classic starts)
python arena.py --seek <your_id> --hide example_student --start-mode stochastic

# Adjust capture distance (catch ghost when distance < 3 instead of default)
python arena.py --seek <your_id> --hide example_student --capture-distance 3

# Allow Pacman an advanced speed multiplier 
# (You may return e.g. (Move.UP, 2) in your PacmanAgent if this is > 1)
python arena.py --seek <your_id> --hide example_student --pacman-speed 2

# Limit observation visibility (test Fog of War)
python arena.py --seek <your_id> --hide example_student --pacman-obs-radius 5 --ghost-obs-radius 3
```

### Using the Run Script

From the Arena directory:

```bash
./run_game.sh --seek <your_id> --hide example_student
```

---

## Debugging Tips

### 1. Add Print Statements

```python
def step(self, map_state, my_position, enemy_position, step_number):
    print(f"Step {step_number}: My pos={my_position}, Enemy pos={enemy_position}")
    
    if enemy_position is not None:
        path = self.bfs(my_position, enemy_position, map_state)
        print(f"  Found path: {[m.name for m in path[:5]]}")  # First 5 moves
        
        if path:
            return path[0]
            
    return Move.STAY
```

### 2. Watch Visualization Slowly

```bash
python arena.py --seek <your_id> --hide example_student --delay 1.0
```

### 3. Test Edge Cases

```python
# Test helper functions independently
agent = PacmanAgent()
test_pos = (10, 10)
test_map = np.zeros((21, 21))  # Empty map

# Test is_valid_position
assert agent._is_valid_position(test_pos, test_map) == True
assert agent._is_valid_position((-1, 10), test_map) == False

# Test apply_move
new_pos = agent._apply_move(test_pos, Move.UP)
assert new_pos == (9, 10)

print("All tests passed!")
```

### 4. Check for Infinite Loops

Make sure your search algorithms terminate:

```python
def bfs(self, start, goal, map_state):
    queue = deque([(start, [])])
    visited = {start}  # IMPORTANT: Track visited nodes!
    
    max_iterations = 10000  # Safety limit
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        # ... rest of BFS
        
    if iterations >= max_iterations:
        print("WARNING: BFS hit iteration limit!")
    
    return [Move.STAY]
```

### 5. Validate Return Values

```python
def step(self, map_state, my_position, enemy_position, step_number):
    move = self.calculate_best_move(...)
    
    # Validate before returning
    if not isinstance(move, Move) and not isinstance(move, tuple):
        print(f"ERROR: Invalid move type: {type(move)}")
        return Move.STAY
    
    return move
```

---

## Common Errors

### Error: "Agent file not found"

**Cause:** Folder name doesn't match the ID you used in command

**Solution:**
```bash
# Check your folder name
ls submissions/

# Make sure it matches exactly
python arena.py --seek exact_folder_name --hide example_student
```

### Error: "Must define a 'PacmanAgent' class"

**Cause:** Class name is wrong or misspelled

**Solution:**
- Class must be named **exactly** `PacmanAgent` or `GhostAgent`
- Check for typos: `PacMan`, `Pacman`, `pacmanAgent` are all WRONG
- Make sure class inherits: `class PacmanAgent(BasePacmanAgent):`

### Error: "Returned invalid move type"

**Cause:** Returning wrong type (string, tuple, None, etc.)

**Solution:**
```python
# ❌ WRONG
return "UP"       # Wrong type
return (0, -1)    # Returning coordinates
return None       # Unhandled fallback

# ✅ CORRECT for Ghost and Pacman
return Move.UP

# ✅ CORRECT for Pacman only (if max speed is configured)
return (Move.UP, 2)
```

### Error: "Pacman requested N steps which exceeds maximum speed"

**Cause:** Returning a speed multiplier tuple where the steps integer is greater than `--pacman-speed` (which defaults to 1). 
**Solution:** Ensure `(Move, steps)` enforces `1 <= steps <= max_speed`.

### Error: Agent crashes with IndexError

**Cause:** Accessing map positions without validation or trying to read coordinates outside boundaries.

**Solution:**
```python
# Always validate before accessing
if self._is_valid_position(pos, map_state):
    value = map_state[pos[0], pos[1]]  # Safe
```

### Error: Agent crashes with TypeError when calculating distances

**Cause:** You may be treating `enemy_position` as a constant, but it turned into `None` due to the observation radius (Fog of War) hiding the enemy.

**Solution:**
```python
# Validate that enemy is present before calculating distances
if enemy_position is not None:
    distance = self._manhattan_distance(my_position, enemy_position)
```

### Error: Agent takes too long / timeout

**Cause:** Inefficient algorithm or infinite loop.

**Solution:**
1. Add visited set to prevent revisiting nodes
2. Limit search depth
3. Use better data structures (heap for A*, deque for BFS)
4. Add iteration limit for safety

---

## Advanced Strategies

### Strategy 1: Path Replanning

Don't recompute the entire path every step. Cache your previous path and only replan if the enemy's position has significantly changed or if the current path runs out.

### Strategy 2: Predictive Movement

Instead of aiming for the enemy's current tile, predict where the enemy will be next. Since Ghosts often try to maximize distance, you can simulate their next move and intercept them.

### Strategy 3: Adversarial Search (Minimax)

You can model the game as a minimizing and maximizing player. Minimax (optionally with Alpha-Beta pruning) allows you to search a few steps ahead to evaluate potential adversarial interactions.

---

## Quick Reference

### Available Moves

```python
Move.UP      # (row-1, col)
Move.DOWN    # (row+1, col)
Move.LEFT    # (row, col-1)
Move.RIGHT   # (row, col+1)
Move.STAY    # (row, col)
```

### Input Parameters to `step()`

- `map_state`: 2D numpy array (`0`=empty, `1`=wall, `-1`=unseen)
- `my_position`: (row, col) tuple
- `enemy_position`: (row, col) tuple, or `None` if hidden by Fog of War.
- `step_number`: int (starts at 1)

### Essential Helper Functions

```python
_is_valid_position(pos, map_state)  # Check if position is valid
_apply_move(pos, move)               # Apply move to position
_get_neighbors(pos, map_state)       # Get valid neighbors
_manhattan_distance(pos1, pos2)      # Calculate distance
```

### Common Algorithms

- **BFS**: Shortest path (optimal for Pacman)
- **A\***: Optimal with heuristic (efficient for Pacman)
- **Greedy**: Fast but not optimal
- **Minimax**: Adversarial search (good for Ghost)

---

## Checklist Before Submission

- [ ] Agent loads without errors
- [ ] Agent doesn't crash during game
- [ ] Agent makes valid moves (returns Move enum, or appropriate tuple for Pacman speed multiplier)
- [ ] Handles unseen areas (`-1`) and `enemy_position` being `None` appropriately
- [ ] Agent performs better than random
- [ ] Agent handles being trapped in corners
- [ ] Agent works for full game length (200 steps)
- [ ] Code is well-documented with comments
- [ ] No print statements in final version (or minimal)

---

## Getting Help

1. **Read this guide** thoroughly.
2. **Check example_student/agent.py** for working baseline code.
3. **Test with `--delay 0.5`** to see what your agent is doing in real-time.
4. **Use print statements** to trace variables, especially `enemy_position`.
5. **Ask your instructor or TA** if you get stuck.

---

## Good Luck! 🎮

Have fun implementing your AI agent! Remember:

- Start simple (get it working first, then add predictions/fogs)
- Test frequently
- Improve incrementally
- Learn from mistakes
- Compete fairly and have fun!

**May the best algorithm win!** 🏆
