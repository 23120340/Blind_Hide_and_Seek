# Blind Hide and Seek

AI agents for the Blind Adversary / Hide and Seek Arena assignment.

The project is organized so the original framework, custom agents, benchmark runner, results, and submission zips are easy to find.

## Structure

```text
Blind_Hide_and_Seek/
|-- README.md
|-- docs/
|   `-- BlindArena-2526-2.pdf
|-- archives/
|   |-- pacman.zip
|   |-- main_agent.zip
|   `-- *.zip
`-- arena/
    |-- requirements.txt
    |-- results/
    |   `-- latest_results.txt
    |-- src/
    |   |-- arena.py
    |   |-- run_tournament.py
    |   |-- environment.py
    |   |-- agent_loader.py
    |   |-- agent_interface.py
    |   `-- visualizer.py
    `-- submissions/
        |-- main_agent/
        |-- codex_agent/
        |-- lookahead_agent/
        |-- hybrid_agent/
        |-- frontier_agent/
        |-- trap_agent/
        |-- montecarlo_agent/
        |-- random_agent/
        `-- example_student/
```

## Main Agent

Use `main_agent` as the current submission candidate.

Path:

```text
arena/submissions/main_agent/agent.py
```

Zip:

```text
archives/main_agent.zip
```

### Design

`main_agent` is a hybrid chosen after comparing several strategies.

It also contains a fixed-map optimization. When the observed wall layout and the start position match the official default map, the agent enables a prior:

- Pacman assumes the Ghost starts at `(9, 10)`.
- Ghost assumes Pacman starts at `(15, 10)`.
- Both roles load the full default wall map instead of slowly rediscovering empty corridors.
- Ghost uses an opening book: `RIGHT, RIGHT, RIGHT, UP, UP, LEFT`, which exits the central row through the right-side vertical corridor before normal safety scoring takes over.

Pacman role: `Main Frontier Hunter`

- Maintains a memory map across turns.
- Updates known cells from the partial 21 x 21 observation.
- If Ghost is visible, runs BFS to the Ghost and uses Pacman's speed 2 when the first path segment is straight.
- If fixed-map mode is active and Ghost is not visible, first goes toward the known/last-known Ghost start. Otherwise selects a frontier target: a known reachable cell near unknown territory, penalized by distance and repeated visits.
- Falls back to the least repeated/high-information visible move.

Ghost role: `Main Escape Hider`

- Maintains memory, visit counts, last known Pacman position, and recent positions.
- Scores each legal move plus `STAY` after the fixed opening ends.
- Penalizes cells Pacman can reach in one speed-2 straight move.
- Rewards high maze distance from Pacman, high local escape options, and short-horizon escape potential.
- Penalizes dead ends, corridors without nearby junctions, recent loops, and standing still.

This is intentionally not a deep minimax tree. The arena is small, but each step must stay under the time limit, so the final agent uses shallow tactical reasoning plus reliable BFS-style search.

## Compared Agents

`codex_agent`: memory map + A*/BFS + heuristic scoring. Stable in partial observability, but slower at catching strong hiders.

`lookahead_agent`: shallow simultaneous-move lookahead. Very fast seeker, but hider performance varies by start.

`hybrid_agent`: lookahead Pacman + stronger safety Ghost. Best in one stochastic benchmark, but weaker than `main_agent` in deterministic round-robin.

`frontier_agent`: frontier exploration seeker + farthest safe cell hider. Strong seeker and useful source for the final Pacman design.

`trap_agent`: tries to chase positions that reduce Ghost escape routes. Good at quick captures, weaker as hider.

`montecarlo_agent`: short random rollouts. Good in selected stochastic cases, but slower and inconsistent against stronger agents.

`random_agent`: valid random baseline.

## Benchmark Commands

Run one match:

```bash
python arena/src/arena.py --seek main_agent --hide example_student --submissions-dir arena/submissions --no-viz --pacman-speed 2 --capture-distance 2 --pacman-obs-radius 5 --ghost-obs-radius 5
```

Compare all custom agents against simple baselines:

```bash
python arena/src/run_tournament.py --candidates codex_agent lookahead_agent hybrid_agent frontier_agent trap_agent montecarlo_agent --opponents example_student random_agent
```

Fair deterministic round-robin for the strongest candidates:

```bash
python arena/src/run_tournament.py --candidates main_agent lookahead_agent frontier_agent codex_agent hybrid_agent --opponents main_agent lookahead_agent frontier_agent codex_agent hybrid_agent trap_agent montecarlo_agent example_student random_agent
```

Fair stochastic round-robin:

```bash
python arena/src/run_tournament.py --candidates main_agent lookahead_agent frontier_agent codex_agent hybrid_agent --opponents main_agent lookahead_agent frontier_agent codex_agent hybrid_agent trap_agent montecarlo_agent example_student random_agent --start-mode stochastic
```

Results are written to:

```text
arena/results/latest_results.txt
```

## Latest Results

Settings used:

```text
max_steps=200
pacman_speed=2
capture_distance=2
pacman_obs_radius=5
ghost_obs_radius=5
```

### Deterministic Round-Robin

Result file:

```text
arena/results/tournament_20260518_162735.txt
```

```text
1. main_agent       win_rate=62.5% wins=10/16 avg_seek=6.2  avg_hide=124.6 tie=118.4
2. hybrid_agent     win_rate=56.2% wins= 9/16 avg_seek=16.6 avg_hide=64.6  tie=48.0
3. lookahead_agent  win_rate=56.2% wins= 9/16 avg_seek=16.4 avg_hide=63.4  tie=47.0
4. codex_agent      win_rate=56.2% wins= 9/16 avg_seek=49.6 avg_hide=66.8  tie=17.1
5. frontier_agent   win_rate=56.2% wins= 9/16 avg_seek=16.4 avg_hide=31.5  tie=15.1
```

### Stochastic Round-Robin

Result file:

```text
arena/results/tournament_20260518_155332.txt
```

```text
1. hybrid_agent     win_rate=68.8% wins=11/16 avg_seek=40.0 avg_hide=123.5 tie=83.5
2. lookahead_agent  win_rate=62.5% wins=10/16 avg_seek=23.9 avg_hide=108.2 tie=84.4
3. main_agent       win_rate=62.5% wins=10/16 avg_seek=41.9 avg_hide=107.1 tie=65.2
4. frontier_agent   win_rate=62.5% wins=10/16 avg_seek=47.8 avg_hide=108.0 tie=60.2
5. codex_agent      win_rate=62.5% wins=10/16 avg_seek=77.4 avg_hide=93.0  tie=15.6
```

### Decision

`main_agent` is selected as the primary submission because:

- The current tournament setting is fixed-map/fixed-start, and `main_agent` exploits that with a safe opening book.
- It is still ranked first in the deterministic round-robin after the fixed-map change.
- Its Pacman role is faster than before: average seek steps improved to `6.2`.
- Its Ghost role survives much longer in deterministic tests: average hide steps improved to `124.6`.
- The trade-off is one fewer deterministic win than the earlier non-opening version, but the tie-break profile is much better for fixed-start rounds.

If the final tournament uses mostly random starts and the opponents resemble the current test pool, `hybrid_agent` is worth keeping as a backup candidate. For a balanced unknown tournament, `main_agent` is the safer primary choice.

## Submission

The assignment expects:

```text
group_id/
`-- agent.py
```

To prepare the final zip, copy or rename:

```text
arena/submissions/main_agent/
```

to your group id, for example:

```text
1/agent.py
```

then zip it as:

```text
1.zip
```

## References

- Berkeley/CS188 multi-agent Pacman ideas: Minimax, Alpha-Beta, Expectimax, evaluation functions.  
  https://github.com/philipp-kurz/CS188_P2_Multi-Agent_Search
- Pacman multi-agent implementation notes.  
  https://github.com/taradalaei/Packman-Multi-agent
- Hide and Seek AI competition structure.  
  https://github.com/acmucsd/hide-and-seek-ai
- Pacman contest strategy notes using A* and BFS.  
  https://github-wiki-see.page/m/s3767707/RMIT-COSC1125-1127-AI-22-Pacman-Contest-Project/wiki/AI-Method-1

All agents in this repository are written for this assignment interface. The references were used for algorithm ideas, not copied code.
