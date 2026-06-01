"""Weight sweep utility for hide-and-seek heuristic tuning.
Usage (from repository root):
    python submissions/agent/weight_sweep.py --games-per-role 4
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

RE_WINNER = re.compile(r"WINNER:\s+(.+)")
RE_STEPS = re.compile(r"Total Steps:\s+(\d+)")


def _parse_csv_floats(raw: str) -> List[float]:
    values: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _run_match(
    src_dir: Path,
    seek_id: str,
    hide_id: str,
    step_timeout: float,
    max_steps: int,
    subprocess_timeout: float,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[int], str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "arena.py",
        "--seek",
        seek_id,
        "--hide",
        hide_id,
        "--no-viz",
        "--start-mode",
        "stochastic",
        "--max-steps",
        str(max_steps),
        "--step-timeout",
        str(step_timeout),
    ]

    effective_timeout = float(subprocess_timeout)
    if effective_timeout <= 0:
        # Approximate worst-case runtime with a safety factor.
        effective_timeout = max(120.0, (float(max_steps) * max(0.1, float(step_timeout)) * 2.5) + 30.0)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(src_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial = f"{exc.stdout or ''}\n{exc.stderr or ''}".strip()
        timeout_output = (
            f"MATCH TIMEOUT after {effective_timeout:.1f}s for seek={seek_id} hide={hide_id}\n"
            f"{partial}"
        )
        return None, None, timeout_output

    output = f"{completed.stdout}\n{completed.stderr}"
    winner = None
    steps = None

    winner_match = RE_WINNER.search(output)
    if winner_match:
        winner = winner_match.group(1).strip()

    steps_match = RE_STEPS.search(output)
    if steps_match:
        steps = int(steps_match.group(1))

    return winner, steps, output


def _evaluate_combo(
    src_dir: Path,
    w1: float,
    w2: float,
    w3: float,
    games_per_role: int,
    step_timeout: float,
    max_steps: int,
    subprocess_timeout: float,
) -> Dict[str, float]:
    seek_wins = 0
    seek_steps: List[int] = []

    ghost_wins = 0
    ghost_steps: List[int] = []
    timeout_count = 0

    for _ in range(games_per_role):
        winner, steps, _ = _run_match(
            src_dir,
            "agent",
            "example_student",
            step_timeout,
            max_steps,
            subprocess_timeout,
            {
                "PACMAN_W1": str(w1),
                "PACMAN_W2": str(w2),
                "PACMAN_W3": str(w3),
            },
        )
        if winner == "agent (Pacman)":
            seek_wins += 1
        if steps is not None:
            seek_steps.append(steps)
        else:
            timeout_count += 1

    for _ in range(games_per_role):
        winner, steps, _ = _run_match(
            src_dir,
            "example_student",
            "agent",
            step_timeout,
            max_steps,
            subprocess_timeout,
            {
                "GHOST_W1": str(w1),
                "GHOST_W2": str(w2),
                "GHOST_W3": str(w3),
            },
        )
        if winner == "agent (Ghost)":
            ghost_wins += 1
        if steps is not None:
            ghost_steps.append(steps)
        else:
            timeout_count += 1

    seek_win_rate = seek_wins / max(1, games_per_role)
    ghost_win_rate = ghost_wins / max(1, games_per_role)

    avg_seek_steps = (
        float(pd.Series(seek_steps, dtype="float64").mean())
        if seek_steps
        else float(max_steps)
    )
    avg_ghost_steps = (
        float(pd.Series(ghost_steps, dtype="float64").mean())
        if ghost_steps
        else 0.0
    )

    # Higher is better: win rates dominate; survival and capture speed provide tie-break.
    step_scale = float(max(1, max_steps))
    score = (
        (2.0 * seek_win_rate)
        + (1.0 * ghost_win_rate)
        + (0.5 * (avg_ghost_steps / step_scale))
        - (0.5 * (avg_seek_steps / step_scale))
    )

    return {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "seek_win_rate": seek_win_rate,
        "ghost_win_rate": ghost_win_rate,
        "avg_seek_steps": avg_seek_steps,
        "avg_ghost_steps": avg_ghost_steps,
        "timeout_count": float(timeout_count),
        "score": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hide-and-seek heuristic weight sweep")
    parser.add_argument("--games-per-role", type=int, default=4)
    parser.add_argument("--w1", type=str, default="2.5,3.0,3.5")
    parser.add_argument("--w2", type=str, default="0.8,1.2,1.6")
    parser.add_argument("--w3", type=str, default="1.2,1.8,2.4")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=1.0,
        help="Per-agent timeout passed to arena (seconds).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per game passed to arena.",
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=float,
        default=0.0,
        help="Hard timeout for each subprocess game; <=0 uses auto estimate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="logs/hide_seek_weight_sweep.csv",
        help="Path to save full sweep results as CSV",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"

    w1_values = _parse_csv_floats(args.w1)
    w2_values = _parse_csv_floats(args.w2)
    w3_values = _parse_csv_floats(args.w3)

    combos = list(itertools.product(w1_values, w2_values, w3_values))
    results: List[Dict[str, float]] = []

    print(f"Running {len(combos)} combos, {args.games_per_role} games per role...")

    for idx, (w1, w2, w3) in enumerate(combos, start=1):
        result = _evaluate_combo(
            src_dir,
            w1,
            w2,
            w3,
            args.games_per_role,
            args.step_timeout,
            args.max_steps,
            args.subprocess_timeout,
        )
        results.append(result)
        print(
            f"[{idx:02d}/{len(combos):02d}] "
            f"W=({w1:.2f},{w2:.2f},{w3:.2f}) "
            f"seek={result['seek_win_rate']:.2f} "
            f"ghost={result['ghost_win_rate']:.2f} "
            f"score={result['score']:.3f}"
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results produced.")
        return 1

    results_df = results_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    top_count = max(1, int(args.top))
    top_df = results_df.head(top_count).copy()

    display_df = top_df.copy()
    for col in ["w1", "w2", "w3", "seek_win_rate", "ghost_win_rate", "score"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.3f}")
    for col in ["avg_seek_steps", "avg_ghost_steps"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.1f}")

    print("\nTop results:")
    print(display_df.to_string(index=False))

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved full sweep table to: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
