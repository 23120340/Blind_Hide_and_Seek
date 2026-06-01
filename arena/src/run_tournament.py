"""
Tournament runner for local Pacman/Ghost agent comparison.

Run from arena/src:
    python run_tournament.py --main codex_agent
    python run_tournament.py --candidates codex_agent lookahead_agent
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
SUBMISSIONS_DIR = ROOT / "submissions"
RESULTS_DIR = ROOT / "results"
ARENA_SCRIPT = SRC_DIR / "arena.py"

EXCLUDED_DEFAULT = {
    "__pycache__",
    "broken_agent",
    "exit_test",
    "slow_agent",
}


@dataclass
class Match:
    seek: str
    hide: str
    winner: str
    steps: int | None
    raw: str


def cleanup_old_result_files(prefix: str, keep_path: Path) -> None:
    for path in RESULTS_DIR.glob(f"{prefix}_*.txt"):
        if path.resolve() != keep_path.resolve():
            path.unlink()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def parse_winner(output: str) -> str:
    lower = output.lower()

    if re.search(r"(winner:|\[winner\])\s+\S+\s+\(pacman\)", lower):
        return "seek"
    if re.search(r"(winner:|\[winner\])\s+\S+\s+\(ghost\)", lower):
        return "hide"

    if "pacman caught the ghost" in lower or "pacman wins by default" in lower:
        return "seek"
    if "ghost successfully evaded" in lower or "ghost wins by default" in lower:
        return "hide"
    if "failed to load ghost" in lower or ("agent load failed" in lower and "ghost" in lower):
        return "seek"
    if "failed to load pacman" in lower or ("agent load failed" in lower and "pacman" in lower):
        return "hide"
    if re.search(r"\bdraw\b", lower):
        return "draw"
    if "agent timed out" in lower or "process timeout" in lower:
        return "timeout"
    return "unknown"


def parse_steps(output: str) -> int | None:
    patterns = [
        r"Total Steps\s*[:\-]?\s*(\d+)",
        r"total steps?\s*[:\-]?\s*(\d+)",
        r"Game Statistics:[\s\S]*?Total Steps\s*[:\-]?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_agents() -> list[str]:
    agents = []
    for path in SUBMISSIONS_DIR.iterdir():
        if not path.is_dir() or path.name in EXCLUDED_DEFAULT:
            continue
        if (path / "agent.py").exists():
            agents.append(path.name)
    return sorted(agents)


def run_match(args, seek_id: str, hide_id: str) -> Match:
    cmd = [
        sys.executable,
        str(ARENA_SCRIPT),
        "--seek",
        seek_id,
        "--hide",
        hide_id,
        "--submissions-dir",
        str(SUBMISSIONS_DIR),
        "--no-viz",
        "--step-timeout",
        str(args.step_timeout),
        "--max-steps",
        str(args.max_steps),
        "--capture-distance",
        str(args.capture_distance),
        "--pacman-speed",
        str(args.pacman_speed),
        "--pacman-obs-radius",
        str(args.pacman_obs_radius),
        "--ghost-obs-radius",
        str(args.ghost_obs_radius),
        "--start-mode",
        args.start_mode,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd,
            cwd=SRC_DIR,
            capture_output=True,
            text=True,
            timeout=args.process_timeout,
            env=env,
        )
        output = strip_ansi(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return Match(seek_id, hide_id, "timeout", None, "PROCESS TIMEOUT")
    except Exception as exc:
        return Match(seek_id, hide_id, "error", None, repr(exc))

    return Match(seek_id, hide_id, parse_winner(output), parse_steps(output), output)


def summarize(candidate: str, matches: list[tuple[Match, str]]) -> dict:
    wins = losses = unknown = 0
    seek_steps = []
    hide_steps = []

    for match, role in matches:
        if role == "seek":
            if match.winner == "seek":
                wins += 1
            elif match.winner in {"hide", "draw"}:
                losses += 1
        else:
            if match.winner == "hide":
                wins += 1
            elif match.winner in {"seek", "draw"}:
                losses += 1

        if match.winner not in {"seek", "hide", "draw"}:
            unknown += 1

        if role == "seek" and match.steps is not None:
            seek_steps.append(match.steps)
        if role == "hide" and match.steps is not None:
            hide_steps.append(match.steps)

    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.0
    avg_seek = sum(seek_steps) / len(seek_steps) if seek_steps else 0.0
    avg_hide = sum(hide_steps) / len(hide_steps) if hide_steps else 0.0

    # Primary: win rate. Tie-breaker mirrors the assignment idea:
    # seek should finish fast, hide should survive long.
    rank_score = win_rate * 1000 + avg_hide - avg_seek
    return {
        "candidate": candidate,
        "total": len(matches),
        "wins": wins,
        "losses": losses,
        "unknown": unknown,
        "win_rate": win_rate,
        "avg_seek": avg_seek,
        "avg_hide": avg_hide,
        "tie_score": avg_hide - avg_seek,
        "rank_score": rank_score,
    }


def format_line(match: Match, role: str) -> str:
    opponent = match.hide if role == "seek" else match.seek
    if role == "seek":
        result = "WIN" if match.winner == "seek" else "LOSE"
        if match.winner == "draw":
            result = "DRAW"
    else:
        result = "WIN" if match.winner == "hide" else "LOSE"
        if match.winner == "draw":
            result = "DRAW"

    if match.winner not in {"seek", "hide", "draw"}:
        result = match.winner.upper()

    steps = f"{match.steps:>3} steps" if match.steps is not None else "N/A steps"
    return f"  vs {opponent:<18} [{role.upper():4}] {result:<8} {steps}"


def evaluate_candidate(args, candidate: str, opponents: list[str]) -> tuple[dict, str]:
    lines = [f"\nCandidate: {candidate}"]
    matches = []

    for opponent in opponents:
        if opponent == candidate:
            continue

        seek_match = run_match(args, candidate, opponent)
        matches.append((seek_match, "seek"))
        lines.append(format_line(seek_match, "seek"))

        hide_match = run_match(args, opponent, candidate)
        matches.append((hide_match, "hide"))
        lines.append(format_line(hide_match, "hide"))

    summary = summarize(candidate, matches)
    lines.append(
        "  Summary: "
        f"wins={summary['wins']}, losses={summary['losses']}, "
        f"win_rate={summary['win_rate']:.1f}%, "
        f"avg_seek={summary['avg_seek']:.1f}, "
        f"avg_hide={summary['avg_hide']:.1f}, "
        f"tie={summary['tie_score']:.1f}"
    )
    return summary, "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Pacman/Ghost agents.")
    parser.add_argument("--main", default="codex_agent", help="Single candidate to evaluate.")
    parser.add_argument("--candidates", nargs="*", help="Evaluate multiple candidates.")
    parser.add_argument("--opponents", nargs="*", help="Opponent agent names. Default: all discovered agents.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-timeout", type=float, default=1.0)
    parser.add_argument("--process-timeout", type=float, default=120.0)
    parser.add_argument("--capture-distance", type=int, default=2)
    parser.add_argument("--pacman-speed", type=int, default=2)
    parser.add_argument("--pacman-obs-radius", type=int, default=5)
    parser.add_argument("--ghost-obs-radius", type=int, default=5)
    parser.add_argument("--start-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--include-risky", action="store_true", help="Include broken/slow/exit agents in auto-discovery.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    discovered = discover_agents()
    if args.include_risky:
        discovered = [
            path.name
            for path in SUBMISSIONS_DIR.iterdir()
            if path.is_dir() and (path / "agent.py").exists()
        ]
        discovered.sort()

    candidates = args.candidates if args.candidates else [args.main]
    opponents = args.opponents if args.opponents else discovered

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"tournament_{now}.txt"

    header = (
        f"Blind Hide and Seek Tournament\n"
        f"Time: {now}\n"
        f"Candidates: {', '.join(candidates)}\n"
        f"Opponents: {', '.join(opponents)}\n"
        f"Settings: max_steps={args.max_steps}, speed={args.pacman_speed}, "
        f"capture={args.capture_distance}, obs=({args.pacman_obs_radius}, {args.ghost_obs_radius}), "
        f"start={args.start_mode}\n"
    )

    print(header)
    blocks = [header]
    summaries = []

    for candidate in candidates:
        summary, block = evaluate_candidate(args, candidate, opponents)
        summaries.append(summary)
        blocks.append(block)
        print(block)

    summaries.sort(key=lambda item: item["rank_score"], reverse=True)
    ranking_lines = ["\nRanking:"]
    for idx, item in enumerate(summaries, start=1):
        ranking_lines.append(
            f"{idx}. {item['candidate']:<18} "
            f"win_rate={item['win_rate']:>5.1f}% "
            f"wins={item['wins']:>2}/{item['wins'] + item['losses']:<2} "
            f"avg_seek={item['avg_seek']:>5.1f} "
            f"avg_hide={item['avg_hide']:>5.1f} "
            f"tie={item['tie_score']:>6.1f}"
        )

    ranking = "\n".join(ranking_lines)
    print(ranking)
    blocks.append(ranking)

    result_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    (RESULTS_DIR / "latest_results.txt").write_text("\n".join(blocks) + "\n", encoding="utf-8")
    cleanup_old_result_files("tournament", result_path)
    print(f"\nSaved: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
