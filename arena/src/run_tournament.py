"""
Tournament runner for local Pacman/Ghost agent comparison.

Run from arena/src:
    python run_tournament.py --main codex_agent
    python run_tournament.py --candidates codex_agent lookahead_agent
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
AGENT_HASH_CACHE: dict[str, str | None] = {}

EXCLUDED_DEFAULT = {
    "__pycache__",
    "6",
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


def cleanup_old_result_files(prefix: str, keep_paths: Path | list[Path]) -> None:
    if isinstance(keep_paths, Path):
        keep_paths = [keep_paths]
    keep_paths = {path.resolve() for path in keep_paths}
    for path in RESULTS_DIR.glob(f"{prefix}_*.*"):
        if path.resolve() not in keep_paths:
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


def unique_ordered(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def filter_agents(agent_ids: list[str], include_risky: bool) -> tuple[list[str], list[str]]:
    if include_risky:
        return unique_ordered(agent_ids), []

    kept = []
    skipped = []
    for agent_id in unique_ordered(agent_ids):
        if agent_id in EXCLUDED_DEFAULT:
            skipped.append(agent_id)
        else:
            kept.append(agent_id)
    return kept, skipped


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def agent_hash(agent_id: str) -> str | None:
    if agent_id not in AGENT_HASH_CACHE:
        path = SUBMISSIONS_DIR / agent_id / "agent.py"
        AGENT_HASH_CACHE[agent_id] = sha1_bytes(path.read_bytes()) if path.exists() else None
    return AGENT_HASH_CACHE[agent_id]


def game_settings_dict(args) -> dict:
    return {
        "max_steps": args.max_steps,
        "step_timeout": args.step_timeout,
        "capture_distance": args.capture_distance,
        "pacman_speed": args.pacman_speed,
        "pacman_obs_radius": args.pacman_obs_radius,
        "ghost_obs_radius": args.ghost_obs_radius,
        "start_mode": args.start_mode,
    }


def match_cache_key(args, seek_id: str, hide_id: str) -> str:
    payload = {
        "seek": seek_id,
        "hide": hide_id,
        "seek_hash": agent_hash(seek_id),
        "hide_hash": agent_hash(hide_id),
        "settings": game_settings_dict(args),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha1_bytes(text.encode("utf-8"))


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


def match_record(match: Match, index: int, mode: str, args=None, cached: bool = False) -> dict:
    record = {
        "index": index,
        "mode": mode,
        "seek": match.seek,
        "hide": match.hide,
        "winner": match.winner,
        "steps": match.steps,
        "raw": match.raw,
        "cached": cached,
    }
    if args is not None:
        record.update(
            {
                "cache_key": match_cache_key(args, match.seek, match.hide),
                "seek_hash": agent_hash(match.seek),
                "hide_hash": agent_hash(match.hide),
                "game_settings": game_settings_dict(args),
            }
        )
    return record


def load_match_cache(args) -> dict[str, dict]:
    if args.no_cache:
        return {}
    cache_path = Path(args.cache_detail)
    if not cache_path.is_absolute():
        cache_path = (Path.cwd() / cache_path).resolve()
    if not cache_path.exists():
        return {}

    try:
        detail = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Cache ignored: failed to read {cache_path}: {exc}", file=sys.stderr)
        return {}

    cache = {}
    for record in detail.get("matches", []):
        key = record.get("cache_key")
        if not key:
            continue
        if record.get("winner") not in {"seek", "hide", "draw", "timeout", "error", "unknown"}:
            continue
        cache[key] = record
    return cache


def record_to_match(record: dict) -> Match:
    return Match(
        record["seek"],
        record["hide"],
        record.get("winner", "unknown"),
        record.get("steps"),
        record.get("raw", ""),
    )


def run_or_cache_match(args, seek_id: str, hide_id: str, cache: dict[str, dict]) -> tuple[Match, bool]:
    key = match_cache_key(args, seek_id, hide_id)
    cached = cache.get(key)
    if cached is not None:
        return record_to_match(cached), True
    return run_match(args, seek_id, hide_id), False


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


def evaluate_candidate_with_details(
    args,
    candidate: str,
    opponents: list[str],
    records: list[dict],
) -> tuple[dict, str]:
    lines = [f"\nCandidate: {candidate}"]
    matches = []

    for opponent in opponents:
        if opponent == candidate:
            continue

        seek_match = run_match(args, candidate, opponent)
        records.append(match_record(seek_match, len(records) + 1, "candidate"))
        matches.append((seek_match, "seek"))
        lines.append(format_line(seek_match, "seek"))

        hide_match = run_match(args, opponent, candidate)
        records.append(match_record(hide_match, len(records) + 1, "candidate"))
        matches.append((hide_match, "hide"))
        lines.append(format_line(hide_match, "hide"))

    summary = summarize(candidate, matches)
    lines.append(format_summary(summary))
    return summary, "\n".join(lines)


def format_summary(summary: dict) -> str:
    return (
        "  Summary: "
        f"wins={summary['wins']}, losses={summary['losses']}, "
        f"win_rate={summary['win_rate']:.1f}%, "
        f"avg_seek={summary['avg_seek']:.1f}, "
        f"avg_hide={summary['avg_hide']:.1f}, "
        f"tie={summary['tie_score']:.1f}"
    )


def records_to_match_map(records: list[dict]) -> dict[tuple[str, str], Match]:
    match_map = {}
    for record in records:
        key = (record["seek"], record["hide"])
        if key in match_map:
            continue
        match_map[key] = Match(
            record["seek"],
            record["hide"],
            record.get("winner", "unknown"),
            record.get("steps"),
            record.get("raw", ""),
        )
    return match_map


def evaluate_from_records(candidate: str, opponents: list[str], match_map: dict[tuple[str, str], Match]) -> tuple[dict, str]:
    lines = [f"\nCandidate: {candidate}"]
    matches = []

    for opponent in opponents:
        if opponent == candidate:
            continue

        seek_match = match_map.get((candidate, opponent))
        if seek_match is None:
            seek_match = Match(candidate, opponent, "missing", None, "MISSING")
        matches.append((seek_match, "seek"))
        lines.append(format_line(seek_match, "seek"))

        hide_match = match_map.get((opponent, candidate))
        if hide_match is None:
            hide_match = Match(opponent, candidate, "missing", None, "MISSING")
        matches.append((hide_match, "hide"))
        lines.append(format_line(hide_match, "hide"))

    summary = summarize(candidate, matches)
    lines.append(format_summary(summary))
    return summary, "\n".join(lines)


def evaluate_same_pool(args, pool: list[str], records: list[dict], on_match=None) -> tuple[list[dict], list[str]]:
    matches_by_agent = {agent_id: [] for agent_id in pool}
    total_matches = len(pool) * (len(pool) - 1)
    completed = 0

    for left_idx, left in enumerate(pool):
        for right in pool[left_idx + 1 :]:
            left_seek = run_match(args, left, right)
            records.append(match_record(left_seek, len(records) + 1, "same-pool"))
            matches_by_agent[left].append((left_seek, "seek"))
            matches_by_agent[right].append((left_seek, "hide"))
            completed += 1
            print(f"  [{completed:>3}/{total_matches}] {left} SEEK vs {right} HIDE -> {left_seek.winner} {left_seek.steps}")
            if on_match is not None:
                on_match()

            right_seek = run_match(args, right, left)
            records.append(match_record(right_seek, len(records) + 1, "same-pool"))
            matches_by_agent[right].append((right_seek, "seek"))
            matches_by_agent[left].append((right_seek, "hide"))
            completed += 1
            print(f"  [{completed:>3}/{total_matches}] {right} SEEK vs {left} HIDE -> {right_seek.winner} {right_seek.steps}")
            if on_match is not None:
                on_match()

    summaries = []
    blocks = []
    for candidate in pool:
        lines = [f"\nCandidate: {candidate}"]
        for match, role in matches_by_agent[candidate]:
            lines.append(format_line(match, role))
        summary = summarize(candidate, matches_by_agent[candidate])
        summaries.append(summary)
        lines.append(format_summary(summary))
        blocks.append("\n".join(lines))
    return summaries, blocks


def add_task(tasks_by_key: dict[tuple[str, str], dict], seek: str, hide: str, candidate: str, role: str, order: int) -> None:
    key = (seek, hide)
    if key not in tasks_by_key:
        tasks_by_key[key] = {
            "seek": seek,
            "hide": hide,
            "credits": [],
            "order": order,
        }
    tasks_by_key[key]["credits"].append((candidate, role, order))


def build_match_tasks(candidates: list[str], opponents: list[str], same_pool: bool) -> list[dict]:
    tasks_by_key: dict[tuple[str, str], dict] = {}
    order = 0

    if same_pool:
        for left_idx, left in enumerate(candidates):
            for right in candidates[left_idx + 1 :]:
                order += 1
                add_task(tasks_by_key, left, right, left, "seek", order)
                add_task(tasks_by_key, left, right, right, "hide", order)

                order += 1
                add_task(tasks_by_key, right, left, right, "seek", order)
                add_task(tasks_by_key, right, left, left, "hide", order)
    else:
        for candidate in candidates:
            for opponent in opponents:
                if opponent == candidate:
                    continue

                order += 1
                add_task(tasks_by_key, candidate, opponent, candidate, "seek", order)

                order += 1
                add_task(tasks_by_key, opponent, candidate, candidate, "hide", order)

    return sorted(tasks_by_key.values(), key=lambda item: item["order"])


def execute_tasks(args, tasks: list[dict], mode: str, records: list[dict], on_match=None) -> list[tuple[dict, Match, bool]]:
    cache = load_match_cache(args)
    total = len(tasks)
    completed = 0
    results = []

    def run_task(task: dict) -> tuple[dict, Match, bool]:
        match, cached = run_or_cache_match(args, task["seek"], task["hide"], cache)
        return task, match, cached

    def consume(task: dict, match: Match, cached: bool) -> None:
        nonlocal completed
        completed += 1
        records.append(match_record(match, len(records) + 1, mode, args=args, cached=cached))
        source = "cache" if cached else "run"
        print(f"  [{completed:>3}/{total}] {match.seek} SEEK vs {match.hide} HIDE -> {match.winner} {match.steps} ({source})")
        results.append((task, match, cached))
        if on_match is not None:
            on_match()

    workers = max(1, args.workers)
    if workers == 1 or total <= 1:
        for task in tasks:
            consume(*run_task(task))
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            consume(*future.result())

    return results


def build_blocks_from_task_results(candidates: list[str], results: list[tuple[dict, Match, bool]]) -> tuple[list[dict], list[str]]:
    matches_by_agent = {candidate: [] for candidate in candidates}
    for task, match, _cached in results:
        for candidate, role, order in task["credits"]:
            if candidate in matches_by_agent:
                matches_by_agent[candidate].append((order, match, role))

    summaries = []
    blocks = []
    for candidate in candidates:
        candidate_matches = sorted(matches_by_agent[candidate], key=lambda item: item[0])
        lines = [f"\nCandidate: {candidate}"]
        for _order, match, role in candidate_matches:
            lines.append(format_line(match, role))
        summary = summarize(candidate, [(match, role) for _order, match, role in candidate_matches])
        summaries.append(summary)
        lines.append(format_summary(summary))
        blocks.append("\n".join(lines))
    return summaries, blocks


def format_ranking(summaries: list[dict]) -> str:
    ordered = sorted(summaries, key=lambda item: item["rank_score"], reverse=True)
    ranking_lines = ["\nRanking:"]
    for idx, item in enumerate(ordered, start=1):
        ranking_lines.append(
            f"{idx}. {item['candidate']:<18} "
            f"win_rate={item['win_rate']:>5.1f}% "
            f"wins={item['wins']:>2}/{item['wins'] + item['losses']:<2} "
            f"avg_seek={item['avg_seek']:>5.1f} "
            f"avg_hide={item['avg_hide']:>5.1f} "
            f"tie={item['tie_score']:>6.1f}"
        )
    return "\n".join(ranking_lines)


def settings_dict(args) -> dict:
    return {
        "max_steps": args.max_steps,
        "step_timeout": args.step_timeout,
        "process_timeout": args.process_timeout,
        "workers": args.workers,
        "cache_detail": args.cache_detail,
        "no_cache": args.no_cache,
        "capture_distance": args.capture_distance,
        "pacman_speed": args.pacman_speed,
        "pacman_obs_radius": args.pacman_obs_radius,
        "ghost_obs_radius": args.ghost_obs_radius,
        "start_mode": args.start_mode,
    }


def write_detail(
    args,
    detail_path: Path,
    latest_detail_path: Path,
    now: str,
    mode: str,
    candidates: list[str],
    opponents: list[str],
    summaries: list[dict],
    records: list[dict],
) -> None:
    payload = {
        "schema_version": 1,
        "time": now,
        "mode": mode,
        "candidates": candidates,
        "opponents": opponents,
        "settings": settings_dict(args),
        "summaries": summaries,
        "matches": records,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    detail_path.write_text(text, encoding="utf-8")
    latest_detail_path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Pacman/Ghost agents.")
    parser.add_argument("--main", default="codex_agent", help="Single candidate to evaluate.")
    parser.add_argument("--candidates", nargs="*", help="Evaluate multiple candidates.")
    parser.add_argument("--all-candidates", action="store_true", help="Evaluate every discovered non-risky agent.")
    parser.add_argument("--opponents", nargs="*", help="Opponent agent names. Default: all discovered agents.")
    parser.add_argument(
        "--same-pool",
        action="store_true",
        help="Use candidates as opponents too, run each unordered pair once per role, and credit both agents.",
    )
    parser.add_argument(
        "--from-detail",
        help="Rebuild summaries/ranking from a saved tournament *_matches.json file without running matches.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-timeout", type=float, default=1.0)
    parser.add_argument(
        "--process-timeout",
        type=float,
        default=300.0,
        help="Maximum wall-clock seconds per match; 300 allows slow 200-step agents to finish.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel arena subprocesses to run.")
    parser.add_argument(
        "--cache-detail",
        default=str(RESULTS_DIR / "latest_match_details.json"),
        help="Reuse matching results from a detail JSON when agent hashes and settings match.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable match-detail cache reuse.")
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

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"tournament_{now}.txt"
    detail_path = RESULTS_DIR / f"tournament_{now}_matches.json"
    latest_detail_path = RESULTS_DIR / "latest_match_details.json"

    if args.from_detail:
        detail_source = Path(args.from_detail)
        if not detail_source.is_absolute():
            detail_source = (Path.cwd() / detail_source).resolve()
        detail = json.loads(detail_source.read_text(encoding="utf-8"))
        detail_candidates = detail.get("candidates", [])
        detail_opponents = detail.get("opponents", detail_candidates)

        if args.all_candidates:
            candidates = detail_candidates
        elif args.candidates:
            candidates = args.candidates
        else:
            candidates = detail_candidates

        opponents = candidates if args.same_pool else (args.opponents if args.opponents else detail_opponents)
        candidates, skipped_candidates = filter_agents(candidates, args.include_risky)
        opponents, skipped_opponents = filter_agents(opponents, args.include_risky)

        match_map = records_to_match_map(detail.get("matches", []))
        mode = "detail-replay same-pool" if args.same_pool else "detail-replay"

        header = (
            f"Blind Hide and Seek Tournament\n"
            f"Time: {now}\n"
            f"Mode: {mode}\n"
            f"From detail: {detail_source}\n"
            f"Candidates: {', '.join(candidates)}\n"
            f"Opponents: {', '.join(opponents)}\n"
            f"Skipped: {', '.join(skipped_candidates + skipped_opponents) if skipped_candidates or skipped_opponents else 'none'}\n"
        )

        print(header)
        blocks = [header]
        summaries = []
        for candidate in candidates:
            summary, block = evaluate_from_records(candidate, opponents, match_map)
            summaries.append(summary)
            blocks.append(block)
            print(block)

        ranking = format_ranking(summaries)
        print(ranking)
        blocks.append(ranking)
        result_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
        (RESULTS_DIR / "latest_results.txt").write_text("\n".join(blocks) + "\n", encoding="utf-8")
        print(f"\nSaved: {result_path}")
        return 0

    if args.all_candidates:
        candidates = discovered
    else:
        candidates = args.candidates if args.candidates else [args.main]
    opponents = candidates if args.same_pool else (args.opponents if args.opponents else discovered)

    candidates, skipped_candidates = filter_agents(candidates, args.include_risky)
    opponents, skipped_opponents = filter_agents(opponents, args.include_risky)
    if not candidates:
        print("No candidates to evaluate after filtering.", file=sys.stderr)
        return 2
    if not opponents:
        print("No opponents to evaluate after filtering.", file=sys.stderr)
        return 2

    mode = "same-pool" if args.same_pool else "candidate"

    header = (
        f"Blind Hide and Seek Tournament\n"
        f"Time: {now}\n"
        f"Mode: {mode}\n"
        f"Candidates: {', '.join(candidates)}\n"
        f"Opponents: {', '.join(opponents)}\n"
        f"Skipped: {', '.join(skipped_candidates + skipped_opponents) if skipped_candidates or skipped_opponents else 'none'}\n"
        f"Settings: max_steps={args.max_steps}, speed={args.pacman_speed}, "
        f"capture={args.capture_distance}, obs=({args.pacman_obs_radius}, {args.ghost_obs_radius}), "
        f"start={args.start_mode}, workers={max(1, args.workers)}, "
        f"cache={'off' if args.no_cache else args.cache_detail}\n"
    )

    print(header)
    blocks = [header]
    summaries = []
    records = []

    def save_match_detail() -> None:
        write_detail(args, detail_path, latest_detail_path, now, mode, candidates, opponents, summaries, records)

    tasks = build_match_tasks(candidates, opponents, args.same_pool)
    print(f"Planned matches: {len(tasks)} unique subprocesses, workers={max(1, args.workers)}")
    task_results = execute_tasks(args, tasks, mode, records, on_match=save_match_detail)
    summaries, run_blocks = build_blocks_from_task_results(candidates, task_results)
    blocks.extend(run_blocks)
    for block in run_blocks:
        print(block)

    ranking = format_ranking(summaries)
    print(ranking)
    blocks.append(ranking)

    result_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    (RESULTS_DIR / "latest_results.txt").write_text("\n".join(blocks) + "\n", encoding="utf-8")
    write_detail(args, detail_path, latest_detail_path, now, mode, candidates, opponents, summaries, records)
    cleanup_old_result_files("tournament", [result_path, detail_path])
    print(f"\nSaved: {result_path}")
    print(f"Saved detail: {detail_path}")
    print(f"Latest detail: {latest_detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
