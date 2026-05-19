"""Measure per-agent RAM usage for Blind Hide and Seek submissions.

The assignment PDF caps the total RAM delta while running at 100MB. This
script launches a fresh Python subprocess for each agent so imports, caches,
and class state from one submission do not contaminate the next measurement.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
SUBMISSIONS_DIR = ROOT / "submissions"
RESULTS_DIR = ROOT / "results"
RAM_LIMIT_MB = 100.0


def rss_mb() -> float:
    """Return current process working-set/RSS in MB without external deps."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wt.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wt.DWORD,
        ]
        get_memory_info.restype = wt.BOOL
        ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            # On newer Windows versions the same API is also exported by
            # kernel32 as K32GetProcessMemoryInfo.
            get_memory_info = ctypes.windll.kernel32.K32GetProcessMemoryInfo
            get_memory_info.argtypes = [
                wt.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wt.DWORD,
            ]
            get_memory_info.restype = wt.BOOL
            ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / (1024 * 1024)

    import resource

    # ru_maxrss is KB on Linux, bytes on macOS. The repo is Windows-first, but
    # this keeps the script usable elsewhere.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def child_measure(agent_id: str, max_steps: int) -> dict:
    base_mb = rss_mb()
    peak_mb = base_mb
    start = time.perf_counter()

    sys.path.insert(0, str(SRC_DIR))
    from agent_loader import AgentLoader  # noqa: PLC0415
    from environment import Environment  # noqa: PLC0415

    framework_mb = rss_mb()
    peak_mb = max(peak_mb, framework_mb)
    loader = AgentLoader(str(SUBMISSIONS_DIR))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pacman = loader.load_agent(agent_id, "pacman", {"pacman_speed": 2})
            ghost = loader.load_agent(agent_id, "ghost")
    except BaseException as exc:  # includes SystemExit from intentionally bad agents
        now_mb = rss_mb()
        return {
            "agent": agent_id,
            "status": "load_error",
            "error": f"{type(exc).__name__}: {exc}",
            "base_mb": base_mb,
            "framework_mb": framework_mb,
            "peak_mb": max(peak_mb, now_mb),
            "total_delta_mb": max(peak_mb, now_mb) - base_mb,
            "agent_delta_mb": max(peak_mb, now_mb) - framework_mb,
            "elapsed_s": time.perf_counter() - start,
        }

    loaded_mb = rss_mb()
    peak_mb = max(peak_mb, loaded_mb)

    env = Environment(
        max_steps=max_steps,
        deterministic_starts=True,
        capture_distance_threshold=2,
        pacman_speed=2,
    )
    env.reset()

    status = "ok"
    error = ""
    steps_done = 0
    try:
        for step in range(1, max_steps + 1):
            pacman_obs, pacman_pos, pacman_enemy = env.get_observation("pacman", 5, 5)
            ghost_obs, ghost_pos, ghost_enemy = env.get_observation("ghost", 5, 5)

            pacman_action = pacman.step(pacman_obs, pacman_pos, pacman_enemy, step)
            pacman_action = loader.validate_agent_move(
                pacman_action, "pacman", agent_id, 2
            )
            ghost_move = ghost.step(ghost_obs, ghost_pos, ghost_enemy, step)
            ghost_move = loader.validate_agent_move(ghost_move, "ghost", agent_id)
            done, _result, _state = env.step(pacman_action, ghost_move)

            steps_done = step
            peak_mb = max(peak_mb, rss_mb())
            if done:
                break
    except BaseException as exc:
        status = "step_error"
        error = f"{type(exc).__name__}: {exc}"
        peak_mb = max(peak_mb, rss_mb())

    total_delta = peak_mb - base_mb
    agent_delta = peak_mb - framework_mb
    return {
        "agent": agent_id,
        "status": status,
        "error": error,
        "steps_done": steps_done,
        "base_mb": base_mb,
        "framework_mb": framework_mb,
        "loaded_mb": loaded_mb,
        "peak_mb": peak_mb,
        "total_delta_mb": total_delta,
        "agent_delta_mb": agent_delta,
        "pass_total_100mb": status == "ok" and total_delta <= RAM_LIMIT_MB,
        "pass_agent_100mb": status == "ok" and agent_delta <= RAM_LIMIT_MB,
        "elapsed_s": time.perf_counter() - start,
    }


def discover_agents() -> list[str]:
    return sorted(
        path.name
        for path in SUBMISSIONS_DIR.iterdir()
        if path.is_dir() and (path / "agent.py").exists() and path.name != "__pycache__"
    )


def run_parent(args: argparse.Namespace) -> int:
    agents = discover_agents() if args.all or not args.agents else args.agents
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    script = Path(__file__).resolve()

    for agent in agents:
        cmd = [
            sys.executable,
            str(script),
            "--child-agent",
            agent,
            "--max-steps",
            str(args.max_steps),
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT.parent),
                env=env,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            output = completed.stdout.strip().splitlines()
            row = json.loads(output[-1]) if output else {}
            if completed.returncode != 0 and row.get("status") == "ok":
                row["status"] = "process_error"
            row.setdefault("agent", agent)
            row.setdefault("returncode", completed.returncode)
            if completed.stderr.strip():
                row["stderr"] = completed.stderr.strip()[-500:]
        except subprocess.TimeoutExpired:
            row = {
                "agent": agent,
                "status": "timeout",
                "error": f"subprocess exceeded {args.timeout}s",
                "pass_total_100mb": False,
                "pass_agent_100mb": False,
            }
        except Exception as exc:
            row = {
                "agent": agent,
                "status": "runner_error",
                "error": f"{type(exc).__name__}: {exc}",
                "pass_total_100mb": False,
                "pass_agent_100mb": False,
            }
        rows.append(row)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"agent_ram_check_{now}.json"
    txt_path = RESULTS_DIR / f"agent_ram_check_{now}.txt"
    latest_json = RESULTS_DIR / "latest_agent_ram_check.json"
    latest_txt = RESULTS_DIR / "latest_agent_ram_check.txt"

    payload = {
        "time": now,
        "ram_limit_mb": RAM_LIMIT_MB,
        "max_steps": args.max_steps,
        "timeout_s": args.timeout,
        "rows": rows,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    latest_json.write_text(json_text + "\n", encoding="utf-8")

    lines = [
        "Blind Hide and Seek Agent RAM Check",
        f"Time: {now}",
        f"RAM limit: total_delta <= {RAM_LIMIT_MB:.0f}MB",
        f"Warmup: up to {args.max_steps} self-play steps per agent",
        "",
        "agent                 status       total_delta  agent_delta  peak_mb  steps  verdict",
    ]
    for row in rows:
        status = row.get("status", "")
        total = row.get("total_delta_mb")
        agent_delta = row.get("agent_delta_mb")
        peak = row.get("peak_mb")
        steps = row.get("steps_done", "")
        verdict = "PASS" if row.get("pass_total_100mb") else "FAIL"
        if status != "ok":
            verdict = "SKIP/FAIL"
        lines.append(
            f"{row.get('agent',''):<21} {status:<12} "
            f"{total if isinstance(total, (int, float)) else float('nan'):>10.1f} "
            f"{agent_delta if isinstance(agent_delta, (int, float)) else float('nan'):>11.1f} "
            f"{peak if isinstance(peak, (int, float)) else float('nan'):>8.1f} "
            f"{str(steps):>5}  {verdict}"
        )
        if row.get("error"):
            lines.append(f"  error: {row['error']}")

    txt = "\n".join(lines) + "\n"
    txt_path.write_text(txt, encoding="utf-8")
    latest_txt.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check RAM use of arena agents")
    parser.add_argument("agents", nargs="*", help="Agent folder names to check")
    parser.add_argument("--all", action="store_true", help="Check all submission folders")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--child-agent", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.child_agent:
        print(json.dumps(child_measure(args.child_agent, args.max_steps), ensure_ascii=False))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
