"""Fast in-process role tournament.

This runner evaluates agents separately as Seek/Pacman and Hide/Ghost. It is
faster than run_tournament.py because it avoids spawning a Python process per
match.
"""

import argparse
import contextlib
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent_loader import AgentLoader
from environment import Environment


ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = ROOT / "submissions"
RESULTS_DIR = ROOT / "results"

EXCLUDED = {"broken_agent", "exit_test", "slow_agent", "simple_agent", "__pycache__"}


@dataclass
class RoleStats:
    wins: int = 0
    losses: int = 0
    errors: int = 0
    steps: list[int] = field(default_factory=list)

    @property
    def games(self):
        return self.wins + self.losses

    @property
    def win_rate(self):
        return self.wins / self.games * 100 if self.games else 0.0

    @property
    def avg_steps(self):
        return sum(self.steps) / len(self.steps) if self.steps else 0.0


def cleanup_old_result_files(prefix, keep_path):
    for path in RESULTS_DIR.glob(f"{prefix}_*.txt"):
        if path.resolve() != keep_path.resolve():
            path.unlink()


def discover_valid_agents(loader):
    valid = []
    invalid = []
    for path in sorted(SUBMISSIONS_DIR.iterdir()):
        if not path.is_dir() or path.name in EXCLUDED or not (path / "agent.py").exists():
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                loader.load_agent(path.name, "pacman", {"pacman_speed": 2})
                loader.load_agent(path.name, "ghost")
            valid.append(path.name)
        except Exception as exc:
            invalid.append((path.name, str(exc)))
    return valid, invalid


def run_match(loader, seek_id, hide_id, args):
    env = Environment(
        max_steps=args.max_steps,
        deterministic_starts=(args.start_mode == "deterministic"),
        capture_distance_threshold=args.capture_distance,
        pacman_speed=args.pacman_speed,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        pacman = loader.load_agent(seek_id, "pacman", {"pacman_speed": args.pacman_speed})
        ghost = loader.load_agent(hide_id, "ghost")

    env.reset()
    for step in range(1, args.max_steps + 1):
        pacman_obs, pacman_pos, pacman_enemy = env.get_observation(
            "pacman", args.pacman_obs_radius, args.ghost_obs_radius
        )
        ghost_obs, ghost_pos, ghost_enemy = env.get_observation(
            "ghost", args.pacman_obs_radius, args.ghost_obs_radius
        )

        try:
            pacman_action = pacman.step(pacman_obs, pacman_pos, pacman_enemy, step)
            pacman_action = loader.validate_agent_move(
                pacman_action, "pacman", seek_id, args.pacman_speed
            )
        except Exception as exc:
            return "hide", step, f"pacman_error:{type(exc).__name__}:{exc}"

        try:
            ghost_move = ghost.step(ghost_obs, ghost_pos, ghost_enemy, step)
            ghost_move = loader.validate_agent_move(ghost_move, "ghost", hide_id)
        except Exception as exc:
            return "seek", step, f"ghost_error:{type(exc).__name__}:{exc}"

        done, result, _ = env.step(pacman_action, ghost_move)
        if done:
            if result == "pacman_wins":
                return "seek", step, ""
            if result == "ghost_wins":
                return "hide", step, ""
            return "draw", step, ""

    return "hide", args.max_steps, ""


def update_stats(stats, seek_id, hide_id, winner, steps, error):
    stats.setdefault(seek_id, {"seek": RoleStats(), "hide": RoleStats()})
    stats.setdefault(hide_id, {"seek": RoleStats(), "hide": RoleStats()})

    stats[seek_id]["seek"].steps.append(steps)
    stats[hide_id]["hide"].steps.append(steps)

    if error:
        if winner == "seek":
            stats[hide_id]["hide"].errors += 1
        elif winner == "hide":
            stats[seek_id]["seek"].errors += 1

    if winner == "seek":
        stats[seek_id]["seek"].wins += 1
        stats[hide_id]["hide"].losses += 1
    elif winner == "hide":
        stats[hide_id]["hide"].wins += 1
        stats[seek_id]["seek"].losses += 1
    else:
        stats[seek_id]["seek"].losses += 1
        stats[hide_id]["hide"].losses += 1


def rank_seek(item):
    agent, role = item
    return (role.win_rate, -role.avg_steps, role.wins)


def rank_hide(item):
    agent, role = item
    return (role.win_rate, role.avg_steps, role.wins)


def format_role_line(index, agent, role, seek=True):
    avg_label = "avg_seek" if seek else "avg_hide"
    return (
        f"{index:>2}. {agent:<18} "
        f"wr={role.win_rate:>5.1f}% "
        f"wins={role.wins:>2}/{role.games:<2} "
        f"{avg_label}={role.avg_steps:>6.1f} "
        f"errors={role.errors}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Fast role-separated tournament")
    parser.add_argument("--agents", nargs="*", help="Agents to include. Default: all valid agents.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--capture-distance", type=int, default=2)
    parser.add_argument("--pacman-speed", type=int, default=2)
    parser.add_argument("--pacman-obs-radius", type=int, default=5)
    parser.add_argument("--ghost-obs-radius", type=int, default=5)
    parser.add_argument("--start-mode", choices=["deterministic", "stochastic"], default="deterministic")
    return parser


def main():
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    loader = AgentLoader(str(SUBMISSIONS_DIR))
    valid, invalid = discover_valid_agents(loader)
    agents = args.agents if args.agents else valid

    stats = {}
    details = []
    for seek_id in agents:
        for hide_id in agents:
            if seek_id == hide_id:
                continue
            winner, steps, error = run_match(loader, seek_id, hide_id, args)
            update_stats(stats, seek_id, hide_id, winner, steps, error)
            details.append((seek_id, hide_id, winner, steps, error))

    seek_ranking = sorted(
        ((agent, roles["seek"]) for agent, roles in stats.items()),
        key=rank_seek,
        reverse=True,
    )
    hide_ranking = sorted(
        ((agent, roles["hide"]) for agent, roles in stats.items()),
        key=rank_hide,
        reverse=True,
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        "Blind Hide and Seek Role Tournament",
        f"Time: {now}",
        f"Agents: {', '.join(agents)}",
        f"Settings: max_steps={args.max_steps}, speed={args.pacman_speed}, "
        f"capture={args.capture_distance}, obs=({args.pacman_obs_radius}, {args.ghost_obs_radius}), "
        f"start={args.start_mode}",
        "",
        "Invalid/skipped:",
    ]
    lines.extend(f"- {name}: {reason}" for name, reason in invalid)
    lines.extend(["", "Seek ranking:"])
    lines.extend(format_role_line(i, agent, role, seek=True) for i, (agent, role) in enumerate(seek_ranking, 1))
    lines.extend(["", "Hide ranking:"])
    lines.extend(format_role_line(i, agent, role, seek=False) for i, (agent, role) in enumerate(hide_ranking, 1))
    lines.extend(["", "Match details:"])
    for seek_id, hide_id, winner, steps, error in details:
        err = f" {error}" if error else ""
        lines.append(f"{seek_id:<18} vs {hide_id:<18} winner={winner:<4} steps={steps}{err}")

    output = "\n".join(lines) + "\n"
    result_path = RESULTS_DIR / f"role_tournament_{now}.txt"
    result_path.write_text(output, encoding="utf-8")
    (RESULTS_DIR / "latest_role_results.txt").write_text(output, encoding="utf-8")
    cleanup_old_result_files("role_tournament", result_path)
    print(output)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
