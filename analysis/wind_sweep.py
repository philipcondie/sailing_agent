#!/usr/bin/env python3
"""Measure wind-direction generalization of a trained policy.

Rolls out greedy episodes over a grid of pinned wind directions (and a
fixed wind speed), then reports the start / round / finish rate per
direction — the objective answer to "does the boat race in every wind?".

Usage:
    python analysis/wind_sweep.py runs/phase1/checkpoints/ckpt_2000000.pt \
        [--directions 16] [--episodes-per 3] [--wind-speed 8] \
        [--out runs/phase1/plots/wind_sweep.png] [--csv ...]

Wind direction is where the wind comes FROM (0 = North, i.e. blowing down
the course; the mark leg is then dead upwind).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sailing_env import SailingEnv
from sailing_env.wrappers import NormalizeObservation
from rl.config import DQNConfig

# Same phase colors as plots.py / animate.py
COLOR_STARTED = "#2a78d6"
COLOR_ROUNDED = "#008300"
COLOR_FINISHED = "#4a3aa7"
INK_SECONDARY = "#52514e"


def load_agent(checkpoint: Path):
    from rl.agent import DQNAgent  # deferred: torch import is slow

    config_path = checkpoint.parent.parent / "config.json"
    config = DQNConfig.load(config_path) if config_path.exists() else DQNConfig()
    env = NormalizeObservation(SailingEnv())
    agent = DQNAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        n_actions=int(env.action_space.n),
        config=config,
    )
    env.close()
    step = agent.load(checkpoint)
    return agent, step


def sweep(agent, directions: int, episodes_per: int, wind_speed: float,
          seed: int = 0) -> list[dict]:
    from rl.evaluate import run_episode

    env = NormalizeObservation(SailingEnv())
    rows = []
    thetas = np.linspace(-np.pi, np.pi, directions, endpoint=False)
    for theta in thetas:
        for ep in range(episodes_per):
            traj = run_episode(
                env, agent, seed=seed + ep,
                options={"wind_direction": float(theta),
                         "wind_speed": wind_speed},
            )
            o = traj["outcome"]
            rows.append({
                "wind_direction": round(float(theta), 4),
                "episode": ep,
                "steps": o["steps"],
                "total_reward": round(o["total_reward"], 2),
                "started": int(o["started"]),
                "rounded": int(o["rounded"]),
                "finished": int(o["finished"]),
                "out_of_bounds": int(o["out_of_bounds"]),
            })
    env.close()
    return rows


def plot(rows: list[dict], out: Path, wind_speed: float) -> None:
    thetas = sorted({r["wind_direction"] for r in rows})
    rates = {"started": [], "rounded": [], "finished": []}
    for theta in thetas:
        sub = [r for r in rows if r["wind_direction"] == theta]
        for k in rates:
            rates[k].append(np.mean([r[k] for r in sub]))

    fig, ax = plt.subplots(figsize=(7, 4.6), subplot_kw={"projection": "polar"})
    # Compass convention: 0 = North (wind from dead upwind), clockwise.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = 2 * np.pi / len(thetas)

    # Nested bars: started behind, finished in front, so all three read.
    ax.bar(thetas, rates["started"], width=width * 0.95, color=COLOR_STARTED,
           alpha=0.35, label="started")
    ax.bar(thetas, rates["rounded"], width=width * 0.72, color=COLOR_ROUNDED,
           alpha=0.6, label="rounded")
    ax.bar(thetas, rates["finished"], width=width * 0.5, color=COLOR_FINISHED,
           alpha=0.95, label="finished")

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["50%", "100%"], color=INK_SECONDARY, fontsize=8)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["N (upwind mark)", "NE", "E", "SE", "S", "SW", "W", "NW"],
                       fontsize=8, color=INK_SECONDARY)
    ax.set_title(f"Greedy success rate by wind direction ({wind_speed:.0f} m/s wind)",
                 fontsize=11)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0.0), fontsize=8,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--episodes-per", type=int, default=3)
    parser.add_argument("--wind-speed", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default: wind_sweep.png beside the checkpoint's run)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write per-episode results as CSV")
    args = parser.parse_args()

    agent, step = load_agent(args.checkpoint)
    print(f"Loaded {args.checkpoint} (global step {step})")

    rows = sweep(agent, args.directions, args.episodes_per, args.wind_speed,
                 seed=args.seed)

    n = len(rows)
    for k in ("started", "rounded", "finished", "out_of_bounds"):
        print(f"  {k:14s} {sum(r[k] for r in rows)}/{n} "
              f"({100 * np.mean([r[k] for r in rows]):.0f}%)")

    run_dir = args.checkpoint.parent.parent
    out = args.out or (run_dir / "plots" / "wind_sweep.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, out, args.wind_speed)
    print(f"wrote {out}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
