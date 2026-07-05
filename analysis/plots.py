#!/usr/bin/env python3
"""Turn a training run directory into diagnostic PNGs for the blog post.

Usage:
    python analysis/plots.py runs/<run_name> [--out runs/<run_name>/plots] [--window 50]

Reads the run directory (see the schema in the module docstring of each
loader below) and writes one PNG per diagnostic into --out. Degrades
gracefully: a missing/short/malformed input file causes that one plot to be
skipped with a printed warning, never a crash.

Only stdlib + numpy + matplotlib are used (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Course geometry — import from sailing_env.env; fall back to literals so
# this script keeps working standalone (e.g. if gymnasium isn't installed).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from sailing_env.env import (
        WORLD_W,
        WORLD_H,
        START_LINE_CENTER,
        START_LINE_HALF_WIDTH,
        BUOY_POS,
        BUOY_RADIUS,
    )
except ImportError:
    WORLD_W = 1000.0
    WORLD_H = 1200.0
    START_LINE_CENTER = np.array([500.0, 100.0])
    START_LINE_HALF_WIDTH = 60.0
    BUOY_POS = np.array([500.0, 900.0])
    BUOY_RADIUS = 25.0


# ---------------------------------------------------------------------------
# Palette (validated categorical set from the dataviz skill, light-surface
# variant — this script only ever renders on a light background).
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed categorical assignment reused across every plot that touches the
# three race phases, so color always means the same thing in the blog post.
COLOR_STARTED = BLUE
COLOR_ROUNDED = GREEN
COLOR_FINISHED = VIOLET

DPI = 150

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "font.size": 10,
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def warn(msg: str) -> None:
    print(f"[plots] warning: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def rolling_mean(values: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Trailing rolling mean, NaN-aware (NaNs are excluded from the window,
    not treated as zero). Returns NaN wherever fewer than min_periods valid
    samples are available in the window."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    mask = ~np.isnan(values)
    vals0 = np.where(mask, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(vals0)))
    ccount = np.concatenate(([0], np.cumsum(mask)))
    for i in range(n):
        lo = max(0, i + 1 - window)
        c = ccount[i + 1] - ccount[lo]
        if c >= min_periods:
            out[i] = (csum[i + 1] - csum[lo]) / c
    return out


def heading_to_vec(heading: float) -> np.ndarray:
    """Compass heading (rad, 0=North, clockwise+) -> unit (x=East, y=North)."""
    return np.array([math.sin(heading), math.cos(heading)])


# ---------------------------------------------------------------------------
# CSV loaders — each returns a dict[str, np.ndarray] or None if the file is
# missing/empty/unreadable. Never raises.
# ---------------------------------------------------------------------------

def _load_csv(path: Path, float_cols, int_cols=()) -> dict | None:
    if not path.exists():
        warn(f"missing file, skipping: {path}")
        return None
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, csv.Error) as e:
        warn(f"could not read {path}: {e}")
        return None

    if not rows:
        warn(f"{path} has no data rows, skipping")
        return None

    cols: dict[str, np.ndarray] = {}
    try:
        for c in float_cols:
            cols[c] = np.array(
                [float(r[c]) if r.get(c, "") != "" else np.nan for r in rows],
                dtype=float,
            )
        for c in int_cols:
            cols[c] = np.array([int(float(r[c])) for r in rows], dtype=int)
    except (KeyError, ValueError) as e:
        warn(f"{path} missing/malformed column ({e}), skipping")
        return None

    return cols


def load_episodes(run_dir: Path):
    return _load_csv(
        run_dir / "episodes.csv",
        float_cols=[
            "total_reward",
            "epsilon",
            "mean_loss",
            "mean_q",
            "wind_speed",
            "wind_direction",
        ],
        int_cols=[
            "episode",
            "global_step",
            "steps",
            "started",
            "rounded",
            "finished",
            "start_step",
            "round_step",
            "finish_step",
        ],
    )


def load_training(run_dir: Path):
    return _load_csv(
        run_dir / "training.csv",
        float_cols=["loss", "mean_q", "max_q", "epsilon", "buffer_fill"],
        int_cols=["global_step"],
    )


def load_evals(run_dir: Path):
    return _load_csv(
        run_dir / "evals.csv",
        float_cols=["total_reward"],
        int_cols=["global_step", "episode", "steps", "started", "rounded", "finished"],
    )


def load_config(run_dir: Path) -> dict | None:
    path = run_dir / "config.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warn(f"could not read {path}: {e}")
        return None


def load_latest_trajectories(run_dir: Path, max_n: int = 6):
    """Load up to max_n eval/*.json trajectories from the largest global_step
    present. Returns (global_step, [traj_dicts]) or (None, []) if none found."""
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        warn(f"missing eval directory, skipping trajectories: {eval_dir}")
        return None, []

    pattern = re.compile(r"traj_step(\d+)_ep(\d+)\.json$")
    by_step: dict[int, list[tuple[int, Path]]] = {}
    for p in sorted(glob.glob(str(eval_dir / "traj_step*_ep*.json"))):
        m = pattern.search(p)
        if not m:
            continue
        step, ep = int(m.group(1)), int(m.group(2))
        by_step.setdefault(step, []).append((ep, Path(p)))

    if not by_step:
        warn(f"no trajectory files found in {eval_dir}, skipping trajectories")
        return None, []

    latest_step = max(by_step)
    files = sorted(by_step[latest_step])[:max_n]

    trajs = []
    for _, path in files:
        try:
            with open(path) as f:
                trajs.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            warn(f"could not read {path}: {e}")

    return latest_step, trajs


# ---------------------------------------------------------------------------
# Plot 1: learning_curve.png
# ---------------------------------------------------------------------------

def plot_learning_curve(episodes, window: int, out_dir: Path) -> None:
    if episodes is None:
        return
    ep = episodes["episode"]
    reward = episodes["total_reward"]
    steps = episodes["steps"].astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(ep, reward, color=BASELINE, linewidth=0.6, alpha=0.6, zorder=1)
    ax1.plot(
        ep,
        rolling_mean(reward, window),
        color=BLUE,
        linewidth=2.0,
        label=f"rolling mean (w={window})",
        zorder=2,
    )
    ax1.set_ylabel("episode total reward")
    ax1.set_title("Learning curve")
    ax1.legend(loc="best")

    ax2.plot(ep, steps, color=BASELINE, linewidth=0.6, alpha=0.6, zorder=1)
    ax2.plot(
        ep,
        rolling_mean(steps, window),
        color=AQUA,
        linewidth=2.0,
        label=f"rolling mean (w={window})",
        zorder=2,
    )
    ax2.set_xlabel("episode")
    ax2.set_ylabel("episode length (steps)")
    ax2.legend(loc="best")

    fig.tight_layout()
    _save(fig, out_dir / "learning_curve.png")


# ---------------------------------------------------------------------------
# Plot 2: race_progress.png
# ---------------------------------------------------------------------------

def plot_race_progress(episodes, window: int, out_dir: Path) -> None:
    if episodes is None:
        return
    ep = episodes["episode"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for col, color, label in (
        ("started", COLOR_STARTED, "started"),
        ("rounded", COLOR_ROUNDED, "rounded"),
        ("finished", COLOR_FINISHED, "finished"),
    ):
        frac = rolling_mean(episodes[col].astype(float), window)
        ax.plot(ep, frac, color=color, linewidth=2.0, label=label)

    ax.set_xlabel("episode")
    ax.set_ylabel(f"rolling fraction of episodes (w={window})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Race progress: did it learn the three phases?")
    ax.legend(loc="best")

    fig.tight_layout()
    _save(fig, out_dir / "race_progress.png")


# ---------------------------------------------------------------------------
# Plot 3: loss_and_q.png
# ---------------------------------------------------------------------------

def plot_loss_and_q(training, out_dir: Path) -> None:
    if training is None:
        return
    gstep = training["global_step"]
    loss = training["loss"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    if np.any(np.isfinite(loss) & (loss > 0)):
        ax1.set_yscale("log")
    ax1.plot(gstep, loss, color=RED, linewidth=1.0, label="loss")
    ax1.set_ylabel("loss")
    ax1.set_title("Training loss and Q-values")
    ax1.legend(loc="upper left")

    ax1_eps = ax1.twinx()
    ax1_eps.plot(
        gstep,
        training["epsilon"],
        color=INK_MUTED,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label="epsilon",
    )
    ax1_eps.set_ylabel("epsilon", color=INK_MUTED)
    ax1_eps.set_ylim(-0.02, 1.02)
    ax1_eps.grid(False)
    ax1_eps.tick_params(axis="y", colors=INK_MUTED)

    # Combined legend for the twin-axis panel.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_eps.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    ax2.plot(gstep, training["mean_q"], color=BLUE, linewidth=1.2, label="mean Q")
    ax2.plot(gstep, training["max_q"], color=ORANGE, linewidth=1.2, label="max Q")
    ax2.set_xlabel("global step")
    ax2.set_ylabel("Q value")
    ax2.legend(loc="best")

    fig.tight_layout()
    _save(fig, out_dir / "loss_and_q.png")


# ---------------------------------------------------------------------------
# Plot 4: start_timing.png
# ---------------------------------------------------------------------------

def plot_start_timing(episodes, window: int, out_dir: Path) -> None:
    if episodes is None:
        return
    started_mask = episodes["started"] == 1
    if not np.any(started_mask):
        warn("no episodes with started==1, skipping start_timing.png")
        return

    ep = episodes["episode"][started_mask]
    start_step = episodes["start_step"][started_mask].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ep, start_step, s=10, color=BASELINE, alpha=0.6, label="episode start step")
    ax.plot(
        ep,
        rolling_mean(start_step, window),
        color=BLUE,
        linewidth=2.0,
        label=f"rolling mean (w={window})",
    )
    ax.axhline(60, color=INK_MUTED, linestyle="--", linewidth=1.2, label="gun fires (step 60)")

    ax.set_xlabel("episode")
    ax.set_ylabel("in-episode step of start-line crossing")
    ax.set_title("Start timing (only episodes that started)")
    ax.legend(loc="best")

    fig.tight_layout()
    _save(fig, out_dir / "start_timing.png")


# ---------------------------------------------------------------------------
# Plot 5: trajectories.png
# ---------------------------------------------------------------------------

RACE_STATE_COLORS = {0: COLOR_STARTED, 1: COLOR_ROUNDED, 2: COLOR_FINISHED}
RACE_STATE_LABELS = {0: "pre-start", 1: "to the mark", 2: "to the finish"}


def _draw_course(ax) -> None:
    x0 = float(START_LINE_CENTER[0] - START_LINE_HALF_WIDTH)
    x1 = float(START_LINE_CENTER[0] + START_LINE_HALF_WIDTH)
    y_line = float(START_LINE_CENTER[1])

    ax.plot([x0, x1], [y_line, y_line], color=INK_SECONDARY, linewidth=1.5, zorder=1)
    ax.scatter(
        [x0, x1],
        [y_line, y_line],
        marker="s",
        s=28,
        color=INK_PRIMARY,
        zorder=3,
    )

    buoy = Circle(
        (float(BUOY_POS[0]), float(BUOY_POS[1])),
        BUOY_RADIUS,
        fill=False,
        edgecolor=INK_MUTED,
        linestyle=":",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(buoy)
    ax.scatter(
        [float(BUOY_POS[0])],
        [float(BUOY_POS[1])],
        marker="^",
        s=40,
        color=INK_PRIMARY,
        zorder=3,
    )

    ax.set_xlim(0, WORLD_W)
    ax.set_ylim(0, WORLD_H)
    ax.set_aspect("equal")


def _outcome_title(traj: dict) -> str:
    outcome = traj.get("outcome", {})
    steps = outcome.get("steps", "?")
    reward = outcome.get("total_reward", float("nan"))
    if outcome.get("finished"):
        return f"finished in {steps} steps, R={reward:.1f}"
    if outcome.get("rounded"):
        return f"rounded, did not finish ({steps} steps), R={reward:.1f}"
    if outcome.get("started"):
        return f"started, did not round ({steps} steps), R={reward:.1f}"
    return f"never started ({steps} steps), R={reward:.1f}"


def plot_trajectories(global_step, trajs, out_dir: Path) -> None:
    if not trajs:
        return

    n = len(trajs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, traj in enumerate(trajs):
        ax = axes_flat[i]
        _draw_course(ax)

        steps = traj.get("steps", [])
        if steps:
            xs = np.array([s["x"] for s in steps])
            ys = np.array([s["y"] for s in steps])
            states = np.array([s.get("race_state", 0) for s in steps])

            for state, color in RACE_STATE_COLORS.items():
                mask = states == state
                if np.any(mask):
                    ax.plot(
                        xs[mask],
                        ys[mask],
                        ".",
                        color=color,
                        markersize=3,
                        linestyle="none",
                        zorder=2,
                    )
            # Connect consecutive points with segments colored by the
            # state at the start of the segment, so the path reads as a
            # continuous track rather than a scatter.
            for j in range(len(xs) - 1):
                c = RACE_STATE_COLORS.get(int(states[j]), INK_MUTED)
                ax.plot(xs[j : j + 2], ys[j : j + 2], color=c, linewidth=1.3, zorder=2)

            ax.scatter([xs[0]], [ys[0]], marker="o", s=25, color=INK_PRIMARY, zorder=4)

        wind_dir = traj.get("wind_direction")
        wind_speed = traj.get("wind_speed")
        if wind_dir is not None:
            # wind_direction is where the wind comes FROM; the arrow shows
            # where it blows TO.
            vec = heading_to_vec(wind_dir + math.pi)
            arrow_len = 90.0
            origin = np.array([WORLD_W * 0.12, WORLD_H * 0.92])
            ax.annotate(
                "",
                xy=tuple(origin + vec * arrow_len),
                xytext=tuple(origin),
                arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, linewidth=1.5),
                zorder=5,
            )
            speed_txt = f"{wind_speed:.1f} m/s" if wind_speed is not None else ""
            ax.annotate(
                speed_txt,
                xy=tuple(origin),
                xytext=(0, -12),
                textcoords="offset points",
                fontsize=8,
                color=INK_SECONDARY,
                ha="center",
            )

        ax.set_title(_outcome_title(traj), fontsize=9)
        ax.set_xlabel("x, East (m)")
        ax.set_ylabel("y, North (m)")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    handles = [
        Line2D([0], [0], color=RACE_STATE_COLORS[s], lw=2, label=RACE_STATE_LABELS[s])
        for s in (0, 1, 2)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)

    fig.suptitle(f"Greedy-policy trajectories at global_step={global_step}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, out_dir / "trajectories.png")


# ---------------------------------------------------------------------------
# Plot 6: eval_curve.png
# ---------------------------------------------------------------------------

def plot_eval_curve(evals, out_dir: Path) -> None:
    if evals is None:
        return

    gstep = evals["global_step"]
    order = np.argsort(gstep, kind="stable")
    checkpoints = []
    seen = set()
    for g in gstep[order]:
        if g not in seen:
            seen.add(g)
            checkpoints.append(g)
    checkpoints = np.array(checkpoints)

    mean_reward = np.zeros(len(checkpoints))
    min_reward = np.zeros(len(checkpoints))
    max_reward = np.zeros(len(checkpoints))
    frac_started = np.zeros(len(checkpoints))
    frac_rounded = np.zeros(len(checkpoints))
    frac_finished = np.zeros(len(checkpoints))

    for i, g in enumerate(checkpoints):
        mask = gstep == g
        rewards = evals["total_reward"][mask]
        mean_reward[i] = np.mean(rewards)
        min_reward[i] = np.min(rewards)
        max_reward[i] = np.max(rewards)
        frac_started[i] = np.mean(evals["started"][mask])
        frac_rounded[i] = np.mean(evals["rounded"][mask])
        frac_finished[i] = np.mean(evals["finished"][mask])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.fill_between(checkpoints, min_reward, max_reward, color=BLUE, alpha=0.18, label="min-max range")
    ax1.plot(checkpoints, mean_reward, color=BLUE, linewidth=2.0, marker="o", markersize=4, label="mean reward")
    ax1.set_ylabel("eval total reward")
    ax1.set_title("Greedy evaluation performance over training")
    ax1.legend(loc="best")

    ax2.plot(checkpoints, frac_started, color=COLOR_STARTED, linewidth=2.0, marker="o", markersize=4, label="started")
    ax2.plot(checkpoints, frac_rounded, color=COLOR_ROUNDED, linewidth=2.0, marker="o", markersize=4, label="rounded")
    ax2.plot(checkpoints, frac_finished, color=COLOR_FINISHED, linewidth=2.0, marker="o", markersize=4, label="finished")
    ax2.set_xlabel("global step")
    ax2.set_ylabel("fraction of eval episodes")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="best")

    fig.tight_layout()
    _save(fig, out_dir / "eval_curve.png")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="path to runs/<run_name>")
    parser.add_argument("--out", type=str, default=None, help="output directory (default: <run_dir>/plots)")
    parser.add_argument("--window", type=int, default=50, help="rolling-mean window in episodes/checkpoints")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[plots] error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else run_dir / "plots"

    config = load_config(run_dir)
    if config is not None:
        print(f"[plots] loaded config.json with {len(config)} keys")

    episodes = load_episodes(run_dir)
    training = load_training(run_dir)
    evals = load_evals(run_dir)
    latest_step, trajs = load_latest_trajectories(run_dir)

    plot_learning_curve(episodes, args.window, out_dir)
    plot_race_progress(episodes, args.window, out_dir)
    plot_loss_and_q(training, out_dir)
    plot_start_timing(episodes, args.window, out_dir)
    plot_trajectories(latest_step, trajs, out_dir)
    plot_eval_curve(evals, out_dir)

    print(f"[plots] done -> {out_dir}")


if __name__ == "__main__":
    main()
