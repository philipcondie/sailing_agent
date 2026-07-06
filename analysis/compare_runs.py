#!/usr/bin/env python3
"""Overlay episodes.csv from multiple training runs for side-by-side comparison.

Usage:
    python analysis/compare_runs.py runs/shaped runs/phase1 [...] \\
        [--out compare.png] [--window 50] [--labels a,b,...]

Reads each run directory's episodes.csv (schema: episode, global_step, steps,
total_reward, started, rounded, finished, start_step, round_step,
finish_step, epsilon, mean_loss, mean_q, wind_speed, wind_direction[, oob]),
parses by header name, and tolerates missing/extra columns (older runs may
lack the trailing oob column). A malformed or short file causes that one run
to be skipped with a printed warning, never a crash.

Produces one PNG with three stacked panels sharing the x-axis (global_step,
so runs with different episode lengths still align):
    1. rolling mean of total_reward
    2. rolling fraction of episodes finished
    3. rolling fraction started (dashed) and rounded (dotted)

Only stdlib + numpy + matplotlib are used (no pandas).
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
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap (matches analysis/plots.py) — not strictly
# needed here since we don't import sailing_env, but keeps the convention
# consistent for anything imported from this package in the future.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Palette — one fixed color per run, in this order. Runs beyond the palette
# length wrap around and print a warning (colors will repeat).
# ---------------------------------------------------------------------------
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

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
    print(f"[compare] warning: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Numeric helpers (same NaN-aware trailing rolling mean as analysis/plots.py)
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


# ---------------------------------------------------------------------------
# CSV loading — tolerant of missing/extra columns (e.g. older runs lack the
# trailing `oob` column). Returns None (with a warning) instead of raising.
# ---------------------------------------------------------------------------

# Columns we actually need; everything else in the file is ignored.
_FLOAT_COLS = ["total_reward"]
_INT_COLS = ["episode", "global_step", "started", "rounded", "finished"]
_REQUIRED_COLS = _FLOAT_COLS + _INT_COLS


def load_run_episodes(run_dir: Path) -> dict | None:
    path = run_dir / "episodes.csv"
    if not path.exists():
        warn(f"missing file, skipping run: {path}")
        return None
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as e:
        warn(f"could not read {path}: {e}")
        return None

    if not rows:
        warn(f"{path} has no data rows, skipping run")
        return None

    missing = [c for c in _REQUIRED_COLS if c not in fieldnames]
    if missing:
        warn(f"{path} missing required column(s) {missing}, skipping run")
        return None

    cols: dict[str, np.ndarray] = {}
    try:
        for c in _FLOAT_COLS:
            cols[c] = np.array(
                [float(r[c]) if r.get(c, "") not in ("", None) else np.nan for r in rows],
                dtype=float,
            )
        for c in _INT_COLS:
            cols[c] = np.array([int(float(r[c])) for r in rows], dtype=int)
    except (KeyError, ValueError) as e:
        warn(f"{path} has malformed data ({e}), skipping run")
        return None

    return cols


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(runs: list[tuple[str, dict]], window: int, out_path: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for i, (label, data) in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        gstep = data["global_step"]

        reward_roll = rolling_mean(data["total_reward"], window)
        ax1.plot(gstep, reward_roll, color=color, linewidth=2.0, label=label)

        finished_roll = rolling_mean(data["finished"].astype(float), window)
        ax2.plot(gstep, finished_roll, color=color, linewidth=2.0, label=label)

        started_roll = rolling_mean(data["started"].astype(float), window)
        rounded_roll = rolling_mean(data["rounded"].astype(float), window)
        ax3.plot(gstep, started_roll, color=color, linewidth=1.6, linestyle="--")
        ax3.plot(gstep, rounded_roll, color=color, linewidth=1.6, linestyle=":")

    ax1.set_ylabel(f"reward, rolling mean (w={window})")
    ax1.set_title("Run comparison")
    ax1.legend(loc="best")

    ax2.set_ylabel(f"fraction finished (w={window})")
    ax2.set_ylim(-0.02, 1.02)

    ax3.set_xlabel("global step")
    ax3.set_ylabel(f"fraction (w={window})")
    ax3.set_ylim(-0.02, 1.02)
    style_handles = [
        Line2D([0], [0], color=INK_SECONDARY, lw=1.6, linestyle="--", label="started"),
        Line2D([0], [0], color=INK_SECONDARY, lw=1.6, linestyle=":", label="rounded"),
    ]
    ax3.legend(handles=style_handles, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="paths to runs/<run_name> directories")
    parser.add_argument("--out", type=str, default="compare_runs.png", help="output PNG path")
    parser.add_argument("--window", type=int, default=50, help="rolling-mean window in episodes")
    parser.add_argument("--labels", type=str, default=None,
                         help="comma-separated labels, one per run_dir (default: dir basename)")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]

    if args.labels is not None:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(run_dirs):
            print(
                f"[compare] error: --labels has {len(labels)} entries but "
                f"{len(run_dirs)} run_dirs were given",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        labels = [d.name.rstrip("/") or str(d) for d in run_dirs]

    if len(run_dirs) > len(PALETTE):
        warn(
            f"{len(run_dirs)} runs but only {len(PALETTE)} palette colors — "
            "colors will repeat across runs"
        )

    runs: list[tuple[str, dict]] = []
    for run_dir, label in zip(run_dirs, labels):
        if not run_dir.is_dir():
            warn(f"not a directory, skipping run: {run_dir}")
            continue
        data = load_run_episodes(run_dir)
        if data is None:
            continue
        runs.append((label, data))

    if not runs:
        print("[compare] error: no valid runs to plot", file=sys.stderr)
        sys.exit(1)

    plot_comparison(runs, args.window, Path(args.out))


if __name__ == "__main__":
    main()
