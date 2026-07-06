#!/usr/bin/env python3
"""Animate a captured race trajectory into a GIF (or MP4) for the blog post.

Usage:
    # a specific trajectory JSON (as written by rl/evaluate.py / train.py):
    python analysis/animate.py runs/shaped/eval/traj_step500000_ep0.json

    # or a run directory — picks the latest eval checkpoint:
    python analysis/animate.py runs/shaped [--episode 0] [--step 500000]

    # output / pacing:
    python analysis/animate.py ... --out race.gif --fps 25 --duration 18

The scene: water, the start/finish line between its two committee marks, the
windward buoy with its rounding radius, a boat (triangle + boom) that leaves
a trail colored by race phase, a wind arrow, and a HUD with the pre-start
countdown, elapsed time, boat speed, and race phase. The final frame holds
for a moment with the outcome stamped on it.

Only stdlib + numpy + matplotlib (Pillow does the GIF encoding).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

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
        PRESTART_SECONDS,
    )
except ImportError:
    WORLD_W, WORLD_H = 1000.0, 1200.0
    START_LINE_CENTER = np.array([500.0, 100.0])
    START_LINE_HALF_WIDTH = 60.0
    BUOY_POS = np.array([500.0, 900.0])
    BUOY_RADIUS = 25.0
    PRESTART_SECONDS = 60.0

# Phase colors — identical to analysis/plots.py so color means the same
# thing across every figure in the post.
PHASE_COLORS = {0: "#2a78d6", 1: "#008300", 2: "#4a3aa7"}
PHASE_NAMES = {0: "pre-start", 1: "to the mark", 2: "to the finish"}

WATER = "#dceefb"
WATER_DEEP = "#c7e2f4"     # subtle band so the water isn't a flat slab
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MARK_COLOR = "#1c1c1c"
BUOY_COLOR = "#e34948"
HULL = "#ffffff"
HULL_EDGE = "#1c1c1c"

# Boat drawn oversize (metres) so it reads at blog-GIF resolution.
BOAT_LEN = 46.0
BOAT_BEAM = 20.0


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def resolve_trajectory(path: Path, step: int | None, episode: int) -> Path:
    """Accept a trajectory JSON directly, or pick one out of a run dir."""
    if path.is_file():
        return path
    eval_dir = path / "eval"
    if not eval_dir.is_dir():
        raise SystemExit(f"{path} is neither a trajectory JSON nor a run directory")
    pattern = re.compile(r"traj_step(\d+)_ep(\d+)\.json$")
    found = {}
    for p in eval_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            found[(int(m.group(1)), int(m.group(2)))] = p
    if not found:
        raise SystemExit(f"no trajectories in {eval_dir}")
    chosen_step = step if step is not None else max(s for s, _ in found)
    key = (chosen_step, episode)
    if key not in found:
        avail = sorted(found)
        raise SystemExit(f"no trajectory for step={chosen_step} ep={episode}; "
                         f"available: {avail}")
    return found[key]


def load_trajectory(path: Path) -> dict:
    traj = json.loads(path.read_text())
    steps = traj["steps"]
    traj["xy"] = np.array([[s["x"], s["y"]] for s in steps], dtype=float)
    traj["heading"] = np.array([s["heading"] for s in steps], dtype=float)
    traj["speed"] = np.array([s["speed"] for s in steps], dtype=float)
    traj["phase"] = np.array([s["race_state"] for s in steps], dtype=int)
    return traj


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def heading_rotation(heading: float) -> np.ndarray:
    """Rotate boat-local coords (y = forward) into world (0 = North, CW+)."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, s], [-s, c]]).T


def boat_vertices(pos: np.ndarray, heading: float) -> np.ndarray:
    """Hull outline: sharp bow, flat transom, slight flare amidships."""
    local = np.array([
        [0.0, 0.62],                     # bow
        [0.36, 0.05], [0.30, -0.38],     # starboard side -> transom corner
        [-0.30, -0.38], [-0.36, 0.05],   # port
    ]) * [BOAT_BEAM, BOAT_LEN]
    return pos + local @ heading_rotation(heading)


def boom_segment(pos: np.ndarray, heading: float, wind_dir: float) -> np.ndarray:
    """Boom/sail line, eased out toward the leeward side.

    The boom hangs opposite the wind: sheeted in when pointing high,
    eased out running downwind — enough physics to read as sailing.
    """
    twa = np.arctan2(np.sin(heading - wind_dir), np.cos(heading - wind_dir))
    side = -np.sign(twa) if twa != 0 else 1.0        # leeward side
    ease = 0.15 + 0.75 * (abs(twa) / np.pi)          # sheet angle fraction
    boom_angle = np.pi + side * ease * (np.pi / 2)   # from bow, boat frame
    tip_local = np.array([np.sin(boom_angle), np.cos(boom_angle)]) * BOAT_LEN * 0.52
    mast_local = np.array([0.0, BOAT_LEN * 0.10])
    R = heading_rotation(heading)
    return np.array([pos + mast_local @ R, pos + tip_local @ R])


def build_scene(traj: dict):
    margin = 40.0
    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    fig.patch.set_facecolor(WATER)
    ax.set_facecolor(WATER)
    ax.set_xlim(-margin, WORLD_W + margin)
    ax.set_ylim(-margin, WORLD_H + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(WATER_DEEP)

    # Faint depth bands so the water has some texture
    for y0 in np.arange(0, WORLD_H, 150):
        ax.axhspan(y0, y0 + 75, color=WATER_DEEP, alpha=0.25, lw=0)

    # Race-area boundary (leaving it terminates the episode)
    ax.add_patch(plt.Rectangle((0, 0), WORLD_W, WORLD_H, fill=False,
                               ec=INK_SECONDARY, lw=0.8, ls=(0, (6, 6)), alpha=0.5))

    # Start/finish line between the two committee marks
    x0 = START_LINE_CENTER[0] - START_LINE_HALF_WIDTH
    x1 = START_LINE_CENTER[0] + START_LINE_HALF_WIDTH
    y_line = START_LINE_CENTER[1]
    ax.plot([x0, x1], [y_line, y_line], color=INK, lw=1.4, ls=(0, (4, 3)))
    ax.plot([x0, x1], [y_line, y_line], "s", color=MARK_COLOR, ms=7)
    ax.annotate("start / finish", (x1 + 18, y_line), color=INK_SECONDARY,
                fontsize=8, va="center")

    # Windward buoy + rounding radius
    ax.add_patch(Circle(BUOY_POS, BUOY_RADIUS, fill=False, ec=BUOY_COLOR,
                        lw=1.0, ls=":", alpha=0.8))
    ax.plot(*BUOY_POS, marker="^", color=BUOY_COLOR, ms=11,
            mec=MARK_COLOR, mew=0.8)
    ax.annotate("mark", (BUOY_POS[0] + 34, BUOY_POS[1]), color=INK_SECONDARY,
                fontsize=8, va="center")

    # Wind arrow (fixed HUD element, top-left) — points the way the wind blows
    wd, ws = traj["wind_direction"], traj["wind_speed"]
    flow = -np.array([np.sin(wd), np.cos(wd)])
    base = np.array([90.0, WORLD_H - 90.0])
    ax.annotate("", xy=base + flow * 70, xytext=base - flow * 70,
                arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, lw=2.2))
    ax.annotate(f"wind {ws:.0f} m/s", (base[0], base[1] - 110),
                color=INK_SECONDARY, fontsize=8, ha="center")

    # Dynamic artists
    trail = LineCollection([], linewidths=2.2, capstyle="round", alpha=0.85)
    ax.add_collection(trail)
    hull = Polygon(np.zeros((5, 2)), closed=True, fc=HULL, ec=HULL_EDGE,
                   lw=1.1, zorder=6)
    ax.add_patch(hull)
    boom, = ax.plot([], [], color=HULL_EDGE, lw=1.6, zorder=7)
    hud = ax.text(0.98, 0.985, "", transform=ax.transAxes, fontsize=9,
                  color=INK, family="monospace", ha="right", va="top", zorder=8,
                  bbox=dict(fc="#fcfcfb", ec="none", alpha=0.75,
                            boxstyle="round,pad=0.35"))
    banner = ax.text(0.5, 0.55, "", transform=ax.transAxes, fontsize=15,
                     color=INK, ha="center", weight="bold", zorder=8)

    # Phase legend along the bottom edge
    handles = [Line2D([], [], color=PHASE_COLORS[p], lw=2.5,
                      label=PHASE_NAMES[p]) for p in (0, 1, 2)]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              frameon=False, handlelength=1.6)

    fig.tight_layout(pad=0.4)
    return fig, ax, trail, hull, boom, hud, banner


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def hud_text(traj: dict, i: int) -> str:
    t = i  # DT = 1 s per step
    gun = max(0.0, PRESTART_SECONDS - t)
    phase = PHASE_NAMES[int(traj["phase"][i])]
    line1 = f"t = {t:4d} s   phase: {phase}"
    if gun > 0:
        line1 += f"   GUN in {gun:.0f} s"
    return line1 + f"\nboat speed {traj['speed'][i]:4.1f} m/s"


def outcome_text(traj: dict) -> str:
    o = traj["outcome"]
    if o.get("finished"):
        return f"FINISHED — {o['steps']} s"
    if o.get("out_of_bounds"):
        return "OUT OF BOUNDS"
    if o.get("rounded"):
        return "rounded, never finished"
    if o.get("started"):
        return "started, never rounded"
    return "never started"


def render(traj: dict, out: Path, fps: int, duration: float, dpi: int,
           hold: float = 1.5) -> None:
    from matplotlib.animation import FFMpegWriter, PillowWriter

    xy, phase = traj["xy"], traj["phase"]
    n = len(xy)
    frames = max(2, min(n, int(fps * duration)))
    idx = np.unique(np.linspace(0, n - 1, frames).astype(int))

    fig, ax, trail, hull, boom, hud, banner = build_scene(traj)

    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    seg_colors = [PHASE_COLORS[int(p)] for p in phase[:-1]]

    if out.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)

    with writer.saving(fig, str(out), dpi=dpi):
        for i in idx:
            trail.set_segments(segments[:i])
            trail.set_color(seg_colors[:i] if i else [])
            hull.set_xy(boat_vertices(xy[i], traj["heading"][i]))
            bx = boom_segment(xy[i], traj["heading"][i], traj["wind_direction"])
            boom.set_data(bx[:, 0], bx[:, 1])
            hud.set_text(hud_text(traj, int(i)))
            writer.grab_frame()
        banner.set_text(outcome_text(traj))
        banner.set_bbox(dict(fc="#fcfcfb", ec=INK_SECONDARY, alpha=0.9,
                             boxstyle="round,pad=0.5"))
        for _ in range(int(hold * fps)):
            writer.grab_frame()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path,
                        help="trajectory JSON, or a run directory")
    parser.add_argument("--step", type=int, default=None,
                        help="eval global_step when path is a run dir (default: latest)")
    parser.add_argument("--episode", type=int, default=0,
                        help="eval episode index when path is a run dir")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .gif or .mp4 (default: <traj name>.gif beside the JSON)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duration", type=float, default=18.0,
                        help="target animation length in seconds (the episode is "
                             "resampled to fit; the outcome frame holds ~1.5 s extra)")
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    traj_path = resolve_trajectory(args.path, args.step, args.episode)
    traj = load_trajectory(traj_path)
    out = args.out or traj_path.with_suffix(".gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    render(traj, out, fps=args.fps, duration=args.duration, dpi=args.dpi)
    o = traj["outcome"]
    print(f"wrote {out}  ({outcome_text(traj)}, reward {o['total_reward']:.1f})")


if __name__ == "__main__":
    main()
