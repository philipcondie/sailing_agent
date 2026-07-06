#!/usr/bin/env python3
"""Visualize a trained DQN's value function and greedy policy over the course.

For a fixed wind and race phase, sweeps a grid of boat positions and, at each
position, evaluates every boat heading to find the state's value (max Q over
actions, maxed over headings) and the heading that achieves it. Answers two
questions side by side: "what does the policy think this part of the course
is worth?" and "which way does it want to point the boat from here?".

Usage:
    python analysis/value_map.py runs/shaped/checkpoints/ckpt_500000.pt \
        [--wind-direction 0.0] [--wind-speed 8.0] [--phase 1] \
        [--grid 45] [--headings 16] [--out value_map.png]

Wind direction is where the wind comes FROM (0 = North). Phase is the
race_state to evaluate under: 0 pre-start, 1 to the mark, 2 to the finish
(pre-start and to-the-finish both navigate toward the start/finish line;
to-the-mark navigates toward the buoy — same target logic as the env).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sailing_env.env import (
    WORLD_W,
    WORLD_H,
    START_LINE_CENTER,
    START_LINE_HALF_WIDTH,
    BUOY_POS,
    BUOY_RADIUS,
    STATE_TO_MARK,
    _polar_speed,
)
from rl.config import DQNConfig

# Same palette as plots.py / animate.py, so color means the same thing
# across every figure in the post.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
WATER = "#dceefb"

PHASE_NAMES = {0: "pre-start", 1: "to the mark", 2: "to the finish"}


# ---------------------------------------------------------------------------
# Agent loading — same pattern as analysis/wind_sweep.py's load_agent
# ---------------------------------------------------------------------------

def load_agent(checkpoint: Path):
    from rl.agent import DQNAgent  # deferred: torch import is slow
    from sailing_env import SailingEnv
    from sailing_env.wrappers import NormalizeObservation

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


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

def evaluate_grid(agent, phase: int, wind_direction: float, wind_speed: float,
                   grid: int, headings: int):
    """Sweep boat position x heading, batched through the Q-network once.

    Returns (xs, ys, value, best_heading) where xs/ys are the grid axes
    (length `grid`) and value/best_heading have shape (len(xs), len(ys)).
    """
    from sailing_env import SailingEnv

    raw_box = SailingEnv().observation_space
    low, high = raw_box.low.astype(np.float32), raw_box.high.astype(np.float32)
    span = high - low

    target = BUOY_POS if phase == STATE_TO_MARK else START_LINE_CENTER

    xs = np.linspace(0.0, WORLD_W, grid, dtype=np.float64)
    ys = np.linspace(0.0, WORLD_H, grid, dtype=np.float64)
    thetas = np.linspace(-np.pi, np.pi, headings, endpoint=False)

    # indexing="ij": X[i, j, k] varies with xs[i], Y with ys[j], H with thetas[k]
    X, Y, H = np.meshgrid(xs, ys, thetas, indexing="ij")

    delta_x = float(target[0]) - X
    delta_y = float(target[1]) - Y
    bearing = np.arctan2(delta_x, delta_y)
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # True wind angle: angle between heading and where the wind comes from,
    # wrapped to [-pi, pi] via the atan2(sin, cos) identity (same trick as
    # analysis/animate.py's boom_segment).
    twa = np.arctan2(np.sin(H - wind_direction), np.cos(H - wind_direction))
    twa_deg = np.degrees(np.abs(twa))
    speed_factor = np.vectorize(_polar_speed)(twa_deg)
    boat_speed = wind_speed * speed_factor

    n = X.size
    obs = np.empty((n, 8), dtype=np.float32)
    obs[:, 0] = H.reshape(-1)
    obs[:, 1] = boat_speed.reshape(-1)
    obs[:, 2] = wind_direction
    obs[:, 3] = wind_speed
    obs[:, 4] = bearing.reshape(-1)
    obs[:, 5] = distance.reshape(-1)
    obs[:, 6] = float(phase)
    obs[:, 7] = 0.0  # seconds_to_gun: race live at the gun for every phase here

    norm_obs = (2.0 * (obs - low) / span - 1.0).astype(np.float32)

    import torch

    with torch.no_grad():
        t = torch.as_tensor(norm_obs, dtype=torch.float32, device=agent.device)
        q = agent.q_net(t)  # (n, n_actions)
        max_q = q.max(dim=1).values.cpu().numpy()

    max_q = max_q.reshape(grid, grid, headings)
    best_idx = np.argmax(max_q, axis=2)
    value = np.take_along_axis(max_q, best_idx[:, :, None], axis=2)[:, :, 0]
    best_heading = thetas[best_idx]

    return xs, ys, value, best_heading


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_course(ax) -> None:
    """Course geometry overlay, shared by both panels."""
    ax.add_patch(
        Rectangle(
            (0, 0), WORLD_W, WORLD_H, fill=False,
            ec=INK_SECONDARY, lw=0.9, ls=(0, (6, 6)), alpha=0.6, zorder=3,
        )
    )

    x0 = float(START_LINE_CENTER[0] - START_LINE_HALF_WIDTH)
    x1 = float(START_LINE_CENTER[0] + START_LINE_HALF_WIDTH)
    y_line = float(START_LINE_CENTER[1])
    ax.plot([x0, x1], [y_line, y_line], color=INK_PRIMARY, linewidth=1.5,
             ls=(0, (4, 3)), zorder=4)
    ax.scatter([x0, x1], [y_line, y_line], marker="s", s=26,
               color=INK_PRIMARY, zorder=5)

    ax.add_patch(
        Circle(
            (float(BUOY_POS[0]), float(BUOY_POS[1])), BUOY_RADIUS,
            fill=False, edgecolor=INK_PRIMARY, linestyle=":", linewidth=1.2,
            zorder=3,
        )
    )
    ax.scatter([float(BUOY_POS[0])], [float(BUOY_POS[1])], marker="^", s=44,
               color=INK_PRIMARY, zorder=5)

    ax.set_xlim(-40, WORLD_W + 40)
    ax.set_ylim(-40, WORLD_H + 40)
    ax.set_aspect("equal")
    ax.set_xlabel("x, East (m)")
    ax.set_ylabel("y, North (m)")


def _wind_arrow(ax, wind_direction: float, wind_speed: float) -> None:
    """Wind arrow annotation, same convention as wind_sweep.py / animate.py:
    wind_direction is where the wind comes FROM, so the arrow points the
    opposite way (where it blows TO)."""
    vec = np.array([np.sin(wind_direction + np.pi), np.cos(wind_direction + np.pi)])
    origin = np.array([WORLD_W * 0.12, WORLD_H * 0.90])
    ax.annotate(
        "",
        xy=tuple(origin + vec * 70),
        xytext=tuple(origin - vec * 70),
        arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, linewidth=1.8),
        zorder=6,
    )
    ax.annotate(
        f"wind {wind_speed:.1f} m/s",
        xy=tuple(origin),
        xytext=(0, -14),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
        ha="center",
    )


def plot_value_map(xs, ys, value, best_heading, phase: int, wind_direction: float,
                    wind_speed: float, step: int, out: Path) -> None:
    fig, (ax_val, ax_pol) = plt.subplots(1, 2, figsize=(13, 6.4))

    for ax in (ax_val, ax_pol):
        ax.set_facecolor(WATER)

    # --- Left panel: value heatmap -----------------------------------------
    mesh = ax_val.pcolormesh(
        xs, ys, value.T, cmap="Blues", shading="auto", zorder=1,
    )
    cbar = fig.colorbar(mesh, ax=ax_val, fraction=0.046, pad=0.04)
    cbar.set_label("state value (max Q)")
    _draw_course(ax_val)
    _wind_arrow(ax_val, wind_direction, wind_speed)
    ax_val.set_title("What the policy values")

    # --- Right panel: policy quiver -----------------------------------------
    ax_pol.pcolormesh(xs, ys, value.T, cmap="Blues", shading="auto",
                       alpha=0.35, zorder=1)
    step_sub = max(1, len(xs) // 15)  # ~every 3rd point on the default grid
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Xs = X[::step_sub, ::step_sub]
    Ys = Y[::step_sub, ::step_sub]
    Hs = best_heading[::step_sub, ::step_sub]
    Vs = value[::step_sub, ::step_sub]
    dx = np.sin(Hs)
    dy = np.cos(Hs)

    spacing = (xs[1] - xs[0]) * step_sub if len(xs) > 1 else 1.0
    arrow_len = spacing * 0.8
    ax_pol.quiver(
        Xs, Ys, dx, dy, Vs,
        cmap="Blues", pivot="mid", angles="xy", scale_units="xy",
        scale=1.0 / arrow_len, width=0.005, edgecolor=INK_PRIMARY,
        linewidth=0.3, zorder=2,
    )
    _draw_course(ax_pol)
    _wind_arrow(ax_pol, wind_direction, wind_speed)
    ax_pol.set_title("Where the policy wants to go")

    phase_name = PHASE_NAMES.get(phase, str(phase))
    wind_dir_deg = np.degrees(wind_direction)
    fig.suptitle(
        f"Value / policy map — {phase_name} — "
        f"wind {wind_dir_deg:.0f}° @ {wind_speed:.1f} m/s — "
        f"checkpoint step {step}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--wind-direction", type=float, default=0.0,
                        help="radians, where the wind comes FROM (0 = North)")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="m/s")
    parser.add_argument("--phase", type=int, default=1, choices=[0, 1, 2],
                        help="race_state to evaluate under")
    parser.add_argument("--grid", type=int, default=45,
                        help="grid resolution per axis over the course")
    parser.add_argument("--headings", type=int, default=16,
                        help="number of boat headings swept at each position")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default: <run_dir>/plots/value_map_phase<P>.png)")
    args = parser.parse_args()

    agent, step = load_agent(args.checkpoint)
    print(f"Loaded {args.checkpoint} (global step {step})")

    xs, ys, value, best_heading = evaluate_grid(
        agent, args.phase, args.wind_direction, args.wind_speed,
        args.grid, args.headings,
    )
    print(f"  value range: [{value.min():.2f}, {value.max():.2f}]")

    run_dir = args.checkpoint.parent.parent
    out = args.out or (run_dir / "plots" / f"value_map_phase{args.phase}.png")

    plot_value_map(
        xs, ys, value, best_heading, args.phase, args.wind_direction,
        args.wind_speed, step, out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
