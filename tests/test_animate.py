"""Smoke test for the trajectory animator."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from analysis.animate import load_trajectory, render, resolve_trajectory


def _fake_trajectory(n=40):
    ys = np.linspace(40, 400, n)
    return {
        "global_step": 123, "episode": 0,
        "wind_direction": 1.2, "wind_speed": 7.5,
        "outcome": {"started": True, "rounded": False, "finished": False,
                    "out_of_bounds": False, "steps": n, "total_reward": -1.0},
        "steps": [
            {"x": 500.0, "y": float(y), "heading": 0.0, "speed": 6.0,
             "action": 2, "reward": -0.05, "race_state": 0 if y < 100 else 1}
            for y in ys
        ],
    }


def test_render_gif_smoke(tmp_path):
    traj_path = tmp_path / "traj_step123_ep0.json"
    traj_path.write_text(json.dumps(_fake_trajectory()))

    traj = load_trajectory(traj_path)
    out = tmp_path / "race.gif"
    render(traj, out, fps=5, duration=2.0, dpi=40, hold=0.4)

    assert out.exists() and out.stat().st_size > 0
    from PIL import Image
    im = Image.open(out)
    assert im.n_frames >= 5   # animation frames + outcome hold


def test_resolve_trajectory_picks_latest_step(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for step in (100, 900, 500):
        p = eval_dir / f"traj_step{step}_ep0.json"
        p.write_text(json.dumps(_fake_trajectory(n=5)))

    picked = resolve_trajectory(tmp_path, step=None, episode=0)
    assert picked.name == "traj_step900_ep0.json"

    with pytest.raises(SystemExit):
        resolve_trajectory(tmp_path, step=900, episode=7)
