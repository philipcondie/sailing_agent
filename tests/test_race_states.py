"""Verify the three race states: PRE_START -> TO_MARK -> TO_FINISH.

Run directly (``python tests/test_race_states.py``) or via pytest.
The tests reach into private attributes to place the boat and set the wind,
so they exercise the state machine without depending on a trained policy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from gymnasium.utils.env_checker import check_env

from sailing_env.env import (
    SailingEnv,
    STATE_PRE_START,
    STATE_TO_MARK,
    STATE_TO_FINISH,
    _PRESTART_SECONDS,
    _START_LINE_CENTER,
    _BUOY_POS,
)


def test_api_compliance():
    check_env(SailingEnv(), skip_render_check=True)


def test_reset_is_pre_start():
    env = SailingEnv()
    obs, info = env.reset(seed=42)
    assert info["race_state"] == STATE_PRE_START
    assert info["boat_pos"][1] < _START_LINE_CENTER[1]   # below the line
    assert obs[7] == _PRESTART_SECONDS                   # full countdown
    assert not info["gun_fired"]


def _rigged_env(seed=42):
    """Env with the boat heading North on a beam reach (10 m/s per step)."""
    env = SailingEnv()
    env.reset(seed=seed)
    env._wind_direction = np.pi / 2   # wind from the East -> TWA 90 heading N
    env._wind_speed = 10.0
    env._boat_heading = 0.0
    return env


def test_full_race_sequence():
    env = _rigged_env()

    # Crossing the line before the gun does not start the race.
    env._boat_pos = np.array([500.0, 95.0], dtype=np.float32)
    _, _, _, _, info = env.step(2)
    assert env._boat_pos[1] > _START_LINE_CENTER[1]
    assert info["race_state"] == STATE_PRE_START

    # Park below the line until the gun fires.
    env._boat_pos = np.array([500.0, 50.0], dtype=np.float32)
    env._wind_speed = 0.0
    while not env._gun_fired():
        obs, _, _, _, info = env.step(2)
    assert info["race_state"] == STATE_PRE_START
    assert obs[7] == 0.0

    # Crossing after the gun starts the race.
    env._wind_speed = 10.0
    env._boat_pos = np.array([500.0, 95.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_MARK
    assert reward == 10.0
    assert abs(obs[4]) < 0.1          # bearing target is now the buoy (North)

    # Reaching the buoy switches to the finish leg.
    env._boat_pos = _BUOY_POS + np.array([0.0, -30.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_FINISH
    assert reward == 20.0
    assert abs(abs(obs[4]) - np.pi) < 0.2   # target is the finish line (South)

    # Crossing the line upward on the finish leg does not finish.
    env._boat_pos = np.array([500.0, 95.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(2)
    assert not terminated

    # Crossing downward finishes the race.
    env._boat_heading = np.pi
    env._boat_pos = np.array([500.0, 105.0], dtype=np.float32)
    _, reward, terminated, _, _ = env.step(2)
    assert terminated
    assert reward == 100.0


def test_crossing_outside_committee_buoys_does_not_start():
    env = _rigged_env(seed=1)
    env._step_count = int(_PRESTART_SECONDS) + 1        # gun already fired
    env._boat_pos = np.array([300.0, 95.0], dtype=np.float32)  # left of line
    _, _, _, _, info = env.step(2)
    assert info["race_state"] == STATE_PRE_START


if __name__ == "__main__":
    test_api_compliance()
    test_reset_is_pre_start()
    test_full_race_sequence()
    test_crossing_outside_committee_buoys_does_not_start()
    print("ALL CHECKS PASSED")
