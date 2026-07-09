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
    PRESTART_SECONDS,
    START_LINE_CENTER,
    BUOY_POS,
    BUOY_RADIUS,
    ROUNDING_CAPTURE_RADIUS,
)


def test_api_compliance():
    check_env(SailingEnv(), skip_render_check=True)


def test_reset_is_pre_start():
    env = SailingEnv()
    obs, info = env.reset(seed=42)
    assert info["race_state"] == STATE_PRE_START
    assert info["boat_pos"][1] < START_LINE_CENTER[1]   # below the line
    assert obs[7] == PRESTART_SECONDS                   # full countdown
    assert not info["gun_fired"]


def _rigged_env(seed=42):
    """Env with the boat heading North on a beam reach (10 m/s per step)."""
    env = SailingEnv()
    env.reset(seed=seed)
    env._wind_direction = np.pi / 2   # wind from the East -> TWA 90 heading N
    env._wind_speed = 10.0
    env._boat_heading = 0.0
    return env


def _arc_positions(bearings_deg, radius=40.0):
    """Boat positions at a given radius around the buoy, at compass bearings
    (0=N, clockwise +). A decreasing bearing sweep is counter-clockwise on the
    map = leaving the mark to port."""
    out = []
    for b in bearings_deg:
        r = np.radians(b)
        out.append(BUOY_POS + radius * np.array([np.sin(r), np.cos(r)], dtype=np.float32))
    return out


def _round_the_mark(env, bearings_deg=None):
    """Teleport the boat step-by-step around the buoy (port rounding) with the
    physics frozen, until the rounding registers. Returns (obs, reward, info)
    from the step that completed the rounding."""
    if bearings_deg is None:
        bearings_deg = range(80, -50, -15)   # CCW sweep of ~130 deg, 15 deg/step
    for pos in _arc_positions(bearings_deg):
        env._boat_pos = pos.astype(np.float32)
        obs, reward, _, _, info = env.step(2)
        if info["race_state"] == STATE_TO_FINISH:
            return obs, reward, info
    raise AssertionError("boat never registered a rounding while walking the arc")


def test_full_race_sequence():
    env = _rigged_env()

    # Crossing the line before the gun does not start the race.
    env._boat_pos = np.array([500.0, 95.0], dtype=np.float32)
    _, _, _, _, info = env.step(2)
    assert env._boat_pos[1] > START_LINE_CENTER[1]
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
    assert 9.0 < reward < 11.0        # start bonus (+ shaping/time terms)
    assert abs(obs[4]) < 0.1          # bearing target is now the buoy (North)

    # Genuinely rounding the buoy switches to the finish leg. Freeze the
    # physics (wind 0) and walk the boat around the mark on the required side
    # (counter-clockwise on the map = leave the mark to port).
    env._wind_speed = 0.0
    obs, reward, info = _round_the_mark(env)
    assert info["race_state"] == STATE_TO_FINISH
    assert 19.0 < reward < 21.0       # rounding bonus (+ shaping/time terms)
    assert abs(abs(obs[4]) - np.pi) < 0.2   # target is the finish line (South)

    # Restore movement for the finish-line crossings below.
    env._wind_speed = 10.0
    env._boat_heading = 0.0

    # Crossing the line upward on the finish leg does not finish.
    env._boat_pos = np.array([500.0, 95.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(2)
    assert not terminated

    # Crossing downward finishes the race.
    env._boat_heading = np.pi
    env._boat_pos = np.array([500.0, 105.0], dtype=np.float32)
    _, reward, terminated, _, info = env.step(2)
    assert terminated
    assert not info["out_of_bounds"]
    assert 99.0 < reward < 101.0      # finish bonus (+ shaping/time terms)


def test_touching_the_mark_is_not_a_rounding():
    """Merely entering (or sitting inside) the buoy radius must not count as a
    rounding — the boat has to travel an arc around the mark."""
    env = _rigged_env()
    env._wind_speed = 0.0                       # freeze physics
    env._race_state = STATE_TO_MARK

    # Sit right on the mark for several steps: no angular sweep, no rounding.
    env._boat_pos = (BUOY_POS + np.array([0.0, -0.5 * BUOY_RADIUS], dtype=np.float32)).astype(np.float32)
    for _ in range(10):
        _, _, _, _, info = env.step(2)
        assert info["race_state"] == STATE_TO_MARK

    # Drive straight through the mark zone (S -> N): a fast clip, not an arc.
    env2 = _rigged_env()                         # wind 10 m/s, heading North
    env2._race_state = STATE_TO_MARK
    env2._boat_pos = (BUOY_POS + np.array([1.0, -ROUNDING_CAPTURE_RADIUS], dtype=np.float32)).astype(np.float32)
    # Enough steps to pass fully through the zone and out the far side.
    for _ in range(int(2 * ROUNDING_CAPTURE_RADIUS / 10) + 2):
        _, _, _, _, info = env2.step(2)
    assert info["race_state"] == STATE_TO_MARK   # a straight punch-through never rounds


def test_rounding_the_wrong_side_is_rejected():
    """Sweeping the mark clockwise (leave-to-starboard) must not count when the
    course requires a port rounding."""
    env = _rigged_env()
    env._wind_speed = 0.0
    env._race_state = STATE_TO_MARK
    # Clockwise sweep = increasing compass bearing = wrong side.
    for pos in _arc_positions(range(-80, 60, 15)):
        env._boat_pos = pos.astype(np.float32)
        _, _, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_MARK


def test_out_of_bounds_terminates_with_penalty():
    env = _rigged_env()
    env._boat_heading = np.pi                  # sail South, off the map
    env._boat_pos = np.array([500.0, 5.0], dtype=np.float32)
    _, reward, terminated, _, info = env.step(2)
    assert terminated
    assert info["out_of_bounds"]
    assert reward < -19.0                      # OOB penalty dominates


def test_progress_shaping_rewards_closing_on_target():
    env = _rigged_env()
    env._race_state = 1                        # TO_MARK: target is the buoy
    env._boat_pos = np.array([500.0, 400.0], dtype=np.float32)
    _, reward_toward, _, _, _ = env.step(2)    # heading North, closing

    env2 = _rigged_env()
    env2._race_state = 1
    env2._boat_heading = np.pi                 # heading South, opening
    env2._wind_direction = -np.pi / 2          # beam reach again
    env2._boat_pos = np.array([500.0, 400.0], dtype=np.float32)
    _, reward_away, _, _, _ = env2.step(2)

    assert reward_toward > 0 > reward_away


def test_crossing_outside_committee_buoys_does_not_start():
    env = _rigged_env(seed=1)
    env._step_count = int(PRESTART_SECONDS) + 1        # gun already fired
    env._boat_pos = np.array([300.0, 95.0], dtype=np.float32)  # left of line
    _, _, _, _, info = env.step(2)
    assert info["race_state"] == STATE_PRE_START


if __name__ == "__main__":
    test_api_compliance()
    test_reset_is_pre_start()
    test_full_race_sequence()
    test_out_of_bounds_terminates_with_penalty()
    test_progress_shaping_rewards_closing_on_target()
    test_crossing_outside_committee_buoys_does_not_start()
    print("ALL CHECKS PASSED")
