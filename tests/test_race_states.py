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
    BOAT_ACCEL,
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
    """Env with the boat heading North on a beam reach (10 m/s per step).

    Speed now has momentum (BOAT_ACCEL lag toward the polar target), so
    tests that teleport the boat and expect immediate 10 m/step movement
    pin ``_boat_speed`` directly rather than relying on the wind alone.
    """
    env = SailingEnv()
    env.reset(seed=seed)
    env._wind_direction = np.pi / 2   # wind from the East -> TWA 90 heading N
    env._wind_speed = 10.0
    env._boat_heading = 0.0
    env._boat_speed = 10.0
    return env


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
    env._boat_speed = 10.0     # boat has momentum: pin it back up to speed
    env._boat_pos = np.array([500.0, 95.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_MARK
    assert 9.0 < reward < 11.0        # start bonus (+ shaping/time terms)
    assert abs(obs[4]) < 0.1          # bearing target is now the buoy (North)

    # Reaching the buoy switches to the finish leg.
    env._boat_pos = BUOY_POS + np.array([0.0, -30.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_FINISH
    assert 19.0 < reward < 21.0       # rounding bonus (+ shaping/time terms)
    assert abs(abs(obs[4]) - np.pi) < 0.2   # target is the finish line (South)

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


def test_no_go_zone_penalized():
    env = _rigged_env()
    env._race_state = 1                        # TO_MARK: buoy dead upwind
    env._wind_direction = 0.0                  # wind from dead ahead (North)
    env._boat_pos = np.array([500.0, 400.0], dtype=np.float32)
    _, reward_pinching, _, _, info = env.step(2)
    assert info["in_no_go"]
    assert env._boat_speed < 10.0              # bleeding speed in irons

    env2 = _rigged_env()                       # beam reach, same spot
    env2._race_state = 1
    env2._boat_pos = np.array([500.0, 400.0], dtype=np.float32)
    _, reward_sailing, _, _, info2 = env2.step(2)
    assert not info2["in_no_go"]
    assert reward_pinching < reward_sailing


def test_reset_options_pin_the_wind():
    env = SailingEnv()
    obs, _ = env.reset(seed=3, options={"wind_direction": 0.5, "wind_speed": 9.0})
    assert obs[2] == np.float32(0.5)
    assert obs[3] == np.float32(9.0)

    # without options the wind still randomizes
    obs, _ = env.reset(seed=3)
    assert not (obs[2] == np.float32(0.5) and obs[3] == np.float32(9.0))


def test_boat_accelerates_gradually():
    """Speed has momentum: it eases toward the polar target instead of
    snapping to it, so a beam reach builds up gradually from a standstill.

    reset() spawns the boat facing North (heading 0) at a random x, so a
    wind from the East puts it on a beam reach (TWA 90) immediately.
    """
    env = SailingEnv()
    env.reset(seed=7)
    env._wind_direction = np.pi / 2   # wind from the East -> TWA 90 at heading 0
    env._wind_speed = 10.0
    assert env._boat_speed == 0.0      # reset() starts the boat at rest

    env.step(2)                        # hold course
    first_step_speed = env._boat_speed
    expected_first_step = 10.0 * BOAT_ACCEL   # gap (10 - 0) closed by BOAT_ACCEL
    assert abs(first_step_speed - expected_first_step) < 1e-5
    assert 0.0 < first_step_speed < 5.0   # well below the ~10 m/s polar target

    speeds = [first_step_speed]
    for _ in range(20):
        env.step(2)
        speeds.append(env._boat_speed)

    # Monotonically increasing, closing in on (never overshooting) the target.
    assert all(b >= a - 1e-9 for a, b in zip(speeds, speeds[1:]))
    assert speeds[-1] > speeds[0]
    assert speeds[-1] <= 10.0 + 1e-6
    assert speeds[-1] > 7.0             # well on its way to the ~10 m/s target


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
    test_boat_accelerates_gradually()
    test_crossing_outside_committee_buoys_does_not_start()
    print("ALL CHECKS PASSED")
