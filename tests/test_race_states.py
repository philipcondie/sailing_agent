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
    MARK_CONTACT_PENALTY,
    SENSE_PORT,
    SENSE_STARBOARD,
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


def _rigged_env(seed=42, required_sense=SENSE_PORT):
    """Env with the boat heading North on a beam reach (10 m/s per step).

    The required rounding side is pinned (default port) so rounding tests are
    deterministic regardless of the reset RNG."""
    env = SailingEnv()
    env.reset(seed=seed)
    env._wind_direction = np.pi / 2   # wind from the East -> TWA 90 heading N
    env._wind_speed = 10.0
    env._boat_heading = 0.0
    env._required_sense = required_sense
    return env


def _sail_across_north_ray(env, sense, y_offset=40.0):
    """Sail the boat across the ray due north of the buoy in one real step
    (the rounding detector works on the step's track segment, so the boat
    must actually move — teleports between steps are invisible to it).
    sense=+1 crosses west-to-east (clockwise round the mark = starboard),
    sense=-1 east-to-west (counter-clockwise = port).
    Returns (obs, reward, info) from the crossing step."""
    env._wind_direction = 0.0                        # wind from the North
    env._wind_speed = 10.0
    env._boat_heading = sense * np.pi / 2            # East / West: beam reach, 10 m/s
    env._boat_pos = np.array(
        [BUOY_POS[0] - sense * 5.0, BUOY_POS[1] + y_offset], dtype=np.float32
    )
    obs, reward, _, _, info = env.step(2)            # 10 m across: 5 m short -> 5 m past
    return obs, reward, info


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

    # Genuinely rounding the buoy switches to the finish leg: sail across
    # the ray north of the mark, east-to-west (leave the mark to port).
    obs, reward, info = _sail_across_north_ray(env, SENSE_PORT)
    assert info["race_state"] == STATE_TO_FINISH
    assert 19.0 < reward < 21.0       # rounding bonus (+ shaping/time terms)
    assert abs(abs(obs[4]) - np.pi) < 0.2   # target is the finish line (South)

    # Restore a beam reach heading North for the finish-line crossings below.
    env._wind_direction = np.pi / 2
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
    """Merely being near (or sitting next to) the buoy must not count as a
    rounding — the taut string through the track has to hook the mark."""
    env = _rigged_env()
    env._wind_speed = 0.0                       # freeze physics
    env._race_state = STATE_TO_MARK

    # Sit right next to the mark for several steps: the string never wraps.
    env._boat_pos = (BUOY_POS + np.array([0.0, -0.5 * BUOY_RADIUS], dtype=np.float32)).astype(np.float32)
    for _ in range(10):
        _, _, _, _, info = env.step(2)
        assert info["race_state"] == STATE_TO_MARK

    # Drive straight past the mark (S -> N): the track never crosses the
    # north ray, so the string pulls free — no rounding.
    env2 = _rigged_env()                         # wind 10 m/s, heading North
    env2._race_state = STATE_TO_MARK
    env2._boat_pos = (BUOY_POS + np.array([1.0, -100.0], dtype=np.float32)).astype(np.float32)
    for _ in range(22):                          # well past the far side
        _, _, _, _, info = env2.step(2)
    assert info["race_state"] == STATE_TO_MARK   # a straight punch-through never rounds


def test_rounding_the_wrong_side_is_rejected():
    """Passing over the top clockwise (leave-to-starboard) must not count when
    the course requires a port rounding."""
    env = _rigged_env(required_sense=SENSE_PORT)
    env._race_state = STATE_TO_MARK
    _, _, info = _sail_across_north_ray(env, SENSE_STARBOARD)
    assert info["race_state"] == STATE_TO_MARK


def test_starboard_rounding_accepted_when_required():
    """When the episode requires a starboard rounding, a clockwise (west-to-
    east) crossing over the top counts."""
    env = _rigged_env(required_sense=SENSE_STARBOARD)
    env._race_state = STATE_TO_MARK
    _, _, info = _sail_across_north_ray(env, SENSE_STARBOARD)
    assert info["race_state"] == STATE_TO_FINISH


def test_port_rounding_rejected_when_starboard_required():
    """A port (counter-clockwise) rounding must not count on a starboard course."""
    env = _rigged_env(required_sense=SENSE_STARBOARD)
    env._race_state = STATE_TO_MARK
    _, _, info = _sail_across_north_ray(env, SENSE_PORT)
    assert info["race_state"] == STATE_TO_MARK


def test_flyby_below_the_mark_is_not_a_rounding():
    """String rule (RRS 28.2): a track that stays between the mark and the
    finish line never hooks the mark, no matter how much bearing it sweeps.
    This straight pass accumulates ~93 deg of sweep and fooled the old
    sweep-based check."""
    env = _rigged_env(required_sense=SENSE_PORT)
    env._race_state = STATE_TO_MARK
    env._wind_direction = 0.0                        # wind from the North
    env._wind_speed = 10.0
    env._boat_heading = np.pi / 2                    # due East: beam reach, 10 m/s
    env._boat_pos = np.array(
        [BUOY_POS[0] - 120.0, BUOY_POS[1] - 60.0], dtype=np.float32
    )
    info = None
    for _ in range(24):                              # straight W->E pass, 60 m below
        _, _, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_MARK


def test_dip_under_hairpin_is_not_a_rounding():
    """West of the mark -> under it -> up the east side is ~180 deg of CCW
    bearing sweep, but the track only crosses x = buoy_x SOUTH of the buoy,
    so the string still pulls free. This path beat any sweep threshold
    below 270 deg."""
    env = _rigged_env(required_sense=SENSE_PORT)
    env._race_state = STATE_TO_MARK
    env._wind_direction = 0.0
    env._wind_speed = 10.0
    env._boat_heading = np.pi / 2                    # East: dip under the mark
    env._boat_pos = np.array(
        [BUOY_POS[0] - 45.0, BUOY_POS[1] - 40.0], dtype=np.float32
    )
    info = None
    for _ in range(9):                               # crosses x = buoy_x at y < buoy_y
        _, _, _, _, info = env.step(2)
    env._wind_direction = np.pi / 2                  # wind from the East
    env._boat_heading = 0.0                          # climb the east side to NE of the mark
    for _ in range(8):
        _, _, _, _, info = env.step(2)
    assert info["race_state"] == STATE_TO_MARK


def test_crossings_are_net_counted():
    """RRS 28.2 judges the whole track: a wrong-way crossing over the top
    must be unwound before a correct-way crossing can complete the rounding."""
    env = _rigged_env(required_sense=SENSE_PORT)
    env._race_state = STATE_TO_MARK

    _, _, info = _sail_across_north_ray(env, SENSE_STARBOARD)   # wrong way: net +1
    assert info["race_state"] == STATE_TO_MARK
    _, _, info = _sail_across_north_ray(env, SENSE_PORT)        # unwinds to net 0 — correct
    assert info["race_state"] == STATE_TO_MARK                  # direction, yet NOT a rounding
    _, reward, info = _sail_across_north_ray(env, SENSE_PORT)   # net -1 = port: rounded
    assert info["race_state"] == STATE_TO_FINISH
    assert 19.0 < reward < 21.0


def test_touching_the_mark_is_penalized_once_per_incident():
    """RRS 31: contact with the buoy costs MARK_CONTACT_PENALTY, once per
    incident — no re-penalty while still in contact, re-armed after leaving."""
    env = _rigged_env()
    env._wind_speed = 0.0
    env._race_state = STATE_TO_MARK

    env._boat_pos = (BUOY_POS + np.array([0.0, -30.0], dtype=np.float32)).astype(np.float32)
    _, _, _, _, info = env.step(2)
    assert info["mark_contacts"] == 0

    env._boat_pos = (BUOY_POS + np.array([0.0, -3.0], dtype=np.float32)).astype(np.float32)
    _, reward, _, _, info = env.step(2)              # first touch
    assert info["mark_contacts"] == 1
    assert MARK_CONTACT_PENALTY - 1.0 < reward < MARK_CONTACT_PENALTY + 1.0

    _, reward, _, _, info = env.step(2)              # still touching: no new penalty
    assert info["mark_contacts"] == 1
    assert reward > -1.0

    env._boat_pos = (BUOY_POS + np.array([0.0, -30.0], dtype=np.float32)).astype(np.float32)
    _, _, _, _, info = env.step(2)                   # clear the mark
    env._boat_pos = (BUOY_POS + np.array([0.0, -3.0], dtype=np.float32)).astype(np.float32)
    _, _, _, _, info = env.step(2)                   # second incident
    assert info["mark_contacts"] == 2


def test_fast_pass_over_the_mark_still_registers_contact():
    """Contact is tested against the step's track segment, not the endpoint —
    a boat covering more than the contact radius per step can't tunnel past."""
    env = _rigged_env()
    env._race_state = STATE_TO_MARK
    env._wind_direction = 0.0
    env._wind_speed = 12.0                           # beam reach: 12 m in one step
    env._boat_heading = np.pi / 2                    # due East, straight at the mark
    env._boat_pos = np.array([BUOY_POS[0] - 6.0, BUOY_POS[1]], dtype=np.float32)
    _, _, _, _, info = env.step(2)                   # endpoints 6 m clear on either side;
    assert info["mark_contacts"] == 1                # the segment passes through the buoy
    assert info["race_state"] == STATE_TO_MARK       # and that is still not a rounding


def test_required_sense_is_observed_and_pinnable():
    """The required side is randomized, exposed in the observation (index 8),
    reported in info, and pinnable via reset options."""
    env = SailingEnv()
    obs, info = env.reset(seed=1, options={"required_sense": SENSE_STARBOARD})
    assert info["required_sense"] == SENSE_STARBOARD
    assert obs[8] == float(SENSE_STARBOARD)

    obs, info = env.reset(seed=1, options={"required_sense": SENSE_PORT})
    assert info["required_sense"] == SENSE_PORT
    assert obs[8] == float(SENSE_PORT)

    # Unpinned resets produce both sides across seeds.
    seen = {env.reset(seed=s)[1]["required_sense"] for s in range(30)}
    assert seen == {SENSE_PORT, SENSE_STARBOARD}


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
