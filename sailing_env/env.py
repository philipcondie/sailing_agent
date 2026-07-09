import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ---------------------------------------------------------------------------
# Course layout  (metres, x = East, y = North)
# ---------------------------------------------------------------------------
WORLD_W = 1000.0
WORLD_H = 1200.0

START_LINE_CENTER = np.array([500.0, 100.0], dtype=np.float32)
START_LINE_HALF_WIDTH = 60.0   # line runs from x=440 to x=560

BUOY_POS = np.array([500.0, 900.0], dtype=np.float32)
BUOY_RADIUS = 25.0             # drawn rounding-zone ring (see ROUNDING_* for the trigger)

# --- Genuine mark rounding -------------------------------------------------
# A rounding is not "touch a radius": the boat must swing an arc *around* the
# buoy, leaving it on the required side. We accumulate the signed change in the
# compass bearing from the buoy to the boat (0=N, clockwise +, matching the
# boat heading convention) while the boat is inside a capture zone. A genuine
# rounding sweeps a wide arc in one direction; a straight clip barely moves the
# bearing, and the per-step contribution is capped so one fast step through the
# zone can't fake it. The sweep resets whenever the boat leaves the zone, so it
# must be earned in a single pass.
ROUNDING_CAPTURE_RADIUS = 100.0        # accumulate the sweep only within this of the buoy
ROUNDING_ARC = math.pi / 2             # 90 deg of travel around the mark counts as rounded
ROUNDING_MAX_STEP_SWEEP = math.radians(20.0)  # per-step cap on the sweep (kills straight clips)
# Which side the mark must be left on is randomized per episode and exposed in
# the observation (required_sense). Compass bearing increases clockwise, so a
# clockwise sweep (+1) leaves the mark to starboard and a counter-clockwise
# sweep (-1) leaves it to port.
SENSE_PORT = -1
SENSE_STARBOARD = 1

# Pre-start box: the boat spawns below the line and manoeuvres until the gun.
SPAWN_POS = np.array([500.0, 40.0], dtype=np.float32)
SPAWN_X_JITTER = 40.0          # random x offset at reset

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
MAX_BOAT_SPEED = 8.0           # m/s (~16 knots)
MAX_WIND_SPEED = 12.0          # m/s (~23 knots)
WIND_SPEED_RANGE = (4.0, 12.0)
TURN_RATE = 0.08               # rad per step
DT = 1.0                       # seconds per step
MAX_STEPS = 3000
MAX_DIST = float(np.sqrt(WORLD_W**2 + WORLD_H**2))

PRESTART_SECONDS = 60.0        # starting gun fires this long after reset

# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------
STEP_PENALTY = -0.05            # per step, encourages finishing quickly
PROGRESS_REWARD_PER_M = 0.01    # per metre of progress toward the leg target
START_BONUS = 10.0
ROUNDING_BONUS = 20.0
FINISH_BONUS = 100.0
OUT_OF_BOUNDS_PENALTY = -20.0   # leaving the race area (also terminates)

# Race states
STATE_PRE_START = 0             # gun not fired / start line not yet crossed
STATE_TO_MARK = 1               # racing: head upwind to the buoy
STATE_TO_FINISH = 2             # buoy rounded: head back to the finish line


class SailingEnv(gym.Env):
    """Sailing race in three phases: start, round the buoy, finish.

    Race states:
        0 PRE_START  — boat manoeuvres below the start line waiting for the
                       gun (fires PRESTART_SECONDS after reset). The race
                       begins when the boat crosses the start line (between
                       the two committee buoys) heading up-course, after the
                       gun. Crossing before the gun does not count.
        1 TO_MARK    — head up the course and round the windward buoy.
        2 TO_FINISH  — head back down and cross the finish line (same line
                       as the start, crossed in the opposite direction).

    Observation (9 floats):
        boat_heading        [-pi, pi]   rad  (0 = North, clockwise positive)
        boat_speed          [0, 8]      m/s
        wind_direction      [-pi, pi]   rad  (direction wind is coming FROM)
        wind_speed          [0, 12]     m/s
        bearing_to_target   [-pi, pi]   rad  (absolute bearing to current target)
        distance_to_target  [0, max]    m
        race_state          {0, 1, 2}   see above
        seconds_to_gun      [0, 60]     countdown to the starting gun (0 after)
        required_sense      {-1, +1}    which side to leave the mark on this
                                        episode: -1 = port, +1 = starboard

    The bearing/distance target is the start line centre in PRE_START and
    TO_FINISH, and the buoy in TO_MARK.

    Actions (Discrete 3):
        0: turn left
        1: turn right
        2: hold course

    Rewards (additive per step):
        -0.05 time penalty; +0.01 per metre of progress toward the current
        leg's target; +10 start, +20 rounding, +100 finish; -20 for leaving
        the race area.

    Termination:
        Boat crosses the finish line (downward) after rounding the buoy,
        or leaves the WORLD_W x WORLD_H race area (out of bounds, penalised).

    Truncation:
        Step count exceeds MAX_STEPS.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # --- Action / observation spaces -----------------------------------
        self.action_space = spaces.Discrete(3)

        obs_low  = np.array([-np.pi, 0.0,          -np.pi, 0.0,          -np.pi, 0.0,    0.0, 0.0,             -1.0], dtype=np.float32)
        obs_high = np.array([ np.pi, MAX_BOAT_SPEED, np.pi, MAX_WIND_SPEED, np.pi, MAX_DIST, 2.0, PRESTART_SECONDS, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # --- Internal state (set properly in reset) ------------------------
        self._boat_pos:      np.ndarray = np.zeros(2, dtype=np.float32)
        self._prev_boat_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._boat_heading:  float = 0.0
        self._boat_speed:    float = 0.0
        self._wind_direction: float = 0.0
        self._wind_speed:    float = 0.0
        self._race_state:    int = STATE_PRE_START
        self._step_count:    int = 0
        self._out_of_bounds: bool = False

        # Which side to leave the mark on this episode (-1 port, +1 starboard),
        # randomized in reset() and part of the observation.
        self._required_sense:     int = SENSE_PORT
        # Mark-rounding sweep accumulator (see _update_mark_rounding)
        self._mark_sweep:         float = 0.0
        self._prev_mark_bearing:  float | None = None

        self._renderer = None

    # -----------------------------------------------------------------------
    # Core gym API
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._boat_pos      = SPAWN_POS.copy()
        self._boat_pos[0]  += float(
            self.np_random.uniform(-SPAWN_X_JITTER, SPAWN_X_JITTER)
        )
        self._prev_boat_pos = self._boat_pos.copy()
        self._boat_heading  = 0.0   # facing North, toward the start line
        self._boat_speed    = 0.0
        self._race_state    = STATE_PRE_START
        self._step_count    = 0
        self._out_of_bounds = False
        self._mark_sweep        = 0.0
        self._prev_mark_bearing = None

        # Required rounding side: random per episode, or pinned via
        # reset(options={"required_sense": -1 or +1}) for reproducible eval.
        if options is not None and "required_sense" in options:
            self._required_sense = SENSE_PORT if options["required_sense"] < 0 else SENSE_STARBOARD
        else:
            self._required_sense = int(self.np_random.choice([SENSE_PORT, SENSE_STARBOARD]))

        self._wind_direction = float(self.np_random.uniform(-np.pi, np.pi))
        self._wind_speed     = float(self.np_random.uniform(*WIND_SPEED_RANGE))

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._prev_boat_pos = self._boat_pos.copy()

        # Steering
        if action == 0:
            self._boat_heading -= TURN_RATE
        elif action == 1:
            self._boat_heading += TURN_RATE
        # action == 2: hold course

        self._boat_heading = _wrap_angle(self._boat_heading)

        self._update_physics()

        # Dead-reckoning position update
        self._boat_pos = (
            self._boat_pos
            + _heading_to_vec(self._boat_heading) * self._boat_speed * DT
        )

        self._step_count += 1

        # Progress toward the current leg's target, for reward shaping.
        # Measured against the state at the start of the step so a phase
        # transition never produces a spurious distance jump.
        target = self._current_target()
        progress = float(
            np.linalg.norm(self._prev_boat_pos - target)
            - np.linalg.norm(self._boat_pos - target)
        )

        # Race progress: PRE_START -> TO_MARK -> TO_FINISH -> finished
        started = False
        rounded = False
        finished = False

        if self._race_state == STATE_PRE_START:
            # The start only counts after the gun and crossing up-course.
            # An early crossing is simply ignored: the boat is still in
            # PRE_START and must come back and cross again after the gun.
            if self._gun_fired() and self._crossed_line(upward=True):
                self._race_state = STATE_TO_MARK
                started = True
        elif self._race_state == STATE_TO_MARK:
            if self._update_mark_rounding():
                self._race_state = STATE_TO_FINISH
                rounded = True
        elif self._race_state == STATE_TO_FINISH:
            # Finish by re-crossing the line downward (from the course side).
            finished = self._crossed_line(upward=False)

        self._out_of_bounds = bool(
            not (0.0 <= self._boat_pos[0] <= WORLD_W)
            or not (0.0 <= self._boat_pos[1] <= WORLD_H)
        )

        terminated = finished or self._out_of_bounds
        truncated = self._step_count >= MAX_STEPS

        reward = self._compute_reward(
            finished, self._out_of_bounds, started, rounded, progress
        )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self._renderer is not None:
            # TODO: teardown pygame / matplotlib renderer
            self._renderer = None

    # -----------------------------------------------------------------------
    # Observation / info
    # -----------------------------------------------------------------------

    def _current_target(self) -> np.ndarray:
        """Navigation target of the current leg (line centre or buoy)."""
        return BUOY_POS if self._race_state == STATE_TO_MARK else START_LINE_CENTER

    def _get_obs(self) -> np.ndarray:
        target = self._current_target()
        delta = target - self._boat_pos
        bearing  = float(np.arctan2(delta[0], delta[1]))   # arctan2(E, N) = bearing
        distance = float(np.linalg.norm(delta))

        return np.array(
            [
                self._boat_heading,
                self._boat_speed,
                self._wind_direction,
                self._wind_speed,
                bearing,
                distance,
                float(self._race_state),
                self._seconds_to_gun(),
                float(self._required_sense),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "boat_pos":       self._boat_pos.copy(),
            "race_state":     self._race_state,
            "step":           self._step_count,
            "gun_fired":      self._gun_fired(),
            "seconds_to_gun": self._seconds_to_gun(),
            "out_of_bounds":  self._out_of_bounds,
            "required_sense": self._required_sense,
        }

    # -----------------------------------------------------------------------
    # Physics
    # -----------------------------------------------------------------------

    def _update_physics(self):
        """Update boat_speed using a polar diagram (TWA → speed factor).

        True wind angle (TWA) is the angle between where the boat points and
        where the wind comes from, in [0°, 180°] (symmetric port/starboard).

        Speed factors at key angles:
            0°   → 0.00  (head to wind)
            40°  → 0.25  (edge of no-go zone)
            90°  → 1.00  (beam reach — fastest)
            150° → 0.70  (broad reach)
            180° → 0.55  (dead run)
        """
        twa_rad = _wrap_angle(self._boat_heading - self._wind_direction)
        twa_deg = math.degrees(abs(twa_rad))
        self._boat_speed = self._wind_speed * _polar_speed(twa_deg)

    # -----------------------------------------------------------------------
    # Race logic
    # -----------------------------------------------------------------------

    def _gun_fired(self) -> bool:
        return self._step_count * DT >= PRESTART_SECONDS

    def _seconds_to_gun(self) -> float:
        return max(0.0, PRESTART_SECONDS - self._step_count * DT)

    def _update_mark_rounding(self) -> bool:
        """Accumulate the boat's swept angle around the mark and report a
        genuine rounding.

        The boat must travel an arc *around* the buoy on the required side,
        not merely touch a radius. We track the signed change in the compass
        bearing from the buoy to the boat (0=N, clockwise +) while the boat is
        within ROUNDING_CAPTURE_RADIUS; each step's contribution is clamped to
        ROUNDING_MAX_STEP_SWEEP so a single fast pass through the zone can't
        register a fake arc. A rounding is recognised once the sweep reaches
        ROUNDING_ARC in this episode's required direction (self._required_sense;
        -1 = port, +1 = starboard). Leaving the capture zone resets the
        accumulator, so the arc must be completed in one pass.
        """
        vec  = self._boat_pos - BUOY_POS
        dist = float(np.hypot(vec[0], vec[1]))
        if dist > ROUNDING_CAPTURE_RADIUS:
            self._mark_sweep = 0.0
            self._prev_mark_bearing = None
            return False

        bearing = float(np.arctan2(vec[0], vec[1]))   # compass: 0=N, CW+
        if self._prev_mark_bearing is not None:
            step = _wrap_angle(bearing - self._prev_mark_bearing)
            step = max(-ROUNDING_MAX_STEP_SWEEP, min(ROUNDING_MAX_STEP_SWEEP, step))
            self._mark_sweep += step
        self._prev_mark_bearing = bearing

        return self._required_sense * self._mark_sweep >= ROUNDING_ARC

    def _crossed_line(self, upward: bool) -> bool:
        """Detect a directed crossing of the start/finish line segment.

        The line is horizontal at y = START_LINE_CENTER[1], spanning
        x ∈ [center_x − half_width, center_x + half_width] between the two
        committee buoys. `upward=True` requires the boat to cross with y
        increasing (the start); `upward=False` with y decreasing (the finish).

        The x-range test uses the interpolated point where the track actually
        intersects the line, so fast diagonal moves are judged correctly.
        """
        x0     = START_LINE_CENTER[0] - START_LINE_HALF_WIDTH
        x1     = START_LINE_CENTER[0] + START_LINE_HALF_WIDTH
        y_line = START_LINE_CENTER[1]

        prev = self._prev_boat_pos.astype(np.float64)
        curr = self._boat_pos.astype(np.float64)

        if upward:
            crossed = prev[1] < y_line <= curr[1]
        else:
            crossed = prev[1] > y_line >= curr[1]
        if not crossed:
            return False

        # x where the track segment intersects y = y_line
        t = (y_line - prev[1]) / (curr[1] - prev[1])
        cross_x = prev[0] + t * (curr[0] - prev[0])

        return bool(x0 <= cross_x <= x1)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(
        self,
        finished: bool,
        out_of_bounds: bool,
        started: bool,
        rounded: bool,
        progress: float,
    ) -> float:
        """Compute the step reward.

        Components (additive):
            time penalty        every step, encourages efficiency
            progress shaping    metres of progress toward the current leg's
                                target — dense signal pulling the boat down
                                the course between the sparse phase bonuses
            phase bonuses       start / rounding / finish
            out-of-bounds       leaving the race area ends the episode

        TODO possible extensions: VMG-based shaping, penalty scaled by how
        late the boat starts after the gun.
        """
        reward = STEP_PENALTY + PROGRESS_REWARD_PER_M * progress
        if started:
            reward += START_BONUS
        if rounded:
            reward += ROUNDING_BONUS
        if finished:
            reward += FINISH_BONUS
        if out_of_bounds:
            reward += OUT_OF_BOUNDS_PENALTY
        return reward

    # -----------------------------------------------------------------------
    # Rendering stub
    # -----------------------------------------------------------------------

    def _render_frame(self):
        """TODO: implement pygame visualisation.

        Suggested elements:
            - Blue water background (WORLD_W × WORLD_H viewport)
            - Start/finish line (white dashed)
            - Buoy marker (red circle at BUOY_POS)
            - Boat (triangle pointing in self._boat_heading direction)
            - Wind arrow (direction + speed HUD)
            - Race state label and step counter
        """
        pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _polar_speed(twa_deg: float) -> float:
    """Speed factor [0, 1] as a function of true wind angle in degrees.

    Continuous at every boundary — no jumps at 40° or 150°.

    Branches:
        [0,  40°]  no-go zone:      linear ramp 0 → 0.25
        [40°, 90°] close-haul/beam: sin-based curve 0.25 → 1.0
        [90°,180°] beam/run:        linear taper 1.0 → 0.55
    """
    a = abs(twa_deg)
    if a < 40.0:
        return (a / 40.0) * 0.25
    elif a <= 90.0:
        t = (a - 40.0) / 50.0
        return 0.25 + 0.75 * math.sin(math.radians(t * 90.0))
    else:
        t = (a - 90.0) / 90.0
        return 1.0 - 0.45 * t


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _heading_to_vec(heading: float) -> np.ndarray:
    """Convert heading (rad, 0=North, CW+) to a unit (x=East, y=North) vector."""
    return np.array([np.sin(heading), np.cos(heading)], dtype=np.float32)
