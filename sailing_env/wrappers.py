import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Rescale every observation component to [-1, 1].

    Uses the wrapped env's Box bounds, so it stays correct if the
    observation layout changes. Neural networks train poorly on raw
    features whose scales differ by orders of magnitude (heading in
    [-pi, pi] next to distance in [0, ~1562] m), which is why this sits
    between SailingEnv and the DQN.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        box = env.observation_space
        assert isinstance(box, spaces.Box), "NormalizeObservation needs a Box space"
        self._low = box.low.astype(np.float32)
        self._span = (box.high - box.low).astype(np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=box.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (2.0 * (obs - self._low) / self._span - 1.0).astype(np.float32)
