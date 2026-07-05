import numpy as np


class ReplayBuffer:
    """Fixed-size ring buffer of (obs, action, reward, next_obs, done).

    Stored as pre-allocated numpy arrays; sampling returns numpy arrays
    that the agent converts to tensors. `done` is True only for real
    terminations (finishing the race), not truncations (timeouts), so
    bootstrapping stays correct on time-limited episodes.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        self.capacity = capacity
        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._full = False
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._pos
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._dones[i] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        if self._pos == 0:
            self._full = True

    def sample(self, batch_size: int) -> dict:
        idx = self._rng.integers(0, len(self), size=batch_size)
        return {
            "obs":      self._obs[idx],
            "actions":  self._actions[idx],
            "rewards":  self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "dones":    self._dones[idx],
        }
