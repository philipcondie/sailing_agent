import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DQNConfig:
    """Hyperparameters and bookkeeping settings for a training run.

    The config is saved as config.json in the run directory so every
    experiment is reproducible from its artifacts alone.
    """

    # Training horizon
    total_steps: int = 500_000

    # Replay buffer
    buffer_size: int = 100_000
    learning_starts: int = 10_000       # env steps before gradient updates begin
    batch_size: int = 64

    # Optimisation
    lr: float = 1e-4
    gamma: float = 0.99
    grad_clip: float = 10.0             # max gradient norm
    train_freq: int = 4                 # env steps between gradient updates
    target_update_interval: int = 1_000 # env steps between target-net syncs

    # Epsilon-greedy exploration (linear decay)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000

    # Network
    hidden_sizes: tuple = (128, 128)

    # Bookkeeping
    seed: int = 0
    device: str = "cpu"
    eval_interval: int = 25_000         # env steps between greedy evaluations
    eval_episodes: int = 3
    checkpoint_interval: int = 50_000
    train_log_interval: int = 100       # gradient steps between training.csv rows

    def epsilon(self, step: int) -> float:
        """Linearly decayed exploration rate at a given env step."""
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "DQNConfig":
        data = json.loads(path.read_text())
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)
