import csv
import json
from pathlib import Path


class CsvLog:
    """Append-only CSV with a fixed column order; header written once."""

    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(columns)

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in self.columns])


class RunLogger:
    """All artifacts of one training run under runs/<run_name>/.

    Layout (a fixed contract — analysis/plots.py reads exactly this):
        config.json      hyperparameters
        episodes.csv     one row per training episode
        training.csv     periodic gradient-step diagnostics
        evals.csv        one row per greedy evaluation episode
        eval/traj_step<N>_ep<I>.json   greedy trajectories
        checkpoints/ckpt_<N>.pt        model + optimizer snapshots
        model.pt         final weights
    """

    EPISODE_COLUMNS = [
        "episode", "global_step", "steps", "total_reward",
        "started", "rounded", "finished",
        "start_step", "round_step", "finish_step",
        "epsilon", "mean_loss", "mean_q", "wind_speed", "wind_direction",
        "required_sense", "oob", "mark_contacts",
    ]
    TRAINING_COLUMNS = [
        "global_step", "loss", "mean_q", "max_q", "epsilon", "buffer_fill",
    ]
    EVAL_COLUMNS = [
        "global_step", "episode", "steps", "total_reward",
        "started", "rounded", "finished", "required_sense", "mark_contacts",
    ]

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.eval_dir = self.run_dir / "eval"
        self.ckpt_dir = self.run_dir / "checkpoints"
        for d in (self.run_dir, self.eval_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.episodes = CsvLog(self.run_dir / "episodes.csv", self.EPISODE_COLUMNS)
        self.training = CsvLog(self.run_dir / "training.csv", self.TRAINING_COLUMNS)
        self.evals = CsvLog(self.run_dir / "evals.csv", self.EVAL_COLUMNS)

    def save_config(self, config) -> None:
        config.save(self.run_dir / "config.json")

    def save_trajectory(self, global_step: int, episode: int, traj: dict) -> Path:
        path = self.eval_dir / f"traj_step{global_step}_ep{episode}.json"
        path.write_text(json.dumps(traj) + "\n")
        return path

    def checkpoint_path(self, global_step: int) -> Path:
        return self.ckpt_dir / f"ckpt_{global_step}.pt"
