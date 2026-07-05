"""Greedy-policy evaluation with full trajectory capture.

Used two ways:
  - called from the training loop at eval checkpoints;
  - as a CLI to inspect a saved checkpoint:
        python -m rl.evaluate runs/<run>/checkpoints/ckpt_50000.pt --episodes 5
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sailing_env import SailingEnv
from sailing_env.wrappers import NormalizeObservation
from rl.config import DQNConfig


def run_episode(env, agent, seed: int | None = None) -> dict:
    """Roll out one greedy episode; returns a trajectory dict.

    The env must be a NormalizeObservation-wrapped SailingEnv: the agent
    sees normalized observations while positions/wind for the trajectory
    come from the underlying env's info dict.
    """
    obs, info = env.reset(seed=seed)
    raw = env.unwrapped

    steps = []
    total_reward = 0.0
    started = rounded = finished = False
    done = False

    while not done:
        action = agent.select_action(obs, epsilon=0.0)
        pos_before = info["boat_pos"]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

        steps.append({
            "x": float(pos_before[0]),
            "y": float(pos_before[1]),
            "heading": float(raw._boat_heading),
            "speed": float(raw._boat_speed),
            "action": int(action),
            "reward": float(reward),
            "race_state": int(info["race_state"]),
        })
        started = started or info["race_state"] >= 1
        rounded = rounded or info["race_state"] >= 2
        finished = finished or terminated

    # Final position so the plotted track reaches the finish.
    steps.append({
        "x": float(info["boat_pos"][0]),
        "y": float(info["boat_pos"][1]),
        "heading": float(raw._boat_heading),
        "speed": float(raw._boat_speed),
        "action": -1,
        "reward": 0.0,
        "race_state": int(info["race_state"]),
    })

    return {
        "wind_direction": float(raw._wind_direction),
        "wind_speed": float(raw._wind_speed),
        "outcome": {
            "started": started,
            "rounded": rounded,
            "finished": finished,
            "steps": len(steps) - 1,
            "total_reward": total_reward,
        },
        "steps": steps,
    }


def evaluate(agent, n_episodes: int, seed: int = 0) -> list[dict]:
    """Run n greedy episodes on a fresh env; returns their trajectories."""
    env = NormalizeObservation(SailingEnv())
    trajs = []
    for i in range(n_episodes):
        trajs.append(run_episode(env, agent, seed=seed + i))
    env.close()
    return trajs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for trajectory JSONs (default: print summary only)")
    args = parser.parse_args()

    from rl.agent import DQNAgent  # deferred: torch import is slow

    config_path = args.checkpoint.parent.parent / "config.json"
    config = DQNConfig.load(config_path) if config_path.exists() else DQNConfig()

    env = NormalizeObservation(SailingEnv())
    agent = DQNAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        n_actions=int(env.action_space.n),
        config=config,
    )
    env.close()
    global_step = agent.load(args.checkpoint)
    print(f"Loaded {args.checkpoint} (global step {global_step})")

    trajs = evaluate(agent, args.episodes, seed=args.seed)
    for i, t in enumerate(trajs):
        o = t["outcome"]
        print(
            f"  ep {i}: reward {o['total_reward']:8.2f}  steps {o['steps']:4d}  "
            f"started={o['started']}  rounded={o['rounded']}  finished={o['finished']}"
        )
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / f"traj_step{global_step}_ep{i}.json"
            path.write_text(json.dumps(t) + "\n")
            print(f"        wrote {path}")


if __name__ == "__main__":
    main()
