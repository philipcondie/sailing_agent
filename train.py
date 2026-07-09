"""Train a DQN sailing agent, from scratch in PyTorch.

Every run writes its artifacts under runs/<run-name>/ (config, per-episode
metrics, gradient diagnostics, greedy-eval trajectories, checkpoints) so a
run can be inspected, plotted, and reproduced after the fact:

    python train.py --run-name first-attempt
    python analysis/plots.py runs/first-attempt
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from rl import DQNAgent, DQNConfig
from rl.evaluate import evaluate
from rl.logger import RunLogger
from sailing_env import SailingEnv
from sailing_env.env import STATE_TO_FINISH, STATE_TO_MARK
from sailing_env.wrappers import NormalizeObservation

_PRINT_EVERY = 20  # episodes between progress lines


def train(config: DQNConfig, run_dir: Path) -> None:
    logger = RunLogger(run_dir)
    logger.save_config(config)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = NormalizeObservation(SailingEnv())
    agent = DQNAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        n_actions=int(env.action_space.n),
        config=config,
    )

    obs, info = env.reset(seed=config.seed)
    global_step = 0
    grad_steps = 0
    episode = 0
    next_eval = config.eval_interval
    next_ckpt = config.checkpoint_interval
    t0 = time.time()

    # Per-episode accumulators
    ep_reward = 0.0
    ep_losses: list[float] = []
    ep_qs: list[float] = []
    ep_start_step = ep_round_step = ep_finish_step = -1
    recent: list[dict] = []  # for progress printing

    while global_step < config.total_steps:
        epsilon = config.epsilon(global_step)
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        global_step += 1
        ep_reward += float(reward)

        # Phase-transition bookkeeping (in-episode step when each occurred)
        if ep_start_step < 0 and info["race_state"] >= STATE_TO_MARK:
            ep_start_step = info["step"]
        if ep_round_step < 0 and info["race_state"] >= STATE_TO_FINISH:
            ep_round_step = info["step"]
        if terminated and not info["out_of_bounds"]:
            ep_finish_step = info["step"]

        # Truncation is a time limit, not a real terminal state: don't let
        # the TD target treat it as one.
        agent.buffer.add(obs, action, reward, next_obs, done=terminated)
        obs = next_obs

        # Gradient updates
        if global_step >= config.learning_starts and global_step % config.train_freq == 0:
            diag = agent.update()
            grad_steps += 1
            ep_losses.append(diag["loss"])
            ep_qs.append(diag["mean_q"])
            if grad_steps % config.train_log_interval == 0:
                logger.training.append({
                    "global_step": global_step,
                    "loss": diag["loss"],
                    "mean_q": diag["mean_q"],
                    "max_q": diag["max_q"],
                    "epsilon": epsilon,
                    "buffer_fill": len(agent.buffer) / config.buffer_size,
                })

        if global_step % config.target_update_interval == 0:
            agent.sync_target()

        # Episode end
        if terminated or truncated:
            row = {
                "episode": episode,
                "global_step": global_step,
                "steps": info["step"],
                "total_reward": round(ep_reward, 4),
                "started": int(ep_start_step >= 0),
                "rounded": int(ep_round_step >= 0),
                "finished": int(ep_finish_step >= 0),
                "start_step": ep_start_step,
                "round_step": ep_round_step,
                "finish_step": ep_finish_step,
                "epsilon": round(epsilon, 4),
                "mean_loss": round(float(np.mean(ep_losses)), 6) if ep_losses else math.nan,
                "mean_q": round(float(np.mean(ep_qs)), 4) if ep_qs else math.nan,
                "wind_speed": round(float(env.unwrapped._wind_speed), 3),
                "wind_direction": round(float(env.unwrapped._wind_direction), 4),
                "required_sense": int(env.unwrapped._required_sense),
                "oob": int(info["out_of_bounds"]),
            }
            logger.episodes.append(row)
            recent.append(row)
            episode += 1

            if episode % _PRINT_EVERY == 0:
                r = recent[-_PRINT_EVERY:]
                sps = global_step / max(1e-9, time.time() - t0)
                print(
                    f"ep {episode:5d}  step {global_step:7d}  "
                    f"reward {np.mean([e['total_reward'] for e in r]):8.2f}  "
                    f"started {np.mean([e['started'] for e in r]):.2f}  "
                    f"rounded {np.mean([e['rounded'] for e in r]):.2f}  "
                    f"finished {np.mean([e['finished'] for e in r]):.2f}  "
                    f"oob {np.mean([e['oob'] for e in r]):.2f}  "
                    f"eps {epsilon:.3f}  {sps:5.0f} steps/s"
                )
                recent = r

            obs, info = env.reset()
            ep_reward = 0.0
            ep_losses, ep_qs = [], []
            ep_start_step = ep_round_step = ep_finish_step = -1

        # Greedy evaluation with trajectory capture
        if global_step >= next_eval:
            next_eval += config.eval_interval
            trajs = evaluate(agent, config.eval_episodes, seed=1_000_000 + global_step)
            for i, traj in enumerate(trajs):
                traj["global_step"] = global_step
                traj["episode"] = i
                o = traj["outcome"]
                logger.evals.append({
                    "global_step": global_step,
                    "episode": i,
                    "steps": o["steps"],
                    "total_reward": round(o["total_reward"], 4),
                    "started": int(o["started"]),
                    "rounded": int(o["rounded"]),
                    "finished": int(o["finished"]),
                    "required_sense": int(traj["required_sense"]),
                })
                logger.save_trajectory(global_step, i, traj)
            mean_r = np.mean([t["outcome"]["total_reward"] for t in trajs])
            print(f"  [eval @ {global_step}] mean greedy reward {mean_r:.2f} "
                  f"over {len(trajs)} episodes")

        # Checkpointing
        if global_step >= next_ckpt:
            next_ckpt += config.checkpoint_interval
            path = logger.checkpoint_path(global_step)
            agent.save(path, global_step)
            print(f"  [checkpoint] {path}")

    agent.save(run_dir / "model.pt", global_step)
    env.close()
    print(f"Done: {episode} episodes, {global_step} steps, "
          f"{time.time() - t0:.0f}s. Artifacts in {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--eps-decay-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = DQNConfig()
    for field in ("total_steps", "seed", "learning_starts", "buffer_size",
                  "eps_decay_steps", "eval_interval", "checkpoint_interval",
                  "device"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)

    run_dir = Path("runs") / args.run_name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists — pick a new --run-name")
    train(config, run_dir)


if __name__ == "__main__":
    main()
