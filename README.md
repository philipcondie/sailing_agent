# sailing_agent

A from-scratch reinforcement-learning project: a Gymnasium environment that
simulates a one-boat sailing race, and a minimal PyTorch DQN that learns to
race it. No RL libraries — the network, replay buffer, and training loop are
all in this repo, instrumented so every run can be inspected and plotted
after the fact.

## The race

The course models a real (single-handed, one-lap) regatta in three phases:

1. **Pre-start** — the boat spawns below the start line and manoeuvres while
   a 60-second countdown runs. After the gun, the race begins the moment the
   boat crosses the start line (between the two committee buoys) heading
   up-course. Early crossings don't count.
2. **To the mark** — sail up the course and round the windward buoy.
3. **To the finish** — sail back down and cross the same line in the
   opposite direction.

The boat's speed comes from a polar diagram: pointing into the wind stalls
it (the "no-go zone"), a beam reach is fastest, dead downwind is slow. The
boat has momentum, though — speed eases toward that polar target rather
than snapping to it, so it accelerates gradually from a standstill and
coasts through the no-go zone during a tack instead of stopping dead. Wind
direction and strength are randomized every episode, so an upwind mark on
one episode may be a downwind mark on the next.

- Observation (8 floats): heading, boat speed, wind direction, wind speed,
  bearing and distance to the current target, race state, seconds to the gun.
- Actions (3): turn left, turn right, hold course.
- Rewards: +10 start, +20 rounding, +100 finish, −0.05 per step, +0.01 per
  metre of progress toward the current leg's target (dense shaping), −0.05
  extra while pinching inside the no-go zone, and −20 with termination for
  sailing out of the race area.

See the `SailingEnv` docstring in `sailing_env/env.py` for details.

## Layout

```
sailing_env/            the Gymnasium environment
  env.py                physics, course geometry, race-state machine
  wrappers.py           NormalizeObservation ([-1, 1] scaling for the net)
rl/                     from-scratch DQN (PyTorch)
  network.py            Q-network MLP
  replay_buffer.py      numpy ring buffer
  agent.py              epsilon-greedy action selection + TD updates
  config.py             DQNConfig dataclass (saved with every run)
  logger.py             run-directory writer (CSV metrics, trajectories)
  evaluate.py           greedy rollouts with trajectory capture (also a CLI)
analysis/
  plots.py              turns a run directory into diagnostic PNGs
  animate.py            renders a captured trajectory as a GIF/MP4 of the race
train.py                training loop
tests/                  environment + RL unit tests
```

## Training

```bash
pip install -r requirements.txt
python train.py --run-name my-first-run
```

Every run writes `runs/<run-name>/`:

| artifact | contents |
| --- | --- |
| `config.json` | full hyperparameter set for the run |
| `episodes.csv` | per-episode reward, length, which race phases were reached and when, epsilon, mean loss/Q |
| `training.csv` | periodic gradient diagnostics (loss, Q stats, buffer fill) |
| `evals.csv` + `eval/*.json` | greedy-policy evaluations with full step-by-step trajectories |
| `checkpoints/*.pt`, `model.pt` | model + optimizer snapshots |

Useful flags for quick experiments: `--total-steps`, `--learning-starts`,
`--eps-decay-steps`, `--eval-interval`, `--seed` (see `python train.py -h`).

## Inspecting a run

```bash
python analysis/plots.py runs/my-first-run          # writes PNGs to runs/my-first-run/plots/
python -m rl.evaluate runs/my-first-run/checkpoints/ckpt_50000.pt --episodes 5

# animate a captured episode (GIF for the blog post; also writes .mp4):
python analysis/animate.py runs/my-first-run                  # latest eval, episode 0
python analysis/animate.py runs/my-first-run/eval/traj_step500000_ep0.json --out race.gif
```

The plots include the learning curve, the rolling fraction of episodes that
start / round the mark / finish (the clearest view of whether the agent has
learned each race phase), loss and Q-value traces, start-timing relative to
the gun, and course maps of greedy trajectories colored by race state.

## Tests

```bash
python -m pytest tests/
```
