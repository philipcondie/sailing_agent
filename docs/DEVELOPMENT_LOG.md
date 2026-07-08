# Development log — sailing RL agent

Working notes for a from-scratch RL project being written up as a blog post:
a Gymnasium sailing-race environment plus a hand-rolled PyTorch DQN, with
heavy instrumentation so every experiment produces plots and GIFs. This file
is the handoff document: it records what exists, why it's shaped that way,
and what to build next. Update it as the project evolves.

## 1. Project state (as of 2026-07-06)

### Merged to main (PRs #1–#4)
1. **Three-phase race environment** (`sailing_env/env.py`). States:
   `PRE_START(0)` — boat spawns below the line, gun fires at
   `PRESTART_SECONDS = 60` sim-seconds; the race starts only on an upward
   crossing of the start line (between the committee buoys) after the gun;
   early crossings are silently ignored. `TO_MARK(1)` — round the windward
   buoy (25 m proximity). `TO_FINISH(2)` — re-cross the line downward.
   Line crossings are direction-enforced and use the interpolated
   track/line intersection. Observation: 8 floats
   `[heading, boat_speed, wind_dir, wind_speed, bearing_to_target,
   distance_to_target, race_state, seconds_to_gun]`; target is the line
   centre in states 0/2, the buoy in state 1. Actions: Discrete(3)
   turn-left / turn-right / hold (`TURN_RATE = 0.08` rad/step, `DT = 1 s`).
   Boat speed comes from a polar diagram (`_polar_speed`): no-go < 40° TWA,
   beam reach fastest, dead run 0.55. Wind randomized per episode
   (direction uniform, speed 4–12 m/s). Course constants are PUBLIC
   module-level names (`WORLD_W`, `BUOY_POS`, …) because analysis code
   imports them.
2. **Rewards** (named constants at top of env.py — deliberately tweakable):
   `STEP_PENALTY -0.05`, `PROGRESS_REWARD_PER_M +0.01` (potential-style
   delta-distance toward the current leg's target, measured against the
   state at the start of the step so phase transitions don't spike),
   `START_BONUS +10`, `ROUNDING_BONUS +20`, `FINISH_BONUS +100`,
   `OUT_OF_BOUNDS_PENALTY -20` (leaving the 1000×1200 m area TERMINATES).
   Timeouts (3000 steps) are truncations, never terminals — the training
   loop does not bootstrap-through-done on them.
3. **From-scratch DQN** (`rl/`): `QNetwork` MLP 8→128→128→3 ReLU;
   numpy ring `ReplayBuffer`; `DQNAgent` (ε-greedy, target net synced every
   1000 steps, Huber TD loss, Adam lr 1e-4, grad-norm clip 10, γ=0.99);
   `DQNConfig` dataclass saved as `config.json` with every run.
4. **Instrumentation** — the run-directory contract (`rl/logger.py`
   docstring is normative): `runs/<name>/{config.json, episodes.csv,
   training.csv, evals.csv, eval/traj_step<N>_ep<I>.json,
   checkpoints/ckpt_<N>.pt, model.pt}`. `episodes.csv` records, per
   episode, which phases were reached and at what in-episode step
   (`start_step/round_step/finish_step`, −1 if never) plus an `oob` column
   (older runs lack it — parse CSVs by header). `analysis/plots.py` renders
   six diagnostic PNGs from a run dir; `analysis/animate.py` renders any
   trajectory JSON into a race GIF/MP4 (boat hull + boom eased by TWA,
   phase-colored trail, countdown HUD, outcome banner); `rl/evaluate.py`
   is both the in-training evaluator and a checkpoint-inspection CLI.
5. **Conventions**: phase colors are fixed everywhere —
   pre-start `#2a78d6`, to-the-mark `#008300`, to-the-finish `#4a3aa7`;
   run-comparison palette order `#2a78d6, #1baf7a, #eda100, #4a3aa7,
   #e34948, #e87ba4`. Analysis scripts: stdlib+numpy+matplotlib only, Agg
   backend, repo-root sys.path bootstrap, import course geometry from
   `sailing_env.env` with literal fallbacks, degrade gracefully on missing
   files. `runs/` and `*.pt` are gitignored.

### Unmerged branches (all pushed to origin)
- **`claude/gymnasium-race-states-wt737w`** — "phase 1": adds
  `NO_GO_PENALTY -0.05`/step while pinching (TWA < `NO_GO_TWA_DEG = 40`),
  `info["in_no_go"]`, `reset(options={"wind_direction":…, "wind_speed":…})`
  for pinned-wind evaluation, and `analysis/wind_sweep.py` (greedy rollouts
  over a compass grid of pinned winds → polar success chart + CSV).
  **No PR opened yet.** Merge this first.
- **`claude/boat-inertia`** — momentum physics: `BOAT_ACCEL = 0.15`,
  `_boat_speed += (target − speed) · BOAT_ACCEL · DT`. Boat accelerates
  gradually and coasts through tacks. REBASED ONTO the phase-1 branch
  (combined `_update_physics` sets `_in_no_go`, then lags toward target)
  — merge it second, after phase 1. Tests adapted (boats pinned to speed
  in rigged tests; no-go test asserts speed *drops*, not an absolute).
- **`claude/value-policy-viz`** — `analysis/value_map.py`: batched
  Q-evaluation over a position×heading grid for a pinned wind/phase →
  two-panel value heatmap + best-heading quiver over the course map.
  Based on main; merges independently.
- **`claude/double-dqn-ablation`** — `DQNConfig.double_dqn` flag +
  `--double-dqn` train flag (online-net argmax, target-net evaluation)
  and `analysis/compare_runs.py` (overlay reward / finish-fraction /
  started+rounded curves for several run dirs on a shared global_step
  axis). Based on main; merges independently.

### Experimental results so far
All CPU, ~1400–2400 env-steps/s (a 500k-step run ≈ 3.5 min; 2M ≈ 23 min).
Run dirs live only in the session container (gitignored) — treat them as
disposable and regenerate as needed.
- **Sparse rewards only** (pre-shaping `runs/demo`, 300k steps): every
  episode a 3000-step timeout; never rounded the buoy. Root cause visible
  in trajectory plots: boat drifts off-world, no gradient toward course.
- **Shaping + OOB termination** (`runs/shaped`, 500k steps): first full
  greedy races from ~375k steps (best: start-loiter → timed start →
  rounding → finish in 461 s, reward +122.8). Wind sweep of final
  checkpoint: 31% started / 31% rounded / **12% finished**, finishes only
  with northerly wind.
- **Phase 1: + no-go penalty, ε decay 500k, 2M steps** (`runs/phase1`):
  late training windows 55–80% full races (at ε=0.05); wind sweep of
  `ckpt_2000000`: 50% started / 42% rounded / **35% finished**, finishes
  from NE/E/SE/W/NW as well as N. Remaining gap: **southerly winds**
  (downwind start + upwind return leg) still mostly fail; 35% of sweep
  episodes end out-of-bounds. Also observed: greedy evals are noisy/worse
  than ε=0.05 behavior at some checkpoints (classic DQN loop-lock;
  worth a look — see future work).

### Gotchas a future agent should know
- Reward-magnitude tests assert RANGES (e.g. `9 < r < 11`), because
  shaping/time terms are additive with the bonuses.
- `_rigged_env()` in tests pins wind AND (post-inertia) `_boat_speed`;
  tests teleport `env._boat_pos` and step once — respect momentum.
- Trained policies are wed to `NormalizeObservation` ([-1,1] from the env
  Box bounds): any obs-layout change invalidates old checkpoints.
- Old `runs/demo` episodes.csv lacks the `oob` column; all CSV readers
  must parse by header name.
- Subagent worktrees are cut from **main**, not the current branch —
  if delegating work that must stack on an unmerged branch, say so
  explicitly (this bit us once: boat-inertia had to be hand-rebased).
- Physics changes (e.g. inertia) invalidate the *learning problem*:
  policies trained pre-inertia won't transfer; retrain after merging.

## 2. Future development plans

### A. Merge queue and consolidation run (do first)
Merge order: phase-1 branch → boat-inertia → viz/ablation branches (any
order). Then retrain a fresh reference policy under the merged physics:
`python train.py --run-name reference --total-steps 2000000
--eps-decay-steps 500000 --buffer-size 200000` and re-run
`analysis/wind_sweep.py` + `analysis/value_map.py` on it. This becomes the
baseline all later experiments compare against (inertia makes prior run
artifacts non-comparable).

### B. Close the southerly-wind gap
Diagnose first: `wind_sweep.py --csv`, filter failing directions, animate
those trajectories (`animate.py` accepts any traj JSON; wind can be pinned
via `run_episode(..., options=…)`). Hypotheses to test, in order:
1. **More exploration where it fails** — a wind curriculum: sample wind
   uniformly but oversample the failing arc (simple: rejection-resample
   from per-direction success EMA kept in the training loop; add a
   `wind_curriculum: bool` config flag).
2. **VMG shaping** — replace/augment delta-distance with velocity-made-good
   toward the target so downwind legs get honest credit for gybing angles.
3. **Longer runs / n-step returns** if credit assignment over the long
   return leg is the issue.
Success metric: ≥70% finish on a 16-direction × 5-episode sweep.

### C. Racing-craft behaviors (phase 2, blog chapter)
1. **Start-timing bonus**: replace flat `START_BONUS` with
   `START_BONUS · exp(−(t_start − 60)/TIMING_TAU)` (suggest TIMING_TAU≈30 s);
   keep a floor (e.g. +2) so late starts still beat not starting.
   `start_timing.png` in plots.py is already built to show the before/after.
2. **Steering-smoothness cost**: `TURN_PENALTY ≈ −0.01` per turn action —
   expect visibly cleaner GIF tracks and possibly slower learning; good
   before/after animation pair.
3. Optional realism: OCS rule (over line at gun → must clear back below
   the line before the start counts — implement as a flag in the state
   machine rather than a reward hack).

### D. Deliberate failure modes (phase 3, blog chapter)
Cheap, high-value content; each is a small env fork + 500k-step run:
1. **Reward speed** (`+k·boat_speed` instead of progress): expect eternal
   beam-reach circling. 2. **Proximity instead of progress**
   (`+k·(1 − d/D)` potential *level*, not delta): expect parking next to
   the buoy. Animate both. Frame as "reward hacking you can see."

### E. Algorithm ablation (uses the double-dqn branch)
Protocol: 3 seeds × {vanilla, double-DQN} × 2M steps under identical
configs (≈ 2.5 h CPU total; consider doing after F). Compare with
`analysis/compare_runs.py`; check whether double-DQN tames the `max_q`
inflation visible in `training.csv`. Natural extensions afterward:
dueling head, n-step returns, prioritized replay — one branch each, same
protocol.

### F. Throughput: vectorized envs
The env is pure numpy; the bottleneck is per-step torch inference with
batch=1. Either (a) numpy-vectorize `SailingEnv` internals into a
`VectorSailingEnv(n_envs)` and batch action selection, or (b) wrap N envs
with Gymnasium's SyncVectorEnv and batch `select_action` across them.
(b) is a one-day change to `train.py` (per-env episode accumulators;
`episodes.csv` semantics unchanged). Target ≥10k steps/s → makes the
seed-matrix ablations in E trivial.

### G. Blog asset pipeline (nice-to-have)
A `make_assets.py` that, given a run dir, regenerates every figure/GIF the
post needs into `assets/` with stable filenames: plots.py output,
wind_sweep, value_map per phase, one finish GIF + one failure GIF, and the
compare_runs figure against the previous reference run. Keeps the post
reproducible from checkpoints.

### H. Endgame: a second boat
Multi-agent match racing (self-play on the start line, port/starboard
right-of-way). Big lift: env becomes 2-boat (obs gains opponent bearing/
distance/heading), reward becomes relative (beat the other boat), training
becomes self-play against frozen past checkpoints. Only attempt after B/E
are settled and vectorization (F) exists. The payoff GIF — two triangles
duelling for the pin end at the gun — closes the blog post.

## 3. Command cheat sheet

```bash
python -m pytest tests/ -q                          # full suite (~4 s)
python train.py --run-name X --total-steps 2000000 \
    --eps-decay-steps 500000 --buffer-size 200000    # reference recipe
python analysis/plots.py runs/X                      # 6 diagnostic PNGs
python analysis/wind_sweep.py runs/X/checkpoints/ckpt_2000000.pt --csv out.csv
python analysis/animate.py runs/X                    # GIF of latest eval ep 0
python analysis/value_map.py runs/X/checkpoints/ckpt_2000000.pt --phase 1
python analysis/compare_runs.py runs/A runs/B        # (double-dqn branch)
python -m rl.evaluate runs/X/checkpoints/ckpt_2000000.pt --episodes 5
```
