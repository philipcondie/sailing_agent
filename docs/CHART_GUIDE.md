# Reading the diagnostic charts

`analysis/plots.py runs/<name>` writes six PNGs into `runs/<name>/plots/`.
This guide explains what each one shows and how to read it. The illustrative
numbers below come from the first 2M-step baseline run; treat them as an
example of the *shape* of a healthy run, not as targets.

The charts fall into two groups:

- **Training-time views** — computed from *every* episode in `episodes.csv`,
  which includes ε-greedy exploration (ε decays to 0.05).
- **Greedy-eval views** — computed from the periodic deterministic (ε=0)
  evaluations in `evals.csv` (3 episodes every 25k steps). These show the
  "true" policy and are noisier per point, so they lag the training-time
  curves.

One color convention runs through all of them:
**blue = started · green = rounded the mark · violet = finished.**

> Note: any chart's "rounded" (green) series depends on the mark-rounding
> rule in force when the run was trained. Runs trained under the old
> touch-the-radius rule are not comparable to runs under the genuine-rounding
> rule — see the development log.

---

## 1. `learning_curve.png` — is it getting better at all?

Two stacked panels over all episodes. Grey is raw per-episode; the bold line
is a 50-episode rolling mean.

- **Top (episode total reward):** in a healthy run this sits flat and negative
  early (the boat flails and sails out of bounds for the −20 penalty), then
  climbs to a noisy positive plateau once it starts completing races. The raw
  spread fans out late — big positive finishes mixed with occasional failures.
- **Bottom (episode length in steps):** rises as the boat learns to survive and
  complete rather than dying quickly out of bounds. Longer is *not* strictly
  better: a clean finish is a few hundred steps, timeouts are 3000, so a high
  mean can hide slow, meandering races.

## 2. `race_progress.png` — did it learn the three phases, and in what order?

The clearest single chart: rolling fraction of episodes that **started /
rounded / finished**.

- The three curves should emerge **in order** — blue (start) lifts first, green
  (round) trails it, violet (finish) last. That staircase is the agent learning
  the race one leg at a time.
- They always stack correctly (started ≥ rounded ≥ finished — you can't finish
  without rounding without starting). The gap between blue and violet is the
  "starts/rounds but doesn't finish" failure rate.

## 3. `loss_and_q.png` — is the network training sanely?

- **Top:** TD **loss** (red, log scale) with **epsilon** (grey dashed, right
  axis) decaying linearly to 0.05 then flat. Loss stays noisy; its spikes tend
  to *grow* over training because the reward targets get bigger once the agent
  starts collecting the +20/+100 bonuses.
- **Bottom:** **mean Q** (blue) typically dips negative early (returns are
  negative while flailing under step penalties), then recovers as it discovers
  the bonuses. **max Q** (orange) climbs with growing spikes near the end. A
  runaway upward max-Q drift is the classic DQN **over-estimation** tendency —
  the thing the double-DQN ablation is meant to tame.

## 4. `start_timing.png` — does it start well?

For episodes that started, the in-episode step at which the boat crossed the
start line (grey scatter + blue rolling mean). The dashed line is the gun at
**step 60**. A good racer crosses *just after* 60.

- If the rolling mean sits well above 60 with wide scatter, the agent starts
  **late and sloppily** — it hasn't learned to time the gun, it just crosses
  whenever it drifts back over the line. This is the motivation for a
  start-timing bonus.

## 5. `trajectories.png` — what do the final greedy races look like?

Three greedy trajectory maps at the latest checkpoint, colored by phase, with
the wind arrow and an outcome title per panel (steps + reward). Read these for
*quality*, not just success:

- A tight, direct up-and-back track with brief pre-start = an efficient race.
- Heavy zig-zag on a leg = tacking upwind (expected in light/foul wind), slower.
- Loops near the start or wide detours = wasted time bleeding the step penalty
  (finishes but low reward).

## 6. `eval_curve.png` — the greedy (deterministic) counterpart

The ε=0 analog of chart #2, built from the 3-episode greedy evals every 25k
steps.

- **Top:** greedy eval reward — mean line plus a min–max band across the 3
  episodes.
- **Bottom:** started / rounded / finished fractions of the eval episodes.

It shows the *true* deterministic policy rather than exploration-blended
behavior, so it lags chart #2 and jumps around more (only 3 episodes per point).
A greedy policy that scores worse than ε=0.05 behavior at some checkpoints is a
known DQN "loop-lock" effect.

---

**The through-line:** #1 and #6 show *that* it learned; #2 shows *what* it
learned and in what order; #3 confirms training stability and flags Q
over-estimation; #4 and #5 surface the remaining weaknesses — start timing and
route efficiency.
