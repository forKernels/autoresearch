# autoresearch — reference-motion tracking policies

An adaptation of karpathy's autoresearch to the WS-1 animation↔policy loop. The
structure is his: one editable file, one read-only file that owns the metric, a
fixed budget per experiment, keep-or-discard, repeat.

The subject is different. Instead of a language model's bits-per-byte, the
subject is a **reinforcement-learning policy that makes a Unitree G1 track a
reference motion**, and the metric is whether the robot finishes the clip or
falls over.

## Setup

1. **Agree a run tag** and create `autoresearch/<tag>` from master. The branch
   must not already exist.
2. **Read the in-scope files.**
   - `newton_policy/prepare.py` — fixed constants and the metric. Read-only.
   - `newton_policy/train.py` — the file you edit.
   - The trainer itself is `newton-lab/src/newton_lab/rl/`. Read it for
     context; see below for what you may and may not touch there.
3. **Verify the reference exists**: `newton-lab/runs/g1_v05.npz`. If missing,
   ask the human to run `tests/capture_g1_reference.py`.
4. **Initialise `results.tsv`** with a header row.
5. **First run establishes the baseline**, unmodified.

## Experimentation

One experiment is **1000 iterations at 512 environments** — about twenty minutes,
so roughly three per hour. Run it as `uv run newton_policy/train.py`.

**The goal: the LOWEST `tracking_error`.**

Mean absolute joint deviation from the reference, in radians, over every step
where the policy was live. **Lower is better** — the opposite direction from the
metric this replaces, so a candidate that improves is one whose delta is
NEGATIVE.

### Why the budget is 1000 and not 300

Measured on `runs/20260825-151727`, one run, one eval seed, budget the only
variable:

```
 300 -> completion 0.2960      750 -> 1.0000     2000 -> 1.0000
 500 -> completion 0.9841     1000 -> 1.0000     7999 -> 1.0000
```

There is a phase transition between 300 and 500. The old budget of 300 did not
merely undertrain — it sat ON the cliff, the steepest part of the curve, where
slightly more training is a great deal more completion. That is why two runs of
an identical config scattered by 0.13 and why six consecutive experiments
resolved nothing at all.

### Why the metric is no longer `completion_rate`

Because it saturates. From ~750 iterations every policy finishes every episode —
1228 of them, zero terminations, identically — so the number cannot tell a good
policy from a perfect one. Below ~500 it is on the cliff and measures mostly
noise. Its useful range is roughly 200 iterations wide.

That makes it the **third** metric on this project bounded by something other
than policy quality: `mean episode length` by the sampling scheme,
`completion_rate` by saturation. Tracking error is bounded by neither. Measured
across the five checkpoints that all complete 100% of episodes, it spans 3.02x.

### Read all three numbers, not just the first

`train.py` reports `root_error_m` and `ee_error_m` beside `tracking_error`, and
they do not move together. Across one run's checkpoints:

```
iter    tracking_err   root_err_m   ee_err_m
 750         0.04160      0.76729    0.03216
2000         0.04742      0.90544    0.02330
7999         0.02233      0.94145    0.01988
```

Joint and end-effector tracking improve with training; **ROOT position tracking
gets steadily worse** - 0.94 m of drift on a 3.06 m walk, still growing at 8000
iterations with `k_root` in the reward the whole time. The policy is buying limb
accuracy with body drift.

So a candidate that lowers `tracking_error` while raising `root_error_m` has
traded, not improved. Say so rather than reporting the win.

### `comparable` is a guard, not a formality

A tracking error measured while the robot keeps falling is not comparable to one
measured while it completes: a falling policy spends more of its time near a
reset, and reference state initialisation puts that reset exactly ON the
reference. Measured — the 300-iteration checkpoint had the LOWEST root error in
the entire budget curve, purely by falling. Ranked naively, it wins.

### What you CAN change

Everything in `newton_policy/train.py`: reward weights, alive bonus, action
penalty, action scale, lookahead horizon, PPO hyperparameters, network shape,
initial policy standard deviation, steps per environment.

### What you CANNOT change

- **`newton_policy/prepare.py`.** It owns the metric, the budget and the
  evaluation environment. Read-only.
- **The observation layout** (`newton_lab/rl/env_base.py`). It IS the ONNX
  contract the desktop bake reads. Changing its order silently breaks a
  consumer you cannot see.
- **The coordinate/DOF index tables** (`indices.py`). Addressing a
  coordinate-space array with DOF indices is an off-by-one that produces no
  error, only a policy that will not train.
- **PADMM solver settings.** Measured and reverted: lowering the iteration cap
  is 3.3x faster, settles within 0.3 mm, trains a whole iteration, and then
  returns an UNCONVERGED solution that throws the robot 8 m through the floor.
- **The reference clip.** Shortening it or slowing it is optimising the exam.

## The rule this harness exists to enforce

**A metric bounded by the sampling scheme is not a performance metric.**

This project spent roughly seven hours moving `mean episode length`. Under
reference state initialisation each episode starts at a uniformly random point
in the clip, so an episode that runs perfectly to the end lasts
`duration - start`, and `E[duration - U(0, duration)] = duration/2` **exactly**.
The number was arithmetically pinned. Five separate interventions — an alive
bonus, robot-scaled reward weights, a slower capture, doubled action authority,
an action-magnitude penalty — all produced it, and 34M→98M training steps moved
it by two.

Pointed at that metric, this loop would have produced a hundred confident null
results overnight.

So: **if a metric will not move, suspect the metric before the fifth
hypothesis.** And when several changes produce the same number, that sameness is
itself the finding.

`terminations()` already returned `failed` and `timed_out` separately the whole
time; nobody printed them.

And then `completion_rate`, which fixed that, turned out to be bounded too — by
saturation rather than by sampling. **Ask of any metric: what ELSE could pin
this number?** Twice now the answer was something other than the policy.

## Running an experiment

Use the runner, not `train.py` directly:

    uv run newton_policy/experiment.py --baseline --note "unmodified"
    uv run newton_policy/experiment.py --note "lookahead 0/100/300/600ms"

It runs the experiment `EVAL_REPEATS` times, compares the mean against the
recorded baseline using the larger of the two spreads as the bar, appends a row
to `results.tsv`, and prints KEEP / DISCARD / NEUTRAL.

**It does not touch git.** Deciding what to keep is yours and should stay
visible in the diff.

**A NEUTRAL is a result.** Record it and move on. Re-running a candidate until
it happens to clear the bar is how a noisy metric gets mistaken for a finding,
and this project has already paid for that lesson once.

## Accepting a change

The physics is **not deterministic**. The WS-1 handoff measured 3% between two
identical DR Legs rollouts and 11% across three. Therefore:

- Run each candidate **twice** (`prepare.EVAL_REPEATS`).
- Accept only if it beats the baseline by **more than the observed spread**.
- A gain inside the noise is not a gain. Record it as neutral and move on.

## Worth trying, in rough order

These are untried or were tried against the broken metric and so say nothing:

1. **`lookahead_seconds`** — currently 0/20/40/80 ms, inherited from a 0.265 m
   robot and never questioned for a 0.76 m one, whose dynamics are slower. A
   humanoid may need to see a fall coming further ahead than 80 ms.
2. **PPO hyperparameters and network shape** — genuinely untouched.
3. **`alive_bonus`, `action_penalty`, `action_scale`** — all measured against
   `mean episode length`, so those results are void. Redo them honestly.
4. **`init_std`** — 0.3 was picked because an action is a residual in units of
   0.4 rad and unit variance shakes the robot apart. Never tuned.

Already measured, so re-running wastes a slot: lowering PADMM iterations
(catastrophic), `collect_solver_info` (10x slower, silently), and more steps
once the policy already completes (34M→98M changed nothing).

## Simplicity

Same criterion as upstream. A small gain that adds ugly complexity is not worth
it; an equal result from deleting code is a win. This codebase has been bitten
repeatedly by constants that belonged to one robot and were carried to another,
so prefer a change that DERIVES a value over one that hardcodes it.

## Output

`train.py` prints:

```
---
tracking_error:   0.02233
root_error_m:     0.94145
ee_error_m:       0.01988
comparable:       True
completion_rate:  1.0000
completed:        1228
terminated:       0
episodes:         1228
training_seconds: 1198.4
iterations:       1000
envs:             512
```

`experiment.py` records all of them. `tracking_error` decides the verdict;
the other three are how you tell an improvement from a trade.
