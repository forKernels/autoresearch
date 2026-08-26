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

One experiment is **300 iterations at 512 environments** — about five minutes at
the measured ~11,200 environment-steps per second, so roughly twelve per hour.
Run it as `uv run newton_policy/train.py`.

**The goal: the highest `completion_rate`.**

Of the episodes that ended, the share that ended by reaching the end of the clip
rather than by falling over.

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

`completion_rate` is bounded by the policy. `terminations()` already returned
`failed` and `timed_out` separately the whole time; nobody printed them.

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
completion_rate:  0.8734
completed:        1203
terminated:       174
episodes:         1377
training_seconds: 298.4
iterations:       300
envs:             512
```

Record `completion_rate` in `results.tsv` along with what changed and whether it
was kept.
