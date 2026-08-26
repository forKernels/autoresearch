# autoresearch: Newton rigid contact and stacking

Adapted from karpathy's `program.md` and from `newton_policy/`, which did this
first for reference-motion policies. The shape is theirs; the feature is
different. Read `../docs`-equivalent context in newton-lab's `CLAUDE.md` if you
have it — this file carries the parts that matter.

## What is being researched

Whether a pile of rigid bodies stays where it was put. This is newton-lab's
most-documented failure and it is the axis that decides how many objects a
scene can hold, which is the thing buyers actually ask for.

## The three files

| file | role |
|---|---|
| `train.py` | you edit this. `SETTINGS` is fair game, all of it. |
| `prepare.py` | **read-only.** Owns the scene, the budget and the metric. |
| `program.md` | this file. The contract. |

**The read-only evaluator is the whole safety property.** An agent that can
reach the metric optimises the metric; an agent that can reach the SCENE
discovers that a pile of four bricks never loses one. So the pile, its
arrangement, the frame count and the wall-clock cap are `prepare.py`'s and are
not yours.

## The metric

`retention` — the share of bodies still finite and above the floor when the
bake ends. Higher is better; 1.0 is everything held.

It passes Thor's test: the knobs move it (measured), and the sampling cannot,
because the scene is built in `prepare.py`. A mean penetration depth was the
tempting alternative and is wrong — a mean hides the failures, and the failures
are the point.

## The budget

60 frames, 48 bricks, and a **90-second wall-clock cap**. A run that exceeds
the cap is a failure, exactly as an OOM is upstream. The cap exists because
without it there is a trivial winning move — raise substeps until nothing is
lost — which is not a finding, it is buying retention with time nobody has.

## The baseline, measured

```
mesh floor, BOX bricks, 16 substeps, 20 iterations, MuJoCo
    4 runs   retention 1.0000   spread 0.0000   ~14.3 s each
```

**This baseline has NO HEADROOM, and that is the honest state of this harness
right now.** It holds 48/48 every time. A loop pointed at it cannot improve
anything and would spend the night recording NEUTRAL.

## What this harness got wrong about itself, twice

Both are recorded because both are the mistake to avoid repeating, and the
second one invalidated everything the first one appeared to find.

**1. It built its own collision pipeline.** Early versions called
`newton.CollisionPipeline(model)` directly. newton-lab does not: it sizes the
rigid contact buffer to `max(16384, bodies * 512)` in `sim.make_pipeline`,
while newton's raw default is 11000. So this harness ran with a smaller contact
buffer than any real bake, and an overflowed buffer does not raise - it warns
and DROPS contacts.

**2. Everything it then "found" was that bug.** With the correct buffer, the
whole picture changes:

| | undersized buffer | correct buffer |
|---|---|---|
| retention | 0.5000 | **1.0000** |
| spread over runs | 0.3333 | **0.0000** |
| wall clock | 32.4 s | **14.3 s** |

So all of these are DEAD and must not be carried forward:

- ~~"more substeps is worse - 4 holds 48/48, 8 and 16 lose 16/48"~~ - dropped
  contacts, not solver behaviour.
- ~~"HULL beats analytic BOX on a mesh floor"~~ - same cause.
- ~~"the metric is hopelessly noisy, spread 0.3333"~~ - the run-to-run spread
  is ZERO on this scene. MuJoCo is far more reproducible here than the policy
  harness's Kamino rollouts, and the noise was the buffer overflowing a
  different number of times per run.
- ~~"the contact buffer overflows at 200 bricks"~~ - it does not, through
  `make_pipeline`.

**The one lesson that survives, and it is worth more than the findings it
destroyed: construct the world the way the PRODUCT constructs it.** A harness
that reaches past the add-on's own setup is not measuring the add-on, and it
will produce confident, reproducible, entirely fictional results.

## What this harness needs next

A scene with real headroom, since the current one is solved. Unmade choices,
roughly in order of promise:

- **More bricks.** 200 still finishes; cost is super-linear (1072 ms/frame at
  48, 2864 at 100, 14505 at 200 - measured with the undersized buffer, so
  re-measure) but the budget cap is the constraint to respect.
- **Impact rather than settling.** newton-lab's own hard case is a wrecking
  ball into a wall, which diverges at every substep count when the bricks are
  sub-kilogram. That has known headroom.
- **Bricks near the mesh-contact size floor** (~17 mm MuJoCo, ~40 mm VBD), so
  the scene sits where the product is known to fail.

Do NOT simply shrink the pile until it fails - `prepare.py` owns the scene
precisely so the metric cannot be gamed by choosing an easier or harder world
to suit the result.

## What is still known, and still stands

From newton-lab, independent of this harness:

- **`iterations` is the contact knob; `rigid_contact_k_start` is not.**
- **Substeps do not fix non-finite under impact.** A 156-brick wall diverged at
  every substep count tried; real-world masses fixed it at every mass ratio.
- **The mesh-contact size floor is ~17 mm under MuJoCo, ~40 mm under VBD.**
- **A PLANE floor holds a pile where a mesh floor historically did not** -
  0/200 against 23/200. Worth re-checking now that the buffer question is
  settled, because that measurement predates it.

## Fenced off — these are contracts, not knobs

- `prepare.py` in its entirety.
- The scene: brick count, sizes, arrangement, drop gap, frame count, cap.
- The metric and `LOST_BELOW`.
- Anything in newton-lab's `src/` — this harness researches SETTINGS, not the
  add-on's source. A change that only works by editing `sim.py` is a finding to
  report, not an experiment to record.

## The noise floor

MuJoCo is not bit-deterministic. newton-lab measured 3% between two identical
Kamino runs and 11% across three; rigid is expected to be tighter, but that is
an expectation and not a measurement. So:

- run each candidate `prepare.EVAL_REPEATS` times
- accept only a gain LARGER than the observed spread
- a gain inside the noise is neutral, not a win

`experiment.py` enforces this. Use it rather than eyeballing single runs.

## Running it

```
python newton_contact/experiment.py --baseline --note "unmodified"
python newton_contact/experiment.py --note "4 substeps"
```

It runs the candidate `EVAL_REPEATS` times, compares the mean against the
baseline using the spread as the bar, and prints a verdict. The verdict is a
recommendation and does not touch git — deciding what to keep stays visible.

Requires Blender's Python (it is the interpreter that has newton and warp):

```
"<blender>/python/bin/python.exe" newton_contact/experiment.py --baseline
```

## The experiment loop

LOOP:

1. Look at the git state.
2. Edit `SETTINGS` in `train.py` with one idea.
3. `git commit`
4. `python newton_contact/experiment.py --note "<what you tried>"`
5. Read the verdict. Record the row in `results.tsv`.
6. If it beat the baseline by more than the spread, keep the commit and it
   becomes the new baseline. Otherwise `git reset` back.

**Ideas worth trying, roughly in order of promise** — the known-results list
above already rules several things out, so start where it is silent:

- the substep curve below 4, and between 4 and 8, since 4 beats 8
- `iterations` at fixed substeps, both directions
- friction between brick and floor, independently
- CAPSULE / CYLINDER / ELLIPSOID bricks, now that they exist
- solver: COUPLED and VBD against MuJoCo on this scene
- density, and whether the mass ratio to the floor matters at all
- combinations of the two known wins (4 substeps AND HULL)

**When a change produces the same number as the baseline three times, suspect
the metric or the knob before the fourth hypothesis.** That lesson cost
`newton_policy` about seven hours; it is written up in newton-lab's
`docs/research/2026-08-26-the-150-step-ceiling-that-was-not.md`.

**NEVER STOP** once the loop has begun. Do not ask whether to continue. If you
run out of ideas, re-read the known-results list for something it does not
cover, or combine two near-misses.
