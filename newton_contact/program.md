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
    run 1  retention 0.6667      run 2  retention 0.3333
    mean   0.5000               spread 0.3333
```

**Read that spread before you read anything else.** Two identical runs differ
by a third of the pile. The metric is quantised in units of ONE LAYER - 16
bricks - because the pile fails by losing whole layers, so 48/48, 32/48, 16/48
and 0/48 are the only values it can take.

That means only LARGE effects are detectable here. A candidate must clear the
baseline by more than one whole layer to count. A one-layer "improvement" seen
once is noise, and `experiment.py` will correctly call it NEUTRAL. Do not
re-run a candidate until it wins.

This is the harness's own first finding, and it is a limitation rather than a
result. Widening the pile would subdivide the step and costs time
super-linearly. That trade is open and unmade.

## What is already known — do NOT spend slots rediscovering this

All measured, on this harness or in newton-lab. **The first two were SINGLE
runs, taken before the noise floor was known, and one whole layer is inside the
noise - so treat them as leads to confirm, not as results.** They are the
obvious first two experiments to run properly.

- **More substeps may not be better here, and single runs suggested it is
  worse.** 4 substeps held 48/48 once; 8 and 16 each lost 16/48 once. One run
  each, and the spread is exactly one layer, so this is a LEAD. If it survives
  four repeats it is a real and counter-intuitive finding; if it does not, it
  was the metric all along.
- **HULL may beat analytic BOX on a mesh floor.** 48/48 against 32/48, one run
  each, at 52 s against 32 s. Same caveat, same status: confirm it.
- **A PLANE floor holds everything** — 48/48, and 0/200 in newton-lab's own
  200-brick measurement where a mesh floor lost 23. PLANE is not a research
  target, it is the known answer. The open question is why MESH fails.
- **Cost is badly super-linear in body count**: 1072 ms/frame at 48 bricks,
  2864 at 100, 14505 at 200. 4.2x the bodies for 13.5x the time.
- **The contact buffer overflows silently at 200 bricks** — "Contact buffer
  overflowed 17580 > 11000" — so contacts are being dropped. Suspect this
  before believing any large-pile result.
- **`iterations` is the contact knob; `rigid_contact_k_start` is not.** The
  latter was tried in newton-lab and changed nothing.
- **Substeps do not fix non-finite under impact.** newton-lab measured a
  156-brick wall diverging at every substep count tried; real-world masses
  fixed it at every mass ratio tried. If doubling substeps twice has not
  helped, the scene is the cause.
- **The mesh-contact size floor is ~17 mm under MuJoCo and ~40 mm under VBD.**
  The bricks here are 45 mm, deliberately clear of it, so this scene is
  measuring STACKING and not the size floor.

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
