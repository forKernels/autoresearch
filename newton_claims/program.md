# autoresearch: are the documented measurements still true?

Adapted from `newton_contact/`, which adapted karpathy's `program.md`, which
`newton_policy/` did first. The shape is theirs. This one is not an
optimisation loop and the difference is the point.

## What is being researched

Whether newton-lab's `CLAUDE.md` still describes the newton that is installed.

Nearly every measurement in that file was taken against newton 1.5.0. The
add-on now ships 1.6.0.dev0. `CLAUDE.md` says so about itself:

> All were measured against 1.5.0 and none re-probed against 1.6.0.dev0, so
> treat any that a change depends on as worth re-checking.

That is a standing invitation and nothing had accepted it. In one session
before this harness existed, four documented claims were found stale or wrong
BY HAND - Kamino's shape gate, CONE under MuJoCo, XPBD's `enable_restitution`,
and a contact-buffer finding that was the harness's own bug. Four in one
session, found by accident, is the argument for doing it on purpose.

## This is not an optimisation loop

| | `newton_policy` / `newton_contact` | here |
|---|---|---|
| moves | one scalar | nothing |
| decides | KEEP / DISCARD / NEUTRAL | AGREE / DISAGREE / COULD NOT TEST |
| output | a better number | a diff to the documentation |
| ends | never | when every claim is checked |

There is no metric to climb, so there is no metric to game - which removes the
failure that both of the other harnesses had to be designed around. What
replaces it is a different failure, and the whole of this file is arranged
against it: **a claim that quietly passes because it never really ran.**

## The three files

| file | role |
|---|---|
| `claims.py` | **read-only.** The registry: each claim's id, source, verbatim quote, and the experiment that settles it. |
| `verify.py` | the runner. Chooses what to run, prints a table, writes `results.tsv`. |
| `program.md` | this file. The contract. |

`claims.py` is read-only for a cousin of the reason `prepare.py` is. Those own
a METRIC and are fenced off because an agent that can reach the metric
optimises the metric. This owns an ASSERTION and the experiment that settles
it, and an agent that can reach the experiment can make any claim come out
true - which is worse, because a fabricated AGREE is indistinguishable from a
real one and gets written into a document that people then trust.

## The verdicts

- **AGREE** - the assertion reproduced.
- **DISAGREE** - it did not, and the detail line carries what happened instead.
  This is the valuable output. It is not a test failure and does not fail the
  process.
- **COULD NOT TEST** - and always with a reason. A claim that raises becomes
  this, carrying the exception. A claim whose solver is missing returns it
  itself. **Never a pass, and never a silent skip.**

`verify.py` exits non-zero only on COULD NOT TEST, because that is the one
outcome meaning this harness has stopped covering something it says it covers.

## The harness re-measures the ASSERTION, not the original fixture

The easiest thing here to misread, so it is stated in `claims.py` as well.

`CLAUDE.md` records `SolverVBD 40 mm -> 0/12 through, 30 mm -> 3/12, 26 mm ->
5/12`. The 3 and the 5 belong to a scene that no longer exists - a particular
grid, spacing, drop height and density, none of them written down. `claims.py`
builds its OWN fixture, states it in code, and tests the part that is
fixture-independent: 40 mm holds everything, the smaller sizes do not, and
26 mm is no better than 30 mm. Both numbers are printed; the direction decides
the verdict.

So a DISAGREE means the ASSERTION failed to reproduce. It never means a count
moved by one. (In the measured run below, that claim came back 0 / 4 / 5
against a documented 0 / 3 / 5, on a fixture built from scratch - which is
closer than the claim needed to be to stand.)

## The rule inherited from `newton_contact`, and why it is repeated here

**Construct the world the way the PRODUCT constructs it.**

Every collision pipeline comes from `sim.make_pipeline`, never
`newton.CollisionPipeline`. The add-on sizes the rigid contact buffer to
`max(16384, bodies * 512)`; newton's own default is 11000; an overflowed buffer
does not raise, it warns and DROPS contacts. `newton_contact/program.md` lists
the four findings that died when this was fixed there, including "the contact
buffer overflows at 200 bricks", which it does not.

The same reasoning routes stepping through `sim.Stepper`, finalizing through
`sim.finalize`, solver construction through `sim.make_stepper`, and calls
`sim.use_coordinate_targets` at bootstrap because `Sim.__init__` does.

Two claims deliberately go around the product, and both say so in place:
`explicit-mass-adds-to-density-mass` calls `builder.add_body` raw, because the
claim is about newton's API and because `call_flexible` would DROP a keyword
that had been removed and turn that into a silent pass; and
`vbd-requires-builder-color` calls `builder.finalize()` raw, because the claim
is about what happens when the product's own precaution is absent.

## The claims, and what they came back as

Measured 2026-08-26 against newton 1.6.0.dev0 / warp 1.16.0 on cuda:0, the
bundled build loaded from `C:\_git\blender_deps\newton`.

| claim | verdict | measured |
|---|---|---|
| `mesh-size-floor-mujoco` | AGREE | 18 mm 0/32 through, 16 mm 32/32 |
| `mesh-size-floor-vbd` | AGREE | 40 mm 0/12, 30 mm 4/12, 26 mm 5/12 |
| `plane-floor-holds-a-pile` | RUNNING | not yet recorded - see below |
| `fabric-drape-ladder` | AGREE | chiffon 0.531 -> leather 0.658, monotonic |
| `cloth-rests-at-its-thickness` | AGREE | 3 mm rests +0.0030, 1 mm rests +0.0010 |
| `explicit-mass-adds-to-density-mass` | AGREE | 1000 kg alone, 1005 kg with `mass=5.0` |
| `joint-qd-is-linear-then-angular` | AGREE | linear slot travels, angular slot spins |
| `vbd-requires-builder-color` | AGREE | uncoloured raises, coloured constructs |
| `kamino-refuses-particles-not-meshes` | AGREE | refuses cloth; rests on a 2-triangle mesh floor at +0.2000 |
| `mujoco-and-vbd-never-read-restitution` | **DISAGREE** | MuJoCo 0.0 -> 0.0%, 0.7 -> 32.7% of the fall |

### The two disagreements

**MuJoCo reads restitution now.** `CLAUDE.md` says "The control ships and the
value reaches the model. MuJoCo and VBD then never read it." Half of that is
still true - VBD rebounds 0.0% at restitution 0.0 and 0.0% at 0.7 - and half of
it is stale. `sim.apply_mujoco_restitution` writes the value onto
`mjw_model.geom_solref` as a contact dampratio after the solver is built, and
it works: 32.7% of the fall regained at 0.7 against 0.0% at 0.0. The document
was written before that existed and the sentence was never revised.

**The pile claim has not finished running as this is written**, and the row
above says so rather than guessing. Two of its five rows were measured while
building the harness - a tagged PLANE at 8 substeps and a mesh floor at 16
substeps both finished with 0/200 lost, where the document records the mesh
floor losing 23/200 - so a disagreement is likely; likely is not measured, and
this section gets rewritten from `results.tsv` and not from an expectation.

**Neither disagreement is a licence to delete the advice.** The mesh-contact
SIZE floor is untouched - 16 mm still goes straight through a mesh floor and
18 mm still does not, under both solvers, exactly as documented. Any move in
the PILE result is a different failure with a different cause.

## Running it

Blender's Python is the interpreter with newton and warp:

```
"C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe" \
    newton_claims/verify.py
```

```
verify.py --list                     the registry, without importing newton
verify.py                            every claim except the slow ones
verify.py --slow                     everything
verify.py --only <id> --only <id>    just these, slow or not
verify.py --deps <dir> --device cpu  elsewhere
```

A default run prints how many slow claims it skipped and names them. That is
taken from newton-lab's `verify_bake.py`, where the same split exists for the
same reason: **a short run that does not say it was short reads as a complete
one.**

`results.tsv` is appended to, one row per claim per run, and is gitignored -
it is a record of runs, not source.

## The budget

There isn't one, and that is a deliberate difference from the other two
harnesses. They cap wall clock because without a cap "raise substeps until
nothing is lost" is a trivial winning move. Nothing is being won here, so the
only cost discipline needed is honesty about what a run costs: each claim
carries a measured `seconds`, `verify.py` prints the total before starting, and
anything that costs minutes rather than seconds is marked `slow` and excluded
by default.

`plane-floor-holds-a-pile` is the only slow claim and it is slow for a reason
that must not be optimised away: 200 bodies in sustained contact is where the
cost goes super-linear, and shrinking the pile to make it fast would measure a
different claim. `CLAUDE.md` is explicit that the failure is a property of the
PILE - the same bricks in a sparse layer stay put.

## What to do with a DISAGREE

Change the document, not this harness. That is the whole output.

Specifically: re-quote the claim with the new measurement and the date and the
newton version, the way `CLAUDE.md` already does for Kamino ("the reason
recorded here for four releases was WRONG ... Re-probed 2026-08-24"). Do not
delete the old number - the history of what was believed is why several of
these claims were catchable at all.

And check whether anything in `src/` depends on the stale claim before
believing the new one. `sim.MESH_CONTACT_MIN_SIZE` hardcodes 0.017 and the bake
warns buyers on the strength of it; a claim that moved and a constant that did
not is a bug, not a documentation edit.

## Fenced off

- `claims.py` in its entirety, including the quotes, the fixtures and the
  verdict rules.
- Anything under `C:\_git\NewtonLab`. This harness READS newton-lab and never
  writes it. A claim that only reproduces after editing `sim.py` is a finding
  to report, not an experiment to record.
- `newton_policy/` and `newton_contact/`.

## What is still open

- **Ten claims is a start, not coverage.** `docs/newton-measured.md` alone has
  several dozen measured facts, and `CLAUDE.md` carries the scene-scale table
  (0.25x / 1x / 4x across four solvers), the non-finite-under-impact table, and
  the MPM granular-versus-liquid finding. None of those are here yet.
- **`docs/newton-measured.md` is a second document with its own drift.**
  `kamino-refuses-particles-not-meshes` already catches one case where
  `CLAUDE.md` was corrected on 2026-08-24 and `newton-measured.md` was not - it
  still says "Kamino refuses MESH (8), CONVEX_MESH (10) and PLANE (1)". Where
  the two documents disagree, a claim can settle which one is right, and that
  is worth more per claim than checking either alone.
- **SolverKamino warns that `make_pipeline` outsizes it.** Every Kamino step
  prints "Newton `rigid_contact_max` (16384) exceeds Kamino
  `model_max_contacts_host` (1000); active contacts may be truncated". The
  add-on does not ship Kamino so nothing is broken today, but it is the same
  SHAPE of problem as the contact-buffer bug that produced four fictional
  findings in `newton_contact` - a buffer mismatch that warns and truncates
  rather than raising - and it would have to be settled before Kamino could
  ever be a solver choice.
- **Nothing checks that a claim's quote still exists in the document.** A claim
  whose source text has been edited away would keep testing a sentence nobody
  ships. Grepping the quote out of `CLAUDE.md` at run time would close that,
  and it is the cheapest remaining improvement.
