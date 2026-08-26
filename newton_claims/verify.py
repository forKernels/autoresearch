"""Re-measure the documented claims against the installed newton.

    "<blender>/python/bin/python.exe" newton_claims/verify.py
    "<blender>/python/bin/python.exe" newton_claims/verify.py --slow
    "<blender>/python/bin/python.exe" newton_claims/verify.py --only vbd-requires-builder-color

It must run under Blender's Python: that is the interpreter with newton and
warp. `claims.py` owns the assertions and the experiments and is read-only;
this file only decides what to run and how to print it.

Deliberately not `experiment.py`. The other two harnesses in this repo compare
a candidate against a baseline and print KEEP / DISCARD / NEUTRAL, because they
are moving a number. There is no number to move here and no baseline to beat -
a claim either reproduces or it does not - so the runner is a table and a TSV,
not a verdict machine.

Two things it will not do, both for the same reason:

- **A claim that raises becomes COULD NOT TEST with the exception**, never a
  pass and never a silent skip. A harness that quietly drops what it cannot run
  reports a clean sweep it did not earn.
- **A default run prints how many slow claims it skipped.** Taken from
  verify_bake.py, where the same split exists and for the same reason: a short
  run that does not say it was short reads as a complete one.
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claims  # noqa: E402  - read-only, owns the assertions

RESULTS = HERE / "results.tsv"
HEADER = "when\tclaim\tsource\tverdict\tseconds\tdetail\n"


def one(ctx, c):
    """Run a single claim. Returns (verdict, detail, seconds).

    Every exception is caught and becomes COULD NOT TEST. A claim that blew up
    is information - it usually means the API it depends on moved - and the
    exception text is the most useful thing this harness can hand back, so it
    is kept rather than reduced to a category.
    """
    started = time.perf_counter()
    try:
        verdict, detail = c.measure(ctx)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:400]
        return claims.UNTESTED, detail, time.perf_counter() - started
    return verdict, detail, time.perf_counter() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", default=os.environ.get("NEWTON_DEPS", claims.DEPS),
                    help="directory holding newton and warp (pip --target)")
    ap.add_argument("--device", default="cuda:0",
                    help="warp device; falls back if this machine has no CUDA")
    ap.add_argument("--only", action="append", default=[], metavar="ID",
                    help="run just this claim; repeatable")
    ap.add_argument("--slow", action="store_true",
                    help="include the claims that cost minutes rather than "
                         "seconds")
    ap.add_argument("--list", action="store_true",
                    help="print the registry and exit, without importing newton")
    args = ap.parse_args()

    if args.list:
        for c in claims.CLAIMS:
            mark = " (slow)" if c.slow else ""
            print(f"{c.id}{mark}\n    {c.source}\n    ~{c.seconds:.0f}s")
        return 0

    wanted = list(claims.CLAIMS)
    if args.only:
        unknown = [i for i in args.only if claims.find(i) is None]
        if unknown:
            print(f"no such claim: {', '.join(unknown)}")
            print("known: " + ", ".join(c.id for c in claims.CLAIMS))
            return 2
        wanted = [claims.find(i) for i in args.only]
        skipped = []
    else:
        skipped = [c for c in wanted if c.slow and not args.slow]
        wanted = [c for c in wanted if c.slow is False or args.slow]

    try:
        ctx = claims.Context(deps=args.deps, device=args.device)
    except ImportError as exc:
        print(f"cannot import newton from {args.deps or '(none)'}: {exc}")
        print("pass --deps <dir> or set NEWTON_DEPS")
        return 2

    v = ctx.versions()
    print(f"newton {v['newton']} / warp {v['warp']}")
    print(f"device {ctx.device}" + (f"  ({ctx.device_note})"
                                    if ctx.device_note else ""))
    prov = ctx.sim.provenance()
    print(f"newton loaded from {prov['path']}"
          + ("  (the bundled build)" if prov["bundled"] else
             f"  (NOT the bundled {ctx.sim.BUNDLED_NEWTON})"))
    budget = sum(c.seconds for c in wanted)
    print(f"{len(wanted)} claims, roughly {budget / 60:.0f} min")
    print()

    if not RESULTS.exists():
        RESULTS.write_text(HEADER)

    rows = []
    for c in wanted:
        print(f".. {c.id}", flush=True)
        verdict, detail, seconds = one(ctx, c)
        rows.append((c, verdict, detail, seconds))
        print(f"{verdict:<14} {c.id}  ({seconds:.1f}s)")
        print(f"               {detail}")
        print(flush=True)

    with RESULTS.open("a") as f:
        when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for c, verdict, detail, seconds in rows:
            f.write(f"{when}\t{c.id}\t{c.source}\t{verdict}\t"
                    f"{seconds:.1f}\t{detail}\n")

    print("=" * 72)
    for c, verdict, detail, seconds in rows:
        print(f"{verdict:<14} {c.id}")
    print("=" * 72)

    tally = {}
    for _c, verdict, _d, _s in rows:
        tally[verdict] = tally.get(verdict, 0) + 1
    print("  ".join(f"{v}: {n}" for v, n in sorted(tally.items())))

    if skipped:
        # Never let a short run read as a complete one.
        print(f"{len(skipped)} slow claim(s) NOT run: "
              + ", ".join(c.id for c in skipped)
              + ". Pass --slow to include them.")

    disagreed = [c.id for c, verdict, _d, _s in rows if verdict == claims.DISAGREE]
    if disagreed:
        print()
        print("The documentation is wrong about: " + ", ".join(disagreed))
        print("That is the output of this harness - a diff to the docs.")

    print(f"recorded in {RESULTS.name}")
    # A DISAGREE is a finding, not a failure, so it does not fail the process.
    # Only a claim that could not run at all does, because that means this
    # harness stopped covering something it claims to cover.
    return 1 if tally.get(claims.UNTESTED) else 0


if __name__ == "__main__":
    raise SystemExit(main())
