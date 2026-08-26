"""Run one experiment honestly and say whether to keep it.

Upstream has no driver script and does not need one: the agent IS the driver,
pointed at `program.md`. What upstream also does not need is this file, because
bits-per-byte on a fixed seed is reproducible and a single number can be
compared to a single number.

Ours cannot. Kamino and MuJoCo are not bit-deterministic - the WS-1 handoff
measured 3% between two identical DR Legs rollouts and 11% across three - so a
single run tells you almost nothing, and a loop that accepts single-run
improvements will spend the night accepting noise. This runs the experiment
`prepare.EVAL_REPEATS` times, compares the mean against the baseline using the
observed spread as the bar, and refuses to call anything inside the noise a
win.

    uv run newton_policy/experiment.py --note "lookahead 0/100/300/600ms"
    uv run newton_policy/experiment.py --baseline --note "unmodified"

The verdict is a recommendation, not an action: it does not touch git. Deciding
what to keep is the agent's job and should stay visible.
"""

import argparse
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare  # noqa: E402  - read-only, owns the metric

RESULTS = HERE / "results.tsv"
HEADER = ("when\tnote\tcompletion_mean\tcompletion_spread\truns\t"
          "baseline\tverdict\tdiff_lines\n")


def run_once():
    """One training run. Returns the completion_rate it reports, or None."""
    proc = subprocess.run(
        [sys.executable, "-u", str(HERE / "train.py")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        print(f"  run FAILED (exit {proc.returncode})\n{tail}\n{proc.stderr[-800:]}")
        return None
    m = re.search(r"^completion_rate:\s*([\d.]+)", proc.stdout, re.M)
    if not m:
        print("  run produced no completion_rate - did train.py print the summary?")
        return None
    return float(m.group(1))


def baseline_from_results():
    """The first recorded row is the baseline, as in upstream's protocol."""
    if not RESULTS.exists():
        return None, None
    rows = [r.split("\t") for r in RESULTS.read_text().splitlines()[1:] if r.strip()]
    for r in rows:
        if len(r) > 6 and r[6] == "BASELINE":
            return float(r[2]), float(r[3])
    return None, None


def diff_lines():
    """How much of train.py changed, so a big win from a big diff is visible."""
    try:
        out = subprocess.run(["git", "diff", "--numstat", "--", "newton_policy/train.py"],
                             cwd=HERE.parent, capture_output=True, text=True).stdout
        if out.strip():
            a, d = out.split()[0], out.split()[1]
            return int(a) + int(d)
    except Exception:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="what changed, in a few words")
    ap.add_argument("--baseline", action="store_true",
                    help="record this run as THE baseline to compare against")
    ap.add_argument("--repeats", type=int, default=prepare.EVAL_REPEATS)
    args = ap.parse_args()

    if not RESULTS.exists():
        RESULTS.write_text(HEADER)

    scores = []
    for i in range(args.repeats):
        print(f"[experiment] run {i + 1}/{args.repeats}: {args.note}")
        s = run_once()
        if s is None:
            print("[experiment] aborting - a run did not complete")
            return 1
        print(f"  completion_rate {s:.4f}")
        scores.append(s)

    mean = statistics.fmean(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    base, base_spread = baseline_from_results()

    if args.baseline or base is None:
        verdict = "BASELINE"
        print(f"\n[experiment] BASELINE completion_rate {mean:.4f}  spread {spread:.4f}")
    else:
        # The bar is the larger of the two spreads. A candidate has to clear
        # the noise of BOTH the thing it is beating and itself - otherwise a
        # noisy candidate wins by being noisy.
        bar = max(spread, base_spread or 0.0)
        delta = mean - base
        if delta > bar:
            verdict = "KEEP"
        elif delta < -bar:
            verdict = "DISCARD"
        else:
            verdict = "NEUTRAL"
        print(f"\n[experiment] {mean:.4f} vs baseline {base:.4f}   "
              f"delta {delta:+.4f}   noise bar {bar:.4f}")
        print(f"[experiment] VERDICT: {verdict}")
        if verdict == "NEUTRAL":
            print("[experiment] inside the noise - this is not a result. "
                  "Record it and move on rather than re-running until it wins.")

    with RESULTS.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t"
                f"{args.note}\t{mean:.4f}\t{spread:.4f}\t{args.repeats}\t"
                f"{'' if base is None else f'{base:.4f}'}\t{verdict}\t"
                f"{diff_lines()}\n")
    print(f"[experiment] recorded in {RESULTS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
