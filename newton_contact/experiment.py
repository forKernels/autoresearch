"""Run one contact experiment honestly and say whether to keep it.

Derived from `newton_policy/experiment.py`, which is where the design lives,
and re-derived from it whenever that one improves so the two stay in lockstep.
Everything that matters here is Thor's: run the candidate more than once,
make the bar the LARGER of the candidate's spread and the baseline's so a
noisy candidate cannot win by being noisy, stream the child rather than
swallowing it, and take the MOST RECENT baseline rather than the first -
because a baseline is only valid for the task it was measured on, and the
scene here changed during development.

What differs is only the metric. This reads `retention` - the share of a
48-brick pile still finite and above the floor after 60 frames - from
`newton_contact/train.py`.

    <blender-python> newton_contact/experiment.py --baseline --note "unmodified"
    <blender-python> newton_contact/experiment.py --note "4 substeps"

It must run under Blender's Python: that is the interpreter with newton and
warp. The verdict is a recommendation, not an action - it does not touch git.
"""

import argparse
import collections
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
HEADER = ("when\tnote\tretention_mean\tretention_spread\truns\t"
          "baseline\tverdict\tdiff_lines\n")


#: Always shown: anything that means the run is in trouble, and the summary.
#: Never throttled - a failure buried behind a throttle is worse than no
#: streaming at all.
#
# Word-bounded on purpose, and the reason is Thor's: an unbounded `nan` matches
# the `.conan2` in warp's USD warning paths and turns this filter into a
# firehose of identical material-binding warnings.
#
# "Contact buffer overflowed" is in here deliberately. It is a WARNING in
# newton, not an error - it drops contacts and carries on - so a run that hits
# it produces a plausible number measured in a world with less contact than it
# should have. That is the single most misleading thing this harness can do,
# and it is exactly what an early version of prepare.py did to itself.
_ALWAYS = re.compile(
    r"Traceback|Error|error|assert|out of memory|Killed|"
    r"NaN|nan|Contact buffer overflowed|TIMEOUT|"
    r"^retention:|^retained:|^lost:|^non_finite_frame:|^wall_seconds:",
)

#: A contact bake prints no per-iteration progress - it is one build followed
#: by a fixed frame loop - so there is nothing here to throttle, and the
#: progress/metric machinery the policy harness needs has been REMOVED rather
#: than left matching nothing. A pattern that cannot fire reads as coverage it
#: does not provide.

def run_once():
    """One bake, streamed.

    Streams rather than capturing, because a driver meant to run unattended
    should not be a black box - a hung or thrashing run has to be visible while
    it is happening, not at the timeout. The output is parsed on the way past,
    so nothing is lost by showing it.

    A contact bake is quieter than a training run: one build, then a fixed
    frame loop. So the filter shows trouble and the summary, and there is no
    progress stream to throttle.
    """
    proc = subprocess.Popen(
        [sys.executable, "-u", str(HERE / "train.py")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    score = None
    tail = collections.deque(maxlen=40)
    for raw in proc.stdout:
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
        tail.append(line)

        m = re.match(r"retention:\s*([\d.]+)", line.strip())
        if m:
            score = float(m.group(1))

        if _ALWAYS.search(line):
            print(f"    {line.strip()}", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print(f"  run FAILED (exit {proc.returncode}). Last lines:")
        for line in tail:
            print(f"    {line}")
        return None
    if score is None:
        print("  run produced no retention - did train.py print the summary?")
        for line in tail:
            print(f"    {line}")
        return None
    return score


def baseline_from_results():
    """The MOST RECENT baseline row, not the first.

    A baseline is only valid for the task it was measured on. Changing the
    reference clip changes the task, so it has to be re-established - and if
    this returned the first row instead, a candidate would be silently compared
    against a baseline measured on a different clip, which is precisely the
    confound the two-run guard exists to prevent.
    """
    if not RESULTS.exists():
        return None, None
    rows = [r.split("\t") for r in RESULTS.read_text().splitlines()[1:] if r.strip()]
    for r in reversed(rows):
        if len(r) > 6 and r[6] == "BASELINE":
            return float(r[2]), float(r[3])
    return None, None


def diff_lines():
    """How much of train.py changed, so a big win from a big diff is visible."""
    try:
        out = subprocess.run(["git", "diff", "--numstat", "--", "newton_contact/train.py"],
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
        print(f"  retention {s:.4f}")
        scores.append(s)

    mean = statistics.fmean(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    base, base_spread = baseline_from_results()

    if args.baseline or base is None:
        verdict = "BASELINE"
        print(f"\n[experiment] BASELINE retention {mean:.4f}  spread {spread:.4f}")
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
