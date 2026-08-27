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
import collections
import os
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
HEADER = ("when\tnote\tscore_mean\tscore_spread\truns\t"
          "baseline\tverdict\tdiff_lines\tscores\ttrack_err\troot_err_m\t"
          "ee_err_m\tcompletion\n")

#: Results measured before the paired design existed are in
#: `results-unpaired.tsv`. They are kept rather than deleted, and they are NOT
#: read as baselines: their bar came from the spread of two runs launched
#: back-to-back in one process, which measured within-invocation noise and was
#: then applied to a comparison spanning an hour. Same-config across-invocation
#: sigma measured 0.082 against a within-invocation 0.044, so every verdict in
#: that file understates its own error bar by roughly two.


#: Always shown: anything that means the run is in trouble, and the final
#: summary. Never throttled - a failure buried behind a throttle is worse than
#: no streaming at all.
#
# Word-bounded on purpose. An unbounded `nan` matches the `.conan2` in warp's
# USD warning paths, which turned this filter into a firehose of 200 identical
# material-binding warnings - the same substring-against-English mistake that
# made newton-lab's check_parked fire on the word "policy" in a docstring.
_ALWAYS = re.compile(
    r"Traceback|\bError\b|\bassert|out of memory|\bKilled\b|"
    r"\bNaN\b|\bnan\b|"
    r"^tracking_score:|^tracking_error:|^root_error_m:|^ee_error_m:|^comparable:|"
    r"^completion_rate:|^completed:|^terminated:|^episodes:|^training_seconds:",
)
#: Progress. Throttled, because 300 iterations x several lines is 1500 lines
#: per run and a hundred experiments would bury the findings in their own logs.
_PROGRESS = re.compile(r"Learning iteration\s+(\d+)/")
#: Per-iteration metrics, shown only alongside a throttled progress line.
_METRIC = re.compile(r"Mean episode (completion_rate|terminated|length):")

#: Show one progress block every N iterations.
_EVERY = 25


def run_once(index):
    """One training run at repeat `index`, streamed.

    `index` reaches `prepare._seed_training` through the environment rather
    than an argument, because the seeding happens at prepare's IMPORT and
    train.py is the file the agent edits - see prepare.TRAIN_SEEDS.

    Streams rather than capturing, because a driver meant to run unattended
    should not be a black box for six minutes at a time - a hung or thrashing
    run has to be visible while it is happening, not at the timeout. The output
    is parsed on the way past, so nothing is lost by showing it.
    """
    proc = subprocess.Popen(
        [sys.executable, "-u", str(HERE / "train.py")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        env={**os.environ, "NEWTON_POLICY_RUN": str(index)},
    )
    score = None
    extra = {}
    tail = collections.deque(maxlen=40)
    verbose = False
    for raw in proc.stdout:
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
        tail.append(line)

        m = re.match(r"tracking_score:\s*([\d.]+)", line.strip())
        if m:
            score = float(m.group(1))
        m = re.match(r"(tracking_error|root_error_m|ee_error_m|completion_rate)"
                     r":\s*([\d.]+)", line.strip())
        if m:
            extra[m.group(1)] = float(m.group(2))
        if line.strip().startswith("comparable:"):
            extra["comparable"] = line.strip().split()[-1] == "True"

        p = _PROGRESS.search(line)
        if p:
            verbose = int(p.group(1)) % _EVERY == 0
            if verbose:
                print(f"    {line.strip()}", flush=True)
            continue
        if _ALWAYS.search(line):
            print(f"    {line.strip()}", flush=True)
        elif verbose and _METRIC.search(line):
            print(f"    {line.strip()}", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print(f"  run FAILED (exit {proc.returncode}). Last lines:")
        for line in tail:
            print(f"    {line}")
        return None
    if score is None:
        print("  run produced no tracking_score - did train.py print the summary?")
        for line in tail:
            print(f"    {line}")
        return None
    return score, extra


def baseline_from_results():
    """Per-seed scores of the MOST RECENT baseline row, not the first.

    Returns the individual run scores rather than a mean, because the
    comparison is PAIRED: candidate run k is compared against baseline run k,
    which shared its seed and therefore its initialisation. Comparing the means
    instead would leave the seed-to-seed spread in both arms, and that spread
    is the term this design exists to cancel.

    A baseline is only valid for the task it was measured on. Changing the
    reference clip changes the task, so it has to be re-established - and if
    this returned the first row instead, a candidate would be silently compared
    against a baseline measured on a different clip, which is precisely the
    confound the two-run guard exists to prevent.
    """
    if not RESULTS.exists():
        return None
    rows = [r.split("\t") for r in RESULTS.read_text().splitlines()[1:] if r.strip()]
    for r in reversed(rows):
        if len(r) > 8 and r[6] == "BASELINE" and r[8].strip():
            return [float(x) for x in r[8].split(",")]
    return None


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

    scores, extras = [], []
    for i in range(args.repeats):
        seed = prepare.TRAIN_SEEDS[i % len(prepare.TRAIN_SEEDS)]
        print(f"[experiment] run {i + 1}/{args.repeats} (seed {seed}): {args.note}")
        got = run_once(i)
        if got is None:
            print("[experiment] aborting - a run did not complete")
            return 1
        s, extra = got
        scores.append(s)
        extras.append(extra)
        print(f"  tracking_score {s:.5f}   root {extra.get('root_error_m', float('nan')):.5f}"
              f"   ee {extra.get('ee_error_m', float('nan')):.5f}"
              f"   completion {extra.get('completion_rate', float('nan')):.4f}")

    # A tracking error measured while the robot keeps falling is not comparable
    # to one measured while it completes - a falling policy spends more of its
    # time near a reset, which reference state initialisation puts ON the
    # reference. Measured: the 300-iteration checkpoint had the LOWEST root
    # error in the whole budget curve purely by falling. Ranked naively it wins.
    if any(e.get("comparable") is False for e in extras):
        print("[experiment] NOT COMPARABLE: a run completed less than "
              f"{prepare.COMPARABLE_COMPLETION:.0%} of its episodes, so its "
              f"tracking error was measured in a different regime. Raise "
              f"prepare.TRAIN_ITERATIONS rather than reading this as a result.")

    mean = statistics.fmean(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    base = baseline_from_results()

    if args.baseline or base is None:
        verdict = "BASELINE"
        print(f"\n[experiment] BASELINE tracking_score {mean:.5f}  spread {spread:.5f}")
        print(f"[experiment] per-seed {', '.join(f'{x:.5f}' for x in scores)}")
    elif len(base) != len(scores):
        print(f"\n[experiment] baseline has {len(base)} runs, this has "
              f"{len(scores)} - cannot pair. Re-run the baseline at "
              f"--repeats {args.repeats}.")
        return 1
    else:
        d = [c - b for c, b in zip(scores, base)]
        mean_d = statistics.fmean(d)
        paired_spread = (max(d) - min(d)) if len(d) > 1 else 0.0
        floor = prepare.PAIRED_NOISE_FLOOR
        bar = max(paired_spread, floor or 0.0)

        print()
        for k, (b, c, dk) in enumerate(zip(base, scores, d)):
            seed = prepare.TRAIN_SEEDS[k % len(prepare.TRAIN_SEEDS)]
            print(f"[experiment] seed {seed}: {b:.5f} -> {c:.5f}   {dk:+.5f}")
        print(f"[experiment] mean paired delta {mean_d:+.5f}   bar {bar:.5f}"
              f"   (paired spread {paired_spread:.5f}, floor "
              f"{'unmeasured' if floor is None else f'{floor:.5f}'})")

        agree = all(x > 0 for x in d) or all(x < 0 for x in d)
        if not agree:
            verdict = "NEUTRAL"
            print("[experiment] seeds disagree on the SIGN - not an effect.")
        # HIGHER IS BETTER. tracking_score is the reward's own tracking
        # PRODUCT, bounded (0, 1], so a POSITIVE delta is the improvement. This
        # direction has now flipped twice - completion_rate up, tracking_error
        # down, tracking_score up - and getting it backwards ranks every result
        # upside down while still printing confident verdicts.
        elif mean_d > bar:
            verdict = "KEEP"
        elif mean_d < -bar:
            verdict = "DISCARD"
        else:
            verdict = "NEUTRAL"

        if floor is None and verdict in ("KEEP", "DISCARD"):
            print(f"[experiment] would call {verdict}, but PAIRED_NOISE_FLOOR "
                  f"is unmeasured, so the bar has no floor and this may be "
                  f"luck. Run the baseline AGAIN as a candidate (a null "
                  f"experiment) and set prepare.PAIRED_NOISE_FLOOR from it.")
            verdict = "UNCALIBRATED"

        print(f"[experiment] VERDICT: {verdict}")
        if verdict == "NEUTRAL":
            print("[experiment] inside the noise - this is not a result. "
                  "Record it and move on rather than re-running until it wins.")

    def avg(key):
        vals = [e[key] for e in extras if isinstance(e.get(key), float)]
        return statistics.fmean(vals) if vals else float("nan")

    base_mean = "" if base is None else f"{statistics.fmean(base):.5f}"
    with RESULTS.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t"
                f"{args.note}\t{mean:.5f}\t{spread:.5f}\t{args.repeats}\t"
                f"{base_mean}\t{verdict}\t{diff_lines()}\t"
                f"{','.join(f'{x:.5f}' for x in scores)}\t"
                f"{avg('tracking_error'):.5f}\t{avg('root_error_m'):.5f}\t"
                f"{avg('ee_error_m'):.5f}\t"
                f"{avg('completion_rate'):.4f}\n")
    print(f"[experiment] recorded in {RESULTS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
