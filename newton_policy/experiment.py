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
HEADER = ("when\tnote\tcompletion_mean\tcompletion_spread\truns\t"
          "baseline\tverdict\tdiff_lines\tscores\n")

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
    r"Traceback|\bError\b|\berror\b|\bassert|out of memory|\bKilled\b|"
    r"\bNaN\b|\bnan\b|"
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
    tail = collections.deque(maxlen=40)
    verbose = False
    for raw in proc.stdout:
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
        tail.append(line)

        m = re.match(r"completion_rate:\s*([\d.]+)", line.strip())
        if m:
            score = float(m.group(1))

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
        print("  run produced no completion_rate - did train.py print the summary?")
        for line in tail:
            print(f"    {line}")
        return None
    return score


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

    scores = []
    for i in range(args.repeats):
        seed = prepare.TRAIN_SEEDS[i % len(prepare.TRAIN_SEEDS)]
        print(f"[experiment] run {i + 1}/{args.repeats} (seed {seed}): {args.note}")
        s = run_once(i)
        if s is None:
            print("[experiment] aborting - a run did not complete")
            return 1
        print(f"  completion_rate {s:.4f}")
        scores.append(s)

    mean = statistics.fmean(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    base = baseline_from_results()

    if args.baseline or base is None:
        verdict = "BASELINE"
        print(f"\n[experiment] BASELINE completion_rate {mean:.4f}  spread {spread:.4f}")
        print(f"[experiment] per-seed {', '.join(f'{x:.4f}' for x in scores)}")
    elif len(base) != len(scores):
        # Pairing is by POSITION, and position is seed. Different lengths mean
        # the two arms did not face the same set of initialisations, so the
        # difference would mix a real effect with a seed swap.
        print(f"\n[experiment] baseline has {len(base)} runs, this has "
              f"{len(scores)} - cannot pair. Re-run the baseline at "
              f"--repeats {args.repeats}.")
        return 1
    else:
        # PAIRED: candidate run k against baseline run k, same seed, so the
        # initialisation cancels instead of contributing to both arms.
        d = [c - b for c, b in zip(scores, base)]
        mean_d = statistics.fmean(d)
        paired_spread = (max(d) - min(d)) if len(d) > 1 else 0.0
        floor = prepare.PAIRED_NOISE_FLOOR
        bar = max(paired_spread, floor or 0.0)

        print()
        for k, (b, c, dk) in enumerate(zip(base, scores, d)):
            seed = prepare.TRAIN_SEEDS[k % len(prepare.TRAIN_SEEDS)]
            print(f"[experiment] seed {seed}: {b:.4f} -> {c:.4f}   {dk:+.4f}")
        print(f"[experiment] mean paired delta {mean_d:+.4f}   bar {bar:.4f}"
              f"   (paired spread {paired_spread:.4f}, floor "
              f"{'unmeasured' if floor is None else f'{floor:.4f}'})")

        # Direction has to agree across seeds. Two differences that clear a bar
        # on average while pointing opposite ways is not an effect, it is two
        # draws from a wide distribution that happened to average well.
        agree = all(x > 0 for x in d) or all(x < 0 for x in d)
        if not agree:
            verdict = "NEUTRAL"
            print("[experiment] seeds disagree on the SIGN - not an effect.")
        elif mean_d > bar:
            verdict = "KEEP"
        elif mean_d < -bar:
            verdict = "DISCARD"
        else:
            verdict = "NEUTRAL"

        # A bar built only from two paired differences can be arbitrarily small
        # by luck, which is the same failure that made the unpaired results
        # unreadable. Until the null experiment measures how far apart two runs
        # of the SAME config on the SAME seed land, no verdict is trustworthy.
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

    base_mean = "" if base is None else f"{statistics.fmean(base):.4f}"
    with RESULTS.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t"
                f"{args.note}\t{mean:.4f}\t{spread:.4f}\t{args.repeats}\t"
                f"{base_mean}\t{verdict}\t{diff_lines()}\t"
                f"{','.join(f'{x:.4f}' for x in scores)}\n")
    print(f"[experiment] recorded in {RESULTS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
