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
import math
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
#: The training entry point, as a module-level name so an end-to-end test can
#: point it at a shortened run. Nothing else should rebind it.
TRAIN_SCRIPT = HERE / "train.py"
HEADER = ("when\tnote\tscore_mean\tscore_spread\truns\t"
          "baseline\tverdict\tdiff_lines\tscores\ttrack_err\troot_err_m\t"
          "ee_err_m\tcompletion\tmodes\tper_seed\n")

#: Index of the `per_seed` column in RESULTS. Derived from HEADER rather than
#: written by hand, so extending the row cannot silently repoint it.
PER_SEED_COL = HEADER.rstrip("\n").split("\t").index("per_seed")

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
        [sys.executable, "-u", str(TRAIN_SCRIPT)],
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
        m = re.match(r"(tracking_error|root_error_m|ee_error_m|ori_error_rad"
                     r"|vel_error|completion_rate):\s*([\d.]+)", line.strip())
        if m:
            extra[m.group(1)] = float(m.group(2))
        if line.strip().startswith("comparable:"):
            extra["comparable"] = line.strip().split()[-1] == "True"
        if line.strip().startswith("mode:"):
            extra["mode"] = line.strip().split()[-1]

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


#: Two-sided 95% critical values of Student's t, by degrees of freedom.
#:
#: Tabulated rather than imported because scipy on this box is an ABI hazard -
#: the apt build is linked against numpy 1 and this project pins numpy 2 - and
#: because a t-table is eight lines and does not need a dependency that can
#: break the harness at 3am. df>20 falls back to the normal limit; nothing here
#: runs anywhere near that many repeats.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
             11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
             16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def paired_test(d):
    """Paired t-test on the per-seed differences. Pure arithmetic, tested.

    Replaces two n=2 heuristics that between them decided every verdict this
    project has recorded:

      1. `all(x > 0) or all(x < 0)` - unanimity of sign. At n=2 this is a coin
         flip on the noisier seed. It is what rejected k_ee: the differences
         were +0.01870 and -0.01524, and the negative one exists because the
         BASELINE drew 0.89421 on that seed while its own null re-run of the
         same seed drew 0.85832. Measured against the null's draw instead, both
         seeds agree and the sign rule passes. A verdict should not depend on
         which of two equally valid baseline draws a candidate was paired with.
      2. `bar = max(paired_spread, floor)` - beat the worse of two draws. Has
         no confidence attached and grows with the outlier it is meant to guard
         against.

    Both are subsumed here: consistency of sign shows up as a small standard
    error, and an outlier inflates it honestly instead of becoming the bar.

    `significant` is |t| > crit at two-sided 95%. Two-sided on purpose: a
    candidate that makes things reliably WORSE is a result worth recording, and
    a one-sided test at the same alpha would be a looser bar for KEEP than the
    number suggests.
    """
    n = len(d)
    mean = statistics.fmean(d) if n else 0.0
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None,
                "t": None, "df": None, "crit": None, "significant": False}
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    df = n - 1
    crit = T_CRIT_95.get(df, 1.96)
    # se == 0 means every seed moved by exactly the same amount. That is either
    # a deterministic effect (mean != 0) or nothing at all (mean == 0); it is
    # not a division to guard against by returning "no result".
    t = (mean / se) if se else (math.inf if mean else 0.0)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": t, "df": df, "crit": crit, "significant": abs(t) > crit}


def mode_fault(modes):
    """What is wrong with this arm's MODES, or None if its runs agree and walk.

    One place, because the baseline branch and the candidate branch have to
    answer it identically. They did not at first: a mixed baseline was refused
    and a uniformly-standing one was recorded as usable, which is the worse of
    the two - every later candidate would then be paired against a robot
    standing still.
    """
    if modes and len(set(modes)) > 1:
        return "MIXED"
    if modes and modes[0] == "standing":
        return "DEGENERATE"
    return None


def verdict(d, floor, modes, direction=+1):
    """The call, from the differences, the floor, and which policies were found.

    Mode comes FIRST and outranks the statistics, because an arm that found two
    different policies did not measure one thing noisily - it measured two
    things. k_root_vel=20 scored 0.33009 and 0.87239; averaging those to 0.60124
    and reporting NEUTRAL says "no effect" about a run that half the time
    collapsed to standing still. That is a bistability to report, not noise to
    average, and the floor does not apply across modes.

    Significance and size are then separate gates, in that order. A difference
    the test cannot resolve is NEUTRAL whatever its size; a difference it CAN
    resolve but that lands under the floor is also NEUTRAL, because resolvable
    and worth keeping are different claims. Without a measured floor the second
    gate cannot be applied at all, so a resolvable effect reports UNCALIBRATED
    rather than borrowing a bar from a statistic it was not measured on.

    `direction` is +1 for higher-is-better and -1 for lower-is-better, and it
    belongs to the METRIC - `prepare.METRICS` - not to this comparison. Callers
    pass it through; nothing here decides it.
    """
    st = paired_test(d)
    fault = mode_fault(modes)
    if fault:
        return fault, st
    if not st["significant"]:
        return "NEUTRAL", st
    if floor is None:
        return "UNCALIBRATED", st
    if abs(st["mean"]) <= floor:
        return "NEUTRAL", st
    # `direction` comes from prepare.METRICS, never from a comparison written
    # here. tracking_score is higher-better and ee_error_m is lower-better, so
    # a hardcoded `> 0` is right for exactly one of the metrics this now tests
    # and silently inverts every verdict on the others.
    return ("KEEP" if st["mean"] * direction > 0 else "DISCARD"), st


def _parse_per_seed(cell):
    """`name:v,v,v;name:v,v,v` -> {name: [floats]}. Empty cell -> {}."""
    out = {}
    for part in cell.strip().split(";"):
        if ":" not in part:
            continue
        name, _, vals = part.partition(":")
        try:
            out[name.strip()] = [float(x) for x in vals.split(",") if x.strip()]
        except ValueError:
            continue
    return out


def baseline_from_results(metric="tracking_score"):
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
        if not (len(r) > 8 and r[6] == "BASELINE" and r[8].strip()):
            continue
        # Rows written before per-seed recording carry only tracking_score, in
        # column 8. They stay usable for that metric and are simply absent for
        # the others - which is the honest answer, not a fallback to a mean.
        #
        # PER_SEED_COL is named rather than inlined: it was written as 13 first
        # (the index before `modes` was added) and silently returned the modes
        # column, which parses to {} and looks exactly like "not recorded".
        per_seed = _parse_per_seed(r[PER_SEED_COL]) if len(r) > PER_SEED_COL else {}
        if metric in per_seed:
            return per_seed[metric]
        if metric != "tracking_score":
            return None
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
    ap.add_argument("--target", default="tracking_score",
                    choices=sorted(prepare.METRICS),
                    help="the metric the VERDICT is read from. Pre-register it: "
                         "one primary, chosen before the run. Every other metric "
                         "is reported as a guard, never as a verdict, so five "
                         "simultaneous tests cannot be mined for a winner.")
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

    # Per-metric, per-run. tracking_score arrives separately because run_once
    # parses it as the score; the rest ride in `extra`.
    per_metric = {"tracking_score": list(scores)}
    for name in prepare.METRICS:
        if name == "tracking_score":
            continue
        vals = [e.get(name) for e in extras]
        if all(isinstance(v, float) for v in vals):
            per_metric[name] = vals

    if args.target not in per_metric:
        print(f"\n[experiment] target {args.target!r} was not reported by every "
              f"run - cannot pair on it.")
        return 1

    modes = [e.get("mode", "unknown") for e in extras]
    if len(set(modes)) > 1:
        print(f"\n[experiment] MIXED MODES: {', '.join(modes)}")
        print("[experiment] this arm found more than one policy, so its seeds "
              "did not measure the same thing and their spread is a "
              "mode-switch rate rather than an error bar. Report the "
              "bistability; do not average across it.")
    elif modes and modes[0] == "standing":
        print(f"\n[experiment] DEGENERATE: every run stands still in the clip "
              f"(root error > {prepare.WALKING_ROOT_ERR_MAX} m) while "
              f"completing it. That is the failure tracking_score exists to "
              f"expose, not a policy to rank.")

    mean = statistics.fmean(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    target_vals = per_metric[args.target]
    base = baseline_from_results(args.target)

    if args.baseline or base is None:
        # A baseline that is bimodal, or that is standing still, would make
        # every future candidate pair against it. `baseline_from_results`
        # matches the verdict string EXACTLY, so the suffix is what keeps a
        # faulty baseline from ever being picked up as one.
        fault = mode_fault(modes)
        call = f"BASELINE-{fault}" if fault else "BASELINE"
        print(f"\n[experiment] {call} tracking_score {mean:.5f}  spread {spread:.5f}")
        print(f"[experiment] per-seed {', '.join(f'{x:.5f}' for x in scores)}")
        if fault:
            print(f"[experiment] NOT recorded as a usable baseline ({fault}) - "
                  f"nothing can be paired against this. Fix the arm and re-run "
                  f"before measuring any candidate against it.")
    elif len(base) != len(target_vals):
        print(f"\n[experiment] baseline has {len(base)} runs, this has "
              f"{len(target_vals)} - cannot pair. Re-run the baseline at "
              f"--repeats {args.repeats}.")
        return 1
    else:
        d = [c - b for c, b in zip(target_vals, base)]
        floor = prepare.PAIRED_NOISE_FLOORS[args.target]
        direction = prepare.METRICS[args.target]
        call, st = verdict(d, floor, modes, direction)

        print()
        arrow = "higher is better" if direction > 0 else "LOWER is better"
        print(f"[experiment] target {args.target} ({arrow})")
        for k, (b, c, dk) in enumerate(zip(base, target_vals, d)):
            seed = prepare.TRAIN_SEEDS[k % len(prepare.TRAIN_SEEDS)]
            print(f"[experiment] seed {seed}: {b:.5f} -> {c:.5f}   "
                  f"{dk:+.5f}   {modes[k]}")
        # HIGHER IS BETTER. tracking_score is the reward's own tracking PRODUCT,
        # bounded (0, 1], so a POSITIVE delta is the improvement. This direction
        # has now flipped twice - completion_rate up, tracking_error down,
        # tracking_score up - and getting it backwards ranks every result upside
        # down while still printing confident verdicts.
        if st["se"]:
            print(f"[experiment] mean paired delta {st['mean']:+.5f}   "
                  f"sd {st['sd']:.5f}   se {st['se']:.5f}   "
                  f"t {st['t']:+.3f} vs crit {st['crit']:.3f} (df {st['df']})")
        else:
            print(f"[experiment] mean paired delta {st['mean']:+.5f}   "
                  f"(n={st['n']} - nothing to estimate a spread from)")
        print("[experiment] floor "
              + ("unmeasured" if floor is None else f"{floor:.5f}"))

        if call == "UNCALIBRATED":
            print("[experiment] the effect RESOLVES, but PAIRED_NOISE_FLOOR is "
                  "unmeasured, so there is no size gate and this may be real "
                  "and negligible. Run the baseline AGAIN as a candidate (a "
                  "null experiment) at these repeats, then set "
                  "prepare.PAIRED_NOISE_FLOOR from its |mean paired delta|.")

        # Guards. Reported, never a verdict: the primary was pre-registered
        # above, and promoting whichever of five deltas happens to clear its
        # bar is exactly the mining this harness exists to prevent.
        for name in sorted(per_metric):
            if name == args.target:
                continue
            gbase = baseline_from_results(name)
            if not gbase or len(gbase) != len(per_metric[name]):
                continue
            gd = [c - b for c, b in zip(per_metric[name], gbase)]
            gst = paired_test(gd)
            way = "+" if prepare.METRICS[name] > 0 else "-"
            better = "better" if gst["mean"] * prepare.METRICS[name] > 0 else "worse"
            mark = "resolves" if gst["significant"] else "in noise"
            print(f"[experiment] guard {name:>15} ({way}) "
                  f"{gst['mean']:+.5f}  {better:>6}, {mark}")

        print(f"[experiment] VERDICT: {call}  [{args.target}]")
        if call == "NEUTRAL":
            print("[experiment] inside the noise - this is not a result. "
                  "Record it and move on rather than re-running until it wins.")

    def avg(key):
        vals = [e[key] for e in extras if isinstance(e.get(key), float)]
        return statistics.fmean(vals) if vals else float("nan")

    base_mean = "" if base is None else f"{statistics.fmean(base):.5f}"
    with RESULTS.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t"
                f"{args.note}\t{mean:.5f}\t{spread:.5f}\t{args.repeats}\t"
                f"{base_mean}\t{call}\t{diff_lines()}\t"
                f"{','.join(f'{x:.5f}' for x in scores)}\t"
                f"{avg('tracking_error'):.5f}\t{avg('root_error_m'):.5f}\t"
                f"{avg('ee_error_m'):.5f}\t"
                f"{avg('completion_rate'):.4f}\t"
                f"{','.join(modes)}\t"
                # Per-seed, per-metric, so ANY metric can be paired later
                # without re-running - a mean cannot be un-averaged.
                + ";".join(f"{n}:" + ",".join(f"{v:.6g}" for v in vs)
                           for n, vs in sorted(per_metric.items())) + "\n")
    print(f"[experiment] recorded in {RESULTS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
