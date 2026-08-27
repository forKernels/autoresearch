"""Tests for the two pieces of verdict logic that are pure arithmetic.

Everything else in this harness needs a GPU and twenty minutes. These two do
not, and they are the parts that decide whether a result is called a result -
so they are the parts worth pinning down. Every number below that is labelled
"measured" is copied from `results.tsv` or `results-noee.tsv`, not invented.

Run: python3 -m pytest newton_policy/test_metric.py -q
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment  # noqa: E402
import prepare  # noqa: E402


# --- mode classification ----------------------------------------------------

#: Measured `root_err_m`, walking arms: the baseline, the null, and k_ee, all
#: at k_root_vel=10. This is the band the guard must never flag.
WALKING_MEASURED = [0.08027, 0.08003, 0.08074, 0.07518, 0.07910, 0.08794]

#: Measured `root_err_m`, standing-still policies. 0.89071 is the tracking_score
#: baseline the k_root_vel work was written to expose; 0.94104 is the
#: 8000-iteration policy an earlier post-mortem recorded as a success.
STANDING_MEASURED = [0.89071, 0.94104]


@pytest.mark.parametrize("root_err", WALKING_MEASURED)
def test_walking_runs_classify_as_walking(root_err):
    assert prepare.classify_mode(root_err) == "walking"


@pytest.mark.parametrize("root_err", STANDING_MEASURED)
def test_standing_runs_classify_as_standing(root_err):
    assert prepare.classify_mode(root_err) == "standing"


def test_threshold_sits_clear_of_both_measured_bands():
    """The guard is only meaningful if it is not near either cluster."""
    assert max(WALKING_MEASURED) < prepare.WALKING_ROOT_ERR_MAX
    assert prepare.WALKING_ROOT_ERR_MAX < min(STANDING_MEASURED)
    # And with real margin, not by a hair.
    assert prepare.WALKING_ROOT_ERR_MAX > 2 * max(WALKING_MEASURED)
    assert prepare.WALKING_ROOT_ERR_MAX < 0.5 * min(STANDING_MEASURED)


def test_boundary_is_inclusive_on_walking():
    assert prepare.classify_mode(prepare.WALKING_ROOT_ERR_MAX) == "walking"
    assert prepare.classify_mode(prepare.WALKING_ROOT_ERR_MAX + 1e-9) == "standing"


def test_infinite_root_error_is_standing_not_a_crash():
    """`evaluate_tracking` returns inf when no step was live."""
    assert prepare.classify_mode(float("inf")) == "standing"


# --- the paired test --------------------------------------------------------

def test_k_ee_measured_differences_are_not_significant():
    """The real k_ee result: n=2, and nowhere near resolvable.

    baseline 0.86985, 0.89421  ->  k_ee 0.88855, 0.87897
    """
    st = experiment.paired_test([+0.01870, -0.01524])
    assert st["n"] == 2
    assert st["mean"] == pytest.approx(0.00173, abs=1e-5)
    assert not st["significant"]


def test_null_measured_differences_are_not_significant():
    """The null experiment must not read as an effect - it IS the control."""
    st = experiment.paired_test([+0.00551, -0.03589])
    assert not st["significant"]


def test_consistent_effect_at_six_seeds_is_significant():
    d = [0.020, 0.021, 0.019, 0.022, 0.018, 0.020]
    st = experiment.paired_test(d)
    assert st["n"] == 6 and st["df"] == 5
    assert st["crit"] == pytest.approx(2.571, abs=1e-3)
    assert st["significant"]


def test_symmetric_noise_at_six_seeds_is_not_significant():
    d = [0.010, -0.010, 0.020, -0.020, 0.005, -0.005]
    assert not experiment.paired_test(d)["significant"]


def test_a_small_but_perfectly_consistent_effect_resolves():
    """This is the case n=2 could not see and six seeds can.

    +0.0017 on every seed is the k_ee effect size. Six seeds agreeing on it
    is a result; two seeds disagreeing on its sign is not.
    """
    st = experiment.paired_test([0.0017] * 5 + [0.0016])
    assert st["significant"]
    assert st["mean"] > 0


def test_zero_variance_with_zero_mean_is_not_an_effect():
    st = experiment.paired_test([0.0, 0.0, 0.0, 0.0])
    assert not st["significant"]


def test_single_run_cannot_be_tested():
    st = experiment.paired_test([0.0173])
    assert st["n"] == 1
    assert not st["significant"]
    assert st["se"] is None


# --- verdicts ---------------------------------------------------------------

WALK6 = ["walking"] * 6
FLOOR = 0.002


def test_mixed_modes_block_a_verdict():
    """k_root_vel=20 measured 0.33009 and 0.87239 - one seed collapsed.

    Averaging those into a mean and calling it NEUTRAL is the bug: the arm did
    not do one thing noisily, it did two different things.
    """
    modes = ["standing", "walking", "walking", "walking", "walking", "walking"]
    v, _ = experiment.verdict([0.02] * 6, FLOOR, modes)
    assert v == "MIXED"


def test_uniformly_standing_arm_is_degenerate_not_a_keep():
    v, _ = experiment.verdict([0.02] * 6, FLOOR, ["standing"] * 6)
    assert v == "DEGENERATE"


def test_consistent_positive_effect_is_a_keep():
    v, _ = experiment.verdict([0.020, 0.021, 0.019, 0.022, 0.018, 0.020],
                              FLOOR, WALK6)
    assert v == "KEEP"


def test_consistent_negative_effect_is_a_discard():
    v, _ = experiment.verdict([-0.020, -0.021, -0.019, -0.022, -0.018, -0.020],
                              FLOOR, WALK6)
    assert v == "DISCARD"


def test_significant_but_below_the_floor_is_neutral():
    """Statistically resolvable and practically nothing is still nothing."""
    d = [0.0002, 0.00021, 0.00019, 0.00022, 0.00018, 0.00020]
    st = experiment.paired_test(d)
    assert st["significant"], "precondition: the test can resolve it"
    v, _ = experiment.verdict(d, FLOOR, WALK6)
    assert v == "NEUTRAL"


def test_unmeasured_floor_refuses_to_call_a_keep():
    """The harness must not rank against a bar it has not measured."""
    v, _ = experiment.verdict([0.020, 0.021, 0.019, 0.022, 0.018, 0.020],
                              None, WALK6)
    assert v == "UNCALIBRATED"


def test_unmeasured_floor_still_reports_neutral_normally():
    """NEUTRAL needs no floor - it is the claim that nothing was shown."""
    v, _ = experiment.verdict([0.010, -0.010, 0.020, -0.020, 0.005, -0.005],
                              None, WALK6)
    assert v == "NEUTRAL"


# --- the configuration these tests assume -----------------------------------

def test_seeds_cover_the_repeats_without_reuse():
    """Six repeats must be six distinct initialisations, not two cycled thrice.

    `_seed_training` indexes TRAIN_SEEDS[run % len], so a short seed list
    silently turns independent paired differences into repeated measures.
    """
    assert len(prepare.TRAIN_SEEDS) >= prepare.EVAL_REPEATS
    assert len(set(prepare.TRAIN_SEEDS)) == len(prepare.TRAIN_SEEDS)


def test_floor_is_unmeasured_after_the_statistic_changed():
    """max|d| over n=2 does not transfer to a t-test over n=6.

    A noise floor belongs to the statistic it was measured on, exactly as it
    belongs to the metric it was measured on. Delete this test when the null
    experiment has been re-run at the new repeat count and the floor re-set.
    """
    assert prepare.PAIRED_NOISE_FLOOR is None


def test_mode_fault_names_the_problem():
    assert experiment.mode_fault(WALK6) is None
    assert experiment.mode_fault(["standing"] * 6) == "DEGENERATE"
    assert experiment.mode_fault(["walking", "standing"]) == "MIXED"
    assert experiment.mode_fault([]) is None


def test_a_faulty_baseline_cannot_be_read_back_as_a_baseline(tmp_path):
    """`baseline_from_results` matches the verdict string exactly.

    That exactness is load-bearing: it is the whole mechanism keeping a
    standing-still or bimodal arm from becoming the thing every later candidate
    is paired against. Not hypothetical - the end-to-end smoke run produces
    exactly this: a 10-iteration policy scores root_err 0.352, classifies as
    standing, and would otherwise have been written as a usable BASELINE.
    """
    results = tmp_path / "results.tsv"
    cols = lambda verdict, scores: (
        "2026-08-27T00:00:00+00:00\tnote\t0.5\t0.01\t2\t\t"
        f"{verdict}\t0\t{scores}\t0.0\t0.0\t0.0\t1.0\tstanding,standing\n")

    results.write_text(experiment.HEADER
                       + cols("BASELINE", "0.11111,0.22222")
                       + cols("BASELINE-DEGENERATE", "0.33333,0.44444")
                       + cols("BASELINE-MIXED", "0.55555,0.66666"))

    old = experiment.RESULTS
    try:
        experiment.RESULTS = results
        # The most recent row is BASELINE-MIXED, then BASELINE-DEGENERATE.
        # Both must be skipped in favour of the older clean one.
        assert experiment.baseline_from_results() == [0.11111, 0.22222]
    finally:
        experiment.RESULTS = old


def test_no_baseline_at_all_returns_none(tmp_path):
    results = tmp_path / "results.tsv"
    results.write_text(experiment.HEADER)
    old = experiment.RESULTS
    try:
        experiment.RESULTS = results
        assert experiment.baseline_from_results() is None
    finally:
        experiment.RESULTS = old
