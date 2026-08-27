"""Fixed constants and the ground-truth metric. READ-ONLY - do not modify.

The counterpart of karpathy's `prepare.py`: it owns the evaluation, and it is
read-only for the same reason. An agent that can reach the metric optimises the
metric.

That is not hypothetical here. This project spent roughly seven hours moving
`mean episode length`, which under reference state initialisation is pinned at
`clip_duration / 2` by arithmetic and cannot be moved by any policy - five
separate interventions all produced the same number. An autonomous loop pointed
at that metric would have produced a hundred confident null results overnight.
Full write-up: newton-lab `docs/research/2026-08-26-the-150-step-ceiling-that-
was-not.md`.

So the metric here is `completion_rate`: of the episodes that ENDED, the share
that ended by reaching the end of the clip rather than by falling over. It is
bounded by the policy, which is the property `mean episode length` lacked.

Evaluation builds its OWN environment with the constants below, so whatever the
training config did does not change how the result is scored.
"""

import sys
from pathlib import Path

# --- fixed, not the agent's to change ---------------------------------------

NEWTON = "/home/lupin4/_git/newton-8ce54fac"      # the WS-1 pinned commit
NEWTON_LAB = "/home/lupin4/_git/newton-lab"
#: Carries four root-relative end-effector tracks (ankles + elbows) as well
#: as joints, so the k_ee reward term has something to score. Changing this
#: changes the TASK - the baseline below was re-measured against it.
REFERENCE = f"{NEWTON_LAB}/runs/g1_v05_ee5.npz"

#: One experiment. ~5 minutes at 512 envs / ~11.2k env-steps per second, which
#: is roughly 12 experiments per hour - the same cadence karpathy's budget gives.
#: Past the cliff. Measured on runs/20260825-151727, one run, one eval seed,
#: budget the only variable:
#:
#:     300 -> 0.2960     500 -> 0.9841     750 -> 1.0000
#:    1000..7999 -> 1.0000, zero terminations, 1228 episodes every time
#:
#: 300 was not merely small, it sat ON the phase transition - the steepest part
#: of the curve, where a little more training is a lot more completion. That is
#: why two runs of an identical config scattered by 0.13 and why six
#: experiments in a row resolved nothing. 1000 is comfortably into saturation
#: with margin, which is also why the metric below is no longer completion.
TRAIN_ITERATIONS = 1000
TRAIN_ENVS = 512

#: Evaluation. Deliberately different from training: more environments than a
#: training batch so the rate is tight, a fixed seed so runs are comparable, and
#: enough steps that an episode starting late in the clip still gets to finish.
EVAL_ENVS = 512
EVAL_STEPS = 400
EVAL_SEED = 20260826

#: Kamino and MuJoCo are not bit-deterministic - the WS-1 handoff measured 3%
#: between two identical DR Legs runs and 11% across three. A single comparison
#: is noise, so a candidate is run twice and only accepted if it beats the
#: baseline by more than the observed spread.
EVAL_REPEATS = 2

#: One training seed per repeat, applied at IMPORT time (see `_seed_training`).
#:
#: These live here, in the read-only file, for the same reason the metric does.
#: An agent that picks its own training seed can seed-hunt: run the candidate
#: on whichever seed flatters it and report that. Fixing them here makes every
#: arm face the same two initialisations, which is what turns a comparison of
#: two noisy means into a PAIRED comparison where the initialisation cancels.
TRAIN_SEEDS = (20260826, 19720305)

#: The paired noise floor: how far apart two runs of the SAME config on the
#: SAME seed land, measured by re-running the baseline against itself.
#: Unset until that null experiment has been run - `experiment.py` refuses to
#: call a KEEP without it, because a bar derived only from two paired
#: differences can be arbitrarily small by luck.
#: MUST be re-measured whenever the metric changes, and stays None until it is
#: - `experiment.py` refuses to return a verdict without it.
#:
#: It was 0.1329, measured by a null experiment on `completion_rate`. Tracking
#: errors run 0.02-0.07, so carrying that number across the metric change would
#: have set a bar wider than the entire range of the new metric: every result
#: NEUTRAL forever, reported with confidence, and nothing to suggest the bar
#: rather than the candidates was the reason. A noise floor belongs to the
#: metric it was measured on, exactly as a baseline belongs to the task it was
#: measured on.
#:
#: Unmeasured for `tracking_score`. It was 0.00950 on `tracking_error` and
#: 0.1329 on `completion_rate` before that - three metrics, three floors, and
#: none of them transferable. `tracking_score` is bounded (0, 1] where
#: `tracking_error` ran 0.02-0.07, so carrying the old number would set a bar
#: a fifth of the entire range wide. A noise floor belongs to the metric it was
#: measured on, exactly as a baseline belongs to the task it was measured on.
#: Measured by the null experiment on `tracking_score` WITH the end-effector
#: factor, at TRAIN_ITERATIONS=1000. Paired differences +0.00551, -0.03589.
#:
#: Read those two numbers before trusting this bar. Seed 20260826 reproduced to
#: +0.00551; seed 19720305 moved -0.03589 on an identical re-run of an
#: identical config. Same seed, same code, 6.5x the spread - the physics
#: nondeterminism `_seed_training` warns about, and NOT mode-mixing: every run
#: in the null sat at root_err_m ~0.08, all of them walking.
#:
#: This is `max |paired difference|` over n=2, which is the weakest part of the
#: harness. The max of two samples has no confidence attached and inherits the
#: worse draw wholesale, so one outlier set a bar ~20x the k_ee effect size. A
#: real paired test needs EVAL_REPEATS well above 2; until then this number is
#: a placeholder that errs toward NEUTRAL, which is the safe direction to err.
PAIRED_NOISE_FLOOR = 0.03589


def _seed_training():
    """Seed the RNGs from TRAIN_SEEDS[NEWTON_POLICY_RUN], at import.

    Called at module import rather than exposed for `train.py` to invoke,
    because `train.py` is the file the agent edits and a call it makes is a
    call it can drop. It must import this module to reach `_rl()`, so this
    runs whatever else it does.

    This does NOT make a run reproducible. MuJoCo is not bit-deterministic and
    the physics still diverges; what it fixes is the network initialisation,
    the initial action distribution and the reset sampling, so two arms start
    from the same place instead of from two different random draws. Whether
    that is the dominant variance term is an empirical question, and the null
    experiment that sets PAIRED_NOISE_FLOOR is what answers it.
    """
    import os
    import random

    import numpy as np
    import torch

    index = int(os.environ.get("NEWTON_POLICY_RUN", "0"))
    seed = TRAIN_SEEDS[index % len(TRAIN_SEEDS)]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[prepare] training seed {seed} (run index {index})", flush=True)
    return seed


TRAIN_SEED = _seed_training()


def _rl():
    """Import newton_lab's rl package without importing the Blender add-on."""
    import importlib.util
    import types

    src = Path(NEWTON_LAB) / "src" / "newton_lab"
    for stub in ("bpy", "bmesh"):
        sys.modules.setdefault(stub, types.ModuleType(stub))
    if str(NEWTON) not in sys.path:
        sys.path.insert(0, str(NEWTON))
    if "nlrl" not in sys.modules:
        pkg = types.ModuleType("nlrl")
        pkg.__path__ = [str(src / "rl")]
        sys.modules["nlrl"] = pkg
    mods = {}
    for name in ("indices", "reward", "contract", "env_base", "reference",
                 "vecenv", "kamino_env", "mujoco_env", "train"):
        spec = importlib.util.spec_from_file_location(
            f"nlrl.{name}", src / "rl" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"nlrl.{name}"] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


def evaluate_tracking(checkpoint, train_cfg, lookahead_seconds):
    """THE METRIC. Roll a checkpoint out and measure how well it TRACKS.

    Returns `tracking_score`, the mean of the reward's own tracking product -
    joint x velocity x root x orientation - over every step where the policy
    was live. HIGHER IS BETTER, and it is bounded in (0, 1].

    It replaces `tracking_error`, mean joint-angle deviation, which was blind
    to the one thing the task is about. Measured: `k_root_vel=10` cut root
    error by 91% (0.879 -> 0.082 m), turned a policy that STEPPED IN PLACE into
    one that walks, and was called DISCARD - because a robot holding still with
    its legs in walking poses tracks joint ANGLES better than one actually
    walking, which contact perturbs. The metric ranked the stationary robot
    above the moving one.

    The product is the right shape because the reward already reconciles
    radians against metres through `k_joint` and `k_root`; the old metric took
    only the joint half of a quantity the reward had already balanced.

    The weights are PINNED HERE and the scorer is built here, never taken from
    the environment under evaluation. The candidate controls that environment's
    reward - `train.py`'s ENV_KWARGS reaches it - and a metric a candidate can
    tune is not a metric. The scorer also carries no `k_ee`, no alive bonus and
    no penalties: those are knobs, and a knob inside the scoring function
    rewards itself.

    `k_root_vel` is NOT scored: it is a candidate's term, and what it BUYS shows
    up in the root factor, which is.

    The END-EFFECTOR term IS scored, and its absence was a mistake worth
    recording. It was left out on the reasoning that "a knob inside the scoring
    function rewards itself" - but that applies to the candidate's TRAINING
    weight, not to the quantity. `k_root` is scored at a fixed weight here while
    `k_root_vel` is the candidate, and that arrangement works; end effectors
    needed exactly the same one. Without it the scorer was
    joint x vel x root x ori, so a `k_ee` candidate could only ever LOWER the
    score - spending scored terms to improve an unscored one. Measured: k_ee cut
    end-effector error 21% on both seeds, consistently, and scored NEUTRAL
    because the improvement was invisible to the metric.

    Note what this makes incomparable: a clip WITHOUT `body_positions` has no
    end-effector factor at all, so its scores cannot be compared against a clip
    that has one. The factor is bounded (0, 1], so adding it can only lower the
    absolute number - which is why the baseline and the floor are re-measured
    whenever the scorer changes.

    Why it replaces `completion_rate`, which is still reported below as a
    guard: completion saturates. Past ~750 iterations every policy finishes
    every episode, so the number is pinned at 1.0 and cannot tell a good policy
    from a perfect one; below ~500 it is on a cliff and measures mostly noise.
    Its useful range is about 200 iterations wide, which is not somewhere a
    research loop can live.

    That makes this the THIRD metric on this project bounded by something other
    than policy quality - `mean episode length` by the sampling scheme,
    `completion_rate` by saturation. Tracking error is bounded by neither: a
    policy that finishes every episode can still track better or worse, which
    is the property the other two lacked.

    `lookahead_seconds` is REQUIRED, and that is the whole point of it.

    Everything else about the evaluation world is fixed here so a candidate
    cannot score itself somewhere easier. The lookahead horizon is the one
    thing that cannot be: it is part of the POLICY'S INPUT CONTRACT, not part
    of the task. Evaluating a policy on a horizon it did not train on is not
    stricter, it is incoherent - the observation vector means something
    different.

    It used to default. `train.py` passed the candidate's horizon and this
    passed nothing, so any candidate that changed the horizon to a different
    tuple of the SAME LENGTH trained on one and was scored on another. Same
    length means same observation width, so nothing raised: `runner.load`
    succeeded and the policy was fed a silently wrong vector. Both lookahead
    experiments run under that bug came back markedly worse, which is exactly
    what an observation mismatch produces, and both were recorded as results.

    Required rather than defaulted because a default is what caused it. A
    caller that has not thought about the horizon should not be able to
    evaluate at all.

    (The same trap, found the same week in `newton_contact`: it built its
    collision pipeline directly instead of through `sim.make_pipeline`, gave
    itself a contact buffer smaller than any real bake gets, and killed four
    confident findings when it was fixed. Construct the world the way the thing
    being measured constructs it.)

    Two things that keep it honest:

    **Steps where an episode ENDED are excluded.** `vecenv.step` resets a done
    environment and returns the post-reset observation, and reference state
    initialisation puts that reset exactly ON the reference - error zero. Left
    in, those steps would reward falling: fall often, reset often, average in
    more zeros. The `live` mask removes them.

    **`completion_rate` is still the guard.** Even with the mask, a policy that
    falls constantly spends more of its time near a reset and less of it far
    down a diverging trajectory. So a tracking error measured while the robot
    is falling over is not comparable to one measured while it completes, and
    `comparable` says so rather than leaving the caller to notice.
    The environment is built here, from the constants above, and NOT from
    whatever the training config used. A candidate cannot make itself look good
    by evaluating in an easier world.
    """
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    m = _rl()
    env = m["mujoco_env"].G1Env(
        reference=m["train"]._PlaceholderReference(), num_envs=EVAL_ENVS)
    env.bind_reference(m["reference"].ReferenceMotion.from_npz(
        REFERENCE, env.actuated_joint_names,
        lookahead_seconds=lookahead_seconds))

    vec = m["vecenv"].DrLegsVecEnv(env)
    runner = OnPolicyRunner(vec, train_cfg, log_dir=None, device=env.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)

    torch.manual_seed(EVAL_SEED)
    obs, _ = vec.reset()
    fell = finished = 0
    joint_sum = root_sum = ee_sum = score_sum = 0.0
    live_steps = 0

    # The scorer, with weights pinned above and built HERE. k_root is in METRES
    # and has to scale to the robot: carrying a 0.265 m robot's value onto a
    # 0.76 m one collapses the product for an ordinary lag.
    scorer = m["reward"].TrackingReward(
        k_root=m["reward"].k_root_for_height(env.standing_height),
        k_ee=m["reward"].k_ee_for_height(env.standing_height))
    has_ee = bool(getattr(env.reference, "has_body_positions", False))

    for _ in range(EVAL_STEPS):
        with torch.inference_mode():
            action = policy(obs)
        obs, _, dones, extras = vec.step(action)
        d = dones.nonzero(as_tuple=False).flatten()
        if len(d):
            t = extras["time_outs"][d]
            finished += int(t.sum())
            fell += int((~t).sum())

        with torch.inference_mode():
            live = ~dones
            n_live = int(live.sum())
            if n_live:
                q = env.actuated_q()[live]
                q_ref = env._ref(env.reference.q)[live]
                joint_sum += float((q - q_ref).abs().mean(dim=-1).sum())

                root = env.root_pose_wxyz()[live]
                root_ref = env._ref(env.reference.root)[live]
                root_sum += float(
                    (root[:, :3] - root_ref[:, :3]).norm(dim=-1).sum())

                # `tm`, not `t` - `t` is the time-outs mask a few lines up.
                tm = scorer.terms(q, q_ref, env.actuated_dq()[live],
                                  env._ref(env.reference.dq)[live],
                                  root, root_ref)
                score = tm["joint"] * tm["vel"] * tm["root"] * tm["ori"]

                if has_ee:
                    ee = env.end_effector_positions()
                    if ee is not None:
                        ee, ee_ref = ee[live], env._ref(env.reference.body)[live]
                        ee_sum += float(
                            (ee - ee_ref).norm(dim=-1).mean(dim=-1).sum())
                        score = score * scorer.end_effector_term(ee, ee_ref)
                score_sum += float(score.sum())
                live_steps += n_live

    ended = fell + finished
    completion = (finished / ended) if ended else 0.0
    return {
        # THE metric. Bounded (0, 1], HIGHER is better.
        "tracking_score": (score_sum / live_steps) if live_steps else 0.0,
        # Kept as a diagnostic, not the goal - it is what ranked a stationary
        # robot above a walking one.
        "tracking_error": (joint_sum / live_steps) if live_steps else float("inf"),
        "root_error_m": (root_sum / live_steps) if live_steps else float("inf"),
        "ee_error_m": (ee_sum / live_steps) if (has_ee and live_steps) else None,
        # The guard, not the goal.
        "completion_rate": completion,
        "comparable": completion >= COMPARABLE_COMPLETION,
        "completed": finished,
        "terminated": fell,
        "episodes": ended,
        "live_steps": live_steps,
        # Recorded so a mismatch is visible in the run's own output rather than
        # having to be inferred from a result that looks merely disappointing.
        "lookahead_seconds": list(lookahead_seconds),
    }


#: A tracking error measured while the robot keeps falling is not comparable to
#: one measured while it completes, so below this the result is flagged rather
#: than silently ranked against results from the other regime.
COMPARABLE_COMPLETION = 0.99


def evaluate_completion(checkpoint, train_cfg, lookahead_seconds):
    """Deprecated alias. `completion_rate` saturates; see evaluate_tracking."""
    return evaluate_tracking(checkpoint, train_cfg, lookahead_seconds)
