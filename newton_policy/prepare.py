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
TRAIN_ITERATIONS = 300
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
PAIRED_NOISE_FLOOR = None


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


def evaluate_completion(checkpoint, train_cfg):
    """THE METRIC. Roll a trained checkpoint out and count how episodes ended.

    Returns a dict with `completion_rate` - of the episodes that ended, the
    share that reached the end of the clip rather than falling.

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
        REFERENCE, env.actuated_joint_names))

    vec = m["vecenv"].DrLegsVecEnv(env)
    runner = OnPolicyRunner(vec, train_cfg, log_dir=None, device=env.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)

    torch.manual_seed(EVAL_SEED)
    obs, _ = vec.reset()
    fell = finished = 0
    for _ in range(EVAL_STEPS):
        with torch.inference_mode():
            action = policy(obs)
        obs, _, dones, extras = vec.step(action)
        d = dones.nonzero(as_tuple=False).flatten()
        if len(d):
            t = extras["time_outs"][d]
            finished += int(t.sum())
            fell += int((~t).sum())

    ended = fell + finished
    return {
        "completion_rate": (finished / ended) if ended else 0.0,
        "completed": finished,
        "terminated": fell,
        "episodes": ended,
    }
