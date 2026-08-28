"""The file the agent edits. Everything here is fair game.

One experiment: build a G1 tracking environment, train for the fixed budget in
`prepare.py`, then hand the checkpoint to `prepare.evaluate_completion`, which
scores it in an environment this file does not control.

    uv run newton_policy/train.py        (or plain python3 with PYTHONPATH set)

What is worth changing, roughly in order of what has NOT been tried:

  * `k_ee` - the end-effector term. Off here. The reward currently scores joint
    ANGLES, and a policy was measured tracking them to 0.033 rad while falling
    over: angles compound down a limb, so two degrees at hip and knee is
    centimetres at the foot. Needs a reference carrying body positions.
  * `lookahead_seconds` - 0/20/40/80 ms, inherited from a 0.265 m robot and
    never questioned for a 0.76 m one. A humanoid may need to see a fall
    further ahead than 80 ms.
  * PPO hyperparameters and the network. Untouched so far.
  * `alive_bonus`, `action_penalty`, `action_scale` - all tried at 150 and
    1500 iterations against a broken metric, so those results say nothing and
    are worth redoing honestly.

What has already been measured, so re-running it wastes a slot:

  * Lowering PADMM iterations is NOT a speedup. It returns an unconverged
    solution and throws the robot through the floor at foot strike. Do not.
  * `collect_solver_info` costs 10x, silently.
  * 34M -> 98M steps changed nothing once the policy was already completing.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402  - read-only, owns the metric


# --- the experiment ---------------------------------------------------------

ENV_KWARGS = dict(
    alive_bonus=0.0,
    action_penalty=0.0,
    # action_scale=0.5,             # the robot's own default if omitted
    # THE CANDIDATE. 4.863 is `reward.k_ee_for_height(0.76)`, the same
    # auto-scaled value the SCORER uses, so training and evaluation weight the
    # term identically.
    #
    # It is 0.0 for a baseline or null arm, and that is not merely "off": the
    # reference HAS body positions, so leaving it unset would auto-scale the
    # end-effector term on and make the control carry the very change it is
    # the control for. reward.py gates on `if self.k_ee`, so 0.0 skips the
    # term entirely rather than multiplying by exp(0)=1.
    k_ee=1.2,
    # THE CANDIDATE. Measured cause: the position term alone produced a policy
    # that tracks joints to 0.048 rad, never falls, and STEPS IN PLACE - root
    # error grew linearly at 0.488 m/s and forward speed was 0.161 against the
    # reference's 0.996. `e_root` is an unbounded squared distance inside an
    # exp and the reward is a PRODUCT, so at 2.7 m of drift the root factor is
    # 2e-8 and the gradient of every other term goes with it. Velocity error
    # does not accumulate, so it still pays where position has gone silent.
    k_root_vel=10.0,
)

LOOKAHEAD_SECONDS = (0.0, 0.02, 0.04, 0.08)

PPO = dict(
    num_learning_epochs=5,
    num_mini_batches=4,
    clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.005,
    learning_rate=1.0e-3,
    max_grad_norm=1.0,
    schedule="adaptive",
    desired_kl=0.01,
)

HIDDEN_DIMS = [512, 256, 128]
INIT_STD = 0.3
NUM_STEPS_PER_ENV = 24


def build_train_cfg():
    net = {"class_name": "MLPModel", "hidden_dims": HIDDEN_DIMS,
           "activation": "elu", "obs_normalization": True}
    actor = dict(net)
    actor["distribution_cfg"] = {"class_name": "GaussianDistribution",
                                 "init_std": INIT_STD, "std_type": "scalar",
                                 "learn_std": True}
    return {
        "algorithm": {"class_name": "PPO", **PPO},
        "actor": actor,
        "critic": dict(net),
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "save_interval": 10_000,
        "empirical_normalization": False,
    }


def main():
    from rsl_rl.runners import OnPolicyRunner

    m = prepare._rl()
    t0 = time.time()

    env = m["mujoco_env"].G1Env(
        reference=m["train"]._PlaceholderReference(),
        num_envs=prepare.TRAIN_ENVS, **ENV_KWARGS)
    env.bind_reference(m["reference"].ReferenceMotion.from_npz(
        prepare.REFERENCE, env.actuated_joint_names,
        lookahead_seconds=LOOKAHEAD_SECONDS))

    out = Path(__file__).resolve().parent / "run"
    out.mkdir(exist_ok=True)
    cfg = build_train_cfg()
    runner = OnPolicyRunner(m["vecenv"].DrLegsVecEnv(env), cfg,
                            log_dir=str(out), device=env.device)
    runner.learn(num_learning_iterations=prepare.TRAIN_ITERATIONS)
    train_s = time.time() - t0

    ckpt = out / f"model_{prepare.TRAIN_ITERATIONS - 1}.pt"
    runner.save(str(ckpt))
    # The horizon MUST be the one this run trained on - see
    # prepare.evaluate_tracking. Passing it is not optional.
    result = prepare.evaluate_tracking(ckpt, build_train_cfg(),
                                       LOOKAHEAD_SECONDS)

    print("\n---")
    # tracking_error is THE metric and LOWER IS BETTER. The other two are
    # reported because they move in opposite directions: measured across one
    # run's checkpoints, joint and end-effector error improve with training
    # while ROOT error gets steadily worse (0.767 -> 0.941 m from iteration 750
    # to 7999). A single number would hide that trade rather than show it.
    print(f"tracking_score:   {result['tracking_score']:.5f}")
    print(f"tracking_error:   {result['tracking_error']:.5f}")
    print(f"root_error_m:     {result['root_error_m']:.5f}")
    print(f"ee_error_m:       {result['ee_error_m'] if result['ee_error_m'] is None else round(result['ee_error_m'], 5)}")
    print(f"ori_error_rad:    {result['ori_error_rad']:.5f}")
    print(f"vel_error:        {result['vel_error']:.5f}")
    print(f"comparable:       {result['comparable']}")
    # WHICH policy this run found. `experiment.py` parses this line: an arm
    # whose seeds report different modes gets no verdict, because its spread is
    # a mode-switch rate and not an error bar.
    print(f"mode:             {result['mode']}")
    print(f"lookahead:        {result['lookahead_seconds']}")
    print(f"completion_rate:  {result['completion_rate']:.4f}")
    print(f"completed:        {result['completed']}")
    print(f"terminated:       {result['terminated']}")
    print(f"episodes:         {result['episodes']}")
    print(f"training_seconds: {train_s:.1f}")
    print(f"iterations:       {prepare.TRAIN_ITERATIONS}")
    print(f"envs:             {prepare.TRAIN_ENVS}")


if __name__ == "__main__":
    main()
