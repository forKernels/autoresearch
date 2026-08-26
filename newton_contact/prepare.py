"""Fixed scene and the ground-truth metric. READ-ONLY - do not modify.

The counterpart of karpathy's `prepare.py` and of `newton_policy/prepare.py`:
it owns the world and the number, and it is read-only for the same reason. An
agent that can reach the metric optimises the metric, and an agent that can
reach the SCENE optimises the scene - it would discover that a pile of four
bricks never loses one.

## What is being researched

Rigid contact and stacking, which is where this stack measurably fails. From
newton-lab's CLAUDE.md, all measured:

    200 bricks @ 45 mm, mesh floor, 16 substeps  ->  23/200 lost
    the same pile on an analytic plane           ->   0/200 lost
    a 156-brick wall hit by a wrecking ball      ->  non-finite at frame 19-30
                                                     at every substep count tried

That is the target. It is also the axis that decides how many objects a scene
can hold, which is the thing buyers ask for.

## The metric, and why this one

`retention` - the share of bodies that are still finite and still above the
floor when the bake ends.

Thor's checklist asks: is there any setting of the KNOBS that moves this, and
is there any setting of the SAMPLING that moves it? The first is yes - substeps,
solver, iterations and collider shape all move it, measurably. The second is no,
BECAUSE the scene is built here and not by the file the agent edits.

A mean penetration depth was the tempting alternative and it is the wrong one:
a mean hides the failures, and the failures are the entire point. `retention`
counts them.

## The budget

A fixed 120 frames, and a wall-clock CAP. Without the cap the loop has a
trivial winning move - raise substeps until nothing is lost - which is not a
finding, it is buying retention with time nobody has. Exceeding the cap is a
failure, exactly as an OOM is upstream.
"""

import os
import sys
import time
import types
from pathlib import Path

import numpy as np

# --- fixed, not the agent's to change ---------------------------------------

NEWTON_LAB = Path(os.environ.get("NEWTON_LAB", r"C:\_git\NewtonLab"))
DEPS = os.environ.get("NEWTON_DEPS", r"C:\_git\blender_deps")

#: One experiment. 60 frames is long enough for the pile to settle and for a
#: loss to show; the cap is what stops "more substeps" being free. Sized from
#: measurement, not taste: at 48 bricks the baseline costs 32 s and the most
#: expensive configuration tried (HULL) costs 52 s, so 90 s admits both and
#: still refuses a configuration that buys retention with time nobody has.
FRAMES = 60
WALL_CAP_SECONDS = 90.0

#: The pile. Bricks well clear of the measured mesh-contact size floor (~17 mm
#: MuJoCo, ~40 mm VBD) so that what is being measured is STACKING, not the
#: size floor - that one is already characterised and is not the open question.
#: 48 bricks, not 200. Cost is badly super-linear in body count - measured
#: 1072 ms/frame at 48, 2864 at 100 and 14505 at 200, so 4.2x the bodies costs
#: 13.5x the time - and a 200-brick pile cannot finish inside any experiment
#: budget worth having. 48 still loses a third of itself at the baseline, which
#: is the headroom the loop needs.
BRICK = (0.090, 0.045, 0.045)      # 90 x 45 x 45 mm
ROWS, PER_ROW, LAYERS = 4, 4, 3    # 48 bricks
DROP = 0.010                       # gap between layers at t=0

#: MuJoCo is not bit-deterministic, and on this scene it is MUCH worse than
#: the policy harness sees. Measured, two identical baseline runs:
#:
#:     run 1  retention 0.6667      run 2  retention 0.3333
#:
#: A spread of 0.3333 on a mean of 0.5000. That is not sloppiness, it is
#: structural: ROWS * PER_ROW = 16 bricks is exactly one LAYER, the pile fails
#: by losing whole layers, and 48/48, 32/48, 16/48 and 0/48 are therefore the
#: ONLY values the metric can take. A four-valued metric with a one-layer step
#: cannot resolve a small effect however many times it is run.
#:
#: Consequences, and they are the honest ones rather than the convenient ones:
#:   * only LARGE effects are detectable here - a candidate must clear the
#:     baseline by more than one whole layer to register as a win.
#:   * more repeats narrow the mean but cannot subdivide the step. Four is a
#:     compromise, not a fix.
#:   * a candidate that "improves" by exactly one layer once is noise.
#:
#: Widening the pile would subdivide the step, and costs time super-linearly -
#: 4.2x the bodies for 13.5x the wall clock. That trade is open and unmade.
EVAL_REPEATS = 4

#: A body is "lost" once it is below this. The floor sits at z=0, so anything
#: meaningfully under it has gone through rather than settled.
LOST_BELOW = -0.05


def load_sim():
    """newton_lab.sim without Blender, the way tests/verify_sim.py does it."""
    import importlib.util

    src = NEWTON_LAB / "src" / "newton_lab"
    bpy = types.ModuleType("bpy")
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules.setdefault("bpy", bpy)
    sys.modules.setdefault("bmesh", types.ModuleType("bmesh"))

    pkg = types.ModuleType("nlc")
    pkg.__path__ = [str(src)]
    sys.modules["nlc"] = pkg
    for name in ("tags", "fabric", "soft", "spring", "wind", "joints", "sim"):
        spec = importlib.util.spec_from_file_location(
            f"nlc.{name}", src / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"nlc.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["nlc.sim"]


def brick_positions():
    """Where every brick starts. Deterministic, and not the agent's to change.

    Built brick-on-brick with a small gap rather than in a loose lattice:
    newton-lab measured that a stack assembled with gaps lands all at once and
    that simultaneous impact is what diverges, so a lattice would be measuring
    the fixture rather than the solver.
    """
    dx, dy, dz = BRICK
    out = []
    for layer in range(LAYERS):
        offset = (dx * 0.5) if layer % 2 else 0.0
        for r in range(ROWS):
            for i in range(PER_ROW):
                out.append((
                    i * dx * 1.02 - (PER_ROW - 1) * dx * 0.51 + offset,
                    r * dy * 1.02 - (ROWS - 1) * dy * 0.51,
                    dz * 0.5 + layer * (dz + DROP),
                ))
    return out


def build_and_bake(settings):
    """Build the fixed pile with the agent's settings and run FRAMES frames.

    `settings` is whatever train.py's `SETTINGS` says - solver, substeps,
    iterations, brick shape, floor kind, density, friction. The SCENE is this
    function's, not the settings'.

    Returns a metrics dict. Raises on a wall-clock overrun.
    """
    sim = load_sim()
    wp, newton = sim.ensure(DEPS)

    device, _note = sim.resolve_device(wp, settings.get("device", "cuda:0"))
    with wp.ScopedDevice(device):
        builder = sim.new_builder(newton, wp, gravity=-9.81)

        floor = settings.get("floor", "PLANE").upper()
        if floor == "GROUND":
            sim.add_ground(builder, wp, newton,
                           vbd=(settings.get("solver") == "VBD"))
        elif floor == "PLANE":
            sim.add_plane_collider(
                builder, wp, newton,
                position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                width=20.0, length=20.0,
                friction=settings.get("floor_friction", 0.9),
            )
        elif floor == "MESH":
            n = 24
            xs = np.linspace(-10.0, 10.0, n)
            verts, tris = [], []
            for j, y in enumerate(xs):
                for i, x in enumerate(xs):
                    verts.append([x, y, 0.0])
            for j in range(n - 1):
                for i in range(n - 1):
                    a = j * n + i
                    tris.append([a, a + 1, a + n + 1])
                    tris.append([a, a + n + 1, a + n])
            sim.add_collider(
                builder, wp, newton,
                np.array(verts, dtype=np.float32),
                np.array(tris, dtype=np.int32),
                friction=settings.get("floor_friction", 0.9),
            )
        else:
            raise ValueError(f"unknown floor {floor!r}")

        dx, dy, dz = BRICK
        hull_v, hull_t = _box_mesh(dx, dy, dz)
        bodies = []
        cache = {}
        for (x, y, z) in brick_positions():
            bodies.append(sim.add_prop(
                builder, wp, newton,
                xform=wp.transform((x, y, z), wp.quat_identity()),
                shape=settings.get("shape", "BOX"),
                dims=BRICK, verts=hull_v, tris=hull_t,
                density=settings.get("density", 1200.0),
                friction=settings.get("friction", 0.9),
                restitution=settings.get("restitution", None),
                mesh_cache=cache,
            ))

        model = sim.finalize(
            builder, has_cloth=False,
            vbd=(settings.get("solver") == "VBD"),
        )
        stepper, solver_name, _warn = sim.make_stepper(
            newton, settings.get("solver", "MUJOCO"), model,
            has_cloth=False, has_rigid=True,
            iterations=settings.get("iterations", 20),
        )

        s0, s1 = model.state(), model.state()
        control = model.control()
        # sim.make_pipeline, NOT newton.CollisionPipeline directly. The
        # add-on sizes the rigid contact buffer to max(16384, bodies * 512);
        # newton's own default is 11000, and constructing the pipeline raw
        # silently gives this harness a smaller buffer than a real bake has.
        # That mistake produced a "contact buffer overflows at 200 bricks"
        # finding here that was the harness's and not the product's.
        pipeline = sim.make_pipeline(newton, model)
        contacts = pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        substeps = int(settings.get("substeps", 16))
        non_finite_frame = None
        started = time.perf_counter()
        for frame in range(FRAMES):
            s0, s1 = stepper.step(
                pipeline, contacts, s0, s1, control,
                dt=1.0 / 24.0, substeps=substeps,
            )
            if frame % 10 == 0 or frame == FRAMES - 1:
                q = s0.body_q.numpy()
                if not np.isfinite(q).all() and non_finite_frame is None:
                    non_finite_frame = frame
                    break
            if time.perf_counter() - started > WALL_CAP_SECONDS:
                raise TimeoutError(
                    f"exceeded the {WALL_CAP_SECONDS:.0f}s cap at frame "
                    f"{frame} - substeps={substeps} solver={solver_name}"
                )
        wall = time.perf_counter() - started

        q = s0.body_q.numpy()
        z = q[np.array(bodies), 2]
        finite = np.isfinite(z)
        retained = int((finite & (z > LOST_BELOW)).sum())
        total = len(bodies)

        return {
            "retention": retained / total,
            "retained": retained,
            "total": total,
            "lost": total - retained,
            "non_finite_frame": (-1 if non_finite_frame is None
                                 else non_finite_frame),
            "lowest_z": float(np.min(z[finite])) if finite.any() else float("nan"),
            "wall_seconds": wall,
            "solver": solver_name,
            "substeps": substeps,
        }


def _box_mesh(dx, dy, dz):
    """A box as triangles, for the HULL and MESH shape paths."""
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float32)
    t = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ], dtype=np.int32)
    return v, t
