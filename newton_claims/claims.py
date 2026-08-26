"""The claims, and the code that re-measures them. READ-ONLY - do not modify.

The counterpart of `newton_policy/prepare.py` and `newton_contact/prepare.py`,
and read-only for a related but not identical reason. Those two own a METRIC,
and are fenced off because an agent that can reach the metric optimises the
metric. This one owns an ASSERTION and the experiment that settles it, and is
fenced off because an agent that can reach the experiment can make any claim
come out true.

## What this harness is, and what it is not

It is not an optimisation loop. karpathy's autoresearch moves one scalar, keeps
or discards, and repeats until the night is over. This takes a CLAIM out of
newton-lab's `CLAUDE.md`, re-measures it against the INSTALLED newton, and
returns one of three verdicts:

    AGREE           the build still does what the document says
    DISAGREE        it does not, and here are the numbers
    COULD NOT TEST  and here is the reason

Its output is a diff to the documentation, not a better number. So it has a
natural end condition - every claim checked - and there is no fake metric for
it to climb.

## Why it exists

Nearly every measurement in `CLAUDE.md` was taken against newton 1.5.0. The
add-on now ships 1.6.0.dev0, and `CLAUDE.md` says so itself: "All were measured
against 1.5.0 and none re-probed against 1.6.0.dev0, so treat any that a change
depends on as worth re-checking." That sentence is a standing invitation and
nothing had accepted it.

In one session before this harness existed, four documented claims were found
stale or wrong by hand: SolverKamino was written up as refusing MESH shapes (it
rests bodies on mesh floors perfectly well), CONE was assumed usable (it builds
and then SolverMuJoCo raises KeyError converting it), XPBD was named as the
bounce solver while being constructed with `enable_restitution` defaulted to
False, and a "contact buffer overflows" finding turned out to be a harness that
bypassed `sim.make_pipeline`. Four in one session, found by hand, is the
argument for doing it systematically.

## The rule that produced the fourth of those, and the one this file obeys

**Construct the world the way the PRODUCT constructs it.**

`newton_contact/program.md` records what happens otherwise, at length. Every
collision pipeline here comes from `sim.make_pipeline`, never from
`newton.CollisionPipeline` directly: the add-on sizes the rigid contact buffer
to `max(16384, bodies * 512)` while newton's own default is 11000, and an
overflowed buffer does not raise - it warns and DROPS contacts. A harness that
reaches past the add-on's own setup is not measuring the add-on, and it will
produce confident, reproducible, entirely fictional results. Four findings died
that way in `newton_contact` alone.

The same reasoning is why the stepping goes through `sim.Stepper`, the finalize
through `sim.finalize`, the solver through `sim.make_stepper`, and why
`sim.use_coordinate_targets` is called at bootstrap the way `Sim.__init__` does
it. Where a claim is about a raw newton API - `add_body(mass=)` is the example
- the call is deliberately raw, and the claim says so.

## The harness re-measures the ASSERTION, not the original fixture

This matters for reading a verdict and is the easiest thing here to get wrong.

`CLAUDE.md` records "SolverVBD 40 mm -> 0/12 through, 30 mm -> 3/12, 26 mm ->
5/12". The 3 and the 5 belong to a scene that no longer exists - a particular
grid, spacing, drop height and density, none of which were written down. This
file builds its OWN fixture, states it in code, and tests the part of the claim
that is fixture-independent: that 40 mm holds everything and that the smaller
sizes do not. Both numbers are printed, and the one that decides the verdict is
the direction, not the count.

So a DISAGREE means the ASSERTION failed to reproduce, and the detail line
carries what was seen instead. It never means "the number moved by one".

## Adding a claim

Quote the document verbatim, name the section, and write the smallest scene
that can settle it. A claim that cannot run must return COULD NOT TEST with a
reason - never a pass. A false pass here is worse than no harness at all,
because it launders a stale claim as a verified one, which is what `CLAUDE.md`
already says about self-checks that cannot fail: "A self-check that cannot fail
is worse than none, because it reads as verified."
"""

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# --- fixed, not the runner's to change --------------------------------------

NEWTON_LAB = Path(os.environ.get("NEWTON_LAB", r"C:\_git\NewtonLab"))
DEPS = os.environ.get("NEWTON_DEPS", r"C:\_git\blender_deps")

#: Everything bakes at Blender's default frame rate, because every number in
#: CLAUDE.md was taken there. A claim measured at 24 fps and re-measured at 60
#: is not the same claim: contacts are sampled once per SUBSTEP, so the frame
#: rate sets the substep duration and with it what the solver can see at all.
FPS = 24.0

AGREE = "AGREE"
DISAGREE = "DISAGREE"
UNTESTED = "COULD NOT TEST"


@dataclass(frozen=True)
class Claim:
    """One documented assertion and the experiment that settles it.

    `quote` is verbatim. Paraphrasing it here would let the harness drift into
    testing something easier than what the document actually promises, which is
    the failure mode this whole file is arranged against.
    """

    id: str
    source: str
    quote: str
    measure: Callable
    slow: bool = False
    #: Roughly what it costs in seconds, from a real run. Advisory only - it is
    #: printed so a run that is going to take a quarter of an hour says so
    #: before it starts rather than after.
    seconds: float = 0.0


CLAIMS = []


def claim(id, source, quote, slow=False, seconds=0.0):
    def register(fn):
        CLAIMS.append(Claim(id=id, source=source, quote=quote.strip(),
                            measure=fn, slow=slow, seconds=seconds))
        return fn
    return register


def find(claim_id):
    for c in CLAIMS:
        if c.id == claim_id:
            return c
    return None


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class Context:
    """newton, warp, the add-on's sim module, and the device to run on.

    Built once and handed to every claim. Importing newton costs a few seconds
    and warp's kernel cache is per-process, so a claim that ran in its own
    interpreter would pay the compile again and the wall-clock column here
    would be mostly compiler.
    """

    def __init__(self, deps=DEPS, device="cuda:0"):
        self.sim = load_sim()
        self.wp, self.newton = self.sim.ensure(deps)
        self.fabric = sys.modules["nlc.fabric"]
        self.tags = sys.modules["nlc.tags"]
        # Sim.__init__ does this before it builds anything. joint_target_q is
        # DOF-shaped by default and newton warns on every finalize that a
        # future release makes it coordinate-shaped; being on the layout the
        # product is on is part of building the world the product's way.
        self.sim.use_coordinate_targets(self.newton)
        self.device, self.device_note = self.sim.resolve_device(self.wp, device)

    def versions(self):
        return self.sim.versions()


def load_sim():
    """newton_lab.sim without Blender, the way tests/verify_sim.py does it.

    Loaded as a package so `from . import tags` resolves. bpy and bmesh are
    stubbed: sim.py only touches bpy lazily inside bpy_path(), and tags.py only
    needs bmesh for convex_hull, so everything exercised here is the real
    module rather than a mock of it.
    """
    import importlib.util

    src = NEWTON_LAB / "src" / "newton_lab"
    bpy = types.ModuleType("bpy")
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules.setdefault("bpy", bpy)
    sys.modules.setdefault("bmesh", types.ModuleType("bmesh"))

    pkg = types.ModuleType("nlc")
    pkg.__path__ = [str(src)]
    sys.modules["nlc"] = pkg
    for name in ("tags", "fabric", "soft", "spring", "wind", "joints",
                 "decompose", "sim"):          # tags first: sim imports it
        spec = importlib.util.spec_from_file_location(
            f"nlc.{name}", src / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"nlc.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["nlc.sim"]


# ---------------------------------------------------------------------------
# Fixtures shared by more than one claim
# ---------------------------------------------------------------------------


def quad(half, z=0.0):
    """A two-triangle square with a +Z normal - exactly what a Blender Plane is.

    Two triangles and not a grid, deliberately. `sim.add_collider` subdivides a
    collider until no edge is longer than a quarter of its bounding-box
    diagonal, so handing it the user's own geometry produces the floor a tagged
    Plane actually produces; handing it a pre-tessellated grid measures a floor
    no buyer has. A 20 m quad comes out as 32 triangles, which is coarse, and
    that coarseness is the product's rather than this harness's.
    """
    verts = np.array([(-half, -half, z), (-half, half, z),
                      (half, -half, z), (half, half, z)], dtype=np.float32)
    tris = np.array([[0, 3, 1], [0, 2, 3]], dtype=np.int32)
    return verts, tris


def flat_sheet(n, size, z):
    """n x n triangulated square in the XY plane at height z."""
    step = size / (n - 1)
    verts = np.array(
        [(i * step - size / 2, j * step - size / 2, z)
         for i in range(n) for j in range(n)],
        dtype=np.float32,
    )
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j + 1, (i + 1) * n + j
            tris += [[a, c, b], [a, d, c]]
    return verts, np.array(tris, dtype=np.int32)


def run(ctx, model, stepper, *, frames, substeps, watch=False):
    """Advance a model and return (leading state, first non-finite frame).

    The pipeline comes from `sim.make_pipeline`. Never call
    `newton.CollisionPipeline` here - see the module docstring, and
    newton_contact/program.md for what it cost the last time.

    `watch` reads body_q back every frame so a divergence can be reported by
    FRAME rather than only as a final NaN, which is the form CLAUDE.md records
    it in. It forces a device sync per frame, which is free next to any solve
    worth watching.
    """
    s0, s1 = model.state(), model.state()
    control = model.control()
    pipeline = ctx.sim.make_pipeline(ctx.newton, model)
    contacts = pipeline.contacts()
    ctx.sim.eval_fk(ctx.newton, model, s0)

    non_finite = -1
    for frame in range(frames):
        s0, s1 = stepper.step(pipeline, contacts, s0, s1, control,
                              dt=1.0 / FPS, substeps=substeps)
        if watch:
            q = s0.body_q.numpy()
            if len(q) and not np.isfinite(q).all():
                non_finite = frame
                break
    return s0, non_finite


def lost_count(state, bodies, below=-0.05):
    """How many of `bodies` are non-finite or below the floor, and every z."""
    z = state.body_q.numpy()[np.asarray(bodies), 2]
    return int((~np.isfinite(z) | (z < below)).sum()), z


def solver_available(ctx, name):
    """Whether this newton has a solver class of that name."""
    return getattr(ctx.newton.solvers, name, None) is not None


def one_line(exc, width=150):
    """An exception as one line, for a table cell.

    Flattened rather than cut at the first newline: newton wraps its refusal
    messages, and SolverKamino puts the part that matters - which feature it
    refused, and how many it found - on the SECOND line. Taking line one alone
    printed "unsupported features:" and stopped, which is the half of the
    sentence that carries no information.
    """
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:width]}"


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------


def _drop_cubes(ctx, *, solver, size, count, drop=0.25, frames=48, substeps=8):
    """Fresh cubes dropped onto a tagged MESH floor. Returns (through, count).

    The fixture CLAUDE.md describes: "Fresh cubes dropped 0.25 m onto a tagged
    mesh floor under MuJoCo, everything else equal". Laid out sparsely, one
    layer, well apart, because the documented threshold was measured that way -
    the same file is explicit that a PILE loses far more and that the warning
    built on these numbers cannot catch that case.
    """
    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton
    vbd = solver == "VBD"

    builder = sim.new_builder(newton, wp, gravity=-9.81)
    verts, tris = quad(1.0)
    sim.add_collider(builder, wp, newton, verts, tris, friction=0.6)

    bodies = []
    per_row = 8
    pitch = max(size * 4.0, 0.08)
    for i in range(count):
        x = (i % per_row - (per_row - 1) * 0.5) * pitch
        y = (i // per_row) * pitch
        bodies.append(sim.add_prop(
            builder, wp, newton,
            xform=wp.transform((x, y, drop + size * 0.5), wp.quat_identity()),
            shape="BOX", dims=(size, size, size), verts=None, tris=None,
            density=1200.0, friction=0.6, restitution=None,
        ))

    model = sim.finalize(builder, has_cloth=False, vbd=vbd)
    stepper, name, _warn = sim.make_stepper(
        newton, solver, model, has_cloth=False, has_rigid=True, iterations=20)
    state, _nf = run(ctx, model, stepper, frames=frames, substeps=substeps)
    through, _z = lost_count(state, bodies)
    return through, name


@claim(
    id="mesh-size-floor-mujoco",
    source="CLAUDE.md, Falling through a collider: three different bugs",
    quote="SolverMuJoCo   18 mm -> 0/32 through      16 mm -> ALL through",
    seconds=25.0,
)
def mesh_size_floor_mujoco(ctx):
    """The size floor under MuJoCo, which `small_prop_warning` is built on.

    Worth re-measuring because a change depends on it in the strongest sense:
    `sim.MESH_CONTACT_MIN_SIZE` hardcodes 0.017, and the bake warns a buyer
    about their props on the strength of it. If the floor moved, the warning is
    either crying wolf or silent when it should not be.
    """
    if not solver_available(ctx, "SolverMuJoCo"):
        return UNTESTED, "this newton has no SolverMuJoCo"

    through_18, name = _drop_cubes(ctx, solver="MUJOCO", size=0.018, count=32)
    through_16, _ = _drop_cubes(ctx, solver="MUJOCO", size=0.016, count=32)
    # Checked after the fact rather than before, because there is no cheaper
    # way: CLAUDE.md records that SolverMuJoCo imports fine without its backend
    # and only raises on construction, so an attribute check cannot detect a
    # missing one. What make_stepper actually returned is the only honest test,
    # and it costs a run to find out.
    if name != "SolverMuJoCo":
        return UNTESTED, (
            f"the MuJoCo backend is missing - make_stepper degraded to {name}")

    detail = (f"18 mm {through_18}/32 through, 16 mm {through_16}/32 through "
              f"(documented 0/32 and 32/32)")
    if through_18 == 0 and through_16 == 32:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="mesh-size-floor-vbd",
    source="CLAUDE.md, Falling through a collider: three different bugs",
    quote="SolverVBD      40 mm -> 0/12 through      30 mm -> 3/12, 26 mm -> 5/12",
    seconds=30.0,
)
def mesh_size_floor_vbd(ctx):
    """The same floor under VBD, which CLAUDE.md puts at more than double.

    The exact 3/12 and 5/12 belong to a fixture that was not written down, so
    the verdict tests the direction: 40 mm holds everything, the smaller sizes
    do not, and 26 mm is no better than 30 mm. See the module docstring on
    re-measuring the assertion rather than the fixture.
    """
    if not solver_available(ctx, "SolverVBD"):
        return UNTESTED, "this newton has no SolverVBD"

    through_40, _ = _drop_cubes(ctx, solver="VBD", size=0.040, count=12)
    through_30, _ = _drop_cubes(ctx, solver="VBD", size=0.030, count=12)
    through_26, _ = _drop_cubes(ctx, solver="VBD", size=0.026, count=12)

    detail = (f"40 mm {through_40}/12, 30 mm {through_30}/12, "
              f"26 mm {through_26}/12 through "
              f"(documented 0/12, 3/12, 5/12)")
    if through_40 == 0 and through_30 > 0 and through_26 >= through_30:
        return AGREE, detail
    return DISAGREE, detail


def _pile(ctx, *, floor, substeps, bricks=200, frames=48):
    """200 bricks at 45 mm on a 20 m floor for 48 frames, as documented.

    Brick-on-brick with a 10 mm gap rather than in a loose lattice: CLAUDE.md
    measured that a stack assembled with gaps lands all at once, and that
    simultaneous impact is itself a cause of divergence, so a lattice would
    measure the fixture instead of the floor.
    """
    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton
    brick = (0.090, 0.045, 0.045)
    dx, dy, dz = brick
    rows, per_row = 5, 8
    layers = -(-bricks // (rows * per_row))

    builder = sim.new_builder(newton, wp, gravity=-9.81)
    if floor == "GROUND":
        sim.add_ground(builder, wp, newton, vbd=False)
    elif floor == "PLANE":
        sim.add_plane_collider(
            builder, wp, newton, position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0), width=20.0, length=20.0,
            friction=0.9)
    elif floor == "BOX":
        # A genuine solid, its top face at z=0. This is the row that refutes
        # "a mesh is a surface, a primitive is a solid" - CLAUDE.md records it
        # diverging much like the mesh, eleven frames later.
        sim.add_box_collider(
            builder, wp, newton, position=(0.0, 0.0, -0.5),
            rotation=(0.0, 0.0, 0.0, 1.0), half_extents=(10.0, 10.0, 0.5),
            friction=0.9)
    elif floor == "MESH":
        verts, tris = quad(10.0)
        sim.add_collider(builder, wp, newton, verts, tris, friction=0.9)
    else:
        raise ValueError(f"unknown floor {floor!r}")

    bodies = []
    for layer in range(layers):
        offset = (dx * 0.5) if layer % 2 else 0.0
        for r in range(rows):
            for i in range(per_row):
                if len(bodies) >= bricks:
                    break
                bodies.append(sim.add_prop(
                    builder, wp, newton,
                    xform=wp.transform((
                        i * dx * 1.02 - (per_row - 1) * dx * 0.51 + offset,
                        r * dy * 1.02 - (rows - 1) * dy * 0.51,
                        dz * 0.5 + layer * (dz + 0.010),
                    ), wp.quat_identity()),
                    shape="BOX", dims=brick, verts=None, tris=None,
                    density=1200.0, friction=0.9, restitution=None,
                ))

    model = sim.finalize(builder, has_cloth=False)
    stepper, _name, _warn = sim.make_stepper(
        newton, "MUJOCO", model, has_cloth=False, has_rigid=True,
        iterations=20)
    state, non_finite = run(ctx, model, stepper, frames=frames,
                            substeps=substeps, watch=True)
    lost, z = lost_count(state, bodies)
    finite = np.isfinite(z)
    # Penetration, not absolute height: a brick resting exactly on the floor
    # has its centre at half its own thickness, and the documented -1.3 mm is
    # how far the worst one is below that.
    worst = (float(np.min(z[finite])) - dz * 0.5) if finite.any() else float("nan")
    return {"lost": lost, "non_finite": non_finite, "worst": worst,
            "total": len(bodies)}


@claim(
    id="plane-floor-holds-a-pile",
    source="CLAUDE.md, A floor under a pile must be a PLANE",
    quote=(
        "mesh floor,      8 substeps -> non-finite at frame 31\n"
        "mesh floor,     16 substeps -> finished, 23/200 lost\n"
        "primitive BOX,   8 substeps -> non-finite at frame 42\n"
        "ground plane,    8 substeps -> finished, 0/200 lost\n"
        "tagged PLANE,    8 substeps -> finished, 0/200 lost, worst -1.3 mm"
    ),
    slow=True,
    seconds=900.0,
)
def plane_floor_holds_a_pile(ctx):
    """The measurement that put PLANE beside BOX and MESH on COLLIDER objects.

    Slow, and it earns it: 200 bodies in sustained contact is where the cost is
    super-linear, and shrinking the pile to make it fast would be measuring a
    different claim. CLAUDE.md is explicit that the failure is a property of
    the PILE - the same bricks in a sparse layer stay put.

    The verdict tests the recommendation the section exists to make: that the
    analytic floors hold the pile and the mesh and box floors do not. The exact
    23/200 and the exact frame numbers are printed and are not required.
    """
    rows = [
        ("mesh floor", "MESH", 8),
        ("mesh floor", "MESH", 16),
        ("primitive BOX", "BOX", 8),
        ("ground plane", "GROUND", 8),
        ("tagged PLANE", "PLANE", 8),
    ]
    seen = {}
    parts = []
    for label, floor, substeps in rows:
        r = _pile(ctx, floor=floor, substeps=substeps)
        seen[(floor, substeps)] = r
        if r["non_finite"] >= 0:
            parts.append(f"{label} @{substeps} non-finite at frame "
                         f"{r['non_finite']}")
        else:
            parts.append(f"{label} @{substeps} finished, {r['lost']}/"
                         f"{r['total']} lost, worst {r['worst'] * 1000:+.1f} mm")
    detail = "; ".join(parts)

    def held(key):
        r = seen[key]
        return r["non_finite"] < 0 and r["lost"] == 0

    def failed(key):
        r = seen[key]
        return r["non_finite"] >= 0 or r["lost"] > 0

    analytic_hold = held(("GROUND", 8)) and held(("PLANE", 8))
    mesh_and_box_fail = failed(("MESH", 8)) and failed(("BOX", 8))
    if analytic_hold and mesh_and_box_fail:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="fabric-drape-ladder",
    source="CLAUDE.md, Fabrics",
    quote="re-measures the ladder (chiffon 0.531 -> leather 0.658, monotonic), "
          "so a change that flattens the library fails rather than shipping",
    seconds=120.0,
)
def fabric_drape_ladder(ctx):
    """Drape a sheet over a sphere and read where the corners settle.

    The test the fabric library exists or does not exist by. A hanging curtain
    measures nothing here - it stays planar and every fabric reads alike, which
    is how the first two attempts at this measured stretch and noise. Bending
    only shows when the cloth has to go round something.
    """
    if not solver_available(ctx, "SolverVBD"):
        return UNTESTED, "this newton has no SolverVBD"

    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton
    radius, height = 0.30, 0.30

    def corner_height(name):
        params = ctx.fabric.resolve(name)
        if params is None:
            raise LookupError(f"no fabric named {name}")
        verts, tris = flat_sheet(21, 1.0, height + radius + 0.15)
        builder = sim.new_builder(newton, wp, gravity=-9.81)
        first, count = sim.add_cloth(
            builder, wp, verts, tris,
            density=params["density"], tri_ke=params["tri_ke"],
            tri_ka=params["tri_ka"], tri_kd=params["tri_kd"],
            edge_ke=params["edge_ke"])
        cfg = sim.shape_config(newton, friction=params["friction"])
        sim.call_flexible(
            builder.add_shape_sphere, -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, height),
                               wp.quat(0.0, 0.0, 0.0, 1.0)),
            radius=radius, cfg=cfg)

        model = sim.finalize(builder, has_cloth=True)
        stepper, _n, _w = sim.make_stepper(
            newton, "VBD", model, has_cloth=True, has_rigid=False)
        # 150 steps at 1/60, which is the cadence the shipping check uses -
        # long enough for the sheet to stop moving on the stiffest fabric.
        s0, s1 = model.state(), model.state()
        control = model.control()
        pipeline = sim.make_pipeline(newton, model)
        contacts = pipeline.contacts()
        sim.eval_fk(newton, model, s0)
        for _ in range(150):
            s0, s1 = stepper.step(pipeline, contacts, s0, s1, control,
                                  dt=1.0 / 60.0, substeps=8)
        p = s0.particle_q.numpy()[first:first + count]
        if not np.isfinite(p).all():
            raise ArithmeticError(f"{name} diverged")
        corners = np.hypot(p[:, 0], p[:, 1]) > 0.45
        return float(p[corners, 2].mean())

    ladder = ["CHIFFON", "SILK", "COTTON", "DENIM", "CANVAS", "LEATHER"]
    heights = [(n, corner_height(n)) for n in ladder]
    detail = ("  ".join(f"{n.lower()} {h:.3f}" for n, h in heights)
              + "  (documented chiffon 0.531 -> leather 0.658)")

    monotonic = all(b > a for (_, a), (_, b) in zip(heights, heights[1:]))
    span = heights[-1][1] - heights[0][1]
    if monotonic and span > 0.08:
        return AGREE, detail
    return DISAGREE, detail + f"; span {span:.3f} m, monotonic {monotonic}"


@claim(
    id="cloth-rests-at-its-thickness",
    source="docs/newton-measured.md, the file CLAUDE.md defers its "
           "measurements to",
    quote="On the defaults the resting height tracks the cloth's own thickness "
          "exactly, 3 mm to +0.003 and 1 mm to +0.001.",
    seconds=30.0,
)
def cloth_rests_at_its_thickness(ctx):
    """Where a dropped sheet comes to rest, against its particle radius.

    A change depends on this: `newton_thickness` is a shipping control and
    `add_cloth` passes it as `particle_radius` precisely because the default of
    0.1 m floated every sheet 100 mm above the floor with nothing exposing it.
    If the relationship stopped being one-to-one, the control stops meaning
    what the UI says it means.
    """
    if not solver_available(ctx, "SolverVBD"):
        return UNTESTED, "this newton has no SolverVBD"

    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton
    params = ctx.fabric.resolve("COTTON")

    def rest(thickness):
        verts, tris = flat_sheet(17, 1.0, 1.0)
        builder = sim.new_builder(newton, wp, gravity=-9.81)
        first, count = sim.add_cloth(
            builder, wp, verts, tris,
            density=params["density"], tri_ke=params["tri_ke"],
            tri_ka=params["tri_ka"], tri_kd=params["tri_kd"],
            edge_ke=params["edge_ke"], particle_radius=thickness)
        floor_v, floor_t = quad(5.0)
        sim.add_collider(builder, wp, newton, floor_v, floor_t, friction=0.5)
        model = sim.finalize(builder, has_cloth=True)
        stepper, _n, _w = sim.make_stepper(
            newton, "VBD", model, has_cloth=True, has_rigid=False)
        s0, s1 = model.state(), model.state()
        control = model.control()
        pipeline = sim.make_pipeline(newton, model)
        contacts = pipeline.contacts()
        sim.eval_fk(newton, model, s0)
        for _ in range(60):
            s0, s1 = stepper.step(pipeline, contacts, s0, s1, control,
                                  dt=1.0 / FPS, substeps=16)
        p = s0.particle_q.numpy()[first:first + count]
        if not np.isfinite(p).all():
            raise ArithmeticError(f"cloth at {thickness} m diverged")
        return float(np.min(p[:, 2]))

    three, one = rest(0.003), rest(0.001)
    detail = (f"3 mm rests at {three:+.4f}, 1 mm rests at {one:+.4f} "
              f"(documented +0.003 and +0.001)")
    # A millimetre of slack on a millimetre-scale claim, and the ordering,
    # which is the part that says the control still does anything at all.
    tracks = abs(three - 0.003) < 0.001 and abs(one - 0.001) < 0.001
    if tracks and three > one:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="explicit-mass-adds-to-density-mass",
    source="CLAUDE.md, Newton API instability - the six that bite most often",
    quote="`add_body(mass=)` **adds to** the density-derived mass rather than "
          "replacing it. Props therefore pass no mass at all.",
    seconds=5.0,
)
def explicit_mass_adds_to_density_mass(ctx):
    """Whether an explicit body mass replaces or accumulates.

    Called RAW rather than through `sim.add_prop`, because the claim is about
    newton's API and add_prop deliberately never passes a mass - and raw rather
    than through `call_flexible`, because call_flexible DROPS a keyword the
    signature will not take, which would turn "the keyword is gone" into a
    silent pass. The signature is inspected first so that case reports itself.

    A unit cube at 1000 kg/m3 has a mass of exactly 1000, which is what makes
    the two possible behaviours 1005 and 5 rather than two similar numbers.
    """
    import inspect

    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton
    builder = sim.new_builder(newton, wp, gravity=-9.81)
    if "mass" not in inspect.signature(builder.add_body).parameters:
        return UNTESTED, "this newton's ModelBuilder.add_body has no mass keyword"

    cfg = sim.shape_config(newton, density=1000.0)
    plain = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0),
                                                wp.quat_identity()))
    builder.add_shape_box(plain, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
    with_mass = builder.add_body(
        xform=wp.transform((4.0, 0.0, 0.0), wp.quat_identity()), mass=5.0)
    builder.add_shape_box(with_mass, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)

    model = sim.finalize(builder, has_cloth=False)
    mass = model.body_mass.numpy()
    a, b = float(mass[plain]), float(mass[with_mass])
    detail = (f"density alone gives {a:.1f} kg, density + mass=5.0 gives "
              f"{b:.1f} kg (adds -> {a + 5:.1f}, replaces -> 5.0)")
    if abs(b - (a + 5.0)) < 0.01:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="joint-qd-is-linear-then-angular",
    source="CLAUDE.md, Newton API instability - the six that bite most often",
    quote="`joint_qd` is LINEAR first then angular - the opposite of warp's "
          "`spatial_vector`. Initial velocity goes there, not into "
          "`state.body_qd`.",
    seconds=10.0,
)
def joint_qd_is_linear_then_angular(ctx):
    """Throw a body, then spin one, and see which slot did which.

    Gravity is off so the displacement is the velocity and nothing else. If the
    layout were angular-first, the throw would spin and the spin would travel,
    which is the exact symptom the note exists to prevent.

    Written through `sim.set_body_velocity` with the state passed, because that
    is what `Sim` does and because without the state the value reaches the
    model and MuJoCo never reads it - the same shape of failure as the dropped
    `density=`.
    """
    if not solver_available(ctx, "SolverMuJoCo"):
        return UNTESTED, "this newton has no SolverMuJoCo"

    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton

    def launch(linear, angular):
        builder = sim.new_builder(newton, wp, gravity=0.0)
        body = sim.add_prop(
            builder, wp, newton,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            shape="BOX", dims=(0.4, 0.4, 0.4), verts=None, tris=None,
            density=400.0, friction=0.5, restitution=0.0)
        model = sim.finalize(builder, has_cloth=False)
        stepper, name, _w = sim.make_stepper(
            newton, "MUJOCO", model, has_cloth=False, has_rigid=True)
        s0, s1 = model.state(), model.state()
        control = model.control()
        pipeline = sim.make_pipeline(newton, model)
        contacts = pipeline.contacts()
        sim.set_body_velocity(model, {body: (linear, angular)}, s0)
        sim.eval_fk(newton, model, s0)
        for _ in range(24):                      # exactly one second
            s0, s1 = stepper.step(pipeline, contacts, s0, s1, control,
                                  dt=1.0 / FPS, substeps=4)
        q = s0.body_q.numpy()[body]
        return np.array(q[:3]), np.array(q[3:]), name

    thrown, throw_quat, name = launch((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    spun, spin_quat, _ = launch((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    if name != "SolverMuJoCo":
        return UNTESTED, f"the MuJoCo backend is missing - got {name}"

    detail = (f"linear slot (2,0,0) -> x={thrown[0]:.3f} spin qx="
              f"{throw_quat[0]:+.3f}; angular slot (3,0,0) -> travel="
              f"{np.linalg.norm(spun):.3f} spin qx={spin_quat[0]:+.3f}")
    linear_translates = abs(thrown[0] - 2.0) < 0.15 and abs(throw_quat[0]) < 0.05
    angular_spins = np.linalg.norm(spun) < 0.05 and abs(spin_quat[0]) > 0.1
    if linear_translates and angular_spins:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="vbd-requires-builder-color",
    source="CLAUDE.md, Newton API instability - the six that bite most often",
    quote="`SolverVBD` raises unless `builder.color()` runs before "
          "`finalize()` on any model containing rigid bodies.",
    seconds=5.0,
)
def vbd_requires_builder_color(ctx):
    """Construct SolverVBD either side of builder.color().

    `sim.finalize` colours whenever has_cloth or vbd is set, so the uncoloured
    half calls `builder.finalize()` raw - the one place in this file where
    going around the product is the point, because the claim is about what
    happens when the product's own precaution is absent.
    """
    if not solver_available(ctx, "SolverVBD"):
        return UNTESTED, "this newton has no SolverVBD"

    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton

    def build(colour):
        builder = sim.new_builder(newton, wp, gravity=-9.81)
        sim.add_ground(builder, wp, newton, vbd=True)
        sim.add_prop(
            builder, wp, newton,
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            shape="BOX", dims=(0.2, 0.2, 0.2), verts=None, tris=None,
            density=500.0, friction=0.5, restitution=None)
        if colour:
            return sim.finalize(builder, has_cloth=False, vbd=True)
        return builder.finalize()

    try:
        newton.solvers.SolverVBD(build(False))
    except Exception as exc:
        raised = one_line(exc, 110)
    else:
        raised = None

    try:
        newton.solvers.SolverVBD(build(True))
    except Exception as exc:
        coloured = one_line(exc)
    else:
        coloured = None

    detail = (f"uncoloured -> {raised or 'constructed with no error'}; "
              f"coloured -> {coloured or 'constructed'}")
    if raised is not None and coloured is None:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="kamino-refuses-particles-not-meshes",
    source="CLAUDE.md, What ships and what does not (corrected 2026-08-24); "
           "docs/newton-measured.md still carries the old claim",
    quote="Geometry type is not in its gate at all. [...] ValueError: "
          "SolverKamino cannot simulate this model due to unsupported "
          "features: particles (found 64) [...] MESH floor (2 tri) + sphere "
          "HELD z=+0.1981",
    seconds=30.0,
)
def kamino_refuses_particles_not_meshes(ctx):
    """Both halves of the correction, because only one of them landed.

    CLAUDE.md was fixed on 2026-08-24 - Kamino refuses PARTICLES, and takes
    mesh colliders perfectly well - but `docs/newton-measured.md` still says
    "Kamino refuses MESH (8), CONVEX_MESH (10) and PLANE (1)". Testing both
    halves is what tells the two documents apart.

    The revisit condition rides on this: CLAUDE.md says to revisit Kamino when
    it takes PARTICLES, not when it takes meshes, so a change would depend on
    which of those is true.
    """
    if not solver_available(ctx, "SolverKamino"):
        return UNTESTED, "this newton has no SolverKamino"

    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton

    # Half one: one cloth sheet, which is the core path this add-on IS.
    verts, tris = flat_sheet(9, 1.0, 1.0)
    builder = sim.new_builder(newton, wp, gravity=-9.81)
    sim.add_cloth(builder, wp, verts, tris, density=0.3, tri_ke=1.0e3,
                  tri_ka=1.0e3, tri_kd=1.0e1, edge_ke=1.0e1)
    cloth_model = sim.finalize(builder, has_cloth=True)
    try:
        newton.solvers.SolverKamino(cloth_model)
    except Exception as exc:
        refusal = one_line(exc, 160)
        mentions_particles = "particle" in str(exc).lower()
    else:
        refusal, mentions_particles = "constructed with no error", False

    # Half two: a sphere dropped 0.6 m onto a two-triangle MESH floor, which
    # is the case newton-measured.md still says it refuses outright.
    builder = sim.new_builder(newton, wp, gravity=-9.81)
    floor_v, floor_t = quad(2.0)
    sim.add_collider(builder, wp, newton, floor_v, floor_t, friction=0.6)
    body = sim.add_prop(
        builder, wp, newton,
        xform=wp.transform((0.0, 0.0, 0.6), wp.quat_identity()),
        shape="SPHERE", dims=(0.4, 0.4, 0.4), verts=None, tris=None,
        density=500.0, friction=0.6, restitution=None)
    model = sim.finalize(builder, has_cloth=False)
    try:
        solver = newton.solvers.SolverKamino(model)
    except Exception as exc:
        return DISAGREE, (
            f"cloth -> {refusal}; but SolverKamino also refused a rigid-only "
            f"model on a mesh floor: {one_line(exc)}")

    # Stepped through the add-on's own Stepper rather than by hand, so the
    # contact buffer and the substep loop are the product's.
    stepper = sim.Stepper(model, rigid=solver, name="SolverKamino")
    state, non_finite = run(ctx, model, stepper, frames=60, substeps=8,
                            watch=True)
    z = float(state.body_q.numpy()[body][2])

    detail = (f"cloth -> {refusal}; sphere on a 2-triangle MESH floor rests at "
              f"z={z:+.4f} (documented +0.1981), non-finite frame {non_finite}")
    held = non_finite < 0 and np.isfinite(z) and abs(z - 0.2) < 0.02
    if mentions_particles and held:
        return AGREE, detail
    return DISAGREE, detail


@claim(
    id="mujoco-and-vbd-never-read-restitution",
    source="CLAUDE.md, What ships and what does not - Restitution / Bounce",
    quote="The control ships and the value reaches the model. MuJoCo and VBD "
          "then never read it. Reported at bake time",
    seconds=90.0,
)
def mujoco_and_vbd_never_read_restitution(ctx):
    """Measure REBOUND, not whether the value arrived.

    Asserting that the setting reaches the model is what the old test did, and
    it always passed - the value arrived every time and nothing read it. So
    this drops a sphere and measures the height it regains as a share of the
    fall, at restitution 0.0 and 0.7 under each solver.

    A solver that bounces materially more at 0.7 than at 0.0 is reading it,
    whatever the document says.
    """
    sim, wp, newton = ctx.sim, ctx.wp, ctx.newton
    DROP, RADIUS = 1.0, 0.2

    def rebound(solver, restitution, frames=96):
        builder = sim.new_builder(newton, wp, gravity=-9.81)
        sim.add_ground(builder, wp, newton, vbd=(solver == "VBD"))
        body = sim.add_prop(
            builder, wp, newton,
            xform=wp.transform((0.0, 0.0, DROP), wp.quat_identity()),
            shape="SPHERE", dims=(RADIUS * 2,) * 3, verts=None, tris=None,
            density=400.0, friction=0.2, restitution=restitution)
        model = sim.finalize(builder, has_cloth=False,
                             vbd=(solver == "VBD"))
        stepper, name, _w = sim.make_stepper(
            newton, solver, model, has_cloth=False, has_rigid=True,
            iterations=20)
        s0, s1 = model.state(), model.state()
        control = model.control()
        pipeline = sim.make_pipeline(newton, model)
        contacts = pipeline.contacts()
        sim.eval_fk(newton, model, s0)
        peak, touched = -9e9, False
        for _ in range(frames):
            s0, s1 = stepper.step(pipeline, contacts, s0, s1, control,
                                  dt=1.0 / FPS, substeps=16)
            z = float(s0.body_q.numpy()[body][2])
            if not np.isfinite(z):
                raise ArithmeticError(
                    f"{solver} went non-finite at restitution {restitution}")
            if z < RADIUS + 0.05:
                touched = True
            if touched:
                peak = max(peak, z)
        return max(0.0, peak - RADIUS) / (DROP - RADIUS), name

    parts, reads = [], []
    for solver in ("MUJOCO", "VBD"):
        dead, name = rebound(solver, 0.0)
        alive, _ = rebound(solver, 0.7)
        parts.append(f"{name} 0.0 -> {dead:.1%} of the fall, 0.7 -> {alive:.1%}")
        # 10% of the fall is the bar the shipping check holds MuJoCo to, and
        # is far above any settling wobble.
        reads.append(alive > dead + 0.10)

    detail = "; ".join(parts)
    if not any(reads):
        return AGREE, detail
    return DISAGREE, (detail + " - a solver the document says ignores "
                               "restitution is bouncing")
