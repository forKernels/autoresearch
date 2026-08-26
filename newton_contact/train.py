"""The file the agent edits. Everything in SETTINGS is fair game.

Run it:

    python newton_contact/train.py

It builds the fixed pile from `prepare.py` with the settings below, bakes the
fixed frame budget, and prints a summary the runner reads.

The scene is NOT here and must not be moved here. Two hundred bricks, their
sizes, their arrangement and the frame count belong to `prepare.py`, because a
loop that can shrink the pile will shrink the pile.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare  # noqa: E402  - read-only, owns the scene and the metric


# --- the agent's knobs ------------------------------------------------------

SETTINGS = {
    # AUTO, MUJOCO, VBD, COUPLED, STYLE3D. With no cloth in the scene AUTO
    # picks MuJoCo, so naming it is the honest baseline.
    "solver": "MUJOCO",

    # Substeps per frame. Raising this is the obvious move and it is NOT free:
    # the wall-clock cap in prepare.py is what makes it cost something.
    "substeps": 16,

    # Solver iterations. newton-lab measured that this is the contact knob and
    # that `rigid_contact_k_start` is not - the latter changed nothing.
    "iterations": 20,

    # BOX, HULL, SPHERE, MESH, CAPSULE, CYLINDER, ELLIPSOID.
    "shape": "BOX",

    # GROUND (newton's analytic half-space), PLANE (a finite analytic plane),
    # or MESH (a triangle floor). MESH is the baseline ON PURPOSE: it is the
    # case that fails, and a baseline that already scores 1.000 gives the loop
    # nothing to optimise. PLANE holds 48/48 and is not a research target.
    "floor": "MESH",

    "density": 1200.0,
    "friction": 0.9,
    "floor_friction": 0.9,
    "restitution": None,

    "device": "cuda:0",
}


def main():
    try:
        m = prepare.build_and_bake(SETTINGS)
    except TimeoutError as exc:
        print(f"TIMEOUT: {exc}")
        return 1

    print("---")
    print(f"retention:        {m['retention']:.6f}")
    print(f"retained:         {m['retained']}/{m['total']}")
    print(f"lost:             {m['lost']}")
    print(f"non_finite_frame: {m['non_finite_frame']}")
    print(f"lowest_z:         {m['lowest_z']:.4f}")
    print(f"wall_seconds:     {m['wall_seconds']:.1f}")
    print(f"solver:           {m['solver']}")
    print(f"substeps:         {m['substeps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
