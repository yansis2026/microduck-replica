"""CPU-only MuJoCo check for the Microduck front-split keyframes.

Loads ``tasks/splits_keyframes.py`` as a plain module (no mjlab/torch import),
poses the full-collision robot at each waypoint's trunk height, and reports the
lowest point of every leg/trunk/head body plus any ground contact. This is the
kinematic feasibility check referenced by ``splits_keyframes.py``.

Usage:
    python scripts/check_split_pose.py
    python scripts/check_split_pose.py --render out/   # needs a GL backend

Exit code is non-zero if the trunk or head touches the floor at any waypoint,
or if a foot is more than 3 mm above/below the floor at the SPLIT waypoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
KEYFRAMES = TRAINING / "src" / "mjlab_microduck" / "tasks" / "splits_keyframes.py"
SCENE = (
    TRAINING
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
    / "scene.xml"
)

FOOT_BODIES = ("ankle_left", "ankle_right")
SHIN_BODIES = ("leg", "leg_2")
UPPER_BODIES = ("upper_leg_left", "upper_leg_right")


def load_keyframes():
    spec = importlib.util.spec_from_file_location("splits_keyframes", KEYFRAMES)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body_subtree(model: mujoco.MjModel, root: int) -> np.ndarray:
    ids = [root]
    for body in range(model.nbody):
        parent = body
        while parent != 0:
            parent = model.body_parentid[parent]
            if parent == root:
                ids.append(body)
                break
    return np.asarray(ids, dtype=int)


def geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, body_ids) -> float:
    """Lowest point over mesh vertices / primitive extents of the given bodies."""
    lows = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in body_ids:
            continue
        gtype = model.geom_type[gid]
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            count = model.mesh_vertnum[mid]
            verts = model.mesh_vert[start : start + count]
            lows.append(float((verts @ mat.T + pos)[:, 2].min()))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            signs = np.array(
                [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
            corners = (signs * model.geom_size[gid]) @ mat.T + pos
            lows.append(float(corners[:, 2].min()))
        else:
            # Sphere/capsule/cylinder: conservative radius-based bound.
            lows.append(float(pos[2] - model.geom_size[gid][0]))
    return min(lows) if lows else float("nan")


def ground_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 != 0 and b2 != 0:
            continue
        body = b1 if b1 != 0 else b2
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"
        geom = c.geom1 if b2 == 0 else c.geom2
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"
        out.append(f"{name}:{geom_name}")
    return sorted(set(out))


def apply_pose(model, data, key_qpos, pose, trunk_z):
    data.qpos[:] = key_qpos
    data.qpos[2] = trunk_z
    for joint_name, value in pose.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise SystemExit(f"unknown joint in keyframe: {joint_name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def render(model, data, path: Path, azimuth: float, elevation: float, distance: float):
    try:
        import mujoco  # noqa: F401  (renderer is part of mujoco)
    except ImportError:  # pragma: no cover
        return False
    try:
        renderer = mujoco.Renderer(model, 480, 640)
    except Exception as exc:  # pragma: no cover - no GL backend
        print(f"[render] skipped: {exc}")
        return False
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.qpos[:3]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        print(f"[render] {path}")
    except ImportError:
        np.save(path.with_suffix(".npy"), rgb)
        print(f"[render] {path.with_suffix('.npy')} (PIL missing)")
    renderer.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=Path, default=None,
                        help="directory for one PNG per waypoint")
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--distance", type=float, default=0.42)
    args = parser.parse_args()

    kf = load_keyframes()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    stand_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_key < 0:
        raise SystemExit("scene.xml has no STAND keyframe")
    key_qpos = np.array(model.key_qpos[stand_key], copy=True)

    body_ids = {}
    for names, key in ((FOOT_BODIES, "foot"), (SHIN_BODIES, "shin"),
                       (UPPER_BODIES, "upper")):
        body_ids[key] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in names]
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    neck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "neck")
    body_ids["trunk"] = [trunk_id]
    body_ids["head"] = body_subtree(model, neck_id)

    heights = dict(
        (round(frac, 6), height)
        for frac, height in ((f, h) for f, h in kf.HEIGHT_WAYPOINTS)
    )

    print(f"model: {SCENE}")
    print(f"waypoints: {[f for f, _ in kf.POSE_WAYPOINTS]}")
    failures = []
    for frac, pose in kf.POSE_WAYPOINTS:
        name = {
            kf.PHASE_STAND: "STAND",
            kf.PHASE_TWIST: "TWIST",
            kf.PHASE_PRESS: "PRESS",
            kf.PHASE_SPLIT: "SPLIT",
        }.get(frac, f"phase {frac:.2f}")
        trunk_z = heights[round(frac, 6)]
        apply_pose(model, data, key_qpos, pose, trunk_z)

        lows = {key: geom_min_z(model, data, ids) for key, ids in body_ids.items()}
        contacts = ground_contacts(model, data)
        print(
            f"\n{name:5s} phase={frac:.2f} trunk_z={trunk_z:.3f}  "
            f"foot={lows['foot']:+.4f} shin={lows['shin']:+.4f} "
            f"upper={lows['upper']:+.4f} trunk={lows['trunk']:+.4f} "
            f"head={lows['head']:+.4f}"
        )
        print(f"      ground contacts: {contacts or 'none'}")

        if abs(lows["foot"]) > 0.003:
            failures.append(
                f"{name}: foot lowest point {lows['foot']:+.4f} m "
                "(expected within 3 mm of the floor)"
            )
        if lows["trunk"] <= 0.0:
            failures.append(f"{name}: trunk touches the floor ({lows['trunk']:+.4f} m)")
        if lows["head"] <= 0.0:
            failures.append(f"{name}: head touches the floor ({lows['head']:+.4f} m)")

        if args.render is not None:
            render(model, data, args.render / f"{name.lower()}.png",
                   args.azimuth, args.elevation, args.distance)

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nOK: all waypoints are kinematically feasible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
