"""CPU-only MuJoCo check for the Microduck front-split keyframes.

Loads ``tasks/splits_keyframes.py`` as a plain module (no mjlab/torch import),
poses the full-collision robot at each waypoint's trunk height, and reports the
lowest point of every leg/trunk/head body plus any ground contact. This is the
kinematic feasibility check referenced by ``splits_keyframes.py``.

Usage:
    python scripts/check_split_pose.py
    python scripts/check_split_pose.py --render out/   # needs a GL backend
    python scripts/check_split_pose.py --timeline show
    python scripts/check_split_pose.py --timeline dance

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
KEYFRAMES_DIR = TRAINING / "src" / "mjlab_microduck" / "tasks"
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


def keyframes_path(timeline: str) -> Path:
    name = "dance_keyframes.py" if timeline == "dance" else "splits_keyframes.py"
    return KEYFRAMES_DIR / name


def load_keyframes(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
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


def foot_sole_up(model: mujoco.MjModel, data: mujoco.MjData, body_ids) -> float:
    """Foot sole normal (foot local x) world z-component over the given bodies.

    ~1.0 means the sole lies flat on the floor; a tilted foot (up on edge)
    reads much lower. Used by the dance check to catch ankle/foot edge-tilting.
    """
    values = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in body_ids:
            continue
        values.append(float(data.geom_xmat[gid].reshape(3, 3)[2, 0]))
    return max(values) if values else float("nan")


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


def apply_pose(model, data, key_qpos, pose, trunk_z, base_roll=0.0):
    data.qpos[:] = key_qpos
    data.qpos[2] = trunk_z
    if base_roll:
        # Roll the free base about world x. The dance weights shift by rolling
        # the pelvis while the soles stay flat and planted; the leg joints are
        # pre-solved so both feet remain level, and in training this base roll
        # emerges from physics.
        half = 0.5 * base_roll
        data.qpos[3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
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
    parser.add_argument("--timeline", choices=("split", "show", "dance"), default="split",
                        help="split: one-way split; show: split/hold/return; "
                             "dance: cyclic alternating weight shift")
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--distance", type=float, default=0.42)
    args = parser.parse_args()

    kf = load_keyframes(keyframes_path(args.timeline))
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
    foot_body_l = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")]
    foot_body_r = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_right")]

    if args.timeline == "show":
        pose_wps = kf.SHOW_POSE_WAYPOINTS
        height_wps = kf.SHOW_HEIGHT_WAYPOINTS
        labels = kf.SHOW_LABELS
    elif args.timeline == "dance":
        pose_wps = kf.DANCE_POSE_WAYPOINTS
        height_wps = kf.DANCE_HEIGHT_WAYPOINTS
        roll_by_frac = dict(kf.DANCE_BASE_ROLL_WAYPOINTS)
        labels = kf.DANCE_LABELS
    else:
        pose_wps = kf.POSE_WAYPOINTS
        height_wps = kf.HEIGHT_WAYPOINTS
        labels = tuple(
            {
                kf.PHASE_STAND: "STAND",
                kf.PHASE_TWIST: "TWIST",
                kf.PHASE_PRESS: "PRESS",
                kf.PHASE_SPLIT: "SPLIT",
            }.get(frac, f"phase {frac:.2f}")
            for frac, _ in pose_wps
        )

    print(f"model: {SCENE}")
    print(f"timeline: {args.timeline}")
    print(f"waypoints: {[f for f, _ in pose_wps]}")
    failures = []
    ref_foot_xy = None
    for index, ((frac, pose), (_, trunk_z), name) in enumerate(
        zip(pose_wps, height_wps, labels)
    ):
        if args.timeline == "dance":
            roll = roll_by_frac[frac]
            apply_pose(model, data, key_qpos, pose, trunk_z, base_roll=roll)
            foot_l = geom_min_z(model, data, foot_body_l)
            foot_r = geom_min_z(model, data, foot_body_r)
            up_l = foot_sole_up(model, data, foot_body_l)
            up_r = foot_sole_up(model, data, foot_body_r)
            xy_l = data.xpos[foot_body_l[0]][:2].copy()
            xy_r = data.xpos[foot_body_r[0]][:2].copy()
            if ref_foot_xy is None:
                ref_foot_xy = (xy_l, xy_r)
            move = max(np.linalg.norm(xy_l - ref_foot_xy[0]),
                       np.linalg.norm(xy_r - ref_foot_xy[1]))
            extra = (f"roll={roll:+.4f} sole_up L/R={up_l:.3f}/{up_r:.3f} "
                     f"foot_move={move * 1000:.1f}mm  ")
        else:
            apply_pose(model, data, key_qpos, pose, trunk_z)
            foot_l = foot_r = None
            up_l = up_r = move = 0.0
            extra = ""

        lows = {key: geom_min_z(model, data, ids) for key, ids in body_ids.items()}
        contacts = ground_contacts(model, data)
        print(
            f"\n{name:13s} phase={frac:.4f} {extra}trunk_z={trunk_z:.3f}  "
            f"foot={lows['foot']:+.4f} shin={lows['shin']:+.4f} "
            f"upper={lows['upper']:+.4f} trunk={lows['trunk']:+.4f} "
            f"head={lows['head']:+.4f}"
        )
        if foot_l is not None:
            print(f"      feet: left={foot_l:+.4f} right={foot_r:+.4f}")
        print(f"      ground contacts: {contacts or 'none'}")

        if args.timeline == "dance":
            for side, value in (("left", foot_l), ("right", foot_r)):
                if abs(value) > 0.003:
                    failures.append(
                        f"{name} (phase {frac:.4f}): {side} foot lowest point "
                        f"{value:+.4f} m (expected within 3 mm of the floor)"
                    )
            for side, up in (("left", up_l), ("right", up_r)):
                if up < 0.99:
                    failures.append(
                        f"{name} (phase {frac:.4f}): {side} sole is tilted "
                        f"(sole normal z {up:.3f}, expected >= 0.99 / flat)"
                    )
            if move > 0.006:
                failures.append(
                    f"{name} (phase {frac:.4f}): a sole moved {move * 1000:.1f} mm "
                    "on the floor (expected <= 6 mm; soles should stay planted)"
                )
        elif abs(lows["foot"]) > 0.003:
            failures.append(
                f"{name} (phase {frac:.4f}): foot lowest point {lows['foot']:+.4f} m "
                "(expected within 3 mm of the floor)"
            )
        if lows["trunk"] <= 0.0:
            failures.append(
                f"{name} (phase {frac:.4f}): trunk touches the floor "
                f"({lows['trunk']:+.4f} m)"
            )
        if lows["head"] <= 0.0:
            failures.append(
                f"{name} (phase {frac:.4f}): head touches the floor "
                f"({lows['head']:+.4f} m)"
            )

        if args.render is not None:
            safe = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
            render(model, data, args.render / f"{index:02d}_{safe}.png",
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
