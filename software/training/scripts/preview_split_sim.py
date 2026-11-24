"""Local MuJoCo preview of the front-split skill (no training stack needed).

This is not the RL policy: plain ``mujoco`` on CPU, driven by the same phase
clock and keyframes as the training task. It lets you watch the motion that
``Mjlab-Splits-Flat-MicroDuck-HD1910`` is rewarded to learn.

Modes:
    kinematic  set qpos along the interpolated keyframes (target motion).
    pd         closed-loop position servo at an HD-1910-like torque limit;
               stock MuJoCo physics, so the holding phase can sag or tip.

Outputs an animated GIF and a labelled filmstrip PNG (Pillow only).

Usage (from ``software/training``):
    python scripts/preview_split_sim.py --mode kinematic --out out/split
    python scripts/preview_split_sim.py --mode pd --out out/split_pd
    python scripts/preview_split_sim.py --mode kinematic --timeline show --out out/show
    python scripts/preview_split_sim.py --mode kinematic --timeline dance --out out/dance
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

TRAINING = Path(__file__).resolve().parents[1]
SCENE = TRAINING / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"
KEYFRAMES_DIR = TRAINING / "src" / "mjlab_microduck" / "tasks"

WIDTH, HEIGHT = 640, 480
RENDER_FPS = 25
CONTROL_DT = 0.02  # 50 Hz, same as the training env
SUBSTEPS = 10  # 0.002 s physics step

# Torque-limit PD gains; the real servo model lives in BAM at training time.
DEFAULT_KP = 4.0
DEFAULT_KV = 0.08
DEFAULT_TORQUE = 1.2


def keyframes_path(timeline: str) -> Path:
    name = "dance_keyframes.py" if timeline == "dance" else "splits_keyframes.py"
    return KEYFRAMES_DIR / name


def load_keyframes(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_pose(stand_pose: dict, pose: dict) -> dict:
    """Waypoints are partial overrides on top of the STAND pose."""
    out = dict(stand_pose)
    out.update(pose)
    return out


def joint_targets(kf, phase: float, stand_pose: dict, waypoints) -> dict:
    """Interpolate the phase waypoints; clamps outside the first/last."""
    target = _full_pose(stand_pose, waypoints[0][1])
    for i in range(1, len(waypoints)):
        f0, p0 = waypoints[i - 1]
        f1, p1 = waypoints[i]
        if phase <= f0:
            continue
        tau = min(max((phase - f0) / max(f1 - f0, 1e-6), 0.0), 1.0)
        a = _full_pose(stand_pose, p0)
        b = _full_pose(stand_pose, p1)
        target = {name: a[name] * (1.0 - tau) + b[name] * tau for name in a}
    return target


def interp_value(phase: float, points) -> float:
    """Piecewise-linear interpolation of scalar ``(frac, value)`` points."""
    value = points[0][1]
    for i in range(1, len(points)):
        f0, v0 = points[i - 1]
        f1, v1 = points[i]
        if phase <= f0:
            continue
        tau = min(max((phase - f0) / max(f1 - f0, 1e-6), 0.0), 1.0)
        value = v0 * (1.0 - tau) + v1 * tau
    return value


def stand_pose(model, kf) -> dict[str, float]:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    key_qpos = np.array(model.key_qpos[key], copy=True)
    out = {}
    for name in kf.JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        out[name] = float(key_qpos[model.jnt_qposadr[jid]])
    return out


def apply_targets(model, data, target: dict[str, float], z: float | None = None) -> None:
    for name, value in target.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    if z is not None:
        data.qpos[2] = z


def make_camera(timeline: str = "split") -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if timeline == "dance":
        # Near-front view: the weight shift sways along y, hidden from the side.
        cam.lookat[:] = [0.0, 0.0, 0.09]
        cam.distance = 0.45
        cam.azimuth = 160.0
        cam.elevation = -6.0
    else:
        cam.lookat[:] = [0.0, 0.0, 0.08]
        cam.distance = 0.42
        cam.azimuth = 90.0  # side view: the split spreads along +x/-x
        cam.elevation = -8.0
    return cam


def set_frame(model, data, target, roll: float, z: float) -> None:
    """Set the full kinematic pose: base xy=0, base roll, joints, height."""
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = [0.0, 0.0, z]
    half = 0.5 * roll
    data.qpos[3:7] = [math.cos(half), math.sin(half), 0.0, 0.0]  # roll about world x
    for name, value in target.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)


def tune_actuators(model, kp, kv, torque) -> None:
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kv
    model.actuator_forcerange[:, 0] = -torque
    model.actuator_forcerange[:, 1] = torque
    model.actuator_ctrlrange[:, 0] = -3.5
    model.actuator_ctrlrange[:, 1] = 3.5
    model.opt.iterations = 30
    model.opt.ls_iterations = 50


def contact_summary(model, data) -> str:
    names = []
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            body = int(model.geom_bodyid[g])
            if body == 0:  # world/floor
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if name:
                names.append(name)
    return ",".join(sorted(set(names))) or "-"


def run(args) -> None:
    kf = load_keyframes(keyframes_path(args.timeline))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    model.opt.timestep = CONTROL_DT / SUBSTEPS
    stand = stand_pose(model, kf)
    cyclic = args.timeline == "dance"
    roll_wps = None
    if args.timeline == "show":
        pose_wps = kf.SHOW_POSE_WAYPOINTS
        height_wps = kf.SHOW_HEIGHT_WAYPOINTS
    elif args.timeline == "dance":
        pose_wps = kf.DANCE_POSE_WAYPOINTS
        height_wps = kf.DANCE_HEIGHT_WAYPOINTS
        roll_wps = kf.DANCE_BASE_ROLL_WAYPOINTS
    else:
        pose_wps = kf.POSE_WAYPOINTS
        height_wps = kf.HEIGHT_WAYPOINTS
    if args.mode == "pd":
        tune_actuators(model, args.kp, args.kv, args.torque)

    # Start from the STAND keyframe at standing height, identity orientation.
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = [0.0, 0.0, 0.12]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    apply_targets(model, data, stand, 0.12)
    mujoco.mj_forward(model, data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = None
    cam = make_camera(args.timeline)
    frames = []
    marks = []

    def max_track_error(target: dict[str, float]) -> float:
        err = 0.0
        for name, value in target.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            err = max(err, abs(data.qpos[model.jnt_qposadr[jid]] - value))
        return err

    total_steps = int(round(args.duration / CONTROL_DT))
    settle_steps = int(round(args.settle / CONTROL_DT))
    frame_every = max(1, int(round(1.0 / (RENDER_FPS * CONTROL_DT))))

    for step in range(total_steps + 1):
        t = step * CONTROL_DT
        if step < settle_steps:
            phase = 0.0
        else:
            raw = (t - args.settle) / args.period
            # The dance cycles: wrap around. The one-way split/show clamp and
            # hold the last pose instead.
            phase = (raw % 1.0) if cyclic else min(raw, 1.0)
        target = joint_targets(kf, phase, stand, pose_wps)
        z = interp_value(phase, height_wps)

        if args.mode == "kinematic":
            if cyclic:
                # Dance: the leg joints are pre-solved so the soles stay flat and
                # planted while the base rolls to sink one hip. The base roll and
                # height come straight from the stored waypoints.
                roll = interp_value(phase, roll_wps)
                set_frame(model, data, target, roll, z)
            else:
                apply_targets(model, data, target, z)
                mujoco.mj_forward(model, data)
        else:
            for name in kf.JOINT_ORDER:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[aid] = target[name]
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        if step % frame_every == 0:
            marks.append(
                (t, phase, data.qpos[2], contact_summary(model, data),
                 max_track_error(target))
            )
            if not args.no_render:
                if renderer is None:
                    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
                renderer.update_scene(data, camera=cam)
                frames.append(Image.fromarray(renderer.render()))

    if args.no_render:
        print(f"mode={args.mode} duration={args.duration:.1f}s (no render)")
        for t, phase, z, contacts, err in marks[:: max(1, len(marks) // 10)]:
            print(
                f"  t={t:5.2f}s phase={phase:5.2f} trunk_z={z:6.3f} "
                f"err={err:5.3f} contacts={contacts}"
            )
        return

    if not frames:
        raise SystemExit("no frames rendered")

    gif_path = out_dir / f"{args.timeline}_{args.mode}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / RENDER_FPS),
        loop=0,
        optimize=True,
    )

    # Filmstrip: 8 evenly spaced panels with phase/time labels.
    picks = [int(round(i * (len(frames) - 1) / 7)) for i in range(8)]
    thumb_w, thumb_h = WIDTH, HEIGHT
    strip = Image.new("RGB", (thumb_w * 4, thumb_h * 2 + 2 * 18), (20, 20, 20))
    draw = ImageDraw.Draw(strip)
    for k, idx in enumerate(picks):
        t, phase, z, _, _ = marks[idx]
        panel = frames[idx].resize((thumb_w, thumb_h))
        x = (k % 4) * thumb_w
        y = (k // 4) * (thumb_h + 18) + 18
        strip.paste(panel, (x, y))
        draw.text((x + 4, y - 15), f"t={t:.2f}s p={phase:.2f} z={z:.3f}", fill=(230, 230, 230))
    strip_path = out_dir / f"{args.timeline}_{args.mode}_strip.png"
    strip.save(strip_path)

    print(f"mode={args.mode} frames={len(frames)} duration={args.duration:.1f}s")
    print(f"gif:      {gif_path}")
    print(f"filmstrip:{strip_path}")
    for t, phase, z, contacts, err in marks[:: max(1, len(marks) // 8)]:
        print(
            f"  t={t:5.2f}s phase={phase:5.2f} trunk_z={z:6.3f} "
            f"err={err:5.3f} contacts={contacts}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("kinematic", "pd"), default="kinematic")
    parser.add_argument("--timeline", choices=("split", "show", "dance"), default="split",
                        help="split: one-way split; show: split/hold/return; "
                             "dance: cyclic alternating weight shift")
    parser.add_argument("--out", default="out/split_preview")
    parser.add_argument("--duration", type=float, default=None,
                        help="defaults to the timeline's episode length")
    parser.add_argument("--period", type=float, default=None,
                        help="defaults to the timeline's phase period")
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kv", type=float, default=DEFAULT_KV)
    parser.add_argument("--torque", type=float, default=DEFAULT_TORQUE)
    args = parser.parse_args()
    if args.duration is None or args.period is None:
        kf = load_keyframes(keyframes_path(args.timeline))
        if args.timeline == "show":
            default_duration = kf.SHOW_EPISODE_LENGTH_S
            default_period = kf.SHOW_PERIOD_S
        elif args.timeline == "dance":
            default_duration = kf.DANCE_EPISODE_LENGTH_S
            default_period = kf.DANCE_PERIOD_S
        else:
            default_duration = 8.0
            default_period = 9.0
        if args.duration is None:
            args.duration = default_duration
        if args.period is None:
            args.period = default_period
    run(args)


if __name__ == "__main__":
    main()
