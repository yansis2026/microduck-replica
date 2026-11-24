"""CPU MuJoCo feasibility of the dance keyframes (no training stack).

Checks the user's requirements directly on the robot model: the soles stay flat
and planted on the floor (they do not tilt or walk), the pelvis stays clear, and
at each sink the loaded hip is lower while the opposite leg is stretched longer.
"""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1]


def _load_checker():
    path = TRAINING / "scripts" / "check_split_pose.py"
    spec = importlib.util.spec_from_file_location("check_split_pose", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rig():
    checker = _load_checker()
    keyframes = checker.load_keyframes(checker.keyframes_path("dance"))
    model = mujoco.MjModel.from_xml_path(str(checker.SCENE))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    assert key >= 0
    key_qpos = np.array(model.key_qpos[key], copy=True)

    def body_id(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    ids = {
        "foot_left": [body_id("ankle_left")],
        "foot_right": [body_id("ankle_right")],
        "hip_left": [body_id("upper_leg_left")],
        "hip_right": [body_id("upper_leg_right")],
        "trunk": [body_id("trunk_base")],
        "head": checker.body_subtree(model, body_id("neck")),
    }
    heights = dict((round(f, 6), h) for f, h in keyframes.DANCE_HEIGHT_WAYPOINTS)
    rolls = dict((round(f, 6), r) for f, r in keyframes.DANCE_BASE_ROLL_WAYPOINTS)
    return checker, keyframes, model, data, key_qpos, ids, heights, rolls


def _metrics(rig, frac):
    checker, keyframes, model, data, key_qpos, ids, heights, rolls = rig
    freq = round(frac, 6)
    pose = dict(keyframes.DANCE_POSE_WAYPOINTS)[freq]
    checker.apply_pose(model, data, key_qpos, pose, heights[freq], base_roll=rolls[freq])
    foot_l = data.xpos[ids["foot_left"][0]].copy()
    foot_r = data.xpos[ids["foot_right"][0]].copy()
    hip_l = data.xpos[ids["hip_left"][0]].copy()
    hip_r = data.xpos[ids["hip_right"][0]].copy()
    return {
        "foot_l": checker.geom_min_z(model, data, ids["foot_left"]),
        "foot_r": checker.geom_min_z(model, data, ids["foot_right"]),
        "up_l": checker.foot_sole_up(model, data, ids["foot_left"]),
        "up_r": checker.foot_sole_up(model, data, ids["foot_right"]),
        "trunk": checker.geom_min_z(model, data, ids["trunk"]),
        "head": checker.geom_min_z(model, data, ids["head"]),
        "xy_l": foot_l[:2].copy(),
        "xy_r": foot_r[:2].copy(),
        "hip_l_z": float(hip_l[2]),
        "hip_r_z": float(hip_r[2]),
        # Hip-to-ankle distance: shrinks on the loaded (sink) side, grows on the
        # stretched (opposite) side.
        "leg_l": float(np.linalg.norm(foot_l - hip_l)),
        "leg_r": float(np.linalg.norm(foot_r - hip_r)),
    }


@pytest.mark.parametrize("index", range(9))
def test_soles_stay_flat_and_planted(rig, index):
    frac = rig[1].DANCE_POSE_WAYPOINTS[index][0]
    m = _metrics(rig, frac)
    # Soles flat (normal points up) and planted within 3 mm of the floor.
    assert m["up_l"] >= 0.99
    assert m["up_r"] >= 0.99
    assert abs(m["foot_l"]) < 0.003
    assert abs(m["foot_r"]) < 0.003
    # The whole dance stays upright; the trunk and head never approach the floor.
    assert m["trunk"] > 0.03
    assert m["head"] > 0.05


def test_soles_do_not_walk(rig):
    # The soles are pinned: no foot centre drifts more than a few mm over the
    # whole groove (the tiny hip_yaw pivot keeps them effectively in place).
    ref = _metrics(rig, 0.0)
    for frac, _ in rig[1].DANCE_POSE_WAYPOINTS:
        m = _metrics(rig, frac)
        move = max(
            float(np.linalg.norm(m["xy_l"] - ref["xy_l"])),
            float(np.linalg.norm(m["xy_r"] - ref["xy_r"])),
        )
        assert move < 0.006, f"sole moved {move * 1000:.1f} mm at phase {frac:.3f}"


def test_sink_lowers_that_hip_and_stretches_the_other_leg(rig):
    # Sink LEFT (i=2): left hip lower, left leg compressed, right leg stretched.
    m2 = _metrics(rig, 0.25)
    assert m2["hip_l_z"] < m2["hip_r_z"]
    assert m2["leg_l"] < m2["leg_r"]
    # Sink RIGHT (i=6): mirror image.
    m6 = _metrics(rig, 0.75)
    assert m6["hip_r_z"] < m6["hip_l_z"]
    assert m6["leg_r"] < m6["leg_l"]


def test_cycle_closes(rig):
    # First and last waypoint are the same pose at the same roll/height, so the
    # phase-wrap between them introduces no jump.
    first = _metrics(rig, 0.0)
    last = _metrics(rig, 1.0)
    for name in ("foot_l", "foot_r", "up_l", "up_r", "hip_l_z", "hip_r_z"):
        assert first[name] == pytest.approx(last[name], abs=1e-6)
    assert np.allclose(first["xy_l"], last["xy_l"], atol=1e-6)
    assert np.allclose(first["xy_r"], last["xy_r"], atol=1e-6)
