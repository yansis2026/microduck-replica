"""CPU MuJoCo feasibility of the split keyframes (no training stack)."""

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
    keyframes = checker.load_keyframes()
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
        "shin_left": [body_id("leg")],
        "trunk": [body_id("trunk_base")],
        "head": checker.body_subtree(model, body_id("neck")),
    }
    return checker, keyframes, model, data, key_qpos, ids


def _pose(rig, frac):
    checker, keyframes, model, data, key_qpos, ids = rig
    heights = dict(
        (round(f, 6), h) for f, h in keyframes.HEIGHT_WAYPOINTS
    )
    pose = dict(keyframes.POSE_WAYPOINTS)[frac]
    checker.apply_pose(model, data, key_qpos, pose, heights[round(frac, 6)])
    return {
        name: checker.geom_min_z(model, data, body_ids)
        for name, body_ids in ids.items()
    }


@pytest.mark.parametrize("index", range(4))
def test_trunk_and_head_stay_clear_of_the_floor(rig, index):
    frac = rig[1].POSE_WAYPOINTS[index][0]
    lows = _pose(rig, frac)
    assert lows["trunk"] > 0.0
    assert lows["head"] > 0.0

    checker, _, model, data, _, _ = rig
    contacts = checker.ground_contacts(model, data)
    assert not any(name.startswith("trunk_base") for name in contacts)
    assert not any(name.startswith("neck") for name in contacts)


@pytest.mark.parametrize("index", range(4))
def test_both_feet_are_down_at_each_waypoint(rig, index):
    frac = rig[1].POSE_WAYPOINTS[index][0]
    lows = _pose(rig, frac)
    assert abs(lows["foot_left"]) < 0.004
    assert abs(lows["foot_right"]) < 0.004


def test_final_split_has_left_forward_right_back_and_shin_contact(rig):
    checker, keyframes, model, data, key_qpos, ids = rig
    # Record which foot-local axes point forward (toe) and up (instep) at
    # STAND, so the back foot's orientation can be checked in world space.
    checker.apply_pose(model, data, key_qpos, keyframes.STAND, 0.115)
    foot_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"
    )
    stand_rot = data.geom_xmat[foot_geom].reshape(3, 3)
    toe_local = stand_rot.T @ np.array([1.0, 0.0, 0.0])
    instep_local = stand_rot.T @ np.array([0.0, 0.0, 1.0])

    _pose(rig, keyframes.PHASE_SPLIT)
    left_x = data.xpos[ids["foot_left"][0]][0]
    right_ankle = data.xpos[ids["foot_right"][0]]
    assert left_x > 0.05
    # The straight back leg sits behind and on the body's right side.
    assert right_ankle[0] < -0.025
    assert right_ankle[1] < -0.05
    # The human front-split back foot: toes point backward (-x) and the instep
    # (top of the foot) faces the floor, so the sole points up.
    rot = data.geom_xmat[foot_geom].reshape(3, 3)
    toe = rot @ toe_local
    instep = rot @ instep_local
    assert toe[0] < -0.85
    assert instep[2] < -0.85
    # Knee is as straight as the linkage allows and the shin lies flat, with
    # the thigh belly as close to the floor as the geometry permits.
    right_shin = checker.geom_min_z(model, data, [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_2")
    ])
    right_thigh = checker.geom_min_z(model, data, [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "upper_leg_right")
    ])
    assert abs(right_shin) < 0.004
    assert right_thigh < 0.02
    assert ids["shin_left"][0] in {
        model.geom_bodyid[data.contact[i].geom1] for i in range(data.ncon)
    } | {
        model.geom_bodyid[data.contact[i].geom2] for i in range(data.ncon)
    }
    assert checker.ground_contacts(model, data)
