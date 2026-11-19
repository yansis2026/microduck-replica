"""Microduck front-split keyframes (left leg forward).

Plain data only: no mjlab/torch imports, so the CPU pose-check script can load
this module without the training stack.

The 14 actuated joints are in the policy/action order used by every microduck
task. Angles are absolute joint positions in radians (0 = straight-leg zero
pose), not offsets from HOME.
"""

from __future__ import annotations

import math

JOINT_ORDER = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

JOINT_INDEX = {name: i for i, name in enumerate(JOINT_ORDER)}

LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_JOINTS = [5, 6, 7, 8]

# Left leg forward (hip pitch negative), right leg straight backward along the
# floor. Verified with MuJoCo FK + floor contacts at trunk z=0.063
# (scripts/check_split_pose.py): both feet touch, the left shin rests on the
# floor, and trunk/head stay clear (19 mm / 83 mm). The back (right) leg keeps
# hip roll/yaw at zero and only rotates the thigh root back, so the thigh points
# backward and down to a knee near the floor. The knee is as straight as this
# linkage allows (its 3D bend bottoms out at ~37 deg at knee 1.0 rad), and the
# shin lies flat. The ankle is plantar-flexed near its limit so the instep (top
# of the foot) faces down and the toes point backward, sole up - the human
# front-split back foot. The thigh's belly reaches ~10 mm off the floor; it
# cannot lie fully flat because the hip sits ~46 mm up and the trunk is pinned at
# 0.063 m by the front leg.
STAND = {}

TWIST = {
    "left_hip_yaw": 0.25,
    "right_hip_yaw": 0.25,
    "left_hip_roll": -0.15,
    "right_hip_roll": 0.15,
    "left_hip_pitch": -0.70,
    "right_hip_pitch": -0.22,
    "left_knee": 0.25,
    "right_knee": 0.0,
    "left_ankle": 0.15,
    "right_ankle": -0.50,
}

PRESS = {
    "left_hip_yaw": 0.10,
    "right_hip_yaw": 0.10,
    "left_hip_roll": -0.05,
    "right_hip_roll": 0.05,
    "left_hip_pitch": -1.20,
    "right_hip_pitch": -0.50,
    "left_knee": 0.05,
    "right_knee": 0.0,
    "left_ankle": -0.25,
    "right_ankle": -0.20,
}

SPLIT = {
    # Front (left) leg: unchanged - hip open, shin flat on the floor.
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0087,
    "left_hip_pitch": math.radians(-89.0),
    "left_knee": math.radians(1.4),
    "left_ankle": math.radians(-34.1),
    # Back (right) leg: thigh root rotated back, knee straight, shin flat,
    # ankle plantar-flexed so the instep faces the floor and toes point back.
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": math.radians(2.3),
    "right_knee": math.radians(74.5),
    "right_ankle": math.radians(-89.4),
}

# Phase milestones over one episode. The last waypoint is clamped, so the
# split is held from SPLIT_PHASE to the end of the episode.
PHASE_STAND = 0.0
PHASE_TWIST = 0.15
PHASE_PRESS = 0.32
PHASE_SPLIT = 0.48

POSE_WAYPOINTS = (
    (PHASE_STAND, STAND),
    (PHASE_TWIST, TWIST),
    (PHASE_PRESS, PRESS),
    (PHASE_SPLIT, SPLIT),
)

# Trunk z (m) with the feet just touching the floor at each waypoint. 0.115 is
# the standing equilibrium; the TWIST/PRESS values are the solver's required
# heights for their bent-leg poses (measured with scripts/check_split_pose.py,
# which fails the build if a foot is more than 3 mm off the floor).
HEIGHT_WAYPOINTS = (
    (PHASE_STAND, 0.115),
    (PHASE_TWIST, 0.114),
    (PHASE_PRESS, 0.095),
    (PHASE_SPLIT, 0.063),
)

EPISODE_LENGTH_S = 8.0


def overrides_by_index(pose: dict[str, float]) -> dict[int, float]:
    """Convert a name-keyed keyframe to the index-keyed form mdp expects."""
    return {JOINT_INDEX[name]: value for name, value in pose.items()}


def pose_waypoints_by_index() -> list[dict]:
    return [
        {"frac": frac, "overrides": overrides_by_index(pose) or None}
        for frac, pose in POSE_WAYPOINTS
    ]


def height_waypoints() -> list[dict]:
    return [{"frac": frac, "height": height} for frac, height in HEIGHT_WAYPOINTS]
