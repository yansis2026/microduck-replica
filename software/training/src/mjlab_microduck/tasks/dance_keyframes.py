"""Microduck in-place weight-shift dance keyframes (sink hip / stretch leg).

Plain data only: no mjlab/torch imports, so the CPU pose-check and preview
scripts can load this module without the training stack.

The motion is a continuous, cyclic, energetic bounce. The robot stands, then
squats a little (both knees fold, dropping the pelvis ~1 cm) with the *loaded*
knee folding far more than the other, returns to the stand, and squats the other
way, so left and right alternate with a centred stand between each side:

    STAND -> sink LEFT (left knee folds most, right knee less) -> STAND
          -> sink RIGHT (right knee folds most, left knee less) -> STAND -> repeat

The head sways with the groove: ``head_roll`` tilts it toward the sunk hip. With
``w = sin(2*pi*i/8)`` over the eight segments, ``w`` is 0 (stand) at the ends and
at the half-cycle, ``+1`` (sink left) at the quarter point, ``-1`` (sink right)
at three-quarters — i.e. stand / sink-left / stand / sink-right / stand.

**Soles stay flat and planted.** The user requirement is that the feet do *not*
tilt or walk — one leg bends slightly ("微屈") while the other stays stretched,
and the *trunk does not sway much*. A naive "flat-foot family"
(``hip_pitch - b/2, knee + b, ankle - b/2``) does NOT keep the sole flat once the
knee bends (the pitch axes are no longer parallel), so it rocks the foot onto
its edge. Instead the nine waypoint poses below were solved with a per-leg IK:
with the base rolled by ``DANCE_BASE_ROLL_WAYPOINTS``, each leg's five joints are
solved so the sole normal points straight up and the foot stays pinned at its
standing (x, y, z). ``hip_yaw`` is bounded to ``+-YAW_BOUND`` so the feet only
pivot a hair instead of toe-ing out.

The "sink" is an energetic squat routed through the knees: both knees fold so
the pelvis visibly bobs down (~1 cm), while the *loaded* knee folds far more
(``~34 deg``) than the other (``~17 deg``). The two legs therefore read as one
bent / one straighter, and left and right alternate. With both feet pinned the
pelvis also rolls slightly toward the sunk hip (``BASE_ROLL_MAX`` ~3 deg). The
visible motion is the big up-down bob, the alternating leg bend, and a lively
head sway (``~10 deg``).

At training time the base roll is not commanded to the joints — it emerges from
physics (both feet grounded) just like the split tasks. The CPU check/preview
reproduce it by applying ``DANCE_BASE_ROLL_WAYPOINTS`` to the free base. See
docs/dance-task.md.

Joint layout (14 actuated joints, policy/action order):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
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

# HOME / STAND joint positions (rad). Matches microduck_constants.HOME_FRAME and
# the STAND keyframe in scene.xml. Left leg forward, right mirrored.
HOME = {
    "left_hip_roll": -0.0873,
    "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}

# ── Timeline constants ───────────────────────────────────────────────────────
DANCE_PERIOD_S = 2.0          # one left->right->left groove (100 control steps)
DANCE_EPISODE_LENGTH_S = 8.0  # four cycles (400 steps)
N_SEGMENTS = 8                # 8 segments -> 9 waypoints; first == last (cyclic)

BASE_ROLL_MAX = 0.050292      # pelvis roll toward the sunk hip (rad, ~2.9 deg)
HEAD_ROLL_MAX = 0.18          # head tilt toward the sunk hip (rad, ~10.3 deg)
YAW_BOUND = 0.04              # max |hip_yaw| the IK used (rad, ~2.3 deg)

# Base z (m). The robot BOBS: both knees fold at each sink (a small energetic
# squat) so the pelvis drops ~1 cm, then rises back to the standing height at the
# centred stands. The dip is symmetric left/right.
_STAND_HEIGHT = 0.11718
_EASE_HEIGHT = 0.110009
_SINK_HEIGHT = 0.107039
DANCE_HEIGHT_WAYPOINTS = tuple(zip(
    (i / N_SEGMENTS for i in range(N_SEGMENTS + 1)),
    (_STAND_HEIGHT, _EASE_HEIGHT, _SINK_HEIGHT, _EASE_HEIGHT, _STAND_HEIGHT,
     _EASE_HEIGHT, _SINK_HEIGHT, _EASE_HEIGHT, _STAND_HEIGHT),
))

# w = sin(2*pi*i/N_SEGMENTS): +1 -> sink left, -1 -> sink right, 0 -> stand.
_W = tuple(math.sin(2.0 * math.pi * i / N_SEGMENTS) for i in range(N_SEGMENTS + 1))

# Base roll (rad) that levels the two feet at each waypoint: negative rolls
# toward the LEFT hip (which sinks), positive toward the RIGHT hip.
DANCE_BASE_ROLL_WAYPOINTS = tuple(
    (i / N_SEGMENTS, -BASE_ROLL_MAX * _W[i]) for i in range(N_SEGMENTS + 1)
)

# ── Measured joint targets ───────────────────────────────────────────────────
# Per-leg absolute joint angles, order (hip_yaw, hip_roll, hip_pitch, knee,
# ankle). Solved by a bounded-yaw per-leg IK (|hip_yaw| <= YAW_BOUND) that pins
# each sole flat (normal up) and planted at its standing (x, y, z). The loaded
# knee is flexed to a target (the "sink"); the other knee stays at its standing
# value, so the two legs read as one bent / one straight. Feet end up flat
# (sole up ~1.00), planted (~1 mm) and nearly static (<=1 mm) across the cycle.
_LEG_ORDER = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")

_STAND = {
    "L": (0.0, HOME["left_hip_roll"], HOME["left_hip_pitch"],
          HOME["left_knee"], HOME["left_ankle"]),
    "R": (0.0, HOME["right_hip_roll"], HOME["right_hip_pitch"],
          HOME["right_knee"], HOME["right_ankle"]),
}

# w > 0 poses (sink the LEFT hip: both knees fold into a small squat, the LEFT
# knee folding far more than the right, which stays relatively straight). The
# w < 0 poses are the exact mirror: swap left/right and negate every angle.
_POSITIVE = {
    "ease": {
        "L": (0.000409, -0.059411, -0.236836, 0.419336, 0.705178),
        "R": (-0.008871, 0.123198, 0.339017, -0.207204, -0.545386),
    },
    "sink": {
        "L": (0.000578, -0.047859, -0.145269, 0.595060, 0.809634),
        "R": (-0.012546, 0.138068, 0.289774, -0.295060, -0.583653),
    },
}

# Per waypoint: (pose family, side sign). sign +1 -> sink left, -1 -> sink right.
_PHASE = (
    ("stand", 0),
    ("ease", 1),
    ("sink", 1),
    ("ease", 1),
    ("stand", 0),
    ("ease", -1),
    ("sink", -1),
    ("ease", -1),
    ("stand", 0),
)


def _mirror(legs: dict[str, tuple[float, ...]]) -> dict[str, tuple[float, ...]]:
    """Sink the RIGHT hip: swap left/right and negate every joint angle."""
    return {
        "L": tuple(-v for v in legs["R"]),
        "R": tuple(-v for v in legs["L"]),
    }


def _pose_at(i: int) -> dict[str, float]:
    """Absolute joint targets for waypoint ``i`` (0..N_SEGMENTS)."""
    family, sign = _PHASE[i]
    if family == "stand":
        legs = _STAND
    else:
        legs = _POSITIVE[family]
        if sign < 0:
            legs = _mirror(legs)
    pose: dict[str, float] = {}
    for side, prefix in (("L", "left_"), ("R", "right_")):
        for name, value in zip(_LEG_ORDER, legs[side]):
            pose[prefix + name] = value
    pose["head_roll"] = HEAD_ROLL_MAX * _W[i]
    return pose


DANCE_POSE_WAYPOINTS = tuple(
    (i / N_SEGMENTS, _pose_at(i)) for i in range(N_SEGMENTS + 1)
)

DANCE_LABELS = (
    "STAND",
    "EASE_L",
    "SINK_L",
    "EASE_L",
    "STAND",
    "EASE_R",
    "SINK_R",
    "EASE_R",
    "STAND",
)


def overrides_by_index(pose: dict[str, float]) -> dict[int, float]:
    """Convert a name-keyed keyframe to the index-keyed form mdp expects."""
    return {JOINT_INDEX[name]: value for name, value in pose.items()}


def dance_pose_waypoints_by_index() -> list[dict]:
    return [
        {"frac": frac, "overrides": overrides_by_index(pose) or None}
        for frac, pose in DANCE_POSE_WAYPOINTS
    ]


def dance_height_waypoints() -> list[dict]:
    return [{"frac": frac, "height": height} for frac, height in DANCE_HEIGHT_WAYPOINTS]
