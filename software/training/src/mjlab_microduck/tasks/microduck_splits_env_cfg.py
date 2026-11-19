"""Microduck front-split task (left leg forward), Feetech HD-1910 model.

The skill follows a human front split:

    STAND -> TWIST (waist/hip twist) -> PRESS (crouch and press down) -> SPLIT
    (both legs extended, feet/shins on the ground), then HOLD.

All four poses live in ``splits_keyframes.py`` as plain data so the CPU-only
pose checker ``scripts/check_split_pose.py`` can validate them without the
training stack. A single deployable phase clock (the ``twist`` command slot,
``[cos(2*pi*p), sin(2*pi*p), 0]``) drives the whole trajectory, exactly like the
ground-pick/sit-stand tasks: the runtime button writes that signal and the same
waypoints are interpolated on the robot.

Kinematic limitation (measured with MuJoCo FK + floor contacts, see
``scripts/check_split_pose.py``): the legs are short relative to the trunk, so
a textbook horizontal split with both feet on the ground is not reachable
without trunk contact. The SPLIT keyframe is the deepest stable compromise
(trunk z ~= 0.063 m, left leg ~89 deg, right leg ~54 deg); shin/knee/instep
contact is allowed, trunk and head contact is not.

Joint layout (14 actuated joints, policy/action order):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

from copy import deepcopy
from dataclasses import fields, replace

from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import get_standup_spec
from . import mdp
from .microduck_hd1910_env_cfg import (
    MicroduckHD1910RlCfg,
    make_microduck_hd1910_velocity_env_cfg,
)
from .splits_keyframes import (
    EPISODE_LENGTH_S,
    LEG_JOINTS,
    NECK_JOINTS,
    PHASE_SPLIT,
    height_waypoints,
    pose_waypoints_by_index,
)

# The phase period must exceed the episode length so the clock never wraps
# back to phase 0 (standing) inside an episode. The target clamps at the SPLIT
# waypoint, so the split is held from phase 0.48 to the end (~0.89 here).
SPLIT_PERIOD_S = 9.0

# The hold window opens when the pose target reaches SPLIT; feet are rewarded
# once the descent is under way (between PRESS and SPLIT).
SPLIT_HOLD_PHASE = PHASE_SPLIT
FEET_GROUND_PHASE = 0.35

RESET_Z = (0.11, 0.12)


def make_microduck_splits_env_cfg(
    play: bool = False,
    rough: bool = False,
):
    """Create the Microduck front-split environment configuration."""
    if rough:
        raise ValueError("Front split currently supports flat terrain only")

    cfg = deepcopy(make_microduck_hd1910_velocity_env_cfg(play=play, rough=False))
    # The walk model has no trunk/head collision geometry, but the split and the
    # press phase bring the trunk close to the floor. Full collisions let the
    # contact sensors see (and reject) any trunk/head ground contact. Joints,
    # masses and HD1910 actuators are identical to the velocity task.
    cfg.scene.entities["robot"].spec_fn = get_standup_spec

    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors += (trunk_ground_cfg, head_ground_cfg)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Rewards: drop walking/velocity-specific terms ────────────────────────
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "head_pose_tracking",
        "body_pose_tracking",
        "head_pose_bias",
    ):
        cfg.rewards.pop(name, None)

    pose_wps = pose_waypoints_by_index()
    height_wps = height_waypoints()

    # ── Rewards: phase-driven split trajectory ───────────────────────────────
    # Legs: eight rad of travel (stand -> split) needs a broad Gaussian; the L1
    # companion keeps a constant gradient at the far end.
    cfg.rewards["split_pose_legs"] = RewardTermCfg(
        func=mdp.split_phase_pose_track,
        weight=8.0,
        params={
            "waypoints": pose_wps,
            "command_name": "twist",
            "std": 0.25,
            "joint_indices": LEG_JOINTS,
        },
    )
    cfg.rewards["split_pose_neck"] = RewardTermCfg(
        func=mdp.split_phase_pose_track,
        weight=1.5,
        params={
            "waypoints": pose_wps,
            "command_name": "twist",
            "std": 0.15,
            "joint_indices": NECK_JOINTS,
        },
    )
    cfg.rewards["split_pose_l1"] = RewardTermCfg(
        func=mdp.split_phase_pose_l1_penalty,
        weight=0.5,
        params={
            "waypoints": pose_wps,
            "command_name": "twist",
            "joint_indices": LEG_JOINTS,
        },
    )

    # Trunk height follows the same clock (0.115 -> 0.114 -> 0.095 -> 0.063 m).
    cfg.rewards["split_height"] = RewardTermCfg(
        func=mdp.split_phase_height_target,
        weight=2.0,
        params={
            "waypoints": height_wps,
            "command_name": "twist",
            "std": 0.02,
        },
    )

    # Keep both feet on the floor once the descent begins, and reward a quiet
    # hold at the bottom (the "split and hold" part of the skill).
    cfg.rewards["split_feet_grounded"] = RewardTermCfg(
        func=mdp.split_phase_feet_grounded,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "start_phase": FEET_GROUND_PHASE,
        },
    )
    cfg.rewards["split_hold_stillness"] = RewardTermCfg(
        func=mdp.split_hold_stillness,
        weight=1.0,
        params={
            "command_name": "twist",
            "start_phase": SPLIT_HOLD_PHASE,
        },
    )

    # Upright trunk and low angular momentum keep the low pose from tipping.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = 0.5
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01

    # Slow, smooth descent and hold. Regularisation is deliberately light so it
    # cannot block the large (~1.5 rad) hip travel during discovery.
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=mdp.joint_torques_l2, weight=-2e-3
    )

    # Trunk/head ground contact is the one failure the user does not allow.
    cfg.rewards["trunk_impact_penalty"] = RewardTermCfg(
        func=mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": trunk_ground_cfg.name, "threshold": 0.5},
    )
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": head_ground_cfg.name, "threshold": 0.5},
    )

    # ── Command: single deployable phase clock in the twist slot ─────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.debug_vis = False
    # The velocity task uses VelocityCommandCommandOnlyCfg, which carries an
    # extra rel_turn_in_place_envs field. Filter to the target dataclass so the
    # phase-command cfg can be built from the base velocity settings.
    phase_fields = {f.name for f in fields(mdp.GroundPickPhaseCommandCfg)}
    phase_kwargs = {
        key: value for key, value in vars(command).items() if key in phase_fields
    }
    phase_kwargs.update(
        class_type=mdp.GroundPickPhaseCommand,
        period=SPLIT_PERIOD_S,
        randomize_phase=False,
    )
    cfg.commands["twist"] = mdp.GroundPickPhaseCommandCfg(**phase_kwargs)

    # Head/body pose commands are not part of this skill; zero their ranges but
    # keep the slots so the observation/action contract matches every other
    # microduck policy (61D obs, 14 actions).
    for name, width in (("head_pose", 4), ("body_pose", 6)):
        if name in cfg.commands:
            cfg.commands[name].ranges = ((-1e-5, 1e-5),) * width
            cfg.commands[name].zero_command_prob = 0.25

    # ── Terminations ─────────────────────────────────────────────────────────
    cfg.terminations["nan_state"].params["sensor_names"] = (
        "feet_ground_contact",
        trunk_ground_cfg.name,
        head_ground_cfg.name,
    )
    # Legs, knees and insteps may rest on the floor; the trunk and head may not.
    for name, sensor_cfg in (
        ("trunk_ground_contact", trunk_ground_cfg),
        ("head_ground_contact", head_ground_cfg),
    ):
        cfg.terminations[name] = TerminationTermCfg(
            func=mdp.sensor_contact_termination,
            time_out=False,
            params={"sensor_name": sensor_cfg.name},
        )

    # ── Reset ────────────────────────────────────────────────────────────────
    cfg.events["reset_base"].params["pose_range"]["z"] = RESET_Z
    if "reset_robot_joints" in cfg.events:
        cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)

    # The low split + full collision model packs contacts close together; match
    # the sit-stand solver settings to avoid contact-solver NaN resets.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # No pushes and no velocity curriculum: this is a controlled stationary skill.
    cfg.events.pop("push_robot", None)
    cfg.curriculum.clear()

    return cfg


MicroduckSplitsRlCfg = replace(
    deepcopy(MicroduckHD1910RlCfg),
    experiment_name="microduck_splits",
    run_name="hd1910_front_split_left_forward",
)
