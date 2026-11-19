"""Front-split task registration, reward wiring, and keyframe interpolation."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import splits_keyframes as kf
from mjlab_microduck.tasks.microduck_splits_env_cfg import (
    SPLIT_PERIOD_S,
    make_microduck_splits_env_cfg,
)

TASK = "Mjlab-Splits-Flat-MicroDuck-HD1910"
BASE = "Mjlab-Velocity-Flat-MicroDuck-HD1910"


def test_task_is_registered_with_hd1910_contract():
    assert TASK in list_tasks()
    cfg = load_env_cfg(TASK)
    base = load_env_cfg(BASE)
    assert cfg.scene.entities["robot"].articulation.actuators == (
        base.scene.entities["robot"].articulation.actuators
    )
    # Full-collision model is required to sense trunk/head ground contact.
    assert cfg.scene.entities["robot"].spec_fn.__name__ == "get_standup_spec"
    assert list(cfg.observations["actor"].terms) == list(
        base.observations["actor"].terms
    )
    assert cfg.actions == base.actions
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    assert cfg.episode_length_s == kf.EPISODE_LENGTH_S
    assert load_rl_cfg(TASK).experiment_name == "microduck_splits"


def test_rewards_track_the_phase_trajectory():
    cfg = make_microduck_splits_env_cfg()
    pose_wps = kf.pose_waypoints_by_index()
    height_wps = kf.height_waypoints()

    legs = cfg.rewards["split_pose_legs"]
    assert legs.func is mdp.split_phase_pose_track
    assert legs.weight == 8.0
    assert legs.params["waypoints"] == pose_wps
    assert legs.params["joint_indices"] == kf.LEG_JOINTS
    assert cfg.rewards["split_pose_neck"].params["joint_indices"] == kf.NECK_JOINTS
    assert cfg.rewards["split_pose_l1"].func is mdp.split_phase_pose_l1_penalty

    height = cfg.rewards["split_height"]
    assert height.func is mdp.split_phase_height_target
    assert height.params["waypoints"] == height_wps
    assert cfg.rewards["split_feet_grounded"].func is mdp.split_phase_feet_grounded
    assert cfg.rewards["split_hold_stillness"].func is mdp.split_hold_stillness
    assert cfg.rewards["split_hold_stillness"].params["start_phase"] == kf.PHASE_SPLIT

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
        assert name not in cfg.rewards


def test_command_is_a_non_randomized_phase_clock():
    cfg = make_microduck_splits_env_cfg()
    command = cfg.commands["twist"]
    assert command.class_type is mdp.GroundPickPhaseCommand
    assert command.period == SPLIT_PERIOD_S
    assert command.randomize_phase is False
    # The clock must not wrap back to standing inside an episode.
    assert SPLIT_PERIOD_S > cfg.episode_length_s
    # Head/body slots stay in the observation contract but are inert.
    for name, width in (("head_pose", 4), ("body_pose", 6)):
        assert cfg.commands[name].ranges == ((-1e-5, 1e-5),) * width


def test_trunk_and_head_contact_are_sensed_and_rejected():
    cfg = make_microduck_splits_env_cfg()
    sensors = {sensor.name for sensor in cfg.scene.sensors}
    assert {"feet_ground_contact", "trunk_ground_contact", "head_ground_contact"} <= sensors
    for name in ("trunk_ground_contact", "head_ground_contact"):
        assert cfg.terminations[name].func is mdp.sensor_contact_termination
        assert cfg.terminations[name].params["sensor_name"] == name
    assert "trunk_impact_penalty" in cfg.rewards
    assert "head_impact_penalty" in cfg.rewards
    assert "push_robot" not in cfg.events
    assert not cfg.curriculum


def test_rough_variant_is_rejected_and_play_builds():
    with pytest.raises(ValueError, match="flat terrain"):
        make_microduck_splits_env_cfg(rough=True)
    assert "split_pose_legs" in make_microduck_splits_env_cfg(play=True).rewards


def test_factory_does_not_mutate_the_velocity_task():
    make_microduck_splits_env_cfg()
    base = load_env_cfg(BASE)
    assert base.scene.entities["robot"].spec_fn.__name__ == "get_walk_spec"


def test_keyframe_layout_partitions_the_14_servo_joints():
    assert len(kf.JOINT_ORDER) == 14
    assert sorted(kf.LEG_JOINTS + kf.NECK_JOINTS) == list(range(14))
    assert not set(kf.LEG_JOINTS) & set(kf.NECK_JOINTS)
    for pose in (kf.TWIST, kf.PRESS, kf.SPLIT):
        assert set(pose) <= set(kf.JOINT_ORDER)
    assert kf.POSE_WAYPOINTS[0] == (kf.PHASE_STAND, kf.STAND)
    assert [frac for frac, _ in kf.POSE_WAYPOINTS] == [
        kf.PHASE_STAND,
        kf.PHASE_TWIST,
        kf.PHASE_PRESS,
        kf.PHASE_SPLIT,
    ]
    # Left leg forward, right leg backward in the final split.
    assert kf.SPLIT["left_hip_pitch"] < 0 and kf.SPLIT["right_hip_pitch"] < 0
    assert abs(kf.SPLIT["left_hip_pitch"]) > abs(kf.SPLIT["right_hip_pitch"])


def test_multistage_interpolation_hits_each_keyframe():
    class Asset:
        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=torch.zeros(2, 14),
                joint_pos=torch.zeros(2, 14),
            )

        def find_joints(self, pattern):
            return list(range(14)), list(kf.JOINT_ORDER)

    robot = Asset()
    env = SimpleNamespace(
        scene={"robot": robot},
        episode_length_buf=torch.zeros(2),
        max_episode_length=400,
    )
    asset_cfg = SceneEntityCfg("robot")
    waypoints = kf.pose_waypoints_by_index()

    def target_at(phase):
        progress = torch.full((2,), phase)
        return mdp._multistage_target_pose(env, asset_cfg, waypoints, progress=progress)

    assert torch.allclose(target_at(0.0), torch.zeros(2, 14))
    for frac, pose in kf.POSE_WAYPOINTS[1:]:
        expected = torch.zeros(14)
        for idx, value in kf.overrides_by_index(pose).items():
            expected[idx] = value
        assert torch.allclose(target_at(frac), expected.expand(2, 14))

    # After the last waypoint the target clamps: the split is held.
    final = torch.zeros(14)
    for idx, value in kf.overrides_by_index(kf.SPLIT).items():
        final[idx] = value
    assert torch.allclose(target_at(0.95), final.expand(2, 14))
