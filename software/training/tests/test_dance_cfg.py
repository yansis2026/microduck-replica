"""In-place dance task registration, reward wiring, and the cyclic keyframes."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import dance_keyframes as kf
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    make_microduck_dance_env_cfg,
)
from mjlab_microduck.tasks.microduck_splits_env_cfg import (
    make_microduck_splits_env_cfg,
)

TASK = "Mjlab-Dance-Flat-MicroDuck-HD1910"
BASE = "Mjlab-Velocity-Flat-MicroDuck-HD1910"

# Indices the dance overrides every waypoint: all five joints of both legs
# (hip_yaw, hip_roll, hip_pitch, knee, ankle) plus head_roll (the head sway).
# The leg joints are pre-solved by IK so the soles stay flat and planted while
# the pelvis rolls; neck pitch/yaw stay at HOME.
#   0,1,2,3,4        = left  hip_yaw/roll/pitch/knee/ankle
#   8                = head_roll
#   9,10,11,12,13    = right hip_yaw/roll/pitch/knee/ankle
DANCE_OVERRIDE_INDICES = {0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13}

LEG_JOINT_NAMES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


def _pose(i: int) -> dict:
    return kf.DANCE_POSE_WAYPOINTS[i][1]


def test_task_is_registered_with_hd1910_contract():
    assert TASK in list_tasks()
    cfg = load_env_cfg(TASK)
    base = load_env_cfg(BASE)
    assert cfg.scene.entities["robot"].articulation.actuators == (
        base.scene.entities["robot"].articulation.actuators
    )
    # Full-collision model keeps the trunk/head contact sensing from the split.
    assert cfg.scene.entities["robot"].spec_fn.__name__ == "get_standup_spec"
    assert list(cfg.observations["actor"].terms) == list(
        base.observations["actor"].terms
    )
    assert cfg.actions == base.actions
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    assert cfg.episode_length_s == kf.DANCE_EPISODE_LENGTH_S
    assert load_rl_cfg(TASK).experiment_name == "microduck_dance"


def test_rewards_track_the_cyclic_dance():
    cfg = make_microduck_dance_env_cfg()
    pose_wps = kf.dance_pose_waypoints_by_index()
    height_wps = kf.dance_height_waypoints()

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
    assert cfg.rewards["split_feet_grounded"].params["start_phase"] == 0.0
    # No stillness segment: the groove never holds still.
    assert "split_hold_stillness" not in cfg.rewards

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


def test_command_is_a_cyclic_phase_clock():
    cfg = make_microduck_dance_env_cfg()
    command = cfg.commands["twist"]
    assert command.class_type is mdp.GroundPickPhaseCommand
    assert command.period == kf.DANCE_PERIOD_S
    assert command.randomize_phase is False
    # Unlike the split/show, the dance *depends* on the clock wrapping: the
    # period is shorter than the episode so the groove repeats.
    assert kf.DANCE_PERIOD_S < cfg.episode_length_s
    for name, width in (("head_pose", 4), ("body_pose", 6)):
        assert cfg.commands[name].ranges == ((-1e-5, 1e-5),) * width


def test_timeline_is_a_closed_cyclic_cycle():
    assert len(kf.DANCE_POSE_WAYPOINTS) == kf.N_SEGMENTS + 1 == 9
    fracs = [frac for frac, _ in kf.DANCE_POSE_WAYPOINTS]
    assert fracs == [i / kf.N_SEGMENTS for i in range(9)]
    assert all(a < b for a, b in zip(fracs, fracs[1:]))

    # First and last waypoint are the same pose, so the cycle wraps seamlessly.
    first = kf.DANCE_POSE_WAYPOINTS[0][1]
    last = kf.DANCE_POSE_WAYPOINTS[-1][1]
    assert set(first) == set(last)
    for name in first:
        assert first[name] == pytest.approx(last[name], abs=1e-9)


def test_sink_left_and_sink_right_are_exact_mirrors():
    # Sink left (i=2, w=+1) is the left/right swap + sign flip of sink right
    # (i=6, w=-1). Same for the two ease waypoints on either side.
    for left_i, right_i in ((2, 6), (1, 5), (3, 7)):
        sink_l = _pose(left_i)
        sink_r = _pose(right_i)
        for side, mirror_side in (("left", "right"), ("right", "left")):
            for joint in LEG_JOINT_NAMES:
                assert sink_l[f"{side}_{joint}"] == pytest.approx(
                    -sink_r[f"{mirror_side}_{joint}"]
                )
        assert sink_l["head_roll"] == pytest.approx(-sink_r["head_roll"])


def test_stand_waypoints_are_the_mirrored_home_pose():
    for i in (0, 4, 8):
        pose = _pose(i)
        for joint in LEG_JOINT_NAMES:
            assert pose[f"left_{joint}"] == pytest.approx(-pose[f"right_{joint}"])
        assert pose["head_roll"] == pytest.approx(0.0, abs=1e-9)
        # At the stand, the leg is exactly the HOME pose.
        for name, value in kf.HOME.items():
            assert pose[name] == pytest.approx(value, abs=1e-9)


def test_head_sways_with_the_weight_shift():
    assert _pose(0)["head_roll"] == pytest.approx(0.0, abs=1e-9)
    assert _pose(4)["head_roll"] == pytest.approx(0.0, abs=1e-9)
    assert _pose(2)["head_roll"] == pytest.approx(+kf.HEAD_ROLL_MAX)
    assert _pose(6)["head_roll"] == pytest.approx(-kf.HEAD_ROLL_MAX)


def test_base_roll_follows_the_groove():
    rolls = [r for _, r in kf.DANCE_BASE_ROLL_WAYPOINTS]
    assert len(rolls) == len(kf.DANCE_POSE_WAYPOINTS) == 9
    assert rolls[0] == pytest.approx(0.0, abs=1e-9)
    assert rolls[4] == pytest.approx(0.0, abs=1e-9)
    # Negative rolls sink the left hip, positive the right.
    assert rolls[2] == pytest.approx(-kf.BASE_ROLL_MAX)
    assert rolls[6] == pytest.approx(+kf.BASE_ROLL_MAX)
    for i in range(9):
        assert rolls[i] == pytest.approx(-rolls[8 - i], abs=1e-9)


def test_base_roll_is_a_free_base_target_not_a_joint():
    # hip_roll IS commanded (the IK uses it to keep the soles flat); the pelvis
    # roll is a separate free-base target that only the CPU check/preview apply
    # and that emerges from physics at training time.
    for _, pose in kf.DANCE_POSE_WAYPOINTS:
        assert "left_hip_roll" in pose
        assert "right_hip_roll" in pose


def test_hip_yaw_stays_within_the_ik_bound():
    for _, pose in kf.DANCE_POSE_WAYPOINTS:
        for side in ("left", "right"):
            assert abs(pose[f"{side}_hip_yaw"]) <= kf.YAW_BOUND + 1e-9


def test_pelvis_dips_at_each_sink_and_recovers_between():
    # Both knees fold into a squat at each sink, so the pelvis bobs down ~1 cm
    # and returns to standing height at the centred stands. The bob is symmetric
    # left/right and still contained (not a collapse).
    assert len(kf.DANCE_HEIGHT_WAYPOINTS) == len(kf.DANCE_POSE_WAYPOINTS)
    assert [f for f, _ in kf.DANCE_HEIGHT_WAYPOINTS] == [
        f for f, _ in kf.DANCE_POSE_WAYPOINTS
    ]
    heights = [h for _, h in kf.DANCE_HEIGHT_WAYPOINTS]
    stand, ease, sink = heights[0], heights[1], heights[2]
    assert heights[0] == heights[4] == heights[8] == pytest.approx(stand, abs=1e-9)
    assert heights[2] == heights[6] == pytest.approx(sink, abs=1e-9)
    assert heights[1] == heights[3] == heights[5] == heights[7] == pytest.approx(
        ease, abs=1e-9
    )
    # Standing is highest, the sink is lowest, and the bob is a clearly visible
    # (but still contained) energetic up-down of a few mm to ~1.5 cm.
    assert sink < ease < stand
    assert 0.10 < stand
    assert 0.006 < stand - sink < 0.02


def test_waypoint_overrides_cover_legs_and_head_roll():
    for wp in kf.dance_pose_waypoints_by_index():
        assert set(wp["overrides"]) == DANCE_OVERRIDE_INDICES


def test_factory_does_not_mutate_the_split_task():
    make_microduck_dance_env_cfg()
    splits = make_microduck_splits_env_cfg()
    assert splits.commands["twist"].period == pytest.approx(9.0)
    assert splits.episode_length_s == pytest.approx(8.0)
    assert "split_hold_stillness" in splits.rewards
    assert splits.rewards["split_feet_grounded"].params["start_phase"] == pytest.approx(0.35)


def test_rough_variant_is_rejected_and_play_builds():
    with pytest.raises(ValueError, match="flat terrain"):
        make_microduck_dance_env_cfg(rough=True)
    assert "split_pose_legs" in make_microduck_dance_env_cfg(play=True).rewards


def test_interpolation_hits_each_waypoint():
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
    waypoints = kf.dance_pose_waypoints_by_index()

    def target_at(phase):
        progress = torch.full((2,), phase)
        return mdp._multistage_target_pose(env, asset_cfg, waypoints, progress=progress)

    for frac, pose in kf.DANCE_POSE_WAYPOINTS:
        expected = torch.zeros(14)
        for idx, value in kf.overrides_by_index(pose).items():
            expected[idx] = value
        assert torch.allclose(target_at(frac), expected.expand(2, 14))
