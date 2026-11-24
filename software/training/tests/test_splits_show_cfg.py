"""SplitsShow task registration, timeline, and reward-window wiring."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import splits_keyframes as kf
from mjlab_microduck.tasks.microduck_splits_env_cfg import (
    FEET_GROUND_PHASE,
    SPLIT_PERIOD_S,
    make_microduck_splits_env_cfg,
)
from mjlab_microduck.tasks.microduck_splits_show_env_cfg import (
    SHOW_FEET_GROUND_PHASE,
    make_microduck_splits_show_env_cfg,
)

TASK = "Mjlab-SplitsShow-Flat-MicroDuck-HD1910"
BASE = "Mjlab-Velocity-Flat-MicroDuck-HD1910"


def test_task_is_registered_with_hd1910_contract():
    assert TASK in list_tasks()
    cfg = load_env_cfg(TASK)
    base = load_env_cfg(BASE)
    assert cfg.scene.entities["robot"].articulation.actuators == (
        base.scene.entities["robot"].articulation.actuators
    )
    # Same full-collision model as the plain split task (trunk/head sensing).
    assert cfg.scene.entities["robot"].spec_fn.__name__ == "get_standup_spec"
    assert list(cfg.observations["actor"].terms) == list(
        base.observations["actor"].terms
    )
    assert cfg.actions == base.actions
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    assert cfg.episode_length_s == kf.SHOW_EPISODE_LENGTH_S
    assert load_rl_cfg(TASK).experiment_name == "microduck_splits_show"


def test_show_timeline_is_strictly_increasing_and_derived_from_seconds():
    fracs = [frac for frac, _ in kf.SHOW_POSE_WAYPOINTS]
    assert len(fracs) == len(kf.SHOW_LABELS) == 9
    assert all(b > a for a, b in zip(fracs, fracs[1:]))
    expected = [
        kf.SHOW_T_STAND_S,
        kf.SHOW_T_TWIST_S,
        kf.SHOW_T_PRESS_S,
        kf.SHOW_T_SPLIT_S,
        kf.SHOW_T_HOLD_END_S,
        kf.SHOW_T_RETURN_PRESS_S,
        kf.SHOW_T_RETURN_TWIST_S,
        kf.SHOW_T_RETURN_STAND_S,
        kf.SHOW_T_SETTLE_END_S,
    ]
    for frac, seconds in zip(fracs, expected):
        assert frac == pytest.approx(seconds / kf.SHOW_PERIOD_S)
    # The clock must not wrap back to standing before the episode ends.
    assert kf.SHOW_PERIOD_S > kf.SHOW_EPISODE_LENGTH_S
    assert fracs[-1] < 1.0


def test_show_reuses_descent_poses_and_retraces_them():
    poses = [pose for _, pose in kf.SHOW_POSE_WAYPOINTS]
    # Descent uses the exact verified poses...
    assert poses[0] is kf.STAND
    assert poses[1] is kf.TWIST
    assert poses[2] is kf.PRESS
    assert poses[3] is kf.SPLIT
    # ...the held split repeats SPLIT, the return retraces PRESS/TWIST, and the
    # final stand repeats STAND for the settle.
    assert poses[4] is kf.SPLIT
    assert poses[5] is kf.PRESS
    assert poses[6] is kf.TWIST
    assert poses[7] is kf.STAND
    assert poses[8] is kf.STAND
    # Height waypoints mirror the pose sequence.
    heights = [height for _, height in kf.SHOW_HEIGHT_WAYPOINTS]
    assert heights == [0.115, 0.114, 0.095, 0.063, 0.063, 0.095, 0.114, 0.115, 0.115]


def test_rewards_track_the_show_waypoints_and_windows():
    cfg = make_microduck_splits_show_env_cfg()
    pose_wps = kf.show_pose_waypoints_by_index()
    height_wps = kf.show_height_waypoints()

    for name in ("split_pose_legs", "split_pose_neck", "split_pose_l1"):
        assert cfg.rewards[name].params["waypoints"] == pose_wps
    assert cfg.rewards["split_pose_legs"].params["joint_indices"] == kf.LEG_JOINTS
    assert cfg.rewards["split_pose_neck"].params["joint_indices"] == kf.NECK_JOINTS
    assert cfg.rewards["split_height"].params["waypoints"] == height_wps

    # Feet stay grounded from the descent onward (also through the return).
    assert (
        cfg.rewards["split_feet_grounded"].params["start_phase"]
        == pytest.approx(SHOW_FEET_GROUND_PHASE)
    )
    assert SHOW_FEET_GROUND_PHASE == pytest.approx(
        FEET_GROUND_PHASE * SPLIT_PERIOD_S / kf.SHOW_PERIOD_S
    )

    # Stillness is windowed over the held split...
    hold = cfg.rewards["split_hold_stillness"]
    assert hold.func is mdp.split_hold_stillness
    assert hold.params["start_phase"] == pytest.approx(kf.SHOW_PHASE_SPLIT)
    assert hold.params["end_phase"] == pytest.approx(kf.SHOW_PHASE_HOLD_END)

    # ...and again over the final stable stand.
    stand = cfg.rewards["split_stand_stillness"]
    assert stand.func is mdp.split_hold_stillness
    assert stand.params["start_phase"] == pytest.approx(kf.SHOW_PHASE_RETURN_STAND)
    assert "end_phase" not in stand.params


def test_rise_bootstrap_and_arrival_damper_are_wired():
    cfg = make_microduck_splits_show_env_cfg()

    rise = cfg.rewards["split_rise_bootstrap"]
    assert rise.func is mdp.split_phase_rise_bootstrap
    assert rise.weight > 0.0
    assert rise.params["start_phase"] == pytest.approx(kf.SHOW_PHASE_HOLD_END)
    assert rise.params["end_phase"] == pytest.approx(kf.SHOW_PHASE_RETURN_STAND)

    # body_ang_vel_at_height returns a positive cost, so the weight is negative.
    damper = cfg.rewards["split_arrival_damper"]
    assert damper.func is mdp.body_ang_vel_at_height
    assert damper.weight < 0.0
    assert damper.params["asset_cfg"].body_names == ("trunk_base",)


def test_command_is_the_longer_non_randomized_phase_clock():
    cfg = make_microduck_splits_show_env_cfg()
    command = cfg.commands["twist"]
    assert command.class_type is mdp.GroundPickPhaseCommand
    assert command.period == kf.SHOW_PERIOD_S == 17.0
    assert command.randomize_phase is False
    assert command.period > cfg.episode_length_s


def test_end_phase_defaults_none_and_factory_isolation():
    # Default keeps the original open-ended gate for the plain split task.
    import inspect

    default = inspect.signature(mdp.split_hold_stillness).parameters["end_phase"]
    assert default.default is None

    # Building the show config must not mutate the plain split config.
    make_microduck_splits_show_env_cfg()
    plain = make_microduck_splits_env_cfg()
    assert plain.episode_length_s == kf.EPISODE_LENGTH_S
    assert plain.commands["twist"].period == SPLIT_PERIOD_S
    assert plain.rewards["split_hold_stillness"].params["start_phase"] == kf.PHASE_SPLIT
    assert "end_phase" not in plain.rewards["split_hold_stillness"].params


def test_rough_variant_is_rejected_and_play_builds():
    with pytest.raises(ValueError, match="flat terrain"):
        make_microduck_splits_show_env_cfg(rough=True)
    assert "split_pose_legs" in make_microduck_splits_show_env_cfg(play=True).rewards


def _pose_target_at(env, asset_cfg, waypoints, phase):
    progress = torch.full((2,), phase)
    return mdp._multistage_target_pose(env, asset_cfg, waypoints, progress=progress)


def test_multistage_interpolation_hits_each_show_keyframe():
    class Asset:
        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=torch.zeros(2, 14),
                joint_pos=torch.zeros(2, 14),
            )

        def find_joints(self, pattern):
            return list(range(14)), list(kf.JOINT_ORDER)

    env = SimpleNamespace(
        scene={"robot": Asset()},
        episode_length_buf=torch.zeros(2),
        max_episode_length=800,
    )
    asset_cfg = SceneEntityCfg("robot")
    waypoints = kf.show_pose_waypoints_by_index()

    assert torch.allclose(_pose_target_at(env, asset_cfg, waypoints, 0.0), torch.zeros(2, 14))
    for frac, pose in kf.SHOW_POSE_WAYPOINTS:
        expected = torch.zeros(14)
        for idx, value in kf.overrides_by_index(pose).items():
            expected[idx] = value
        assert torch.allclose(
            _pose_target_at(env, asset_cfg, waypoints, frac), expected.expand(2, 14)
        )

    # After the last waypoint the target stays at the final STAND pose.
    assert torch.allclose(_pose_target_at(env, asset_cfg, waypoints, 0.99), torch.zeros(2, 14))
