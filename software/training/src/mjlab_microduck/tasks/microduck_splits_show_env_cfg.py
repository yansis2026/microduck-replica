"""Microduck front-split showcase task (left leg forward), Feetech HD-1910.

Extends the plain split task into a full demo:

    STAND -> TWIST -> PRESS -> SPLIT -> (hold 5 s)
          -> PRESS -> TWIST -> STAND -> (stand still 2 s)

The descent keeps the plain split's physical timing; the return retraces the
same poses, and the episode ends after a stable stand. The whole sequence is
driven by the same deployable phase clock (the ``twist`` command slot,
``[cos(2*pi*p), sin(2*pi*p), 0]``), with period ``SHOW_PERIOD_S`` long enough
that the clock never wraps back to standing inside an episode.

This factory reuses ``make_microduck_splits_env_cfg`` (sensors, terminations,
reset, solver, actuator contract) and only swaps the phase period, the
waypoints, and the hold/return reward windows. See docs/splits-show-task.md.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import mdp
from .microduck_hd1910_env_cfg import MicroduckHD1910RlCfg
from .microduck_splits_env_cfg import (
    FEET_GROUND_PHASE,
    SPLIT_PERIOD_S,
    make_microduck_splits_env_cfg,
)
from .splits_keyframes import (
    LEG_JOINTS,
    NECK_JOINTS,
    SHOW_EPISODE_LENGTH_S,
    SHOW_PERIOD_S,
    SHOW_PHASE_HOLD_END,
    SHOW_PHASE_RETURN_STAND,
    SHOW_PHASE_SPLIT,
    show_height_waypoints,
    show_pose_waypoints_by_index,
)

# The descent timing is unchanged, so rescale the plain split's feet-window
# threshold from its 9 s period to the showcase's longer period.
SHOW_FEET_GROUND_PHASE = FEET_GROUND_PHASE * SPLIT_PERIOD_S / SHOW_PERIOD_S

# Return-to-stand window: from the end of the held split until the stand pose
# is reached again.
SHOW_RISE_START_PHASE = SHOW_PHASE_HOLD_END
SHOW_RISE_END_PHASE = SHOW_PHASE_RETURN_STAND


def make_microduck_splits_show_env_cfg(play: bool = False, rough: bool = False):
    if rough:
        raise ValueError("Front split showcase currently supports flat terrain only")

    # Reuses the split task's full-collision model, contact sensors, reset,
    # solver settings, terminations, and HD-1910 actuator contract.
    cfg = make_microduck_splits_env_cfg(play=play, rough=False)

    cfg.episode_length_s = SHOW_EPISODE_LENGTH_S
    cfg.commands["twist"] = replace(cfg.commands["twist"], period=SHOW_PERIOD_S)

    pose_wps = show_pose_waypoints_by_index()
    height_wps = show_height_waypoints()

    # These three terms share one waypoint list in the split cfg; set each
    # explicitly so the descent, hold, return, and settle are all tracked.
    for name in ("split_pose_legs", "split_pose_neck", "split_pose_l1"):
        cfg.rewards[name].params["waypoints"] = pose_wps
    cfg.rewards["split_height"].params["waypoints"] = height_wps

    # Feet stay down from the descent onward (through the return and stand).
    cfg.rewards["split_feet_grounded"].params["start_phase"] = SHOW_FEET_GROUND_PHASE

    # Stillness only during the 5 s held split...
    cfg.rewards["split_hold_stillness"].params["start_phase"] = SHOW_PHASE_SPLIT
    cfg.rewards["split_hold_stillness"].params["end_phase"] = SHOW_PHASE_HOLD_END

    # ...and again during the final stable stand.
    cfg.rewards["split_stand_stillness"] = RewardTermCfg(
        func=mdp.split_hold_stillness,
        weight=1.5,
        params={
            "command_name": "twist",
            "start_phase": SHOW_PHASE_RETURN_STAND,
        },
    )

    # Damp residual wobble around the upright stand (positive cost -> negative
    # weight). The tilt gate keeps the arrival-to-vertical free.
    cfg.rewards["split_arrival_damper"] = RewardTermCfg(
        func=mdp.body_ang_vel_at_height,
        weight=-0.03,
        params={
            "height_low": 0.10,
            "height_high": 0.113,
            "tilt_full_deg": 15.0,
            "tilt_zero_deg": 40.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Early gradient for the hard part: rising out of the full split. Smooth
    # rise is optimal (rates capped), and it is off during descent/hold.
    cfg.rewards["split_rise_bootstrap"] = RewardTermCfg(
        func=mdp.split_phase_rise_bootstrap,
        weight=0.5,
        params={
            "command_name": "twist",
            "start_phase": SHOW_RISE_START_PHASE,
            "end_phase": SHOW_RISE_END_PHASE,
            "max_height": 0.12,
            "max_vz": 0.08,
        },
    )

    return cfg


MicroduckSplitsShowRlCfg = replace(
    deepcopy(MicroduckHD1910RlCfg),
    experiment_name="microduck_splits_show",
    run_name="hd1910_splits_show_left_forward",
)
