"""Microduck in-place dance task (alternating weight shift), HD-1910.

    STAND -> sink LEFT hip / stretch right leg -> STAND
          -> sink RIGHT hip / stretch left leg -> STAND -> repeat
    (head sways toward the sunk hip)

The motion is a continuous, periodic groove: the robot shifts weight onto one
bent leg while the other stays straight and stretches, alternating left/right
with a centred stand between sides. Unlike the one-way split task, the phase
clock *must* wrap (period << episode length) so the dance keeps cycling for the
whole episode. The same phase clock (the ``twist`` command slot,
``[cos(2*pi*p), sin(2*pi*p), 0]``) drives reward targets, exactly like the
split/show tasks: the runtime button writes that signal and the same waypoints
are interpolated on the robot.

The keyframes themselves live in ``dance_keyframes.py`` as plain data so the
CPU-only pose checker ``scripts/check_split_pose.py --timeline dance`` can
validate them without the training stack. See docs/dance-task.md.

This factory reuses ``make_microduck_splits_env_cfg`` (deep-copied: full
collision model, contact sensors, reset, solver settings, terminations, and the
HD-1910 actuator contract) and only swaps the timeline, command period, and
the feet/hold reward windows.
"""

from copy import deepcopy
from dataclasses import replace

from .microduck_hd1910_env_cfg import MicroduckHD1910RlCfg
from .microduck_splits_env_cfg import make_microduck_splits_env_cfg
from .dance_keyframes import (
    DANCE_EPISODE_LENGTH_S,
    DANCE_PERIOD_S,
    dance_height_waypoints,
    dance_pose_waypoints_by_index,
)


def make_microduck_dance_env_cfg(play: bool = False, rough: bool = False):
    """Create the Microduck in-place dance environment configuration."""
    if rough:
        raise ValueError("Dance currently supports flat terrain only")

    # Reuses the split task's full-collision model, contact sensors, reset,
    # solver settings, terminations, and HD-1910 actuator contract.
    cfg = make_microduck_splits_env_cfg(play=play, rough=False)

    cfg.episode_length_s = DANCE_EPISODE_LENGTH_S
    # Short period so the clock wraps and the bob repeats through the episode.
    cfg.commands["twist"] = replace(
        cfg.commands["twist"], period=DANCE_PERIOD_S, randomize_phase=False
    )

    pose_wps = dance_pose_waypoints_by_index()
    height_wps = dance_height_waypoints()

    # These three terms share one waypoint list in the split cfg; set each
    # explicitly so the whole cyclic dance is tracked.
    for name in ("split_pose_legs", "split_pose_neck", "split_pose_l1"):
        cfg.rewards[name].params["waypoints"] = pose_wps
    cfg.rewards["split_height"].params["waypoints"] = height_wps

    # Both feet stay on the floor for the whole cycle.
    cfg.rewards["split_feet_grounded"].params["start_phase"] = 0.0

    # Continuous rhythm: there is no still-hold segment, so drop the stillness
    # reward (it would fight the bob).
    cfg.rewards.pop("split_hold_stillness", None)

    return cfg


MicroduckDanceRlCfg = replace(
    deepcopy(MicroduckHD1910RlCfg),
    experiment_name="microduck_dance",
    run_name="hd1910_dance_weight_shift",
)
