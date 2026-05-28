"""
Manual Override Module for the Traffic Signal System.

Allows a traffic officer (or external code) to bypass the DQN agent
and directly control the signal. Useful for:
  - Accidents
  - Road construction
  - Special events / VIP convoys
  - Testing / debugging
"""

from environment import TrafficEnv


def override_signal(env: TrafficEnv, lane: int, green_duration: int) -> dict:
    """
    Force a specific lane to green for a given duration, bypassing the DQN.

    Parameters
    ----------
    env : TrafficEnv
        The active traffic environment instance.
    lane : int
        Lane index to give the green signal (0 or 1).
    green_duration : int
        Number of simulation steps to keep the override active.

    Returns
    -------
    dict
        Confirmation with override details.

    Raises
    ------
    ValueError
        If the lane index or duration is invalid.
    """
    if lane < 0 or lane >= env.config.num_lanes:
        raise ValueError(
            f"Invalid lane {lane}. Must be 0 to {env.config.num_lanes - 1}."
        )
    if green_duration <= 0:
        raise ValueError("Green duration must be a positive integer.")

    env.set_signal(lane, green_duration)

    return {
        "status": "override_active",
        "lane": lane,
        "duration": green_duration,
        "message": (
            f"Manual override activated: Lane {lane} -> GREEN "
            f"for {green_duration} steps."
        ),
    }


def cancel_override(env: TrafficEnv) -> dict:
    """
    Cancel an active manual override, returning control to the DQN agent.

    Parameters
    ----------
    env : TrafficEnv
        The active traffic environment instance.

    Returns
    -------
    dict
        Confirmation of cancellation.
    """
    was_active = env.override_active
    env.override_active = False
    env._override_remaining = 0
    env._phase_locked = False

    return {
        "status": "override_cancelled" if was_active else "no_override_active",
        "message": (
            "Manual override cancelled. AI agent will resume control."
            if was_active
            else "No override was active."
        ),
    }
