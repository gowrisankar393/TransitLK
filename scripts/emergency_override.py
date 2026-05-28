"""
Emergency Override Module for the Traffic Signal System.

Allows external code to preempt the signal for an emergency vehicle
and clear a lane immediately.
"""

from environment import TrafficEnv


def trigger_emergency(env: TrafficEnv, lane: int, green_duration: int) -> dict:
    """
    Activate emergency preemption for a given lane and duration.

    Parameters
    ----------
    env : TrafficEnv
        The active traffic environment instance.
    lane : int
        Lane index to give the green signal (0 or 1).
    green_duration : int
        Number of simulation steps to keep the emergency active.

    Returns
    -------
    dict
        Confirmation with emergency details.

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

    env.set_emergency(lane, green_duration)

    return {
        "status": "emergency_active",
        "lane": lane,
        "duration": green_duration,
        "message": (
            f"Emergency preemption activated: Lane {lane} -> GREEN "
            f"for {green_duration} steps."
        ),
    }


def cancel_emergency(env: TrafficEnv) -> dict:
    """
    Cancel an active emergency preemption, returning control to the agent.

    Parameters
    ----------
    env : TrafficEnv
        The active traffic environment instance.

    Returns
    -------
    dict
        Confirmation of cancellation.
    """
    was_active = env.emergency_active
    env.cancel_emergency()

    return {
        "status": "emergency_cancelled" if was_active else "no_emergency_active",
        "message": (
            "Emergency preemption cancelled. AI agent will resume control."
            if was_active
            else "No emergency was active."
        ),
    }
