"""
Traffic Signal Environment for DQN-based control.

Simulates a Sri Lankan main road with two lanes in one direction.
Vehicles arrive randomly (Poisson process) and queue up when the
signal is red. The agent controls which lane gets the green signal
and for how long.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional


@dataclass
class Vehicle:
    """Represents a single vehicle in the simulation."""
    arrival_time: int          # timestep when the vehicle arrived
    departure_time: int = -1   # timestep when it departed (-1 = still waiting)


@dataclass
class LaneConfig:
    """Configuration for a single lane."""
    arrival_rate: float = 3.0        # Poisson λ — avg vehicles arriving per step
    saturation_rate: int = 2         # max vehicles that can depart per green step


@dataclass
class EnvConfig:
    """Full environment configuration."""
    num_lanes: int = 2
    episode_length: int = 500        # steps per episode
    min_green_duration: int = 5      # minimum green phase in steps
    short_green: int = 10            # "short" green duration
    long_green: int = 20             # "long" green duration
    yellow_phase: int = 2            # amber transition steps
    lane_configs: List[LaneConfig] = field(default_factory=lambda: [
        LaneConfig(arrival_rate=3.0, saturation_rate=2),   # Lane 0 — heavier traffic
        LaneConfig(arrival_rate=2.0, saturation_rate=2),   # Lane 1 — lighter traffic
    ])


class TrafficEnv:
    """
    Gym-style traffic signal environment.

    State:  [queue_lane_0, queue_lane_1]  — normalised queue lengths
    Actions:
        0 — Green for Lane 0, short duration
        1 — Green for Lane 0, long duration
        2 — Green for Lane 1, short duration
        3 — Green for Lane 1, long duration
    Reward: −(total waiting vehicles) each step
    """

    # ------------------------------------------------------------------ #
    #  Construction / Reset                                                #
    # ------------------------------------------------------------------ #

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.num_actions = self.config.num_lanes * 2   # 2 durations per lane

        # State that gets reset every episode
        self._queues: List[List[Vehicle]] = []
        self._current_step: int = 0

        # Signal state
        self._green_lane: int = 0              # which lane currently has green
        self._green_remaining: int = 0         # steps left in current green phase
        self._yellow_remaining: int = 0        # steps left in yellow transition
        self._phase_locked: bool = False       # True during yellow / min-green

        # Emergency preemption
        self.emergency_active: bool = False
        self._emergency_lane: int = 0
        self._emergency_remaining: int = 0

        # Manual override
        self.override_active: bool = False
        self._override_lane: int = 0
        self._override_remaining: int = 0

        # Bookkeeping for metrics
        self._total_arrived: int = 0
        self._total_departed: int = 0
        self._step_queue_log: List[List[int]] = []       # per-step queue snapshot
        self._step_departed_log: List[int] = []           # per-step departures
        self._step_arrived_log: List[int] = []            # per-step arrivals
        self._vehicle_wait_times: List[int] = []          # wait time of each departed vehicle

        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Set the random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode. Returns the initial state."""
        self._queues = [[] for _ in range(self.config.num_lanes)]
        self._current_step = 0

        self._green_lane = 0
        self._green_remaining = self.config.short_green
        self._yellow_remaining = 0
        self._phase_locked = True       # locked during initial green

        self.emergency_active = False
        self._emergency_lane = 0
        self._emergency_remaining = 0

        self.override_active = False
        self._override_lane = 0
        self._override_remaining = 0

        self._total_arrived = 0
        self._total_departed = 0
        self._step_queue_log = []
        self._step_departed_log = []
        self._step_arrived_log = []
        self._vehicle_wait_times = []

        return self._get_state()

    # ------------------------------------------------------------------ #
    #  Core simulation step                                                #
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one simulation timestep.

        Parameters
        ----------
        action : int
            Agent's chosen action (0-3). Ignored while phase is locked,
            a manual override is active, or emergency preemption is active.

        Returns
        -------
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict   — extra data (queue lengths, departed, arrived, etc.)
        """
        # 1. Handle arrivals (vehicles arrive regardless of signal)
        step_arrived = self._process_arrivals()

        # 2. Handle signal logic (override → yellow → agent action)
        self._process_signal(action)

        # 3. Handle departures (vehicles leave if their lane is green)
        step_departed = self._process_departures()

        # 4. Record metrics
        queue_snapshot = self.get_queue_lengths()
        self._step_queue_log.append(queue_snapshot.copy())
        self._step_departed_log.append(step_departed)
        self._step_arrived_log.append(step_arrived)

        # 5. Compute reward
        total_waiting = sum(queue_snapshot)
        reward = -float(total_waiting)

        # 6. Advance step
        self._current_step += 1
        done = self._current_step >= self.config.episode_length

        info = {
            "step": self._current_step,
            "queue_lengths": queue_snapshot,
            "departed_this_step": step_departed,
            "arrived_this_step": step_arrived,
            "green_lane": self._green_lane,
            "green_remaining": self._green_remaining,
            "emergency_active": self.emergency_active,
            "emergency_lane": self._emergency_lane if self.emergency_active else None,
            "override_active": self.override_active,
            "total_arrived": self._total_arrived,
            "total_departed": self._total_departed,
        }

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _process_arrivals(self) -> int:
        """Generate random vehicle arrivals for each lane. Returns total arrived."""
        total = 0
        for i, lane_cfg in enumerate(self.config.lane_configs):
            n = self._rng.poisson(lane_cfg.arrival_rate)
            for _ in range(n):
                self._queues[i].append(Vehicle(arrival_time=self._current_step))
            total += n
        self._total_arrived += total
        return total

    def _process_signal(self, action: int):
        """Manage signal transitions: emergency → override → yellow → agent decision."""

        # --- Emergency preemption takes highest precedence ---
        if self.emergency_active:
            self._green_lane = self._emergency_lane
            self._emergency_remaining -= 1
            if self._emergency_remaining <= 0:
                self.emergency_active = False
                self._yellow_remaining = self.config.yellow_phase
                self._phase_locked = True
            return

        # --- Manual override takes precedence ---
        if self.override_active:
            self._green_lane = self._override_lane
            self._override_remaining -= 1
            if self._override_remaining <= 0:
                self.override_active = False
                self._yellow_remaining = self.config.yellow_phase
                self._phase_locked = True
            return

        # --- Yellow transition in progress ---
        if self._yellow_remaining > 0:
            self._yellow_remaining -= 1
            self._phase_locked = self._yellow_remaining > 0
            return

        # --- Current green phase still active ---
        if self._green_remaining > 0:
            self._green_remaining -= 1
            if self._green_remaining > 0:
                return
            # Green just ended — start yellow if agent wants to switch
            # (we'll interpret the action below)

        # --- Accept new agent action ---
        lane, duration = self._decode_action(action)

        if lane != self._green_lane:
            # Switching lanes → insert yellow phase first
            self._yellow_remaining = self.config.yellow_phase
            self._phase_locked = True
            self._green_lane = lane

        self._green_remaining = duration

    def _process_departures(self) -> int:
        """Let vehicles depart from the green lane. Returns total departed."""
        if self._yellow_remaining > 0 or self.override_active or self.emergency_active:
            # During yellow: no departures (simplified safety)
            # During override: departures happen on the override lane
            # During emergency: departures happen on the emergency lane
            if self.emergency_active:
                return self._depart_from_lane(self._emergency_lane)
            if self.override_active:
                return self._depart_from_lane(self._override_lane)
            return 0

        return self._depart_from_lane(self._green_lane)

    def _depart_from_lane(self, lane: int) -> int:
        """Remove up to saturation_rate vehicles from the given lane."""
        sat = self.config.lane_configs[lane].saturation_rate
        departed = 0
        while self._queues[lane] and departed < sat:
            vehicle = self._queues[lane].pop(0)
            vehicle.departure_time = self._current_step
            wait = vehicle.departure_time - vehicle.arrival_time
            self._vehicle_wait_times.append(wait)
            self._total_departed += 1
            departed += 1
        return departed

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """
        Decode action integer into (lane, green_duration).

        0 → Lane 0, short    1 → Lane 0, long
        2 → Lane 1, short    3 → Lane 1, long
        """
        lane = action // 2
        duration = (
            self.config.short_green if action % 2 == 0
            else self.config.long_green
        )
        return lane, duration

    def _get_state(self) -> np.ndarray:
        """Return the current state vector (normalised queue lengths)."""
        max_q = 50.0   # normalisation constant
        queues = [len(q) / max_q for q in self._queues]
        return np.array(queues, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Public queries                                                      #
    # ------------------------------------------------------------------ #

    def get_queue_lengths(self) -> List[int]:
        """Return raw queue lengths for each lane."""
        return [len(q) for q in self._queues]

    def get_episode_stats(self) -> Dict:
        """Return summary statistics for the completed episode."""
        avg_queue = (
            np.mean([sum(q) for q in self._step_queue_log])
            if self._step_queue_log else 0.0
        )
        avg_wait = (
            np.mean(self._vehicle_wait_times)
            if self._vehicle_wait_times else 0.0
        )
        throughput = self._total_departed
        efficiency = (
            self._total_departed / self._total_arrived
            if self._total_arrived > 0 else 0.0
        )
        return {
            "avg_queue_length": float(avg_queue),
            "avg_waiting_time": float(avg_wait),
            "throughput": int(throughput),
            "efficiency": float(efficiency),
            "total_arrived": int(self._total_arrived),
            "total_departed": int(self._total_departed),
        }

    # ------------------------------------------------------------------ #
    #  Manual override interface                                           #
    # ------------------------------------------------------------------ #

    def set_signal(self, lane: int, duration: int):
        """
        Force a signal change (used by manual_override module).

        Parameters
        ----------
        lane : int       Lane index (0 or 1).
        duration : int   How many steps this override lasts.
        """
        assert 0 <= lane < self.config.num_lanes, f"Invalid lane {lane}"
        assert duration > 0, "Duration must be positive"
        if self.emergency_active:
            raise RuntimeError("Manual override unavailable during emergency preemption.")
        self.override_active = True
        self._override_lane = lane
        self._override_remaining = duration
        self._green_lane = lane
        self._green_remaining = 0
        self._yellow_remaining = 0
        self._phase_locked = True

    # ------------------------------------------------------------------ #
    #  Emergency preemption interface                                       #
    # ------------------------------------------------------------------ #

    def set_emergency(self, lane: int, duration: int):
        """
        Force an emergency preemption on a lane (e.g., ambulance/fire).

        Parameters
        ----------
        lane : int       Lane index (0 or 1).
        duration : int   How many steps this emergency lasts.
        """
        assert 0 <= lane < self.config.num_lanes, f"Invalid lane {lane}"
        assert duration > 0, "Duration must be positive"
        self.emergency_active = True
        self._emergency_lane = lane
        self._emergency_remaining = duration
        self.override_active = False
        self._override_remaining = 0
        self._green_lane = lane
        self._green_remaining = 0
        self._yellow_remaining = 0
        self._phase_locked = True

    def cancel_emergency(self):
        """Cancel an active emergency preemption."""
        self.emergency_active = False
        self._emergency_remaining = 0
        self._phase_locked = False
