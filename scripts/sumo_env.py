"""
SUMO Environment for DQN-based traffic control.

This class bridges the Python DQN agent with the SUMO simulation
using the TraCI API.
"""

import os
import sys
import numpy as np
from typing import Tuple, Dict, List, Optional

# Ensure SUMO_HOME is set
# Try to detect SUMO_HOME if missing
if "SUMO_HOME" not in os.environ:
    # Standard installation paths for SUMO on Windows
    standard_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for path in standard_paths:
        if os.path.exists(path):
            os.environ["SUMO_HOME"] = path
            break

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if os.path.exists(tools):
        sys.path.append(tools)
else:
    # We'll try to import anyway, but warn the user if it fails
    pass

try:
    import traci
    import sumolib
except ImportError:
    traci = None
    sumolib = None


class SumoTrafficEnv:
    """
    Gym-style environment for SUMO-based traffic control.
    
    State: [queue_north, queue_south, queue_east, queue_west] (normalized)
    Actions:
        0 - Green for North-South
        1 - Green for East-West
    """

    def __init__(
        self,
        config_file: str = "sumo_configs/crossroad.sumocfg",
        use_gui: bool = False,
        episode_length: int = 500,
        yellow_duration: int = 3,
        step_duration: int = 5,  # simulation steps per agent action
        sumo_seed: Optional[int] = None,
        gui_delay_ms: int = 0,
    ):
        self.config_file = config_file
        self.use_gui = use_gui
        self.episode_length = episode_length
        self.yellow_duration = yellow_duration
        self.step_duration = step_duration
        self.sumo_seed = sumo_seed
        self.gui_delay_ms = max(0, int(gui_delay_ms))
        
        # State tracking
        self.current_step = 0
        self.total_departed = 0
        self.total_arrived = 0
        self._queue_history: List[List[int]] = []
        self._total_waiting_time = 0.0
        self._waiting_time_steps = 0

        # Emergency preemption (axis: 0 = N-S, 1 = E-W)
        self.emergency_active = False
        self._emergency_axis = 0
        self._emergency_remaining = 0
        
        # SUMO IDs
        self.tl_id = "J1"
        self.incoming_lanes = ["E0_0", "-E1_0", "E2_0", "-E3_0"]
        
        # Initial check
        if traci is None:
            raise ImportError("TraCI not found. Please follow the instructions in the implementation plan.")

    def reset(self) -> np.ndarray:
        """Starts or resets the SUMO simulation."""
        # Close existing connection if any
        try:
            traci.close()
        except Exception:
            pass

        # Start SUMO
        sumo_binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")
        cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
        ]
        if self.sumo_seed is not None:
            cmd += ["--seed", str(self.sumo_seed)]
        # In GUI mode, start simulation immediately (don't wait for play button)
        if self.use_gui:
            cmd += ["--start", "true", "--quit-on-end", "true"]
            if self.gui_delay_ms > 0:
                cmd += ["--delay", str(self.gui_delay_ms)]
        
        traci.start(cmd)
        
        self.current_step = 0
        self.total_departed = 0
        self.total_arrived = 0
        self._queue_history = []
        self._total_waiting_time = 0.0
        self._waiting_time_steps = 0
        self.emergency_active = False
        self._emergency_axis = 0
        self._emergency_remaining = 0
        
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes one action period in SUMO.
        
        Action 0 -> Green N-S (Phase 0)
        Action 1 -> Green E-W (Phase 2)
        """
        departed_this_step = 0
        arrived_this_step = 0

        # 1. Map higher-level action to SUMO phase (emergency overrides agent)
        if self.emergency_active:
            target_phase = 0 if self._emergency_axis == 0 else 2
        else:
            target_phase = 0 if action == 0 else 2
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        
        # 2. Handle transitions (Yellow) - only if switching between green phases
        current_green = current_phase if current_phase in (0, 2) else (0 if current_phase == 1 else 2)
        if target_phase != current_green:
            # Insert yellow before switching axis
            yellow_phase = 1 if current_green == 0 else 3
            traci.trafficlight.setPhase(self.tl_id, yellow_phase)
            
            # Step for yellow duration
            for _ in range(self.yellow_duration):
                traci.simulationStep()
                departed, arrived = self._update_metrics()
                departed_this_step += departed
                arrived_this_step += arrived
        
        # 3. Set target phase and advance simulation
        traci.trafficlight.setPhase(self.tl_id, target_phase)
        for _ in range(self.step_duration):
            traci.simulationStep()
            departed, arrived = self._update_metrics()
            departed_this_step += departed
            arrived_this_step += arrived
            self.current_step += 1
            
        # 4. Get results
        next_state = self._get_state()
        queue_lengths = self._get_raw_queues()
        self._queue_history.append(queue_lengths)
        reward = -sum(queue_lengths)  # penalty for wait times

        if self.emergency_active:
            self._emergency_remaining -= 1
            if self._emergency_remaining <= 0:
                self.emergency_active = False
        
        done = self.current_step >= self.episode_length
        info = {
            "step": self.current_step,
            "queue_lengths": queue_lengths,
            "departed_this_step": departed_this_step,
            "arrived_this_step": arrived_this_step,
            "total_arrived": self.total_arrived,
            "total_departed": self.total_departed,
            "phase": target_phase,
            "emergency_active": self.emergency_active,
            "emergency_axis": self._emergency_axis if self.emergency_active else None,
        }
        
        if done:
            traci.close()
            
        return next_state, float(reward), done, info

    def _get_state(self) -> np.ndarray:
        """Returns normalized queue lengths for all incoming lanes."""
        max_q = 20.0  # normalization factor
        queues = self._get_raw_queues()
        normalized = [min(1.0, q / max_q) for q in queues]
        return np.array(normalized, dtype=np.float32)

    def _get_raw_queues(self) -> List[int]:
        """Gets number of waiting vehicles on each incoming lane."""
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes]

    def _update_metrics(self) -> Tuple[int, int]:
        """Internal bookkeeping — track departed/arrived vehicles and waiting times."""
        departed_this_step = traci.simulation.getDepartedNumber()
        arrived_this_step = traci.simulation.getArrivedNumber()
        self.total_departed += departed_this_step
        self.total_arrived += arrived_this_step
        # Accumulate waiting time across all incoming lanes
        for lane in self.incoming_lanes:
            self._total_waiting_time += traci.lane.getWaitingTime(lane)
            self._waiting_time_steps += 1
        return departed_this_step, arrived_this_step

    def get_episode_stats(self) -> Dict:
        """Return summary statistics for the completed episode (matches TrafficEnv API)."""
        if self._queue_history:
            all_queues = [sum(q) for q in self._queue_history]
            avg_queue = float(np.mean(all_queues))
        else:
            avg_queue = 0.0

        avg_wait = (
            self._total_waiting_time / self._waiting_time_steps
            if self._waiting_time_steps > 0 else 0.0
        )
        throughput = self.total_arrived
        efficiency = throughput / max(1, self.total_departed + throughput)

        return {
            "avg_queue_length": avg_queue,
            "avg_waiting_time": avg_wait,
            "throughput": throughput,
            "efficiency": efficiency,
            "total_arrived": self.total_arrived,
            "total_departed": self.total_departed,
        }

    # ------------------------------------------------------------------ #
    #  Emergency preemption interface                                       #
    # ------------------------------------------------------------------ #

    def set_emergency(self, axis: int, duration: int):
        """
        Force an emergency preemption on an axis (0 = N-S, 1 = E-W).

        Parameters
        ----------
        axis : int       Axis index (0 or 1).
        duration : int   How many agent-steps this emergency lasts.
        """
        assert axis in (0, 1), "Axis must be 0 (N-S) or 1 (E-W)."
        assert duration > 0, "Duration must be positive."
        self.emergency_active = True
        self._emergency_axis = axis
        self._emergency_remaining = duration

    def cancel_emergency(self):
        """Cancel an active emergency preemption."""
        self.emergency_active = False
        self._emergency_remaining = 0

    def close(self):
        """Safely close traci."""
        try:
            traci.close()
        except:
            pass
