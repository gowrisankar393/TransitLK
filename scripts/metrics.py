"""
Performance Metrics Tracker for the Traffic Signal System.

Tracks and visualises four key metrics across training episodes:
  1. Average Queue Length   — mean total vehicles waiting per step
  2. Average Waiting Time   — mean timesteps a vehicle waits before departing
  3. Throughput             — total vehicles that passed through per episode
  4. Efficiency             — vehicles passed / vehicles arrived
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers / notebooks)
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class MetricsTracker:
    """
    Records per-step data within an episode and computes episode-level
    performance metrics.  Maintains a history of all episodes for
    plotting learning curves.
    """

    def __init__(self):
        # --- Per-step accumulators (reset each episode) ---
        self._step_queues: List[int] = []          # total queue length each step
        self._step_departed: List[int] = []
        self._step_arrived: List[int] = []
        self._wait_times: List[int] = []           # individual vehicle wait times

        # --- Episode history ---
        self.history: List[Dict] = []

        # --- Loss tracking ---
        self.episode_losses: List[float] = []
        self._current_losses: List[float] = []

    # ------------------------------------------------------------------ #
    #  Per-step recording                                                  #
    # ------------------------------------------------------------------ #

    def record_step(
        self,
        queue_lengths: List[int],
        vehicles_departed: int,
        vehicles_arrived: int,
        wait_times: Optional[List[int]] = None,
    ):
        """
        Record data for a single simulation step.

        Parameters
        ----------
        queue_lengths : list of int
            Current queue length of each lane.
        vehicles_departed : int
            Vehicles that departed this step.
        vehicles_arrived : int
            Vehicles that arrived this step.
        wait_times : list of int, optional
            Wait times of vehicles that departed this step.
        """
        self._step_queues.append(sum(queue_lengths))
        self._step_departed.append(vehicles_departed)
        self._step_arrived.append(vehicles_arrived)
        if wait_times:
            self._wait_times.extend(wait_times)

    def record_loss(self, loss: float):
        """Record a training loss value for the current episode."""
        self._current_losses.append(loss)

    # ------------------------------------------------------------------ #
    #  Episode boundary                                                    #
    # ------------------------------------------------------------------ #

    def end_episode(self, env_stats: Optional[Dict] = None) -> Dict:
        """
        Finalise the current episode metadata, store it, and reset
        per-step accumulators.

        Parameters
        ----------
        env_stats : dict, optional
            Episode stats from TrafficEnv.get_episode_stats().
            If provided, these are used directly; otherwise we compute
            from our own logs.

        Returns
        -------
        dict   Episode metrics.
        """
        if env_stats:
            metrics = {
                "episode": len(self.history) + 1,
                "avg_queue_length": env_stats["avg_queue_length"],
                "avg_waiting_time": env_stats["avg_waiting_time"],
                "throughput": env_stats["throughput"],
                "efficiency": env_stats["efficiency"],
                "total_arrived": env_stats["total_arrived"],
                "total_departed": env_stats["total_departed"],
            }
        else:
            total_arrived = sum(self._step_arrived)
            total_departed = sum(self._step_departed)
            metrics = {
                "episode": len(self.history) + 1,
                "avg_queue_length": float(np.mean(self._step_queues)) if self._step_queues else 0.0,
                "avg_waiting_time": float(np.mean(self._wait_times)) if self._wait_times else 0.0,
                "throughput": total_departed,
                "efficiency": total_departed / total_arrived if total_arrived > 0 else 0.0,
                "total_arrived": total_arrived,
                "total_departed": total_departed,
            }

        # Average loss for the episode
        if self._current_losses:
            metrics["avg_loss"] = float(np.mean(self._current_losses))
            self.episode_losses.append(metrics["avg_loss"])
        else:
            metrics["avg_loss"] = None

        self.history.append(metrics)

        # Reset accumulators
        self._step_queues = []
        self._step_departed = []
        self._step_arrived = []
        self._wait_times = []
        self._current_losses = []

        return metrics

    # ------------------------------------------------------------------ #
    #  History access                                                      #
    # ------------------------------------------------------------------ #

    def get_history(self) -> List[Dict]:
        """Return the full list of episode metrics."""
        return self.history

    def get_latest(self) -> Optional[Dict]:
        """Return the most recent episode's metrics."""
        return self.history[-1] if self.history else None

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def plot_metrics(self, save_path: str = "metrics_plot.png", show: bool = False):
        """
        Generate a 2×2 grid of learning-curve plots and save to disk.

        Parameters
        ----------
        save_path : str   File path for the saved PNG.
        show : bool       If True, also display interactively.
        """
        if not self.history:
            print("[Metrics] No data to plot.")
            return

        episodes = [m["episode"] for m in self.history]
        avg_q = [m["avg_queue_length"] for m in self.history]
        avg_w = [m["avg_waiting_time"] for m in self.history]
        throughput = [m["throughput"] for m in self.history]
        efficiency = [m["efficiency"] for m in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "DQN Traffic Signal Control — Learning Curves",
            fontsize=16, fontweight="bold",
        )

        # ---- Average Queue Length ----
        ax = axes[0, 0]
        ax.plot(episodes, avg_q, color="#e74c3c", linewidth=1.2, alpha=0.7)
        if len(avg_q) >= 10:
            ax.plot(
                episodes, self._smooth(avg_q),
                color="#e74c3c", linewidth=2.5, label="Smoothed",
            )
        ax.set_title("Average Queue Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Queue Length")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # ---- Average Waiting Time ----
        ax = axes[0, 1]
        ax.plot(episodes, avg_w, color="#3498db", linewidth=1.2, alpha=0.7)
        if len(avg_w) >= 10:
            ax.plot(
                episodes, self._smooth(avg_w),
                color="#3498db", linewidth=2.5, label="Smoothed",
            )
        ax.set_title("Average Waiting Time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Wait (steps)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # ---- Throughput ----
        ax = axes[1, 0]
        ax.plot(episodes, throughput, color="#2ecc71", linewidth=1.2, alpha=0.7)
        if len(throughput) >= 10:
            ax.plot(
                episodes, self._smooth(throughput),
                color="#2ecc71", linewidth=2.5, label="Smoothed",
            )
        ax.set_title("Throughput (Vehicles Passed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Vehicles")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # ---- Efficiency ----
        ax = axes[1, 1]
        ax.plot(episodes, efficiency, color="#9b59b6", linewidth=1.2, alpha=0.7)
        if len(efficiency) >= 10:
            ax.plot(
                episodes, self._smooth(efficiency),
                color="#9b59b6", linewidth=2.5, label="Smoothed",
            )
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal")
        ax.set_title("Efficiency (Passed / Arrived)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Ratio")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Metrics] Plot saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_loss(self, save_path: str = "loss_plot.png"):
        """Plot the training loss curve."""
        if not self.episode_losses:
            print("[Metrics] No loss data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        eps = list(range(1, len(self.episode_losses) + 1))
        ax.plot(eps, self.episode_losses, color="#e67e22", linewidth=1.2, alpha=0.7)
        if len(self.episode_losses) >= 10:
            ax.plot(
                eps, self._smooth(self.episode_losses),
                color="#e67e22", linewidth=2.5, label="Smoothed",
            )
        ax.set_title("Training Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Metrics] Loss plot saved to {save_path}")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_history(self, path: str = "metrics_history.json"):
        """Save the episode history to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Metrics] History saved to {path}")

    def load_history(self, path: str = "metrics_history.json"):
        """Load episode history from a JSON file."""
        with open(path, "r") as f:
            self.history = json.load(f)
        print(f"[Metrics] Loaded {len(self.history)} episodes from {path}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _smooth(values: list, window: int = 20) -> list:
        """Simple moving average for smoothing noisy curves."""
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(values[start : i + 1]))
        return smoothed
