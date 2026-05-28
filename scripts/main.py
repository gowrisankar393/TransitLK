"""
TransitLK Traffic Signal Control — Main Entry Point

Modes:
    train   — Train the DQN agent and save checkpoint + metrics
    eval    — Load a trained model and evaluate (no exploration)
    demo    — Run one episode with manual and emergency preemption injected mid-way

Usage:
    python scripts\\main.py --mode train --episodes 500
    python scripts\\main.py --mode eval  --model checkpoint.pth
    python scripts\\main.py --mode demo
"""

import argparse
import random
import os
import sys
import time
import tempfile
import xml.etree.ElementTree as ET
from contextlib import nullcontext

import numpy as np

from environment import TrafficEnv, EnvConfig
from dqn_agent import DQNAgent
from metrics import MetricsTracker
from manual_override import override_signal, cancel_override
from emergency_override import trigger_emergency

# SUMO Bridge (Optional)
try:
    from sumo_env import SumoTrafficEnv
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False


# ====================================================================== #
#  Training                                                                #
# ====================================================================== #

SUMO_SCENARIO_PROFILES = {
    "base": {"f_WE": 1.0, "f_EW": 1.0, "f_NS": 1.0, "f_SN": 1.0},
    "peak": {"f_WE": 1.5, "f_EW": 1.5, "f_NS": 1.5, "f_SN": 1.5},
    "ns_heavy": {"f_WE": 0.7, "f_EW": 0.7, "f_NS": 1.8, "f_SN": 1.8},
    "ew_heavy": {"f_WE": 1.8, "f_EW": 1.8, "f_NS": 0.7, "f_SN": 0.7},
}


def _build_randomized_sumo_config(
    temp_dir: str,
    episode: int,
    rng: random.Random,
):
    """Create an episode-specific SUMO config with randomized demand."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_route = os.path.join(repo_root, "sumo_configs", "crossroad.rou.xml")
    net_file = os.path.join(repo_root, "sumo_configs", "crossroad.net.xml")

    scenario_name = rng.choice(list(SUMO_SCENARIO_PROFILES.keys()))
    profile = SUMO_SCENARIO_PROFILES[scenario_name]
    global_jitter = rng.uniform(0.9, 1.1)

    route_tree = ET.parse(base_route)
    route_root = route_tree.getroot()

    for flow in route_root.findall("flow"):
        flow_id = flow.attrib["id"]
        base_vph = float(flow.attrib["vehsPerHour"])
        factor = profile.get(flow_id, 1.0) * global_jitter
        flow.attrib["vehsPerHour"] = str(max(10, int(round(base_vph * factor))))

    route_file = os.path.join(temp_dir, f"train_ep_{episode:04d}.rou.xml")
    route_tree.write(route_file, encoding="utf-8", xml_declaration=True)

    cfg_root = ET.Element("configuration")
    cfg_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    cfg_root.set(
        "xsi:noNamespaceSchemaLocation",
        "http://sumo.dlr.de/xsd/sumoConfiguration.xsd",
    )
    input_el = ET.SubElement(cfg_root, "input")
    ET.SubElement(input_el, "net-file", value=net_file)
    ET.SubElement(input_el, "route-files", value=route_file)
    time_el = ET.SubElement(cfg_root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value="3600")
    report_el = ET.SubElement(cfg_root, "report")
    ET.SubElement(report_el, "verbose", value="false")
    ET.SubElement(report_el, "no-step-log", value="true")

    cfg_file = os.path.join(temp_dir, f"train_ep_{episode:04d}.sumocfg")
    ET.ElementTree(cfg_root).write(cfg_file, encoding="utf-8", xml_declaration=True)
    return cfg_file, scenario_name


def train(
    episodes: int = 500,
    checkpoint_path: str = "checkpoint.pth",
    metrics_dir: str = "results",
    use_sumo: bool = False,
    use_gui: bool = False,
    sumo_randomize_demand: bool = False,
    sumo_seed: int = None,
    sumo_gui_delay_ms: int = 0,
):
    """Train the DQN agent on the traffic environment."""

    os.makedirs(metrics_dir, exist_ok=True)
    model_prefix = "sumo_" if use_sumo else "internal_"
    checkpoint_filename = (
        f"{model_prefix}checkpoint.pth"
        if checkpoint_path == "checkpoint.pth"
        else checkpoint_path
    )
    best_model_path = os.path.join(metrics_dir, checkpoint_filename)
    final_model_path = os.path.join(metrics_dir, f"{model_prefix}final_model.pth")

    if use_sumo:
        if not SUMO_AVAILABLE:
            print("Error: SUMO or TraCI not found. Please install them first.")
            return
        env = SumoTrafficEnv(
            use_gui=use_gui,
            sumo_seed=sumo_seed,
            gui_delay_ms=sumo_gui_delay_ms,
        )
        state_dim = 4  # North, South, East, West
        action_dim = 2 # N-S Green, E-W Green
        env_name = "SUMO (4-Way Intersection)"
        if sumo_randomize_demand:
            env_name = "SUMO (4-Way Intersection, Randomized Demand)"
        episode_steps = env.episode_length
    else:
        config = EnvConfig()
        env = TrafficEnv(config)
        state_dim = config.num_lanes
        action_dim = config.num_lanes * 2
        env_name = "Internal (Sri Lankan 2-Lane)"
        episode_steps = config.episode_length

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    tracker = MetricsTracker()

    print("=" * 65)
    print(f"  TransitLK - DQN Traffic Signal Control  (Training)")
    print("=" * 65)
    print(f"  Environment: {env_name}")
    print("=" * 65)
    print(f"  Device  : {agent.device}")
    print(f"  Episodes: {episodes}")
    print(f"  Env steps/episode: {episode_steps}")
    print(f"  Action space: {agent.action_dim}")
    print("=" * 65)
    print()

    best_efficiency = 0.0
    start_time = time.time()
    rng = random.Random(sumo_seed) if sumo_seed is not None else random.Random()
    context = (
        tempfile.TemporaryDirectory(prefix="sumo_train_")
        if use_sumo and sumo_randomize_demand
        else nullcontext(None)
    )

    with context as temp_sumo_dir:
        for ep in range(1, episodes + 1):
            scenario_name = None
            if use_sumo and sumo_randomize_demand:
                cfg_file, scenario_name = _build_randomized_sumo_config(
                    temp_dir=temp_sumo_dir,
                    episode=ep,
                    rng=rng,
                )
                episode_seed = (sumo_seed + ep) if sumo_seed is not None else None
                env = SumoTrafficEnv(
                    config_file=cfg_file,
                    use_gui=use_gui,
                    sumo_seed=episode_seed,
                    gui_delay_ms=sumo_gui_delay_ms,
                )

            state = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # Agent picks an action
                action = agent.select_action(state)

                # Environment steps
                next_state, reward, done, info = env.step(action)

                # Store transition & learn
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    tracker.record_loss(loss)

                # Record step metrics
                tracker.record_step(
                    queue_lengths=info["queue_lengths"],
                    vehicles_departed=info["departed_this_step"],
                    vehicles_arrived=info["arrived_this_step"],
                )

                episode_reward += reward
                state = next_state

            # End of episode bookkeeping
            env_stats = env.get_episode_stats()
            ep_metrics = tracker.end_episode(env_stats)

            # Decay exploration
            agent.decay_epsilon()

            # Update target network periodically
            if ep % agent.target_update == 0:
                agent.update_target_network()

            # Save best model
            if ep_metrics["efficiency"] > best_efficiency:
                best_efficiency = ep_metrics["efficiency"]
                agent.save(best_model_path)

            # Logging
            if ep % 10 == 0 or ep == 1:
                elapsed = time.time() - start_time
                scenario_suffix = (
                    f" | Scenario: {scenario_name}"
                    if scenario_name is not None
                    else ""
                )
                print(
                    f"Episode {ep:4d}/{episodes} | "
                    f"Reward: {episode_reward:8.1f} | "
                    f"AvgQ: {ep_metrics['avg_queue_length']:6.2f} | "
                    f"AvgWait: {ep_metrics['avg_waiting_time']:6.2f} | "
                    f"Throughput: {ep_metrics['throughput']:5d} | "
                    f"Efficiency: {ep_metrics['efficiency']:.3f} | "
                    f"eps: {agent.epsilon:.3f} | "
                    f"Time: {elapsed:.1f}s"
                    f"{scenario_suffix}"
                )

    # Final save (even if not best)
    agent.save(final_model_path)
    tracker.save_history(os.path.join(metrics_dir, "metrics_history.json"))
    tracker.plot_metrics(os.path.join(metrics_dir, "metrics_plot.png"))
    tracker.plot_loss(os.path.join(metrics_dir, "loss_plot.png"))

    print()
    print("=" * 65)
    print("  Training complete!")
    print(f"  Best efficiency: {best_efficiency:.4f}")
    print(f"  Results saved to: {metrics_dir}/")
    print("=" * 65)


# ====================================================================== #
#  Evaluation                                                              #
# ====================================================================== #

def evaluate(
    model_path: str,
    episodes: int = 20,
    use_sumo: bool = False,
    use_gui: bool = False,
    sumo_gui_delay_ms: int = 0,
):
    """Evaluate a trained model with no exploration (eps = 0)."""

    if use_sumo:
        env = SumoTrafficEnv(use_gui=use_gui, gui_delay_ms=sumo_gui_delay_ms)
        state_dim = 4
        action_dim = 2
    else:
        config = EnvConfig()
        env = TrafficEnv(config)
        state_dim = config.num_lanes
        action_dim = config.num_lanes * 2

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    agent.load(model_path)
    tracker = MetricsTracker()

    print("=" * 65)
    print("  TransitLK - DQN Traffic Signal Control  (Evaluation)")
    print("=" * 65)
    print(f"  Model : {model_path}")
    print(f"  Episodes: {episodes}")
    print("=" * 65)
    print()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            next_state, reward, done, info = env.step(action)

            tracker.record_step(
                queue_lengths=info["queue_lengths"],
                vehicles_departed=info["departed_this_step"],
                vehicles_arrived=info["arrived_this_step"],
            )
            state = next_state

        env_stats = env.get_episode_stats()
        ep_metrics = tracker.end_episode(env_stats)
        print(
            f"Episode {ep:3d} | "
            f"AvgQ: {ep_metrics['avg_queue_length']:6.2f} | "
            f"AvgWait: {ep_metrics['avg_waiting_time']:6.2f} | "
            f"Throughput: {ep_metrics['throughput']:5d} | "
            f"Efficiency: {ep_metrics['efficiency']:.3f}"
        )

    # Summary
    history = tracker.get_history()
    print()
    print("-" * 65)
    print("  Evaluation Summary (averaged over all episodes)")
    print("-" * 65)
    print(f"  Avg Queue Length : {np.mean([h['avg_queue_length'] for h in history]):.2f}")
    print(f"  Avg Waiting Time : {np.mean([h['avg_waiting_time'] for h in history]):.2f}")
    print(f"  Avg Throughput   : {np.mean([h['throughput'] for h in history]):.1f}")
    print(f"  Avg Efficiency   : {np.mean([h['efficiency'] for h in history]):.4f}")
    print("-" * 65)


# ====================================================================== #
#  Manual Override Demo                                                    #
# ====================================================================== #

def demo(
    use_sumo: bool = False,
    use_gui: bool = False,
    sumo_gui_delay_ms: int = 0,
    demo_steps: int = None,
):
    """
    Run a single episode demonstrating manual override mid-simulation.
    """

    effective_delay_ms = sumo_gui_delay_ms
    if use_sumo and use_gui and sumo_gui_delay_ms == 0:
        # Slow default for recording in SUMO demo mode
        effective_delay_ms = 200

    if use_sumo:
        env = SumoTrafficEnv(
            use_gui=use_gui,
            gui_delay_ms=effective_delay_ms,
            episode_length=demo_steps or 500,
        )
        state_dim = 4
        action_dim = 2
        env_name = "SUMO (4-Way Intersection)"
    else:
        config = EnvConfig(episode_length=demo_steps or 500)
        env = TrafficEnv(config)
        state_dim = config.num_lanes
        action_dim = config.num_lanes * 2
        env_name = "Internal (Sri Lankan 2-Lane)"

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    # Try to load a trained model; fall back to random if none exists
    model_prefix = "sumo_" if use_sumo else "internal_"
    model_candidates = [
        os.path.join("results", f"{model_prefix}checkpoint.pth"),
        os.path.join("results", "checkpoint.pth"),
        os.path.join("results", f"{model_prefix}final_model.pth"),
        os.path.join("results", "final_model.pth"),
    ]
    loaded_model_path = None
    for candidate in model_candidates:
        if not os.path.exists(candidate):
            continue
        try:
            agent.load(candidate)
            loaded_model_path = candidate
            break
        except RuntimeError:
            print(f"[Demo] Skipping incompatible model: {candidate}")

    if loaded_model_path:
        print(f"[Demo] Loaded model: {loaded_model_path}")
        use_epsilon = 0.0
    else:
        print(
            f"[Demo] No trained model found in {model_candidates} - using random agent."
        )
        use_epsilon = 1.0

    print("=" * 65)
    print("  TransitLK - Traffic Demo")
    print("=" * 65)
    print(f"  Environment: {env_name}")

    state = env.reset()
    done = False
    step = 0

    while not done:
        step += 1

        # --- Inject manual override / emergency preemption ---
        if use_sumo:
            max_agent_steps = max(1, env.episode_length // env.step_duration)
            emergency_step = max(5, min(80, max_agent_steps // 2))
            emergency_duration = max(5, min(20, max_agent_steps // 3))
            if step == emergency_step:
                env.set_emergency(axis=0, duration=emergency_duration)
                print(
                    f"\n  >>> STEP {step}: Emergency preemption (N-S) "
                    f"for {emergency_duration} steps.\n"
                )
        else:
            if step == 200:
                result = override_signal(env, lane=1, green_duration=50)
                print(f"\n  >>> STEP {step}: {result['message']}\n")

            if step == 230:
                result = cancel_override(env)
                print(f"\n  >>> STEP {step}: {result['message']}\n")

            if step == 260:
                result = trigger_emergency(env, lane=0, green_duration=40)
                print(f"\n  >>> STEP {step}: {result['message']}\n")

        action = agent.select_action(state, epsilon=use_epsilon)
        next_state, reward, done, info = env.step(action)
        state = next_state

        # Print every 50 steps
        if step % 50 == 0 or step == 1:
            q = info["queue_lengths"]
            if use_sumo:
                ctrl = "EMERGENCY (SUMO)" if info["emergency_active"] else "AI (SUMO)"
                phase = f"Phase {info['phase']}"
                print(f"  Step {step:4d} | Queues (N,S,E,W): {q} | Signal: {phase} | Control: {ctrl}")
            else:
                if info["emergency_active"]:
                    ctrl = "EMERGENCY"
                elif info["override_active"]:
                    ctrl = "OVERRIDE"
                else:
                    ctrl = "AI"
                print(
                    f"  Step {step:4d} | Queues: {q} | "
                    f"Green: Lane {info['green_lane']} | "
                    f"Control: {ctrl} | Departed: {info['total_departed']}"
                )

    stats = env.get_episode_stats()
    print()
    print("-" * 65)
    print("  Demo Episode Summary")
    print("-" * 65)
    print(f"  Avg Queue Length : {stats['avg_queue_length']:.2f}")
    print(f"  Avg Waiting Time : {stats['avg_waiting_time']:.2f}")
    print(f"  Throughput       : {stats['throughput']}")
    print(f"  Efficiency       : {stats['efficiency']:.4f}")
    print("-" * 65)


# ====================================================================== #
#  CLI Entry Point                                                         #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="TransitLK - AI-Based Traffic Signal Control (Phase 1, Approach A)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "demo"],
        default="train",
        help="Run mode: train | eval | demo",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training/evaluation episodes (default: 500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/checkpoint.pth",
        help="Path to model checkpoint for eval mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pth",
        help="Checkpoint filename for train mode (saved inside results/)",
    )
    parser.add_argument(
        "--sumo",
        action="store_true",
        help="Use realistic SUMO simulation instead of internal math-env",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable SUMO visual interface (requires --sumo)",
    )
    parser.add_argument(
        "--sumo-randomize-demand",
        action="store_true",
        help="Randomize SUMO traffic demand profile each training episode",
    )
    parser.add_argument(
        "--sumo-seed",
        type=int,
        default=None,
        help="Optional SUMO/random seed for reproducible runs",
    )
    parser.add_argument(
        "--sumo-gui-delay-ms",
        type=int,
        default=0,
        help="Delay per SUMO GUI simulation step in milliseconds (for slow playback)",
    )
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=None,
        help="Episode length in steps for demo mode (default: 500)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(
            episodes=args.episodes,
            checkpoint_path=args.checkpoint,
            use_sumo=args.sumo,
            use_gui=args.gui,
            sumo_randomize_demand=args.sumo_randomize_demand,
            sumo_seed=args.sumo_seed,
            sumo_gui_delay_ms=args.sumo_gui_delay_ms,
        )
    elif args.mode == "eval":
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        evaluate(
            model_path=args.model,
            episodes=args.episodes,
            use_sumo=args.sumo,
            use_gui=args.gui,
            sumo_gui_delay_ms=args.sumo_gui_delay_ms,
        )
    elif args.mode == "demo":
        demo(
            use_sumo=args.sumo,
            use_gui=args.gui,
            sumo_gui_delay_ms=args.sumo_gui_delay_ms,
            demo_steps=args.demo_steps,
        )


if __name__ == "__main__":
    main()
