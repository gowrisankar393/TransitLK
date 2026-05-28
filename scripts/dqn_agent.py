"""
Deep Q-Network (DQN) Agent for traffic signal control.

Implements a standard DQN with:
  - 2-layer MLP Q-network
  - Target network with soft/hard updates
  - Experience replay buffer
  - Epsilon-greedy exploration with decay
"""

import random
import numpy as np
from collections import deque
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ====================================================================== #
#  Q-Network                                                              #
# ====================================================================== #

class QNetwork(nn.Module):
    """
    Simple feed-forward Q-network.

    Architecture:
        Input(state_dim) → 128 → 128 → Output(action_dim)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ====================================================================== #
#  Replay Buffer                                                          #
# ====================================================================== #

class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ====================================================================== #
#  DQN Agent                                                              #
# ====================================================================== #

class DQNAgent:
    """
    DQN Agent for traffic signal control.

    Hyperparameters (all configurable via constructor):
        gamma           discount factor              0.99
        lr              learning rate                0.001
        batch_size      minibatch size               64
        buffer_capacity replay buffer size           10 000
        buffer_warmup   min samples before training  500
        epsilon_start   initial exploration rate      1.0
        epsilon_end     final exploration rate        0.01
        epsilon_decay   multiplicative decay/episode  0.995
        target_update   episodes between hard update  10
        tau             soft-update coefficient       1.0 (=hard update)
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 4,
        gamma: float = 0.99,
        lr: float = 0.001,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        buffer_warmup: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 10,
        tau: float = 1.0,
        device: Optional[str] = None,
    ):
        # Auto-detect device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_warmup = buffer_warmup
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.tau = tau

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimiser & loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()   # Huber loss — more stable than MSE

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Tracking
        self.train_step_count = 0

    # ------------------------------------------------------------------ #
    #  Action selection                                                    #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray   Current state vector.
        epsilon : float      Override epsilon (use self.epsilon if None).

        Returns
        -------
        int   Chosen action index.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    #  Experience storage                                                  #
    # ------------------------------------------------------------------ #

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train_step(self) -> Optional[float]:
        """
        Sample a minibatch and perform one gradient update.

        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.buffer) < self.buffer_warmup:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(states_t)                          # (B, A)
        q_taken = q_values.gather(1, actions_t.unsqueeze(1))     # (B, 1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            next_q_max = next_q.max(dim=1)[0]                   # (B,)
            target = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        loss = self.loss_fn(q_taken.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        return loss.item()

    # ------------------------------------------------------------------ #
    #  Target network updates                                              #
    # ------------------------------------------------------------------ #

    def update_target_network(self):
        """
        Update the target network.

        If tau == 1.0 this is a hard copy.
        Otherwise it's a Polyak (soft) update: θ_t ← τ·θ + (1−τ)·θ_t
        """
        if self.tau >= 1.0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            for tp, sp in zip(
                self.target_net.parameters(), self.q_net.parameters()
            ):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Save the Q-network weights and training state."""
        torch.save({
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
        }, path)
        print(f"[DQN] Model saved to {path}")

    def load(self, path: str):
        """Load Q-network weights and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.train_step_count = checkpoint.get("train_step_count", 0)
        print(f"[DQN] Model loaded from {path}")
