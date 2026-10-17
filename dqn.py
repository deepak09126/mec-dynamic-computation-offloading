"""DQN and non-learning policies for the simulated MEC environment."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import random
from typing import Deque, Optional
import numpy as np
import torch
from torch import nn
from mec_sim import STATE_DIM


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 20_000
    target_sync_steps: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 6_000
    hidden_size: int = 128


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.data: Deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    def __len__(self): return len(self.data)
    def add(self, state, action, reward, next_state, done):
        self.data.append((np.array(state, dtype=np.float32), int(action), float(reward), np.array(next_state, dtype=np.float32), bool(done)))
    def sample(self, size): return self.rng.sample(self.data, size)


class QNetwork(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
    def forward(self, x): return self.layers(x)


class DQNAgent:
    def __init__(self, seed: int = 0, config: Optional[DQNConfig] = None, device: str = "cpu"):
        self.config, self.device = config or DQNConfig(), torch.device(device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.online, self.target = QNetwork(self.config.hidden_size).to(self.device), QNetwork(self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay, self.rng, self.steps = ReplayBuffer(self.config.replay_capacity, seed + 1), random.Random(seed + 2), 0
    @property
    def epsilon(self):
        progress = min(1.0, self.steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
    def act(self, state, valid_actions=None, explore=False):
        valid = np.ones(4, dtype=bool) if valid_actions is None else np.asarray(valid_actions, dtype=bool)
        if not valid.any(): raise ValueError("no valid actions")
        if explore and self.rng.random() < self.epsilon: return int(self.rng.choice(np.flatnonzero(valid).tolist()))
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            q[~torch.as_tensor(valid, device=self.device)] = -torch.inf
            return int(q.argmax().item())
    def observe(self, state, action, reward, next_state, done): self.replay.add(state, action, reward, next_state, done)
    def learn(self):
        if len(self.replay) < self.config.batch_size: return None
        b = self.replay.sample(self.config.batch_size)
        s, a, r, ns, d = zip(*b)
        s = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=self.device); ns = torch.as_tensor(np.asarray(ns), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device).unsqueeze(1); r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1); d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        q = self.online(s).gather(1, a)
        with torch.no_grad():
            # Availability is part of state.  Mask target values as well as
            # behaviour actions, so Bellman backups cannot bootstrap through a
            # remote destination unavailable in the successor state.
            next_q = self.target(ns)
            next_valid = torch.cat((torch.ones((ns.shape[0], 1), dtype=torch.bool, device=self.device), ns[:, 10:13] >= 0.5), dim=1)
            next_q = next_q.masked_fill(~next_valid, -torch.inf)
            target = r + self.config.gamma * next_q.max(1, keepdim=True).values * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q, target); self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(), 10.0); self.optimizer.step()
        self.steps += 1
        if self.steps % self.config.target_sync_steps == 0: self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
    def save(self, path):
        torch.save({"model": self.online.state_dict(), "config": asdict(self.config), "steps": self.steps}, path)
    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True); self.online.load_state_dict(data["model"]); self.target.load_state_dict(data["model"]); self.steps = data.get("steps", 0)


class LocalPolicy:
    def act(self, state, valid_actions=None, explore=False): return 0
class RandomPolicy:
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def act(self, state, valid_actions=None, explore=False):
        valid = np.ones(4, bool) if valid_actions is None else np.asarray(valid_actions, bool)
        return int(self.rng.choice(np.flatnonzero(valid).tolist()))
class LatencyAwarePolicy:
    """Chooses the valid destination with lowest estimated latency from state."""
    def act(self, state, valid_actions=None, explore=False):
        s = np.asarray(state); valid = np.ones(4, bool) if valid_actions is None else np.asarray(valid_actions, bool)
        size, cycles, cpu, bw = s[0] * 1.5, s[1] * 6, max(.1, s[4]), max(.5, s[5] * 60)
        estimates = np.array([cycles * 95 / cpu, size * 8 / bw * 1000 + s[7] * 300 + cycles * 13, size * 8 / bw * 1000 + s[8] * 300 + cycles * 9, size * 8 / bw * 1000 + s[9] * 250 + cycles * 5])
        estimates[~valid] = np.inf
        return int(estimates.argmin())
