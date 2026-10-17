"""Seeded, simulated MEC offloading environment.

All timing and energy values emitted by this module are model estimates, not
physical measurements.  Latencies are milliseconds and energy is mJ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple
import random
import numpy as np

LOCAL, EDGE_A, EDGE_B, CLOUD = range(4)
ACTION_NAMES = ("Local phone (simulated)", "Edge A", "Edge B", "Cloud")
STATE_DIM = 13


@dataclass(frozen=True)
class MECConfig:
    horizon: int = 30
    # Reward is dimensionless: latency/150 ms + energy/100 mJ + penalties.
    latency_scale_ms: float = 150.0
    energy_scale_mj: float = 100.0
    latency_weight: float = 1.0
    energy_weight: float = 0.35
    deadline_penalty: float = 2.0
    failure_penalty: float = 4.0
    train_bandwidth: Tuple[float, float] = (12.0, 40.0)
    # Interactive MEC tasks: a 20 KB--1 MB input can plausibly meet the
    # sub-second deadlines on a tens-of-Mbps link; larger inputs remain useful
    # for the congestion stress profiles.
    task_size_mb: Tuple[float, float] = (0.02, 1.0)
    task_cycles_g: Tuple[float, float] = (0.3, 5.0)
    deadline_ms: Tuple[float, float] = (120.0, 800.0)


class MECOffloadingEnv:
    """Sequential MDP with queue backlog, correlated network and outages."""

    def __init__(self, config: Optional[MECConfig] = None, profile: str = "train"):
        self.config = config or MECConfig()
        self.profile = profile
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()
        self.t = 0
        self.task: Dict[str, float] = {}
        self.bandwidth_mbps = 24.0
        self.jitter_ms = 3.0
        self.queue_a_ms = 8.0
        self.queue_b_ms = 15.0
        self.cloud_rtt_ms = 85.0
        self.edge_a_up = True
        self.edge_b_up = True
        self.cloud_up = True

    def reset(self, seed: Optional[int] = None, profile: Optional[str] = None):
        if profile is not None:
            self.profile = profile
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.t = 0
        if self.profile == "congestion":
            self.bandwidth_mbps, self.queue_a_ms, self.queue_b_ms, self.cloud_rtt_ms = 5.0, 55.0, 75.0, 115.0
        elif self.profile == "outage":
            self.bandwidth_mbps, self.queue_a_ms, self.queue_b_ms, self.cloud_rtt_ms = 1.5, 35.0, 25.0, 140.0
        elif self.profile == "unseen":
            self.bandwidth_mbps, self.queue_a_ms, self.queue_b_ms, self.cloud_rtt_ms = 8.0, 25.0, 35.0, 125.0
        else:
            lo, hi = self.config.train_bandwidth
            self.bandwidth_mbps = self.rng.uniform(lo, hi)
            self.queue_a_ms, self.queue_b_ms, self.cloud_rtt_ms = 8.0, 15.0, 85.0
        self.jitter_ms = self.rng.uniform(1.0, 5.0)
        self.edge_a_up = self.edge_b_up = self.cloud_up = True
        self.task = self._sample_task()
        return self._state(), self.info()

    def _sample_task(self) -> Dict[str, float]:
        c = self.config
        return {
            "size_mb": self.rng.uniform(*c.task_size_mb),
            "cpu_gcycles": self.rng.uniform(*c.task_cycles_g),
            "deadline_ms": self.rng.uniform(*c.deadline_ms),
            "battery_level": self.rng.uniform(0.25, 1.0),
            "local_cpu_available": self.rng.uniform(0.45, 1.0),
        }

    def _advance_exogenous_state(self) -> None:
        # Correlated random walk.  The `unseen` profile deliberately permits
        # lower bandwidth, more jitter, and outages absent from training.
        if self.profile == "congestion":
            center, sigma, outage = 5.0, 1.2, 0.02
        elif self.profile == "outage":
            center, sigma, outage = 1.5, 0.5, 0.30
        elif self.profile == "unseen":
            center, sigma, outage = 7.0, 5.0, 0.12
        else:
            center, sigma, outage = 26.0, 3.0, 0.0
        self.bandwidth_mbps = max(0.5, min(60.0, 0.75 * self.bandwidth_mbps + 0.25 * center + self.rng.gauss(0, sigma)))
        self.jitter_ms = max(0.5, min(40.0, self.jitter_ms + self.rng.gauss(0, 1.5)))
        self.edge_a_up = self.rng.random() >= outage
        self.edge_b_up = self.rng.random() >= outage * 0.8
        self.cloud_up = self.rng.random() >= outage * 0.25

    def valid_actions(self) -> np.ndarray:
        return np.array([True, self.edge_a_up, self.edge_b_up, self.cloud_up], dtype=bool)

    def _state(self) -> np.ndarray:
        d = self.task["deadline_ms"]
        return np.array([
            self.task["size_mb"] / 1.5, self.task["cpu_gcycles"] / 6.0,
            min(2.0, 200.0 / d), self.task["battery_level"], self.task["local_cpu_available"],
            min(1.0, self.bandwidth_mbps / 60.0), min(1.0, self.jitter_ms / 40.0),
            min(1.0, self.queue_a_ms / 300.0), min(1.0, self.queue_b_ms / 300.0),
            min(1.0, self.cloud_rtt_ms / 250.0), float(self.edge_a_up), float(self.edge_b_up), float(self.cloud_up),
        ], dtype=np.float32)

    def step(self, action: int):
        if not 0 <= int(action) < 4:
            raise ValueError("action must be an integer in [0, 3]")
        action = int(action)
        task = dict(self.task)
        valid = bool(self.valid_actions()[action])
        size, cycles = task["size_mb"], task["cpu_gcycles"]
        network = queue_delay = 0.0
        failed = not valid
        if action == LOCAL:
            compute = cycles * 95.0 / task["local_cpu_available"]
            energy = cycles * 32.0
        elif not valid:
            # Failed offload consumes an attempted transmission and expires.
            network, compute, energy = 120.0, 0.0, size * 4.0
        else:
            uplink = size * 8.0 / max(self.bandwidth_mbps, 0.5) * 1000.0
            if action == EDGE_A:
                queue_delay, compute, energy = self.queue_a_ms, cycles * 13.0, size * 7.0 + cycles * 4.0
                self.queue_a_ms += compute
            elif action == EDGE_B:
                queue_delay, compute, energy = self.queue_b_ms, cycles * 9.0, size * 8.0 + cycles * 4.5
                self.queue_b_ms += compute
            else:
                queue_delay, compute, energy = 10.0, cycles * 5.0, size * 11.0 + cycles * 3.0
                uplink += self.cloud_rtt_ms
            network = uplink + self.jitter_ms
        latency = max(1.0, network + queue_delay + compute)
        missed = latency > task["deadline_ms"]
        c = self.config
        reward = -(c.latency_weight * latency / c.latency_scale_ms + c.energy_weight * energy / c.energy_scale_mj
                   + (c.deadline_penalty if missed else 0.0) + (c.failure_penalty if failed else 0.0))
        # Servers work while this task elapses, so decisions affect later queues.
        self.queue_a_ms = max(0.0, self.queue_a_ms - latency * 0.35)
        self.queue_b_ms = max(0.0, self.queue_b_ms - latency * 0.35)
        self.t += 1
        done = self.t >= self.config.horizon
        self._advance_exogenous_state()
        if not done:
            self.task = self._sample_task()
        info = {
            "destination": ACTION_NAMES[action], "latency_ms": round(latency, 3),
            "network_delay_ms": round(network, 3), "queue_delay_ms": round(queue_delay, 3),
            "compute_delay_ms": round(compute, 3), "estimated_energy_mj": round(energy, 3),
            "estimated_energy_units": round(energy, 3),  # legacy API alias
            "deadline_missed": missed, "failed": failed, "reward": round(reward, 5),
            "queue_at_decision": round(queue_delay, 3), "action_valid": valid,
        }
        return self._state(), float(reward), done, False, info

    def info(self) -> Dict[str, Any]:
        return {"bandwidth": self.bandwidth_mbps, "queue_a": self.queue_a_ms, "queue_b": self.queue_b_ms,
                "cloud_ms": self.cloud_rtt_ms, "burst": self.profile in ("congestion", "outage", "unseen"),
                "profile": self.profile, "task": dict(self.task), "valid_actions": self.valid_actions().tolist()}

    def config_dict(self):
        return asdict(self.config)
