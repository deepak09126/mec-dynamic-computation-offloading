import numpy as np
from dqn import DQNAgent, LatencyAwarePolicy
from mec_sim import EDGE_A, LOCAL, MECConfig, MECOffloadingEnv

def test_reset_reproducibility():
    one, two = MECOffloadingEnv(), MECOffloadingEnv(); s1, i1 = one.reset(seed=42); s2, i2 = two.reset(seed=42)
    assert np.array_equal(s1, s2) and i1["task"] == i2["task"]
    a = LatencyAwarePolicy().act(s1, i1["valid_actions"])
    assert one.step(a)[1:] == two.step(a)[1:]
def test_transition_changes_state():
    env = MECOffloadingEnv(); state, _ = env.reset(seed=4); next_state, reward, done, _, info = env.step(EDGE_A)
    assert next_state.shape == (13,) and np.isfinite(reward) and not done and info["destination"] == "Edge A" and not np.array_equal(state, next_state)
def test_actions_and_invalid_offload_failure():
    env = MECOffloadingEnv(profile="outage"); env.reset(seed=1)
    try: env.step(8); assert False
    except ValueError: pass
    env.edge_a_up = False; _, reward, _, _, info = env.step(EDGE_A)
    assert info["failed"] and not info["action_valid"] and reward < -4 and env.valid_actions()[LOCAL]
def test_deadline_reward_and_terminal_state():
    env = MECOffloadingEnv(MECConfig(horizon=1)); env.reset(seed=3); env.task["deadline_ms"] = 1.0
    _, reward, done, _, info = env.step(LOCAL)
    assert done and info["deadline_missed"] and reward <= -env.config.deadline_penalty
def test_reward_units_and_weights_are_applied():
    env = MECOffloadingEnv(); env.reset(seed=3)
    env.task.update(size_mb=0.1, cpu_gcycles=1.0, deadline_ms=99999.0, battery_level=1.0, local_cpu_available=1.0)
    _, reward, _, _, info = env.step(LOCAL)
    expected = -(info["latency_ms"] / 150.0 + 0.35 * info["estimated_energy_mj"] / 100.0)
    assert abs(reward - expected) < 1e-6
def test_agent_masks_unavailable_action():
    assert DQNAgent(seed=11).act(np.zeros(13, dtype=np.float32), [True, False, False, False]) == LOCAL
def test_dashboard_compatibility_endpoints():
    from app import app
    client = app.test_client()
    assert client.get("/api/status").status_code == 200
    assert client.post("/api/network", json={"mode": "congestion"}).status_code == 200
    response = client.post("/api/task", json={"force_destination": "Edge A"})
    assert response.status_code == 200 and "latency_ms" in response.get_json()
