"""Reproducible train/evaluate command line workflow for MEC DQN experiments."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from dqn import DQNAgent, LatencyAwarePolicy, LocalPolicy, RandomPolicy
from mec_sim import MECOffloadingEnv

RESULT_DIR = Path("artifacts")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def run_episode(env, policy, seed, explore=False):
    state, info = env.reset(seed=seed); total = latency = energy = misses = failures = 0.0
    while True:
        action = policy.act(state, info["valid_actions"], explore=explore)
        next_state, reward, done, _, step_info = env.step(action)
        if isinstance(policy, DQNAgent) and explore:
            policy.observe(state, action, reward, next_state, done); policy.learn()
        total += reward; latency += step_info["latency_ms"]; energy += step_info["estimated_energy_mj"]; misses += step_info["deadline_missed"]; failures += step_info["failed"]
        state, info = next_state, env.info()
        if done: break
    n = env.config.horizon
    return {"average_latency_ms": latency/n, "average_estimated_energy_mj": energy/n, "deadline_miss_rate": misses/n, "task_failure_rate": failures/n, "cumulative_reward": total}

def aggregate(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

def train(episodes=500, seed=1, checkpoint=RESULT_DIR / "dqn.pt", agent=None):
    """Train a new agent by default, or continue the supplied dashboard agent."""
    seed_everything(seed); agent = agent or DQNAgent(seed); env = MECOffloadingEnv(profile="train")
    rows = [run_episode(env, agent, seed + i, explore=True) for i in range(episodes)]
    checkpoint.parent.mkdir(parents=True, exist_ok=True); agent.save(checkpoint)
    summary = {"episodes": episodes, "seed": seed, "training_summary": aggregate(rows), "checkpoint": str(checkpoint), "note": "All latency and energy figures are simulated estimates."}
    (checkpoint.parent / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return agent, summary

def evaluate(agent, episodes=100, seed=100_007, profile="train"):
    # Identical episode seeds make every policy see the same generated scenarios.
    policies = {"dqn": agent, "local_only": LocalPolicy(), "random": RandomPolicy(seed), "latency_aware": LatencyAwarePolicy()}
    all_rows = {}; seeds = [seed + i for i in range(episodes)]
    for name, policy in policies.items(): all_rows[name] = [run_episode(MECOffloadingEnv(profile=profile), policy, s) for s in seeds]
    return {name: aggregate(rows) for name, rows in all_rows.items()}

def save_evaluation(summary, profile, path=RESULT_DIR / "evaluation.json"):
    path.parent.mkdir(parents=True, exist_ok=True); document = {"profile": profile, "simulated_estimates": True, "results": summary}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", *next(iter(summary.values())).keys()]); w.writeheader()
        for policy, metrics in summary.items(): w.writerow({"policy": policy, **metrics})

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train"); t.add_argument("--episodes", type=int, default=500); t.add_argument("--seed", type=int, default=1); t.add_argument("--checkpoint", default="artifacts/dqn.pt")
    e = sub.add_parser("evaluate"); e.add_argument("--episodes", type=int, default=100); e.add_argument("--seed", type=int, default=100_007); e.add_argument("--checkpoint", default="artifacts/dqn.pt"); e.add_argument("--profile", choices=["train", "congestion", "outage", "unseen"], default="train")
    a = p.parse_args()
    if a.command == "train": print(json.dumps(train(a.episodes, a.seed, Path(a.checkpoint))[1], indent=2))
    else:
        agent = DQNAgent(a.seed); agent.load(a.checkpoint); result = evaluate(agent, a.episodes, a.seed, a.profile); out = RESULT_DIR / f"evaluation_{a.profile}.json"; save_evaluation(result, a.profile, out); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
