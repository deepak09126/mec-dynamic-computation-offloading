"""Flask dashboard retained for the interactive MEC demonstration."""
from __future__ import annotations
import hashlib
from pathlib import Path
import threading
import time
from flask import Flask, jsonify, render_template_string, request
from dqn import DQNAgent
from experiment import train
from mec_sim import ACTION_NAMES, MECOffloadingEnv

app = Flask(__name__); lock = threading.Lock()
agent = DQNAgent(seed=7); checkpoint = Path("artifacts/dqn.pt")
if checkpoint.exists(): agent.load(checkpoint)
agent_kind = "DQN (PyTorch; simulated MEC model)"
live_env = MECOffloadingEnv(profile="train"); live_state, _ = live_env.reset(seed=17)
metrics = {"requests": 0, "deadline_misses": 0, "failures": 0, "history": [], "trained_episodes": 0}

def _profile(mode): return {"normal": "train", "random": "train", "congestion": "congestion", "outage": "outage"}.get(mode, "train")

@app.route("/")
def index(): return render_template_string(PAGE, agent_kind=agent_kind)

@app.route("/api/status")
def status():
    with lock: return jsonify({"agent": agent_kind, "torch_available": True, "metrics": metrics, "network": live_env.info(), "destinations": list(ACTION_NAMES), "simulated_estimates": True})

@app.route("/api/task", methods=["POST"])
def task():
    global live_state
    forced = (request.get_json(silent=True) or {}).get("force_destination")
    with lock:
        action = ACTION_NAMES.index(forced) if forced in ACTION_NAMES else agent.act(live_state, live_env.valid_actions(), explore=False)
        current_task = dict(live_env.task); next_state, _, done, _, outcome = live_env.step(action)
        started = time.perf_counter(); digest = hashlib.sha256(repr(current_task).encode()).hexdigest()
        for i in range(1500): digest = hashlib.sha256((digest + str(i)).encode()).hexdigest()
        backend_ms = (time.perf_counter() - started) * 1000
        live_state = live_env.reset()[0] if done else next_state
        metrics["requests"] += 1; metrics["deadline_misses"] += int(outcome["deadline_missed"]); metrics["failures"] += int(outcome["failed"])
        record = {"time": time.strftime("%H:%M:%S"), **{k: round(v, 3) for k, v in current_task.items()}, **outcome, "backend_workload_ms": round(backend_ms, 3), "result_hash": digest[:16], "state": [round(float(x), 3) for x in live_state]}
        metrics["history"].insert(0, record); metrics["history"] = metrics["history"][:40]
        return jsonify(record)

@app.route("/api/network", methods=["POST"])
def network():
    global live_state
    with lock:
        live_state, _ = live_env.reset(profile=_profile((request.get_json(silent=True) or {}).get("mode", "normal")))
        return jsonify(live_env.info())

@app.route("/api/train", methods=["POST"])
def train_api():
    global agent
    episodes = max(20, min(3000, int((request.get_json(silent=True) or {}).get("episodes", 300))))
    with lock:
        agent, _ = train(episodes=episodes, seed=7, checkpoint=checkpoint, agent=agent); metrics["trained_episodes"] += episodes
        return jsonify({"episodes_added": episodes, "total_episodes": metrics["trained_episodes"], "agent": agent_kind, "checkpoint": str(checkpoint)})

@app.route("/api/reset", methods=["POST"])
def reset():
    with lock:
        metrics.update(requests=0, deadline_misses=0, failures=0, history=[], trained_episodes=metrics["trained_episodes"])
        return jsonify({"ok": True})

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>MEC Live Lab</title><style>body{margin:auto;max-width:1040px;padding:24px;background:#0b1020;color:#eaf0ff;font:15px system-ui}button{padding:9px;margin:3px;background:#6385f5;border:0;border-radius:7px;font-weight:700}pre{background:#121a2e;padding:12px;overflow:auto}small{color:#b7c4e5}</style></head><body><h1>MEC Live Lab</h1><p>Sequential DQN offloading controller. Network, queue, latency, and energy are <b>simulated estimates</b>; backend timing is a local measurement only.</p><p id="agent">{{ agent_kind }}</p><button onclick="network('normal')">Normal</button><button onclick="network('congestion')">Congestion</button><button onclick="network('outage')">Link stress</button><button onclick="network('random')">Random state</button><br><button onclick="task()">Send task & let agent decide</button><button onclick="task('Edge A')">Force Edge A</button><button onclick="task('Edge B')">Force Edge B</button><button onclick="train()">Train +300 episodes</button><button onclick="reset()">Reset metrics</button><h2>Latest task</h2><pre id="result">Awaiting task.</pre><h2>Status</h2><pre id="status">Loading…</pre><script>async function api(p,d){let r=await fetch(p,{method:d?'POST':'GET',headers:{'Content-Type':'application/json'},body:d?JSON.stringify(d):null});if(!r.ok)throw Error(await r.text());return r.json()}async function refresh(){let x=await api('/api/status');status.textContent=JSON.stringify(x,null,2)}async function task(force_destination){result.textContent=JSON.stringify(await api('/api/task',force_destination?{force_destination}:{}),null,2);refresh()}async function network(mode){await api('/api/network',{mode});refresh()}async function train(){result.textContent='Training…';result.textContent=JSON.stringify(await api('/api/train',{episodes:300}),null,2);refresh()}async function reset(){await api('/api/reset',{});refresh()}refresh()</script></body></html>'''
if __name__ == "__main__":
    print("MEC Live Lab: http://127.0.0.1:5000"); app.run(host="0.0.0.0", port=5000, debug=False)
