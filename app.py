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
agent = DQNAgent(seed=7); checkpoint = Path("artifacts/dashboard_dqn.pt")
if checkpoint.exists(): agent.load(checkpoint)
agent_kind = "DQN (PyTorch; simulated MEC model)"
live_env = MECOffloadingEnv(profile="train"); live_state, _ = live_env.reset(seed=17)
metrics = {"requests": 0, "deadline_misses": 0, "failures": 0, "history": [], "trained_episodes": 0}
training = {"running": False, "episodes": 0, "message": "Ready"}

def _profile(mode): return {"normal": "train", "random": "random", "congestion": "congestion", "outage": "outage"}.get(mode, "train")

@app.route("/")
def index(): return render_template_string(PAGE, agent_kind=agent_kind)

@app.route("/api/status")
def status():
    with lock:
        return jsonify({"agent": agent_kind, "torch_available": True, "metrics": dict(metrics),
                        "network": live_env.info(), "destinations": list(ACTION_NAMES),
                        "training": dict(training), "simulated_estimates": True})

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
        if training["running"]:
            return jsonify({"error": "Training is already running", "training": dict(training)}), 409
        training_seed = 7 + metrics["trained_episodes"]
        training.update(running=True, episodes=episodes, message="Training in background...")

    def run_training():
        global agent
        try:
            trained_agent, _ = train(episodes=episodes, seed=training_seed, checkpoint=checkpoint, agent=agent)
            with lock:
                agent = trained_agent
                metrics["trained_episodes"] += episodes
                training.update(running=False, episodes=0, message=f"Finished {episodes} episodes")
        except Exception as error:
            with lock:
                training.update(running=False, episodes=0, message=f"Training failed: {error}")

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"accepted": True, "episodes": episodes, "message": training["message"]}), 202

@app.route("/api/reset", methods=["POST"])
def reset():
    with lock:
        metrics.update(requests=0, deadline_misses=0, failures=0, history=[], trained_episodes=metrics["trained_episodes"])
        return jsonify({"ok": True})

PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEC Live Lab</title>
<style>
body{margin:0;background:#0b1020;color:#eaf0ff;font:15px system-ui}
main{max-width:1100px;margin:auto;padding:24px}.hero,.card{background:#121a2e;border:1px solid #263555;border-radius:14px;padding:20px;margin-bottom:16px}
h1{margin:0 0 6px;font-size:30px}h2{margin:0 0 12px;font-size:18px}
.muted{color:#aebcdb}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.buttons{display:flex;flex-wrap:wrap;gap:8px}button{padding:11px 15px;background:#6385f5;color:white;border:0;border-radius:8px;font-weight:700;cursor:pointer}
button:hover{background:#82a0ff}button:disabled{opacity:.5;cursor:wait}.danger{background:#b95768}.pill{display:inline-block;padding:5px 10px;border-radius:99px;background:#263555}
pre{background:#080d19;padding:14px;border-radius:8px;overflow:auto;white-space:pre-wrap;min-height:80px}
#message{min-height:22px;color:#9fe3b1}.error{color:#ff9ba7!important}
label{display:block;margin:10px 0 5px}.number{width:90px;padding:9px;border-radius:6px;border:1px solid #526589;background:#0b1020;color:white}
</style></head>
<body><main>
<section class="hero"><h1>MEC Live Lab</h1>
<p class="muted">Interactive computation-offloading demo. Latency, energy, queues, and network conditions are simulated estimates.</p>
<span class="pill">{{ agent_kind }}</span> <span id="connection" class="pill">Connecting...</span>
<p id="message">Loading dashboard status...</p></section>
<div class="grid">
<section class="card"><h2>1. Choose network condition</h2><p class="muted">This changes the simulated environment for the next task.</p>
<div class="buttons"><button onclick="setNetwork('normal')">Normal</button><button onclick="setNetwork('congestion')">Congestion</button><button onclick="setNetwork('outage')">Outage</button><button onclick="setNetwork('random')">Random sample</button></div></section>
<section class="card"><h2>2. Send a task</h2><p class="muted">Let the DQN decide, or force a destination.</p>
<div class="buttons"><button onclick="sendTask()">Let agent decide</button><button onclick="sendTask('Edge A')">Force Edge A</button><button onclick="sendTask('Edge B')">Force Edge B</button><button onclick="sendTask('Cloud')">Force Cloud</button></div></section>
<section class="card"><h2>3. Train the agent</h2><p class="muted">Training runs in the background. The page remains responsive.</p>
<label for="episodes">Episodes</label><input id="episodes" class="number" type="number" min="20" max="3000" value="300">
<div class="buttons"><button onclick="startTraining()">Start training</button><button class="danger" onclick="resetMetrics()">Reset metrics</button></div></section>
</div>
<section class="card"><h2>Latest task result</h2><pre id="result">No task sent yet.</pre></section>
<section class="card"><h2>Live status</h2><pre id="status">Waiting for the server...</pre></section>
</main>
<script>
const $=id=>document.getElementById(id);
let busy=false;
async function api(path,data){
  const response=await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json'},body:data?JSON.stringify(data):undefined});
  const text=await response.text(); let payload={}; try{payload=text?JSON.parse(text):{}}catch(_){payload={error:text}};
  if(!response.ok) throw Error(payload.error||`Server returned ${response.status}`);
  return payload;
}
function showError(error){$('message').textContent=`Error: ${error.message}`;$('message').className='error';}
function setBusy(value){busy=value;document.querySelectorAll('button').forEach(button=>button.disabled=value);}
async function refresh(){
  try{const data=await api('/api/status');$('status').textContent=JSON.stringify(data,null,2);$('connection').textContent=data.training.running?'Training...':'Connected';$('message').textContent=data.training.message;$('message').className='';}
  catch(error){$('connection').textContent='Disconnected';showError(error);}
}
async function sendTask(destination){
  if(busy)return;setBusy(true);$('message').textContent='Sending simulated task...';
  try{$('result').textContent=JSON.stringify(await api('/api/task',destination?{force_destination:destination}:{}),null,2);await refresh();}
  catch(error){showError(error);}finally{setBusy(false);}
}
async function setNetwork(mode){
  if(busy)return;setBusy(true);$('message').textContent=`Switching to ${mode}...`;
  try{await api('/api/network',{mode});await refresh();}catch(error){showError(error);}finally{setBusy(false);}
}
async function startTraining(){
  if(busy)return;const episodes=Math.max(20,Math.min(3000,Number($('episodes').value)||300));setBusy(true);$('message').textContent='Starting background training...';
  try{const data=await api('/api/train',{episodes});$('result').textContent=JSON.stringify(data,null,2);await refresh();}catch(error){showError(error);}finally{setBusy(false);}
}
async function resetMetrics(){
  if(busy)return;setBusy(true);try{await api('/api/reset',{});$('result').textContent='Metrics reset.';await refresh();}catch(error){showError(error);}finally{setBusy(false);}
}
refresh();setInterval(refresh,3000);
</script></body></html>'''
if __name__ == "__main__":
    print("MEC Live Lab: http://127.0.0.1:5000"); app.run(host="0.0.0.0", port=5000, debug=False)
