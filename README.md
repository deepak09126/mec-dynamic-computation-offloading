# MEC Live Lab — DRL Computation Offloading Prototype

A phone-accessible, interactive research prototype for studying computation offloading under stochastic network conditions.

## What is real and what is simulated?

- **Real:** responsive web application, HTTP requests from the phone, controller logic, backend CPU workload, checksum output, and request logging.
- **Simulated:** mobile uplink bandwidth, edge/cloud network delays, queue lengths, estimated energy, and end-to-end MEC latency.
- **Learning:** uses a compact DQN when PyTorch is installed. Without PyTorch, it falls back to a tabular Q-learning baseline; the UI/API reports which agent is active.

This is an educational prototype, not a physical 5G/MEC benchmark. It must not be described as a real-world network latency measurement.

## Run on your computer

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Open it on your Android phone

1. Connect the phone and computer to the same trusted Wi-Fi network.
2. Find the computer's local IPv4 address:
   - Windows: run `ipconfig` and look for the Wi-Fi adapter's IPv4 address.
   - macOS/Linux: use your network settings or `ip addr`.
3. On the phone, open `http://COMPUTER-IP:5000`, replacing `COMPUTER-IP` with the local address.
4. If it cannot connect, check that the computer firewall allows Python on your private network. Do not expose this development server directly to the public internet.

Example only: `http://192.168.1.20:5000`

## Demo sequence

1. Send task & let agent decide.
2. Switch to Congestion or Link stress.
3. Send more tasks and observe decisions and deadline misses.
4. Train +300 episodes.
5. Compare with forced Edge A/Edge B actions.
6. Explain the difference between simulated MEC latency and measured backend workload time.

## API

- `GET /api/status` — agent, network state, metrics
- `POST /api/task` — submit a task; optional `{"force_destination":"Edge A"}`
- `POST /api/network` — `{"mode":"normal"}`, `congestion`, `outage`, or `random`
- `POST /api/train` — `{"episodes":300}`
- `POST /api/reset` — clear request metrics

## Research integrity / next steps

## Reproducible sequential DQN workflow (Windows)

```powershell
.venv\Scripts\activate
python experiment.py train --episodes 500 --seed 7 --checkpoint artifacts\dqn.pt
python experiment.py evaluate --checkpoint artifacts\dqn.pt --episodes 100 --seed 100007 --profile train
python experiment.py evaluate --checkpoint artifacts\dqn.pt --episodes 100 --seed 100007 --profile congestion
python experiment.py evaluate --checkpoint artifacts\dqn.pt --episodes 100 --seed 100007 --profile outage
python experiment.py evaluate --checkpoint artifacts\dqn.pt --episodes 100 --seed 100007 --profile unseen
python -m pytest -q
python app.py
```

To run the five independent training seeds and evaluate every seed on all four
profiles, then aggregate the results into paper-ready tables and figures:

```powershell
python run_multiseed.py
python aggregate_results.py
```

The multi-seed command writes per-run JSON/CSV files under
`runs/seed<S>/<profile>/`. The aggregation command reads those files and
writes `results_tables.tex`, `fig_latency.png`, `fig_energy.png`,
`fig_miss.png`, and `fig_gap.png`. Use `python aggregate_results.py
--placeholder` when preparing a paper before real runs are available.

The simulator is a sequential MDP. Its 13-value state includes normalized task size and CPU cycles, deadline pressure, battery level, local CPU availability, uplink bandwidth, jitter, Edge A/Edge B queue loads, cloud RTT, and Edge A/Edge B/cloud availability. Actions are local execution, Edge A, Edge B, and cloud. Every transition is recorded as `(state, action, reward, next_state, done)` and remote availability masks unavailable actions for policies.

Reward is dimensionless: `-(latency_ms / 150 + 0.35 * energy_mJ / 100 + 2 * deadline_miss + 4 * failed_offload)`. Latency is a simulated ms estimate, energy is a simulated mJ estimate, 150 ms is the latency scaling reference, and 100 mJ is the energy scaling reference. The weights and penalties are configurable simulation assumptions, not calibrated measurements.

Training uses only the `train` profile. Evaluation begins at a disjoint seed range; all evaluated policies receive identical episode seeds. `congestion`, `outage`, and `unseen` are stress tests, with `unseen` including bandwidth/jitter/outage dynamics excluded from training. The commands save model checkpoints and machine-readable JSON/CSV results under `artifacts/`. Reports list average latency, estimated energy, deadline-miss rate, task-failure rate, and cumulative reward. They are simulated estimates and do not establish physical performance, novelty, or guaranteed superiority.

Do not claim novelty merely because this app combines a phone interface and DRL. Related work already studies DRL-based MEC offloading and digital-twin-assisted learning. The contribution should be established through a literature review and a clearly defined experiment, for example evaluating adaptation to previously unseen burst/outage patterns, with DQN and non-learning baselines, multiple random seeds, confidence intervals, and ablation studies.

Before using results in a paper:
- Validate the delay/energy model and document every assumption.
- Add tests for the simulator and learning loop.
- Run repeated, seeded experiments.
- Separate training conditions from unseen test conditions.
- Report actual outcomes only; do not fabricate result tables.
- Consider replacing the toy model with a calibrated simulator or a real edge node for stronger claims.

## Suggested repository description

"Interactive phone-accessible MEC offloading testbed with a virtual stochastic network, edge/cloud destinations, a DQN controller, and reproducible experiment endpoints."
