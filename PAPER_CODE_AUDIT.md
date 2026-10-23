# Paper-to-Code Audit

## Final verdict

**Code-level alignment: partial and conditional.** The repository implements a
13-state, four-action sequential MEC simulator and a DQN with the documented
network and optimization defaults. Several concrete repository inconsistencies
were corrected in this audit. A full paper-alignment verdict is not possible
because the paper named in the task is not present in this repository, in the
attached snapshot, or among the project files inspected. Therefore the paper's
equations, tables, exact evaluation seed-generation procedure, and reported
metrics remain unverified.

**Numerical reproduction: not established.** Existing `artifacts/`, `runs/`,
and `results_backup/` files were not relabeled or treated as fresh evidence.
The complete five-seed, 500-episode training plus four-profile, 100-episode
evaluation protocol was not rerun during this audit.

## Verified implementation map

| Paper requirement or repository claim | Original implementation | Correction made | Verification evidence |
|---|---|---|---|
| 13-dimensional state and four destinations | `mec_sim.py` defines `STATE_DIM = 13`, four actions, normalized task/network/availability features | None; source paper equations unavailable for independent comparison | Existing simulator tests pass |
| DQN 13 -> 128 -> 128 -> 4, ReLU | `dqn.py` `QNetwork` | None | Constructor inspected; smoke training completed |
| Replay capacity 20,000; batch 64; Adam 0.001; gamma 0.99 | `DQNConfig` | None | Configuration inspected and training smoke test completed |
| Huber loss, target sync 250, gradient clipping 10 | `DQNAgent.learn` | None | Configuration/code inspected |
| Availability masking | Behavior policy masks invalid actions; Bellman target masks successor availability from state features | None | `test_agent_masks_unavailable_action` passes |
| Checkpoint resume | Only model, config, and update count were saved originally; replay and optimizer state were lost | Checkpoints now persist target network, optimizer, replay data, replay RNG, agent RNG, and update count using safe plain-list serialization | `test_checkpoint_restores_training_state` and manual non-empty replay round trip pass |
| Random dashboard state | `app.py` mapped `random` to the training profile | Added a distinct `random` simulator profile with independently sampled bandwidth, queues, and cloud RTT; dashboard now selects it | `test_random_profile_is_not_training_profile` passes |
| Dashboard training isolation | Dashboard used `artifacts/dqn.pt` and seed 7 for every request | Dashboard uses `artifacts/dashboard_dqn.pt` and advances its training seed based on accumulated dashboard episodes | Source inspection; API compatibility test passes |
| README fallback claim | README claimed a tabular fallback although `dqn.py` imports PyTorch unconditionally | README now states PyTorch is required and no fallback exists | Documentation/source consistency inspection |
| Unseen jitter claim | README claimed unseen jitter dynamics were excluded from training, although jitter is stochastic in all profiles | README now describes unseen as a distribution/outage shift and states jitter is present in training too | Source/documentation comparison |
| Five training seeds and 100007 evaluation base | `run_multiseed.py` uses seeds 1--5 and evaluation seeds `100007 + i` | None; exact paper seed-generation rule cannot be checked without the paper | Source inspection only |

## Files changed

- `mec_sim.py`: added an independently sampled `random` profile.
- `dqn.py`: made checkpoints resume optimizer, target, replay, and RNG state;
  retained compatibility with older model-only checkpoints.
- `app.py`: separated dashboard checkpoints from experiment checkpoints,
  advanced dashboard seeds, and wired the random profile.
- `experiment.py`: changed the single-run training default seed to 1, matching
  the first documented training seed; multi-seed orchestration remains in
  `run_multiseed.py`.
- `README.md`: removed the unsupported fallback claim and corrected profile
  and reproducibility descriptions.
- `tests/test_mec.py`: added regression coverage for random profiles and
  checkpoint training-state restoration.
- `PAPER_CODE_AUDIT.md`: this audit record.

## Tests and smoke checks

Executed with the repository virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Result: **9 passed**.

Additional checks:

```powershell
.venv\Scripts\python.exe -c "...save/load a non-empty replay..."
```

Result: **replay checkpoint round trip passed**.

```powershell
.venv\Scripts\python.exe experiment.py train --episodes 2 --seed 1 --checkpoint artifacts\smoke_audit.pt
```

Result: **training smoke test completed and wrote a checkpoint**. This
two-episode run is a validation artifact, not a paper result.

The system Python installation did not have pytest; validation used the
project's existing `.venv`.

## Reproduction commands

From a clean output directory, install `requirements.txt`, then run:

```powershell
.venv\Scripts\python.exe run_multiseed.py
.venv\Scripts\python.exe aggregate_results.py
```

The orchestrator trains seeds 1, 2, 3, 4, and 5 for 500 episodes and evaluates
each checkpoint for 100 episodes on `train`, `congestion`, `unseen`, and
`outage`, using the configured base seed `100007`. It writes run data under
`runs\seed<S>\` and aggregation outputs in the repository root.

For a single smoke or inspection run:

```powershell
.venv\Scripts\python.exe experiment.py train --episodes 500 --seed 1 --checkpoint artifacts\dqn.pt
.venv\Scripts\python.exe experiment.py evaluate --checkpoint artifacts\dqn.pt --episodes 100 --seed 100007 --profile train
```

## Unverified requirements and limitations

1. The authoritative paper file is missing, so exact equations, feature
   semantics/scales, baseline decision logic, discount-factor provenance,
   evaluation seed derivation, and reported-number comparisons cannot be
   certified.
2. Existing generated results were not regenerated in this audit and must not
   be presented as newly reproduced results.
3. The simulator explicitly emits model estimates rather than physical
   latency or energy measurements.
4. The repository's current `run_multiseed.py` performs the requested
   five-seed/four-profile loop, but completion and numerical comparison were
   not rerun here due to the scope and cost of the full experiment.
5. Whether the paper requires a different random-profile definition or a
   different seed derivation remains an unresolved paper-dependent ambiguity.

