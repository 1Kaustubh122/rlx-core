# Reinforcement Learning Toolkit (RLTK)

**Production-grade RL for industrial autonomy. UNDER ACTIVE DEVELOPMENT**
Train in **JAX/Flax/Optax**, export to **ONNX**, run **deterministically in C++20** with safety guards. Designed to plug into real robots and machines alongside classical control and sensor-fusion stacks.

---

## Why RLTK

Most RL repos stop at toy sims. RLTK is built for **factory floors, yards, ports, mines, and agri**:

* **IO-agnostic contract:** numeric observations in, numeric actions out.
* **Deterministic C++ runtime:** fixed tick, no heap after init, WCET budgets.
* **Safety at runtime:** action clamps, rate/jerk limits, novelty/timing monitors, instant fallback signals.
* **Portable deployment:** ROS2, industrial PCs, and—with proper optimization—PLC/embedded integration via C ABI.
* **Auditability:** ablation configs, seeds, metrics, artifacts, ONNX + `metadata.json` contract.

---

## Architecture

**Training (Python/JAX)** -> **Export (ONNX + metadata)** -> **Runtime (C++20)** -> **Adapters (ROS2/PLC/embedded)**

* **Training stack:** JAX/Flax/Optax + Hydra configs + W\&B tracking.
* **Export layer:** frozen params -> ONNX, with strict op set and numerics parity tests.
* **Runtime:** lightweight C++20 inference loop with safety hooks and deterministic scheduling.
* **Operational modes:** Shadow / Residual / Primary / Cooperative.
* **Integration:** thin adapters feed obs/actions to ICTK/SFTK pipelines.

---

## Capabilities

* **Agents (JAX):** modular policies, losses, buffers, model-based rollouts.
* **Experiment engine:** Hydra configs, grid/sweep runners, seeded determinism, W\&B artifacts.
* **Data & logging:** episode stores, KPI reporters, exportable artifacts.
* **Export & parity:** ONNX export, Python↔C++ inference parity tests (tolerance configurable).
* **Runtime safety:** clamps, rate/jerk limits, novelty/timing monitors, fallback trip signals.
* **Modes:**

  * **Shadow:** observe only, log counterfactual actions.
  * **Residual:** add bounded residual on top of classical controller.
  * **Primary:** direct control within safety envelope.
  * **Cooperative:** blend with classical policy via scheduler.

---

## Algorithms & Roadmap (Phases)

**Legacy baseline (complete/archival):**

1. **Tabular:** Bandits, MDP planning, MC, TD, n-step.
2. **Function Approximation:** NN-Q, NN-SARSA, NN n-step Q.
3. **DQN family:** Vanilla, Double, Dueling, PER, Rainbow.
4. **Policy Gradients:** REINFORCE, A2C, TRPO, PPO.
5. **Actor-Critic:** TD3, SAC, REDQ.
#### *Phase 6+ (current and forward, production-grade):*
6. **Model-Based RL:** MBPO, PETS, PlaNet.
7. **World Models:** Dreamer (V1/V2/V3).
8. **Latent/World-Model Methods:** MuZero, SimPLe.
9. **Meta-RL:** MAML, RL².
10. **Hierarchical RL:** Options, Feudal variants.
11. **Offline/IL:** BC, DAgger, CQL, BRAC.
12. **Multi-Agent:** I-DQN, QMIX, MADDPG.
13. **Exploration:** ICM, RND, NGU, Go-Explore.
14. **Inverse/Imitation:** GAIL, AIRL, SQIL.
15. **Transformer RL:** Decision Transformer, Trajectory Transformer, Gato-style.
16. **Robust Sim2Real:** Domain Randomization, EPOpt, Adversarial RL.
17. **Real-Time RL:** latency-bounded inference, scheduler integration.
18. **Multi-Task/Transfer:** shared backbones, adapters, fine-tune heads.
19. **Safe/Risk-Sensitive:** constrained RL, risk-aware PPO/SAC.

> Phases 1–5 remain as **archival references**. Phase 6+ is the **active, production-grade** track.

---

## Operational KPIs (gates)

A model is **deployable** only if it meets:

* **Parity:** Python vs C++ outputs match within `≤1e-5` L∞ for fixed test cases.
* **Latency:** p95 inference `≤2 ms` on reference hardware.
* **Safety:** no observed clamp/rate violations in validation suites; novelty alarms within budget.
* **Stability:** zero runtime allocations after init; no missed ticks at target Hz.
* **KPI uplift:** beats classical baseline on scenario KPIs (e.g., autonomy ratio, lateral RMS, stop distance, disengagements/hour, throughput delta).

---

## Deployment Model Contract

Artifacts per trained policy:

* `model.onnx` — fixed opset, static shapes preferred.
* `metadata.json` — minimal schema:

```json
{
  "abi_version": "rltk-v1",
  "obs_dim": 24,
  "act_dim": 4,
  "obs_norm": {"mean": [...], "std": [...]},
  "act_bounds": {"min": [...], "max": [...]},
  "tick_hz": 100.0,
  "preprocess": ["clip:obs:-10,10", "scale:act:-1,1"],
  "training_commit": "abcdef1234",
  "onnx_sha256": "…",
  "kpi_ref": "reports/run_2025-08-28.json",
  "license": "MIT"
}
```

**Runtime guarantees:**

* Fixed-step scheduler at `tick_hz`, bounded WCET.
* No heap allocs after init; pre-allocated workspaces.
* Safety hooks: clamp -> rate/jerk -> novelty -> fallback signal cascade.

**PLC/embedded note:** deployment is **inference-only**. With small nets and static shapes, ONNX or generated kernels can meet real-time budgets on industrial PCs and many modern controllers. Integration via C ABI.

---

## Tech Stack

* **Train:** Python 3.10+, JAX/Flax/Optax, Hydra, W\&B.
* **Sim:** Isaac Sim / Isaac Lab for industrial scenes.
* **Export:** ONNX.
* **Runtime:** C++20, ONNX Runtime, C ABI for integration.
* **Adapters:** ROS2 nodes; PLC/embedded bridges where applicable.

---



## Safety & Determinism (Runtime)

* **Init:** load ONNX, allocate all buffers, bind threads, warm up.
* **Tick:** acquire obs -> normalize -> infer -> clamp -> rate/jerk limit -> emit action.
* **Monitors:** deadline misses, novelty scores, out-of-range inputs.
* **Fallback:** on violation, raise signal for ICTK controller takeover.
* **WCET harness:** included tests run warm/cold cycles and write CSVs.

---

## Status & Milestones

* **Mid Sept:** Base runtime + export + parity + safety hooks ready.
* **Sept 30:** On-vehicle validation (golf cart) with Shadow/Residual modes.

---

## Contributing

Focused roadmap. PRs welcome for: determinism, export tooling, runtime safety, adapters, tests. Follow `CONTRIBUTING.md`.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgments

Built to operate alongside [**ICTK**](https://github.com/1Kaustubh122/industrial-control-toolkit) and [**SFTK**](https://github.com/1Kaustubh122/sensor-fusion-toolkit)  with thin ROS2/PLC adapters for end-to-end autonomy.

---
