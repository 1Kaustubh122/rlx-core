# rlx-core

> Modular, production-ready Reinforcement Learning stack for robotics and factory automation. Built to scale from research to real factory floors—no shortcuts, no toy code, full control.

---

## 🧠 Overview

`rlx-core` is my ground-up, multi-phase RL framework for industrial robotics. The repo is split into clear phases—from basic RL to world models and sim2real pipelines.  
Phases 1–5 are foundational, focused on speed and learning. Phase 6 onward: everything gets rebuilt for **production and deployment** at real-world scale.

---

## ⚡️ Tech Stack

- **Phases 1–5:**  
  - Custom GridWorlds, DeepMind Control Suite, PyTorch.  
  - Fast prototyping, no heavy frameworks.  
  - Good enough for getting RL right, not pretending to be "enterprise" before it matters.

- **Phase 6+ (Current and Future):**  
  - **JAX (Flax/Haiku/Optax):** State-of-the-art RL/ML stack for high performance and composability.
  - **Hydra:** Configs and ablations—everything is controlled and reproducible.
  - **Weights & Biases:** Experiment tracking, metrics, artifacts—no more black-box runs.
  - **Isaac Sim (+ Isaac Lab for rapid prototyping):** Realistic factory/cell/robot simulation, built for sim2real.
  - **ONNX + C++/ONNX Runtime:** Deployment outside Python; real-time inference for robot controllers and factory systems.
  - **CI/CD, Docker, Regression Tests:** Everything tested and reproducible from the start.

---

## ⚠️ Stack Transition Notice

**Phases 1–5:**  
- Legacy stack. Custom environments and PyTorch for rapid learning and prototyping.
- Not productionized, not modular, not suitable for real-world deployment as-is.

**Phase 6 onward:**  
- Full-stack, modular, ablation-ready, deployment-focused.  
- If you want to see the difference, compare any agent/phase before and after.

*This is deliberate. I don’t waste time rewriting history—my focus is on forward progress and system-level execution. No backfilling, no fake consistency.*

---

## 🗂️ Repository Structure

```text
rlx-core/
│
├── phase_01_tabular/                # Bandits, MDP, MC, TD, N_Step, Planning
├── phase_02_function_approx/        # NN_Q_learningm, NN_SARSA, NN_N_Step_Q
├── phase_03_dqn/                    # Vanilla DQN, Double DQN, Dueling DQN, PER DQN, Rainbow DQN
├── phase_04_pg/                     # REINFORCE, A2C, TRPO, PPO
├── phase_05_actor_critic/           # TD3, SAC, REDQ
*Will be added soon from here*
├── phase_06_model_based/            # Dyna-Q FA, MBPO, PETS, PlaNet
├── phase_07_dreamer/                # DreamerV1, V2, V3
├── phase_08_world_models/           # MuZero, SimPLe
├── phase_09_meta_rl/                # MAML, RL²
├── phase_10_hierarchical_rl/        # Options, Feudal Networks
├── phase_11_offline_rl/             # BC, DAgger, CQL, BRAC
├── phase_12_multi_agent_rl/         # I-DQN, QMIX, MADDPG
├── phase_13_exploration/            # ICM, RND, NGU, Go-Explore
├── phase_14_inverse_rl_imitation/   # GAIL, AIRL, SQIL
├── phase_15_transformer_rl/         # Decision Transformer, Gato, Trajectory Transformer
├── phase_16_robust_sim2real_rl/     # Domain Randomization, EPOpt, Robust Adversarial RL
├── phase_17_real_time_rl/           # Real-time policy inference/deployment
├── phase_18_multi_task_transfer/    # Multi-task, transfer architectures
├── phase_19_safe_risk_sensitive_rl/ # Constrained RL, risk-sensitive PPO/SAC
│
├── core/                            # Agent, env, buffer, wrappers, utils
├── configs/                         # Hydra configs for all ablations/experiments
├── scripts/                         # Training, evaluation, deployment scripts
├── deploy/                          # ONNX models, C++ inference, Dockerfiles
├── docs/                            # Papers, diagrams, design notes
```
---

## 🚀 Features
*Will be added soon*
* **Everything modular:** Agents, envs, replay buffers, loss functions—swap and ablate with configs, not code hacks.
* **Production logging:** Every run, config, and artifact is logged and tracked (W\&B).
* **Isaac Sim native:** Ready for direct deployment in Nvidia’s industrial sim pipeline. ROS2 bridgeable, real-world tasks supported.
* **ONNX export + C++ runtime:** All policies exportable for deployment in real systems—no Python bottleneck, no excuses.
* **CI/CD enforced:** Every module and deployment step is tested—if it breaks, the pipeline fails.
* **Sim2Real Ready:** Sim environments support domain randomization, sensor/actuator noise, and policy transfer.

---

## 🏁 Getting Started

### 1. **Requirements**

* Python 3.9+
* JAX, Flax, Optax, Hydra, W\&B (see `requirements.txt`)
* Isaac Sim (see Nvidia docs)
* ONNX, onnxruntime (Python + C++ for deployment)
* Docker (for containerized deployment)
* GPU (strongly recommended for Isaac Sim and JAX RL)

### 2. **Setup**

```bash
# Python deps
conda create -n rlx-core python=3.10
conda activate rlx-core
pip install -r requirements.txt
# Isaac Sim install—follow Nvidia’s [official docs](https://docs.nvidia.com/isaac/isaac-sim/latest/installation.html)
```

### 3. **Training / Experiments**
*Will be added soon*
* See `scripts/` for launchers and ablation runners.
* Example:

  ```bash
  python scripts/train.py --config-name=mbpo.yaml
  ```
* W\&B project link: *Will be added soob*

### 4. **Deployment**
*Will be added soon*
* See `deploy/` for ONNX models and C++ inference code.
* Real-world deployment via Docker containers and ONNX Runtime API.

---

## 📚 Documentation
*Will be added soon*
* All canonical papers (PPO, DreamerV3, MBPO, MuZero, etc.) in `docs/papers/`
* Architecture diagrams and execution flows in `docs/diagrams/`
* Design notes and ablation results in `docs/`

---

## 📢 Roadmap

* **Phases 1–5:** Locked. No further upgrades or refactoring.
* **Phase 6+:** Ongoing.

  * Model-based RL: MBPO, PETS, PlaNet, Dreamer (JAX).
  * Full integration with Isaac Sim and real-world deployment pipeline.
  * Sim2real, policy export, ONNX/C++ integration, ablation sweeps, and CI/CD coverage.

---

## 👷 Contributing

This repo is solo-built and tightly controlled—pull requests only considered for non-trivial improvements or major bugfixes. If you want to use/extend, fork away.

---

## 📜 License

MIT License. Free for any use with attribution. Commercial support possible—contact for details.

---

**I build RL for real robots and factories.
If you’re tired of toy benchmarks and academic bloat, start here.**

