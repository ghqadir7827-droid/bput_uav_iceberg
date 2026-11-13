# PHASE 4K — PRE-RUN INTEGRITY & AUDIT REPORT

**Experiment:** Phase 4K — Symmetry-Standoff Root-Cause Audit and Isolated Curriculum Experiment  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Directory:** `experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/`  
**Execution Timestamp:** 2026-10-29T21:02:00+05:30  
**Auditor:** Senior Reinforcement Learning Engineer, Autonomous UAV Navigation Researcher, Simulation-Validation Auditor  

---

## 1. Experimental Mission & Objectives

Phase 4J-G definitively demonstrated that the current Behavior Cloning (BC) baseline controller (`bc_policy_best.pt`) is **not** time-budget starved. Extending the episode horizon from 500 steps ($25.0\text{ s}$) to 600 steps ($30.0\text{ s}$) produced zero additional goal successes ($24/50 \to 24/50$), with net forward progress across the 26 failed seeds averaging only $+5.7\text{ mm}$ over 100 extra steps.

The primary failure mode is **persistent standoff before centered obstacles**, where the policy outputs near-zero lateral velocity ($v_y \approx 0.0\text{ m/s}$) while requesting forward velocity ($v_x \approx +0.65\text{ m/s}$), causing the canonical safety supervisor to safely clamp forward motion ($v_x \to 0.10\text{ m/s}$).

**Phase 4K Mission:**
1. Investigate the causal root of this symmetry-induced standoff across training data, observation representations, expert behavior, and network architectures.
2. Determine whether an isolated curriculum-based policy candidate is justified.
3. Prioritize rigorous causal diagnosis over blind model training.

---

## 2. Mandatory Constraints & Compliance Checklist

- [x] **Canonical Benchmark Preserved:** Episode horizon remains strictly 500 steps ($25.0\text{ s}$ at $\Delta t = 0.05\text{ s}$).
- [x] **Canonical Safety Supervisor Unmodified:** `controllers/safety_supervisor.py` preserved (SHA-256 verified).
- [x] **Baseline Checkpoint Unmodified:** `models/checkpoints/bc_policy_best.pt` preserved (SHA-256 verified).
- [x] **Zero Overwriting of Historical Artifacts:** Phase 4J reports, manifests, CSVs, and traces preserved.
- [x] **Canonical Contracts Unaltered:** Observation space (23 floats) and action space ($[-1, 1]^4$) remain strictly unchanged.
- [x] **Held-Out Test Set Untouched:** Seeds `90000–90099` strictly sealed; no training, tuning, or candidate selection on held-out seeds.
- [x] **Zero Hardware Commands:** No physical flight commands or communication with RadioLink flight controller.
- [x] **Safety Thresholds Invariant:** Emergency braking ($0.30\text{ m}$), retreat ($0.15\text{ m}$), and boundary buffer ($0.25\text{ m}$) remain invariant.
- [x] **Isolation:** All new code, datasets, checkpoints, logs, and reports isolated in `experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/`.

---

## 3. Cryptographic Provenance

| Component | Path | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **BC Baseline Model** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **PRESERVED** |
| **Canonical Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **PRESERVED** |
| **PPO Baseline Checkpoint** | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **UNMODIFIED** |
| **Held-Out Test Set** | Seeds `90000–90099` | Sealed and untouched. | **SEALED** |

---

## 4. Verified Baseline Benchmark Metrics (to Preserve)

The verified performance of `models/checkpoints/bc_policy_best.pt` on development seeds `2000–2049` under $d_{\text{caut}} = 0.50\text{ m}$:
- **Success Rate:** 24 / 50 (48.0%, 95% Wilson CI [34.8%, 61.5%])
- **Obstacle Collisions:** 0
- **Boundary Collisions:** 0
- **Obstacle Envelope Violations (<0.08m):** 0
- **Boundary Envelope Violations (<0.08m):** 0
- **Absolute Min Obstacle Clearance:** $0.1070\text{ m}$
- **Absolute Min Boundary Clearance:** $0.2537\text{ m}$
- **Mean Forward Progress:** $6.525\text{ m}$
- **Episode Duration:** $22.99\text{ s}$

---

## 5. Execution Protocol

1. **Experiment A:** Source and Training-Data Audit (examine `models/dataset.py`, `models/train_bc.py`, `data/demonstrations/bc_train_10m.npz`, and generation scripts).
2. **Experiment B:** Controlled Symmetry Standoff Reproduction and Diagnostics.
3. **Experiment C:** Avoidance Curriculum Design.
4. **Experiment D:** Isolated Training Pilot (only if justified by the audit).
5. **Experiment E:** Controlled Development Evaluation (seeds `2000–2049`).
6. **Full Deliverables and Promotion Audit.**

