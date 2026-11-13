# PHASE 4J-G — PRE-RUN INTEGRITY & EXPERIMENT AUDIT

**Experiment:** Phase 4J-G — Horizon Sensitivity and Policy-Supervisor Compatibility Audit  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Directory:** `experiments/phase4j/phase4jg_horizon_control_audit_20261029_192000/`  
**Execution Timestamp:** 2026-10-29T19:20:00+05:30  
**Auditor:** Senior Reinforcement Learning Engineer, Simulation-Validation Auditor  

---

## 1. Experimental Mission & Objectives

The primary objective of Phase 4J-G is to resolve the fundamental architectural question identified across Phases 4J-E and 4J-F:
**Is the current Behavior Cloning (BC) navigation system primarily limited by its fixed episode time budget (25.0s / 500 steps), by inefficient obstacle navigation (policy/control-limited), or by conflicts between policy commands and supervisor interventions?**

### Context from Previous Phases:
1. **Phase 4J-E (PPO Audit & Controlled Comparison):**
   - Established that on development seeds `2000–2049` under $d_{\text{caut}} = 0.50\text{ m}$, the frozen BC baseline achieved **24/50 (48.0%)** goal completions, substantially outperforming PPO baseline (**6/50 (12.0%)**, McNemar $p = 0.000534$).
   - Revealed that the primary failure mode of BC was timeouts (safe hover/standoff near obstacles).
2. **Phase 4J-F (Proactive Lateral Guidance Evaluation):**
   - Implemented an isolated proactive lateral guidance supervisor to break centered-obstacle deadlocks.
   - Evaluated 50 paired episodes on seeds `2000–2049`.
   - Resulted in **Conclusion B (No meaningful improvement)**: Baseline achieved 24/50, Lateral Guidance achieved 23/50 ($p = 1.000000$).
   - Key finding: While lateral guidance eliminated standoff at Pillar 1 (reducing Pillar 1 stalls from 6 to 4 and increasing median progress from $6.324\text{ m}$ to $7.199\text{ m}$), the lateral detours consumed $1.0–2.0\text{ s}$ per obstacle. The delays accumulated and pushed downstream obstacles into timeout under the 500-step horizon. Furthermore, an adversarial dynamic tension was identified between BC restoring commands and supervisor lateral overrides.

---

## 2. Cryptographic Integrity Audit

All core files, models, and environments have been hashed and verified prior to running any experiment:

| Component | Path | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **BC Checkpoint** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **PRESERVED** |
| **Canonical Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **PRESERVED** |
| **PPO Baseline Checkpoint** | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **UNMODIFIED** |
| **Reserved Held-Out Dataset** | Seeds `90000–90099` | Sealed and untouched. Zero runs executed on held-out seeds. | **SEALED** |

---

## 3. Environment, Sensor, and Dynamics Specifications

- **Corridor Benchmark:** 10m canonical 3-pillar corridor
  - Dimensions: $x \in [0.0, 10.0]\text{ m}$, width $2.0\text{ m}$ ($y \in [-1.0, +1.0]\text{ m}$), height $2.0\text{ m}$ ($z \in [0.0, 2.0]\text{ m}$).
  - Drone Radius: $r_{\text{drone}} = 0.15\text{ m}$.
  - Safety Envelope Margin: $0.080\text{ m}$.
- **Pillars:** Three cylindrical obstacles of diameter $0.30\text{ m}$ placed at $x \approx 3.0\text{ m}$, $5.0\text{ m}$, $7.0\text{ m}$.
- **Goal Criteria:** Distance to $[9.5, 0.0, 1.0]\text{ m} \le 0.30\text{ m}$ with terminal velocity $\|\mathbf{v}\| \le 0.35\text{ m/s}$.
- **Simulation Timestep:** $\Delta t = 0.05\text{ s}$ ($20\text{ Hz}$).
- **Observation Space:** 23-dimensional deployable vector:
  - $[0:12]$: 12 rangefinder rays normalized by $4.0\text{ m}$.
  - $[12:15]$: Global position $[x, y, z]$.
  - $[15:18]$: Linear velocity $[v_x, v_y, v_z]$.
  - $[18:22]$: Orientation quaternion $[q_x, q_y, q_z, q_w]$.
  - $[22]$: Normalized goal distance.
- **Action Space:** 4-dimensional normalized action in $[-1, 1]^4$, mapped to:
  - $v_x \in [0.0, 0.80]\text{ m/s}$
  - $v_y \in [-0.50, +0.50]\text{ m/s}$
  - $v_z \in [-0.30, +0.30]\text{ m/s}$
  - $\dot{\psi} \in [-0.7854, +0.7854]\text{ rad/s}$ ($[-45^\circ, +45^\circ]/\text{s}$).

---

## 4. Experimental Plan

### Experiment A: Isolated Horizon Sensitivity Sweep
- Compare:
  - **Configuration A:** 500 steps ($25.0\text{ s}$)
  - **Configuration B:** 600 steps ($30.0\text{ s}$)
- Seeds: `2000–2049` (50 paired development episodes).
- Critical Paired Determinism Requirement: The initial 500 steps of each 600-step episode must identically match the 500-step episode. Any divergence must be verified and confirmed zero.
- Primary Questions:
  - How many additional episodes complete between steps 501 and 600?
  - Does the longer horizon introduce any new collisions or envelope violations?
  - For episodes that still fail at step 600, where and why do they fail?

### Experiment B: Failure and Time-Budget Classification
- Classify all failed episodes across the standardized 8-class taxonomy:
  1. Insufficient remaining time after safe obstacle avoidance (productive progress until cutoff)
  2. Persistent centered-obstacle standoff (velocity clamped at obstacle)
  3. Repeated braking or retreat
  4. Lateral oscillation
  5. Inefficient or excessive lateral detour
  6. Boundary-induced stall
  7. Goal alignment or terminal-state failure
  8. Other verified failure mode

### Experiment C: Policy-Supervisor Compatibility Audit
- Quantify interaction between raw policy actions and supervisor filtering:
  - Lateral command sign agreement vs opposition ($\text{sign}(v_{y, \text{raw}}) \neq \text{sign}(v_{y, \text{safe}})$).
  - Magnitude of opposing commands.
  - Duration and correlation with progress stalls.
  - Evaluation of the training data distribution hypothesis.

---

## 5. Pre-Run Compliance Check

- [x] Canonical supervisor `controllers/safety_supervisor.py` is unmodified.
- [x] Model weights are frozen and verified by SHA-256.
- [x] Held-out seeds `90000–90099` are untouched and protected.
- [x] No physical flight commands or communication enabled.
- [x] Pytest suite passes 138/138 tests.

