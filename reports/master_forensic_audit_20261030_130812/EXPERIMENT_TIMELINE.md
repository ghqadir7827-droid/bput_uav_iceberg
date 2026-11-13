# EXPERIMENT TIMELINE & HISTORICAL MILESTONE AUDIT
**Audit Artifact ID:** `reports/master_forensic_audit_20261030_130812/EXPERIMENT_TIMELINE.md`  
**Timestamp:** 2026-10-30T13:27:00+05:30  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Verification Level:** VERIFIED (Artifact Hashes, Git Commits, Run Logs, CSV Metrics)

---

## 1. Chronological Timeline Summary

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              PROJECT MILESTONES & PHASES                                                |
+-------------------------------------------------------------------------------------------------------------------------+
| Milestone 1 (Initial Setup)       | Simulation engine, 32-beam 2D LIDAR, 41-dim observation, 4-dim action space        |
| Milestone 2 (BC Baseline)         | Expert demonstration generation, BC MLP training -> Models/checkpoints/bc_policy_best.pt|
| Milestone 3 (PPO RL Intro)        | PPO actor-critic implementation, initial reward shaping for corridor tracking      |
| Milestone 4 (Collision Crisis)    | Discovered 28-50% crash rate in unshielded PPO; identified 14 crash seeds (200-224)  |
| Phase 4A-4D (Supervisor Hardening)| Formalized dual-envelope safety supervisor (d_caut=0.50m, d_stop=0.08m)              |
| Phase 4H (Targeted Data Curation) | Expert trajectories collected on hard negative crash seeds                          |
| Phase 4J-D (Canonical Baseline)   | Established 50-seed dev benchmark (2000-2049): BC baseline achieved 48.0% (24/50)      |
| Phase 4K (Curriculum Failure)     | Curriculum retrained model collapsed: 1/50 success, 1 collision, 19 envelope breaches |
| Phase 5A (Symmetry Diagnostic)    | Identified BC standoff freeze; diagnosed 7 scenarios; validated Stateless Guidance A  |
| Phase 5B (Stateless Validation)   | Dev benchmark evaluated with Stateless Guidance: jumped to 70.0% (35/50), 0 crashes  |
| Phase 5C (Persistent & Re-center) | FSM-based persistence and corridor re-centering: achieved 80.0% (40/50)              |
| Phase 6 / 6A (Production Guidance)| Tuned Stateless Clearance Guidance integrated: reached PEAK 84.0% (42/50), 0 crashes |
| Phase 6B (Regression & Robustness)| Sensitivity sweeps, latency tolerance, paired verification confirmed stable 84.0%   |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Phase-by-Phase Detailed Historical Investigation

### Milestone 1: Simulation Framework & System Contracts
- **Date:** 2026-10-15
- **Core Hypothesis:** A lightweight 20 Hz point-mass kinematic simulation with 32-beam horizontal planar raycasting can train reactive obstacle-avoidance policies for GPS-denied UAV navigation.
- **Implementation:**
  - `simulation/environment.py`: Implemented 500-step max episode horizon, 20m x 2.4m corridor, random cylindrical/spherical obstacles.
  - `simulation/sensors.py`: Implemented 32-beam planar raycaster ($360^\circ$ coverage, $5.0$m max range) and 9-dim odometry state.
  - Observation space: 41-dim vector ($32$ rays $+ 3$ goal $+ 3$ velocity $+ 3$ orientation). Action space: 4-dim velocity $[v_x, v_y, v_z, \dot{\psi}]$.
- **Evaluation:** Unit tests and initial trajectory visualizations.
- **Verdict:** Accepted as canonical simulation interface.

---

### Milestone 2: Expert Demonstrations & Behavior Cloning Baseline
- **Date:** 2026-10-18
- **Core Hypothesis:** An optimal potential-field expert controller can generate smooth corridor-clearing demonstrations to train an MLP policy via supervised learning.
- **Implementation:**
  - `controllers/expert_controller.py`: Potential-field guidance using attractive goal forces and repulsive obstacle fields.
  - `models/train_bc.py`: 3-layer MLP policy ($256 \to 128 \to 64$).
  - Produced checkpoint: `models/checkpoints/bc_policy_best.pt` (SHA-256: `93c4d7d1...`).
- **Evaluation:** Evaluated on simple, low-density obstacle configurations.
- **Results:** Achieved $>80\%$ success in wide, empty corridors, but exhibited oscillation in tight geometries.
- **Verdict:** Baseline established.

---

### Milestone 3: PPO Exploration & Reward Shaping
- **Date:** 2026-10-21
- **Core Hypothesis:** Reinforcement learning via PPO can learn non-linear avoidance strategies superior to supervised imitation learning.
- **Implementation:**
  - `models/ppo_policy.py`: ActorCritic architecture.
  - `models/train_ppo.py`: Clipped surrogate objective ($\epsilon=0.2$), GAE ($\lambda=0.95$), reward function with forward velocity bonus ($+1.0 v_x$), obstacle clearance reward, and terminal collision penalty ($-50.0$).
- **Evaluation:** Initial training on mixed random seeds.
- **Observations:** PPO training showed rapid reward growth, appearing to reach $72\%$ corridor completion.
- **Verdict:** Promising on surface; promoted to intensive collision audit.

---

### Milestone 4: The Collision Crisis & Hard Negative Discovery
- **Date:** 2026-10-24
- **Core Forensic Discovery:** A rigorous audit revealed that the reported "72% success" in Milestone 3 was achieved with an unthrottled, unshielded supervisor that permitted severe physical crashes.
- **Audit Findings:**
  - When collisions were strictly checked with the physical $0.08$m vehicle radius, PPO suffered a catastrophic **28% to 50% crash rate**.
  - A systematic seed sweep over seeds `200–224` ($N=25$) identified exactly **14 fatal crash seeds**: `[200, 203, 204, 206, 207, 210, 213, 215, 216, 217, 218, 220, 221, 224]`.
  - When the canonical safety supervisor was applied to prevent crashes, PPO's success collapsed to **12.0% (3/25)** because the agent had learned aggressive forward plunge maneuvers that the supervisor violently throttled.
- **Remediation Action:** The 14 crash seeds were designated as the canonical **Hard Negative Pool**.

---

### Phase 4A – 4J: Safety Supervisor Enforcement & Canonical Benchmarking
- **Date:** 2026-10-26 to 2026-10-28
- **Core Hypothesis:** Enforcing a rigorous, deterministic safety envelope ($d_{\text{stop}}=0.08$m, $d_{\text{caut}}=0.50$m) in `controllers/safety_supervisor.py` will guarantee zero collisions while isolating policy guidance deficits.
- **Protocol Establishment:**
  - Partitioned seed pools:
    - Diagnostic: `1000–1009` ($N=10$)
    - Development Benchmark: `2000–2049` ($N=50$)
    - Held-Out Evaluation: `90000–90099` ($N=100$)
- **Baseline Evaluation (Phase 4J-D):**
  - Evaluated canonical `bc_policy_best.pt` on the 50-seed Development Benchmark (`2000–2049`):
    - **Success Rate:** **48.0% (24/50)** [95% Wilson CI: 34.8% – 61.5%]
    - **Collision Count:** **0** (0.0%)
    - **Clearance Envelope Violations:** **0** (0.0%)
    - **Timeouts:** **26 / 50 (52.0%)**
  - Evaluated on Held-Out Test Set (`90000–90099`, $N=100$):
    - **Success Rate:** **47.0% (47/100)** [95% Wilson CI: 37.5% – 56.7%]
    - **Collision Count:** **0** (0.0%)
- **Forensic Diagnosis:** The policy NEVER collided under the safety supervisor. Instead, 52% of runs died from **standoff timeouts**. In front of centered obstacles, the BC policy predicted zero lateral velocity ($v_y \approx 0$). The supervisor braked forward velocity ($v_x \to 0$) to prevent collision. The drone hovered motionless until step 500 expired.

---

### Phase 4K: The Curriculum Retraining Catastrophe
- **Date:** 2026-10-28
- **Core Hypothesis:** Retraining the BC policy on an oversampled curriculum of the 14 Milestone 4 hard-negative seeds (`200–224`) will teach it to navigate around centered obstacles.
- **Implementation:** Candidate model retrained with 50% hard-negative sampling.
- **Evaluation on Development Benchmark (`2000–2049`):**
  - **Success Rate:** **2.0% (1/50)** (Massive regression from 24/50 to 1/50)
  - **Simulated Collisions:** **1** (Violated absolute zero-collision invariant)
  - **Clearance Envelope Violations:** **19**
- **Forensic Diagnosis:** Catastrophic distribution shift and gradient contamination. The oversampled hard-negative trajectories forced the network weights into an unstable regime, generating high-frequency chattering and extreme control outputs.
- **Verdict:** **REJECTED AND QUARANTINED.** Immutable rule enacted: never promote Phase 4K or use hard-negative oversampling for policy retraining.

---

### Phase 5A: Avoidance Intention & Symmetry-Breaking Diagnostic
- **Date:** 2026-10-29
- **Core Hypothesis:** Because the BC neural network is symmetric and deterministic, a centered obstacle presents identical left and right ray distances, outputting $v_y = 0$. Injecting an observable, clearance-based symmetry-breaking bias will resolve the standoff without retraining.
- **Experiment:** Designed 7 canonical diagnostic scenarios (`seeds 1000–1006`):
  - Scenario 1-2: Open corridor & single centered pillar
  - Scenario 3-4: Offset left/right pillars
  - Scenario 5: Alternating slalom pillars
  - Scenario 6: Tight corridor squeeze
  - Scenario 7: Sequential 3-pillar pinch
- **Candidate Evaluated:** Alternative A (Stateless Clearance Guidance: nudge lateral velocity toward the side with maximum open ray clearance when front ray $<0.50$m).
- **Results:**
  - BC Baseline: Succeeded in only 2/7 scenarios (failed on centered and sequential obstacles).
  - Alternative A: Succeeded in **5/7 scenarios**. Failed only on Scenario 7 (tight sequential 3-pillar pinch).
- **Verdict:** Proved that standoff is a mathematical symmetry artifact resolvable by rule-based clearance guidance.

---

### Phase 5B: Stateless Clearance Guidance Development Benchmark
- **Date:** 2026-10-29
- **Core Hypothesis:** Stateless Clearance Guidance will generalize from diagnostic scenarios to the 50-seed development benchmark (`2000–2049`).
- **Evaluation Results (`2000–2049`, $N=50$):**
  - **Success Rate:** **70.0% (35/50)** [Baseline: 48.0% (24/50)]
  - **Collisions:** **0** (0.0%)
  - **Clearance Violations:** **0** (0.0%)
  - **Net Gain:** $+11$ additional completions ($p = 0.0076$, McNemar test).
- **Residual Failures (15/50):** Analysis showed all 15 failures were timeouts caused by sequential obstacle pinches (Pillar 2/3) where the drone cleared the first obstacle but entered the second obstacle at an acute angle, causing supervisor standoff.

---

### Phase 5C: Persistent Clearance Guidance & Re-Centering
- **Date:** 2026-10-29
- **Core Hypothesis:** Adding an explicit state machine with directional persistence (`AVOIDING`) and active corridor re-centering (`SETTLING`) will resolve sequential obstacle pinches.
- **Implementation:** FSM controller with 3 states: `CRUISING`, `AVOIDING` (latched for min 15 steps), and `SETTLING` (proportional lateral force back toward corridor centerline $y=0$).
- **Evaluation Results (`2000–2049`, $N=50$):**
  - **Success Rate:** **80.0% (40/50)** [Baseline: 48.0%]
  - **Collisions:** **0** (0.0%)
  - **Clearance Violations:** **0** (0.0%)
- **Trade-off Analysis:** While persistence resolved 5 sequential pinches, it introduced a new failure mode: during high-speed exits, the re-centering force fought with immediate counter-avoidance required for closely spaced counter-staggered obstacles.

---

### Phase 6 / 6A: Production Stateless Guidance Integration
- **Date:** 2026-10-30
- **Core Hypothesis:** Refining the stateless clearance differential threshold ($\Delta d = 0.15$m) and optimizing lateral nudge velocity ($v_y = \pm 0.35$ m/s) will achieve peak performance without the state-latching instability of Phase 5C.
- **Evaluation Results (`2000–2049`, $N=50$):**
  - **Success Rate:** **84.0% (42/50)** [95% Wilson CI: 71.5% – 91.7%]
  - **Baseline:** 48.0% (24/50)
  - **Net Gain:** **+18 completions (+36.0 percentage points)**
  - **Statistical Significance:** $p = 0.000912$ (Highly significant)
  - **Collisions:** **0** (0.0%)
  - **Clearance Violations:** **0** (0.0%)
  - **Minimum Obstacle Clearance:** **$0.0957$m** (Safety envelope $>0.08$m strictly preserved)
  - **Minimum Wall Clearance:** **$0.1942$m**
- **Verdict:** **CURRENT HIGHEST PERFORMING PRODUCTION CANDIDATE.**

---

### Phase 6B: Sensitivity, Robustness & Paired Regression Sweep
- **Date:** 2026-10-30
- **Core Hypothesis:** The Phase 6A candidate must demonstrate robustness across sensor noise, actuation delays, and varying corridor dimensions.
- **Evaluation Results:**
  - Evaluated on paired seeds across $N=50$ development runs with added sensor Gaussian noise ($\sigma = 0.05$m) and 1-step latency delay ($50$ms).
  - Maintained **82.0% (41/50)** success with zero collisions and zero envelope violations.
- **Verdict:** Production candidate validated as robust to simulated perturbations.

