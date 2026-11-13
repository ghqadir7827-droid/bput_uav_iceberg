# PHASE 5C — PERSISTENT CLEARANCE GUIDANCE AND RE-CENTERING VALIDATION
## Master Final Audit, Benchmark Evaluation & Architectural Determination Report

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Isolation Directory:** `experiments/phase5c/phase5c_persistent_recenter_20261029_225000/`
- **Benchmark Horizon:** Canonical 500 steps ($25.0\text{ s}$ at $\Delta t = 0.05\text{ s}$)
- **Supervisor Caution Distance:** Canonical $d_{\text{caut}} = 0.50\text{ m}$
- **Evaluation Dataset:** Development seeds `2000–2049` (50 matched episodes per controller)
- **Held-Out Test Reservation:** Seeds `90000–90099` strictly preserved and untouched

---

## 1. Executive Summary & Definitive Recommendation

### Final Phase 5C Determination

$$\mathbf{\text{FINAL DECISION: C. STATELESS GUIDANCE REMAINS PREFERRED}}$$

> Persistent guidance does not deliver a meaningful improvement and introduces severe operational and safety failures across the multi-obstacle development benchmark. While stateless clearance guidance achieves a robust, statistically significant **70.0% goal success rate** (net +11 wins over baseline, $p = 0.0433$) with zero collisions and zero envelope violations, persistent guidance with re-centering collapses to **34.0% success** (losing 22 episodes to stateless guidance, $p = 0.0005$), chokes flight velocity with a $37.15\%$ throttled fraction, and introduces an obstacle clearance-envelope violation ($0.0223\text{ m}$ in Seed 2001). Therefore, **Stateless Clearance Guidance (Controller B)** remains the definitively preferred controller architecture.

---

## 2. Integrity Verification & Safeguard Compliance

Before and after the benchmark evaluation, cryptographic hashes and workspace boundaries were audited:

| Component | Target File | Pre-Run SHA-256 | Post-Run SHA-256 | Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| **Safety Supervisor** | `controllers/safety_supervisor.py` | `9f84af228...` | `9f84af228...` | **MATCH - UNMODIFIED** |
| **Frozen BC Baseline** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d...` | `93c4d7d1d...` | **MATCH - UNMODIFIED** |
| **Held-Out Seeds** | `90000–90099` | Sealed | Sealed | **UNTOUCHED** |
| **Hardware Bus** | RadioLink / PX4 Actuators | Inactive | Inactive | **SAFE** |
| **Test Suite Baseline**| Pytest Suite (`pytest`) | 138 / 138 pass | 138 / 138 pass | **ALL PASSED (2.68s)** |

---

## 3. Resolution of Phase 5B Reporting Discrepancies

Prior to benchmarking, reported figures from Phase 5B were mathematically audited ([`PHASE_5C_STATE_MACHINE_AUDIT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/PHASE_5C_STATE_MACHINE_AUDIT.md)):
1. **The 7.936 m vs 8.155 m Progress Metric:**
   - **$7.936\text{ m}$** was the exact mean progress across the full **50-episode development set (`seeds 2000–2049`)**.
   - **$8.155\text{ m}$** was the mean progress across the **10-seed subset (`seeds 2000–2009`)** used during threshold sensitivity testing.
   - Restricting the paired benchmark to seeds `2000–2009` yields exactly **$8.155\text{ m}$**. The discrepancy was purely an artifact of sample size ($N=50$ vs $N=10$).
2. **Progress vs True Goal Success:**
   - In compliance with project standards, reaching $X > 8.0\text{ m}$ (clearing all 3 pillars) or entering the goal sphere is not sufficient for success. Success strictly requires entering the goal radius ($d_{\text{goal}} \le 0.50\text{ m}$) with terminal speed bounded at $\|\mathbf{v}\| \le 0.35\text{ m/s}$.

---

## 4. Paired Three-Controller Development Benchmark (Seeds 2000–2049)

All three controllers were evaluated across all 50 matched development seeds under identical simulation physics, 500-step horizon, and canonical supervisor thresholds ($d_{\text{caut}} = 0.50\text{ m}$). Per-seed data is recorded in [`paired_three_controller_results.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/paired_three_controller_results.csv) and summarized in [`controller_comparison_summary.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/controller_comparison_summary.json):

### Comprehensive Benchmark Results Table

| Evaluation Metric | Controller A (Frozen Baseline) | Controller B (Stateless Guidance) | Controller C (Persistent Guidance) |
| :--- | :---: | :---: | :---: |
| **Goal Successes** | 24 / 50 | **35 / 50** | 17 / 50 |
| **Goal Success Rate (%)** | 48.0% | **70.0%** | 34.0% |
| **Wilson 95% CI** | `[34.8%, 61.5%]` | **`[56.2%, 80.9%]`** | `[22.4%, 47.8%]` |
| **Obstacle Collisions** | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** |
| **Boundary Collisions** | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** |
| **Obstacle Envelope Violations (<0.08m)**| **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | **1 Violation (Seed 2001)** |
| **Boundary Envelope Violations (<0.08m)**| **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** |
| **Minimum Obstacle Clearance** | 0.1070 m | **0.0870 m (Safe)** | **0.0223 m (UNSAFE)** |
| **Minimum Boundary Clearance** | 0.2537 m | 0.1724 m | 0.1544 m |
| **Mean Forward Progress** | 6.525 m | **7.936 m (+21.6%)** | 6.870 m |
| **Mean Episode Duration** | 22.99 s | **21.74 s (-5.4%)** | 24.25 s |
| **Mean Throttled Fraction (%)** | 27.57% | **20.18% (-7.4%)** | 37.15% (+9.6% Choked) |
| **Pillar 1 Clearance Rate (%)** | 88.0% (44/50) | 96.0% (48/50) | **100.0% (50/50)** |
| **Pillar 2 Clearance Rate (%)** | 68.0% (34/50) | **94.0% (47/50)** | 76.0% (38/50) |
| **Pillar 3 Clearance Rate (%)** | 48.0% (24/50) | **78.0% (39/50)** | 52.0% (26/50) |
| **Terminal Alignment Failures** | 0 | 1 (Seed 2041, dist: 0.40m) | 0 |

### Paired Statistical Comparison Analysis

$$\begin{array}{l|c|c|c|c}
\text{Paired Comparison} & \text{Wins (2nd)} & \text{Losses (2nd)} & \text{Net Difference} & \text{Exact Binomial } p\text{-value} \\
\hline
\text{Controller B vs Controller A} & \mathbf{18} & \mathbf{7} & \mathbf{+11\text{ Wins}} & \mathbf{p = 0.0433\text{ (Statistically Significant)}} \\
\text{Controller C vs Controller B} & \mathbf{4} & \mathbf{22} & \mathbf{-18\text{ Losses}} & \mathbf{p = 0.0005\text{ (Significant Degradation)}} \\
\text{Controller C vs Controller A} & \mathbf{7} & \mathbf{14} & \mathbf{-7\text{ Losses}} & \mathbf{p = 0.1892\text{ (Inferior to Baseline)}} \\
\end{array}$$

---

## 5. Forensic Diagnosis of Controller C Collapse

A detailed failure analysis ([`persistent_guidance_failure_analysis.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/persistent_guidance_failure_analysis.md)) and state audit ([`state_transition_metrics.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/state_transition_metrics.csv)) reveal why persistent guidance failed:

### 1. The Multi-Obstacle Latch Trap
- The exit condition for `AVOID_LEFT` and `AVOID_RIGHT` required:
  $$C_{\text{front}} > 1.25\text{ m} \quad \text{AND} \quad \Delta x \ge 0.65\text{ m}$$
- In a corridor with pillars spaced $2.0–2.5\text{ m}$ apart, the forward rays detect the next pillar before $C_{\text{front}}$ can ever exceed $1.25\text{ m}$.
- The state machine became trapped in avoidance mode until the 50-step timeout ($2.5\text{ seconds}$).
- Across all 50 episodes, Controller C spent **$70.0\%$ of its flight time locked in avoidance states** (`AVOID_LEFT`: $42.0\%$, `AVOID_RIGHT`: $28.0\%$).
- This continuous lateral override drove the UAV directly into the corridor walls ($|Y| \approx 0.60–0.75\text{ m}$), triggering massive supervisor boundary repulsions and forward speed throttling ($37.15\%$ throttled time).

### 2. The Near-Collision Envelope Violation (Seed 2001)
- In Seed 2001, Controller C recorded an obstacle clearance of **$0.0223\text{ m}$** at Step 281 ($X = 5.145\text{ m}, Y = -0.55\text{ m}$).
- Trapped in `AVOID_LEFT` setpoint ($v_y = +0.25\text{ m/s}$), the UAV executed an oblique diagonal traverse across the face of Pillar 2, skimming its surface at $2.2\text{ cm}$.
- Under project safety rules, any configuration that violates the $0.080\text{ m}$ clearance envelope is **strictly disqualified from promotion**.

### 3. Re-Centering Preemption Churn
- The `SETTLING` state averaged only **$36.0\text{ steps}$ ($1.8\text{ s}$)** per episode across an average of **5.3 preemptions**.
- Re-centering was aborted almost immediately as downstream pillars triggered $C_{\text{front}} < 0.85\text{ m}$, introducing severe velocity chattering without restoring centerline positioning.

---

## 6. Safety and Promotion Audit

| Promotion Criterion | Requirement | Controller B (Stateless) | Controller C (Persistent) | Promotion Status |
| :--- | :--- | :---: | :---: | :---: |
| Obstacle Collisions | Zero tolerated | **0 / 50 (PASS)** | **0 / 50 (PASS)** | Both pass |
| Boundary Collisions | Zero tolerated | **0 / 50 (PASS)** | **0 / 50 (PASS)** | Both pass |
| Obstacle Clearance Envelope | Zero $< 0.080\text{ m}$ | **0 / 50 (Min: 0.0870m)** | **1 Violation (Min: 0.0223m)** | **C DISQUALIFIED** |
| Boundary Clearance Envelope | Zero $< 0.080\text{ m}$ | **0 / 50 (Min: 0.1724m)** | **0 / 50 (Min: 0.1544m)** | Both pass |
| Development Goal Completion | Non-inferiority | **70.0% vs 48.0% ($p=0.043$)** | **34.0% vs 48.0% (Inferior)** | **B APPROVED** |
| Multi-Obstacle Passage | Multiple pillars | P1: 96%, P2: 94%, P3: 78% | P1: 100%, P2: 76%, P3: 52% | **B APPROVED** |

---

## 7. Phase 5C Deliverables & Clickable Links

All Phase 5C evaluation logs, scripts, and diagnostic reports are organized in the workspace:

| Deliverable | File Link | Description |
| :--- | :--- | :--- |
| **Pre-Run Audit** | [`PHASE_5C_PRE_RUN_AUDIT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/PHASE_5C_PRE_RUN_AUDIT.md) | Cryptographic checksums and environment boundaries |
| **State Machine Audit** | [`PHASE_5C_STATE_MACHINE_AUDIT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/PHASE_5C_STATE_MACHINE_AUDIT.md) | Formal state specifications, transitions, and reporting reconciliation |
| **Paired Benchmark Results** | [`paired_three_controller_results.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/paired_three_controller_results.csv) | 150 paired episode records across seeds 2000..2049 |
| **State Transition Metrics** | [`state_transition_metrics.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/state_transition_metrics.csv) | Per-seed state occupancy, preemptions, and clearances for Controller C |
| **Comparative Summary** | [`controller_comparison_summary.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/controller_comparison_summary.json) | Statistical metrics, Wilson CIs, and McNemar exact binomial tests |
| **Failure Analysis Report** | [`persistent_guidance_failure_analysis.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/persistent_guidance_failure_analysis.md) | Forensic investigation of the 22 discordant failure episodes |
| **Benchmark Test Harness** | [`run_phase5c_benchmark.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/run_phase5c_benchmark.py) | Full executable evaluation harness |
| **Episode JSON Traces** | [`traces/`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/traces) | High-resolution per-step episode telemetry logs |
| **Master Final Report** | [`PHASE_5C_FINAL_REPORT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase5c/phase5c_persistent_recenter_20261029_225000/PHASE_5C_FINAL_REPORT.md) | This master audit and determination document |

---

## 8. Summary & Next Steps (Phase 5D Roadmap)

1. **Retain Controller B (Stateless Guidance) as the Sole Primary Candidate:**
   - Stateless guidance is verified as the only controller architecture that significantly outperforms the baseline (**70.0% vs 48.0%**, $p = 0.0433$) while preserving all safety invariants.
   - Stateful guidance architectures with rigid distance exit conditions are conclusively rejected.
2. **Terminal Goal Landing Soft-Landing Transition:**
   - The single terminal alignment failure in Controller B (Seed 2041, final goal distance $0.404\text{ m}$) occurred because guidance remained active near the goal. Relaxing lateral guidance for $X > 8.5\text{ m}$ to allow nominal baseline deceleration can elevate success toward $> 72\%$.
3. **Formal Held-Out Generalization Evaluation (Seeds `90000–90099`):**
   - With the development benchmark firmly established and the stateful hypothesis thoroughly audited and ruled out, Controller B is fully qualified for held-out generalization testing.

