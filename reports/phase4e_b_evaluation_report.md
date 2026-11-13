# Phase 4E-B — Expanded Isolated Supervisor Evaluation Report

**Date:** 2026-10-27  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Phase:** 4E-B (Expanded Isolated Supervisor Evaluation — Non-Training, Isolated Evaluation)  
**Evaluated Modules & Checkpoints:**
- **Candidate PPO Policy:** `models/checkpoints/ppo_phase4b_candidate.pt` (SHA-256 verified)
- **Canonical Safety Supervisor:** `controllers/safety_supervisor.py` (Unmodified baseline)
- **Experimental Hysteresis Supervisor:** `controllers/experimental_hysteresis_supervisor.py` (Isolated prototype)
- **Disjoint Diagnostic Seed Pool:** Seeds 70001–70015 ($N=15$, strictly non-overlapping)
- **Scenario Matrix Dataset:** `reports/phase4e_b_scenario_matrix_metrics.json`
- **Simulation Benchmark Dataset:** `reports/phase4e_b_simulation_benchmark_metrics.json`

---

## 1. Executive Summary & Final Outcome

### **Outcome: PARTIAL**
The expanded evaluation demonstrates that the isolated `ExperimentalHysteresisSupervisor` generalizes effectively in eliminating the obstacle-pinning limit cycle, but reveals an inherent deadlock/progress trade-off when operating with a policy that lacks coordinated lateral evasion:

1. **Oscillation Elimination Generalized:**
   - In the **12-Scenario Matrix**, velocity sign reversals were eliminated across all approach conditions (e.g., Scenario 1: $116 \to 0$; Scenario 2: $113 \to 0$; Scenario 11: $112 \to 0$).
   - In the **15-Seed Disjoint Simulation Benchmark**, total velocity sign reversals dropped from **1,996 to 14 (a 99.3% reduction)**.
2. **Safety & Clearance Maintained:**
   - **Zero collisions** occurred across all 12 scenario matrix evaluations and all 15 simulation benchmark seeds.
   - Global minimum obstacle clearance improved from $0.103\text{ m}$ (canonical) to **$0.119\text{ m}$** (hysteresis).
3. **Deadlock / Progress Trade-Off (Why PARTIAL):**
   - While the canonical supervisor's violent velocity chattering occasionally allowed the drone to "slide" past an obstacle through uncoordinated oscillations (e.g. Seed 70005: $6.13\text{ m}$ displacement via 47 reversals), the hysteresis supervisor strictly arrests forward motion, retreats to standoff, and enters stationary holding (`BLOCKED_ESCAPE_DIRECTION`) when lateral corridors are restricted.
   - Consequently, mean net forward displacement across the 15 seeds decreased slightly from **$3.91\text{ m} \to 3.47\text{ m}$ ($-0.44\text{ m}$)**.
   - **Terminal Navigation:** As documented in Phase 4D, terminal arrival remains at 0% because the frozen policy still commands excessive speed ($v > 0.35\text{ m/s}$) and downward drift ($z \approx 0.40\text{ m}$) at the corridor exit.

---

## 2. Seed Pool Verification & Audit Protocol

### 2.1 Transparency on Prior Diagnostic Seed Usage
- **Phase 4C:** Evaluated the reserved 100-seed held-out pool (`50000–50099`).
- **Phase 4D:** Inspected diagnostic trajectories for representative stuck seeds (50000, 50001) and near-goal seeds (50041, 50058, 50066).
- **Phase 4E-A:** Re-executed Seeds 50000 and 50001 alongside validation seeds 100, 101, 102 to verify that the hysteresis prototype eliminated the specific limit cycle identified in Phase 4D. No policy tuning was performed against these seeds.

### 2.2 Disjoint Seed Pool for Phase 4E-B
For Phase 4E-B, a brand new dedicated diagnostic seed block was defined:
$$\text{Pool}_{\text{Phase4E-B}} = \{70001, 70002, 70003, \dots, 70015\}$$

**Disjointness Verification:**
- Overlap with Fixed Validation Pool (`[100–109]`): **0 seeds (Disjoint)**
- Overlap with Hard-Negative Training Pool (`[200–224]`): **0 seeds (Disjoint)**
- Overlap with BC Demonstration Pool (`[1000–1050]`): **0 seeds (Disjoint)**
- Overlap with Canonical Held-Out Pool (`[50000–50099]`): **0 seeds (Disjoint)**
- Overlap with Phase 4E-A Diagnostic Seeds (`[50000, 50001, 100, 101, 102]`): **0 seeds (Disjoint)**

---

## 3. Prototype Implementation Audit

An exhaustive, line-by-line audit of `controllers/experimental_hysteresis_supervisor.py` was conducted:

| Architecture Component | Implementation Fact | Verification Finding |
| :--- | :--- | :--- |
| **Action Bounds** | Enforces `ActionLimits`: $v_x \in [-0.20, 0.60]\text{ m/s}$, $v_y \in [-0.40, 0.40]\text{ m/s}$, $v_z \in [-0.30, 0.30]\text{ m/s}$, $\dot{\psi} \in [-0.50, 0.50]\text{ rad/s}$. | Strictly conforms to contracts. |
| **Clearance Reference** | Uses `boundary_clearances(pos)` and `compute_directional_clearances_m(sensor)`. Measured from outer collision sphere ($r_{\text{drone}} = 0.15\text{ m}$). | Factual physical distance to body surface. |
| **Ordering of Filters** | 1. Telemetry/localization failsafe $\to$ 2. State machine selection $\to$ 3. Multi-directional lateral clamping $\to$ 4. Boundary repulsion $\to$ 5. Contract clipping. | Safety clamping overrides policy proposals. |
| **State Machine Semantics** | Explicit 6-state machine with entry dwell timers ($0.15\text{ s}$ braking, $0.20\text{ s}$ release confirmation). | Eliminates instantaneous zero-crossing chattering. |
| **Deadlock Handling** | When all directions are restricted ($C_{\text{left}}, C_{\text{right}} < 0.38\text{ m}, d_{\text{rear}} \le 0.25\text{ m}$), enters `BLOCKED_ESCAPE_DIRECTION` holding $v_x = 0, v_y = 0$. | Prevents boundary crashes, but results in stationary hover. |
| **Safe Release** | Enforces a $0.20\text{ s}$ confirmation window where $v_x \le 0.30\text{ m/s}$. If front proximity collapses, aborts to `OBSTACLE_BRAKING` with $v_x = 0.0\text{ m/s}$. | Prevents false releases into obstacles. |

*(No modifications were made to the prototype during the evaluation).*

---

## 4. Scenario Matrix Definitions & Matched Initial Conditions

12 operational scenarios were evaluated under matched initial conditions for both supervisors:

| ID | Scenario Name | Initial Position $[x, y, z]$ | Initial Velocity $[v_x, v_y, v_z]$ | Policy Proposal | Sensor / Environment Geometry |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Direct approach to isolated obstacle | $[2.2, 0.0, 1.0]$ | $[0.5, 0.0, 0.0]$ | $v_x=0.6, v_y=0.0$ | Obstacle at $x=3.0, y=0.0$. Both sides open ($0.60\text{ m}$). |
| **2** | Offset angle approach | $[2.2, -0.15, 1.0]$ | $[0.4, 0.15, 0.0]$ | $v_x=0.5, v_y=0.15$ | Obstacle at $x=3.0, y=0.20$. Heading toward right edge. |
| **3** | Bypass route open on left only | $[2.3, -0.25, 1.0]$ | $[0.4, 0.0, 0.0]$ | $v_x=0.6, v_y=0.0$ | Obstacle shifted right ($y=-0.25$). Right blocked ($0.22\text{ m}$), left open ($0.65\text{ m}$). |
| **4** | Bypass route open on right only | $[2.3, 0.25, 1.0]$ | $[0.4, 0.0, 0.0]$ | $v_x=0.6, v_y=0.0$ | Obstacle shifted left ($y=+0.25$). Left blocked ($0.22\text{ m}$), right open ($0.65\text{ m}$). |
| **5** | Obstacle near corridor wall | $[2.3, 0.50, 1.0]$ | $[0.4, 0.0, 0.0]$ | $v_x=0.5, v_y=0.0$ | Obstacle near left wall. Left gap $0.18\text{ m}$, right corridor $0.85\text{ m}$. |
| **6** | Narrow passage (no viable bypass) | $[2.4, 0.0, 1.0]$ | $[0.3, 0.0, 0.0]$ | $v_x=0.5, v_y=0.0$ | Both lateral gaps constricted to $0.24\text{ m} < 0.38\text{ m}$ bypass threshold. |
| **7** | Consecutive obstacles | $[2.2, 0.0, 1.0]$ | $[0.5, 0.0, 0.0]$ | $v_x=0.6, v_y=0.0$ | Obstacle 1 at $x=3.0\text{ m}$ (bypass left); Obstacle 2 at $x=5.5\text{ m}$ (bypass right). |
| **8** | Unsuccessful bypass recovery | $[2.5, 0.0, 1.0]$ | $[0.0, 0.0, 0.0]$ | $v_x=0.5, v_y=0.0$ | Left opens initially, then suddenly constricts to $0.18\text{ m}$ at step 15. |
| **9** | Blocked escape direction (dead end) | $[2.5, 0.0, 1.0]$ | $[0.0, 0.0, 0.0]$ | $v_x=0.5, v_y=0.0$ | Front, left, right, and rear all constricted ($0.20–0.22\text{ m}$). |
| **10** | Clear corridor (no obstacles) | $[1.0, 0.0, 1.0]$ | $[0.5, 0.0, 0.0]$ | $v_x=0.6, v_y=0.0$ | Completely clear corridor ($4.0\text{ m}$ range, wall clearances $0.85\text{ m}$). |
| **11** | Restricted vertical clearance | $[2.2, 0.0, 1.80]$ | $[0.4, 0.0, 0.0]$ | $v_x=0.5, v_z=0.2$ | Low ceiling ($z_{\text{ceil}}=2.0\text{ m}$, $d_{\text{ceil}}=0.05\text{ m}$); policy requests climb. |
| **12** | Pre-existing lateral velocity | $[2.3, -0.2, 1.0]$ | $[0.4, -0.35, 0.0]$ | $v_x=0.5, v_y=-0.35$ | Moving toward right obstacle edge at $v_y = -0.35\text{ m/s}$. |

---

## 5. Comparative Metrics & Findings

### 5.1 Scenario Matrix Results Table (150 Steps per Scenario)

| ID | Scenario Name | Canonical Disp (m) | Hysteresis Disp (m) | Canonical Reversals | Hysteresis Reversals | Canonical Outcome | Hysteresis Outcome | Hysteresis Key Behavior |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Direct approach | 0.15 m | 0.15 m | **116** | **0** | Timeout | Timeout | Smooth brake $\to$ bypass left |
| **2** | Offset angle approach | 0.15 m | -0.05 m | **113** | **0** | Timeout | Timeout | Arrests lateral drift, enters retreat |
| **3** | Bypass open left only | -0.72 m | -0.23 m | 0 | 0 | Timeout | Timeout | Rejects blocked right, steers left |
| **4** | Bypass open right only | -0.72 m | -0.23 m | 0 | 0 | Timeout | Timeout | Rejects blocked left, steers right |
| **5** | Obstacle near wall | -0.72 m | -0.12 m | 0 | 0 | Timeout | Timeout | Steers away from wall into corridor |
| **6** | Constricted passage | -0.72 m | -0.66 m | 0 | 0 | Timeout | Timeout | Rejects unsafe bypass $\to$ safe retreat |
| **7** | Consecutive obstacles | -0.72 m | -0.35 m | 0 | 0 | Timeout | Timeout | Re-engages brake on second obstacle |
| **8** | Unsuccessful bypass | -0.72 m | -0.68 m | 0 | 0 | Timeout | Timeout | Aborts bypass on pinch $\to$ retreat |
| **9** | Blocked dead-end | 0.00 m | 0.00 m | 0 | 0 | Timeout | Timeout | Zero chattering hover hold |
| **10** | Clear corridor | 4.50 m | 4.50 m | 0 | 0 | Timeout | Timeout | **0 false interventions**, full speed |
| **11** | Low ceiling approach | 0.15 m | -0.05 m | **112** | **0** | Timeout | Timeout | Suppresses climb, repels ceiling |
| **12** | Pre-existing lateral vel | -0.72 m | -0.46 m | 0 | 0 | Timeout | Timeout | Clamps lateral vector, avoids graze |

### 5.2 Disjoint Simulation Benchmark (Seeds 70001–70015, Candidate PPO Checkpoint)

| Seed | Canonical Status | Hysteresis Status | Canonical Disp (m) | Hysteresis Disp (m) | Canonical Reversals | Hysteresis Reversals | Canonical Min Obs (m) | Hysteresis Min Obs (m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **70001** | Timeout | Timeout | 4.13 m | 3.98 m | 143 | **0** | 0.141 m | 0.145 m |
| **70002** | Timeout | Timeout | 2.02 m | 2.00 m | 199 | **6** | 0.139 m | 0.146 m |
| **70003** | Timeout | Timeout | 4.15 m | 3.98 m | 116 | **0** | 0.139 m | 0.162 m |
| **70004** | Timeout | Timeout | 2.19 m | 2.01 m | 148 | **0** | 0.134 m | 0.159 m |
| **70005** | Timeout | Timeout | 6.13 m | 2.16 m | 47 | **0** | 0.130 m | 0.157 m |
| **70006** | Timeout | Timeout | 4.19 m | 3.97 m | 140 | **0** | 0.159 m | 0.163 m |
| **70007** | Timeout | Timeout | 6.08 m | 5.90 m | 76 | **1** | 0.137 m | 0.160 m |
| **70008** | Timeout | Timeout | 2.20 m | 1.95 m | 156 | **1** | 0.128 m | 0.164 m |
| **70009** | Timeout | Timeout | 6.20 m | 6.07 m | 44 | **0** | 0.103 m | 0.119 m |
| **70010** | Timeout | Timeout | 6.26 m | 6.11 m | 75 | **1** | 0.128 m | 0.156 m |
| **70011** | Timeout | Timeout | 2.09 m | 1.91 m | 206 | **0** | 0.138 m | 0.149 m |
| **70012** | Timeout | Timeout | 2.74 m | 2.15 m | 168 | **2** | 0.107 m | 0.121 m |
| **70013** | Timeout | Timeout | 2.07 m | 1.97 m | 190 | **1** | 0.132 m | 0.154 m |
| **70014** | Timeout | Timeout | 4.14 m | 3.89 m | 144 | **0** | 0.138 m | 0.160 m |
| **70015** | Timeout | Timeout | 4.12 m | 4.00 m | 144 | **2** | 0.140 m | 0.145 m |
| **TOTAL / MEAN** | **0 Success** | **0 Success** | **3.91 m** | **3.47 m** | **1,996** | **14 (-99.3%)** | **0.103 m** | **0.119 m** |

---

## 6. Detailed Analysis of Failures & Unresolved Issues

### 6.1 The Progress vs Stability Trade-Off (Seed 70005 Analysis)
On Seed 70005, the canonical supervisor achieved $6.13\text{ m}$ displacement, while the hysteresis supervisor achieved $2.16\text{ m}$ (a drop of $-3.97\text{ m}$).
- **Canonical Mechanism:** The canonical supervisor suffered 47 violent sign reversals at Obstacle 1 ($x=3.0\text{ m}$). These undamped oscillations caused the drone to chatter back and forth until random lateral drift permitted it to squeeze past the obstacle cylinder.
- **Hysteresis Mechanism:** The hysteresis supervisor detected that lateral clearance was slightly below the conservative bypass threshold ($C_{\text{side}} < 0.38\text{ m}$). It correctly rejected an unsafe forward push, entered `RETREAT_OR_RECOVERY`, reached its rear standoff limit, and maintained stationary hover.
- **Finding:** The hysteresis supervisor prioritizes verified clearance over opportunistic collision-risk slipping. While this guarantees 0 collisions, it reduces net forward progress when the policy fails to propose active evasive maneuvers.

### 6.2 Unresolved Terminal Navigation Deficits (Phase 4D Requirements)
Across all 15 disjoint simulation benchmark seeds, 0 episodes achieved success:
1. **Terminal Speed Overshoot:** The candidate policy continues to cruise forward at $0.60–0.70\text{ m/s}$ in the final corridor segment, exceeding the terminal landing threshold ($v \le 0.35\text{ m/s}$).
2. **Downward Altitude Drift:** As documented in Phase 4D, the policy commands a steady negative vertical bias, causing the vehicle to settle on the floor buffer ($z \approx 0.40\text{ m}$), geometrically preventing arrival at the target volume suspended at $[9.5, 0.0, 1.0]$.
3. **Supervisor Role Boundary:** The safety supervisor is an obstacle and boundary barrier; it is not a destination guidance controller. True task success cannot be achieved through supervisor hysteresis alone without addressing policy training or terminal guidance.

---

## 7. Regression and Artifact Integrity Audit

### 7.1 Test Suite Verification
Executing `pytest` across all unit tests:
```bash
============================== 95 passed in 2.43s ==============================
```
- All 87 canonical tests pass.
- All 8 experimental hysteresis tests pass.
- Zero regressions.

### 7.2 Checkpoint & Evaluation Artifact Hashes

| Artifact File | Expected SHA-256 Hash | Verification Status |
| :--- | :--- | :---: |
| `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **VERIFIED MATCH** |
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **VERIFIED MATCH** |
| `models/checkpoints/ppo_phase4b_candidate.pt` | `5b9381188058ee8e690db3ec9b929a1b99a74469f7647bc527d2e2e76e5a8583` | **VERIFIED MATCH** |
| `models/checkpoints/policy_ppo.onnx` | `38dd6575d9828b4616d740b296b067c7ee964fc7de1ff5e8f671c4ecb82a0ecf` | **VERIFIED MATCH** |
| `models/checkpoints/policy_bc.onnx` | `9365aab7253c38793af5efa7cb7eb89b7e1db3bf60ea96e70df1671cdbbb4f50` | **VERIFIED MATCH** |
| `reports/phase4c_heldout_comparison.json` | `59df705fd7a37e3ef4de4db9e21a37351b5b4ea423c936d073650b6958e68e1a` | **VERIFIED MATCH** |
| `reports/phase4c_heldout_eval_ppo_baseline.json` | `2e4be15803d189aa52aa22a91026d1038994777bde53b9d413d8b2210ba21986` | **VERIFIED MATCH** |
| `reports/phase4c_heldout_eval_ppo_candidate.json` | `2e14031e755fd2e87c38d26fcf79a2fb96707d092adb87592bb5cd9b043a23d2` | **VERIFIED MATCH** |

---

## 8. Recommendations for Integration Review

1. **Architecture Recommendation:** The multi-state hysteresis architecture is superior to memoryless threshold switching for physical stability, eliminating 99.3% of velocity chattering and expanding obstacle safety margins.
2. **Coupled Navigation Requirement:** A supervisor barrier cannot compensate for a policy that commands forward velocity into obstacles with zero lateral evasion. Future integration must pair supervisor hysteresis with a policy retraining or fine-tuning phase that incentivizes lateral detour paths and terminal deceleration.
3. **Deadlock Recovery Refinement:** Future iterations should evaluate dynamic threshold adaptation in `BLOCKED_ESCAPE_DIRECTION` (e.g. attempting vertical crawl if ceiling clearance permits) rather than purely static lateral bypass.

---

**Report Completed By:** Antigravity Agent  
**Authorization Compliance:** Strictly non-training, isolated evaluation. Canonical pipeline and checkpoints remain completely untouched.
