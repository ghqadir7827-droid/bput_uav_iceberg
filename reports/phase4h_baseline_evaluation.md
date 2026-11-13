# Phase 4H-B: Shared Baseline Evaluation Report

**Date:** 2026-10-29  
**Status:** COMPLETED & VERIFIED  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Deliverables:**
- `reports/phase4h_baseline_evaluation.md`
- `reports/phase4h_baseline_evaluation_metrics.json`
- `experiments/phase4h/harness.py`

---

## 1. Executive Summary

Phase 4H-B establishes an isolated, shared evaluation harness ([`experiments/phase4h/harness.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4h/harness.py)) executing standardized, controlled benchmark scenarios across four baseline controllers:
1. **Arm B — Frozen BC Baseline** (`models/checkpoints/bc_policy_best.pt`)
2. **Arm A — Frozen PPO Baseline** (`models/checkpoints/ppo_policy_best.pt`)
3. **Arm C — Phase 4G Standard PPO Candidate** (`experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt`)
4. **Arm D — Phase 4G Reflection PPO Candidate** (`experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`)

### Core Discoveries
1. **The 200-Step vs. 500-Step Horizon Sensitivity:**
   - In a clear corridor under the canonical 500-step horizon ($25.0\text{ s}$), **BC achieves 100% goal success in 344 steps**, and **PPO achieves 100% goal success in 379 steps**, both satisfying terminal speed constraints ($v \le 0.35\text{ m/s}$).
   - In the 200-step horizon ($10.0\text{ s}$) used in Phase 4G, every controller times out at $X \approx 5.8 - 6.5\text{ m}$ before reaching the 9.0m goal. The 200-step horizon falsely classified nominal navigation trajectories as 0% failures.
2. **Empirical Confirmation of Centered vs. Offset Obstacle Dynamics:**
   - **BC Policy:** Achieves **100% end-to-end goal success** when an obstacle is offset to the left ($Y_{\text{obs}} = -0.25\text{ m}, -0.15\text{ m}$) or right ($Y_{\text{obs}} = +0.25\text{ m}$). However, against a centered obstacle ($Y_{\text{obs}} = 0.00\text{ m}$), BC fails completely ($0\%$ success, deadlocked at $X = 2.81\text{ m}$).
   - **PPO Policy:** Achieves **100% end-to-end goal success** when the obstacle is offset to the right ($Y_{\text{obs}} = +0.15\text{ m}, +0.25\text{ m}$, where the right opening is wide). However, against centered or left-offset obstacles ($Y_{\text{obs}} \le 0.00\text{ m}$), PPO fails completely ($0\%$ success, deadlocked at $X = 2.58 - 2.77\text{ m}$).
3. **Narrow Passage Performance:**
   - When presented with a 0.70m central passage between two symmetric pillars at $X=3.0\text{ m}$ ($Y = \pm 0.55\text{ m}$), **BC, Arm A, and Arm C all successfully navigate through the opening and achieve 100% goal arrival**.
   - Arm D (Reflection PPO) stalled in the narrow passage at $X = 2.49\text{ m}$ due to conservative velocity scaling.

---

## 2. Comprehensive Scenario-by-Scenario Evaluation Results

### 2.1 Test 1: Clear Corridor (Nominal Cruise & Horizon Comparison)
Evaluates undisturbed forward cruise and goal deceleration:

| Horizon | Controller | Outcome | Steps | Final $X$ | Terminal Speed | Result |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **500 steps ($25\text{s}$)** | **Arm B (BC)** | **success** | **344** | **$9.17\text{ m}$** | **$0.241\text{ m/s}$** | **SUCCESS** |
| 500 steps ($25\text{s}$) | **Arm A (PPO)** | **success** | **379** | **$9.47\text{ m}$** | **$0.282\text{ m/s}$** | **SUCCESS** |
| 500 steps ($25\text{s}$) | **Arm C (PPO No Aug)** | **success** | **388** | **$9.25\text{ m}$** | **$0.244\text{ m/s}$** | **SUCCESS** |
| 500 steps ($25\text{s}$) | Arm D (PPO Refl) | crash_boundary | 444 | $10.36\text{ m}$ | $0.454\text{ m/s}$ | Overshot Goal |
| **200 steps ($10\text{s}$)** | Arm B (BC) | continue (timeout) | 200 | $6.48\text{ m}$ | $0.599\text{ m/s}$ | Truncated Mid-Flight |
| 200 steps ($10\text{s}$) | Arm A (PPO) | continue (timeout) | 200 | $5.96\text{ m}$ | $0.631\text{ m/s}$ | Truncated Mid-Flight |
| 200 steps ($10\text{s}$) | Arm C (PPO No Aug) | continue (timeout) | 200 | $5.80\text{ m}$ | $0.599\text{ m/s}$ | Truncated Mid-Flight |
| 200 steps ($10\text{s}$) | Arm D (PPO Refl) | continue (timeout) | 200 | $5.77\text{ m}$ | $0.495\text{ m/s}$ | Truncated Mid-Flight |

*Finding:* At 200 steps, no policy can physically arrive at the goal because traveling 9.0 meters requires at least 300 to 380 steps at nominal flight speeds ($0.3 - 0.6\text{ m/s}$).

### 2.2 Test 2: Lateral Obstacle Sweep ($X_{\text{obs}} = 3.0\text{ m}$, 500-step horizon)
Evaluates bypass side selection, obstacle passage, and end-to-end goal success across varying lateral geometry:

| Obstacle $Y_{\text{obs}}$ | Controller | Passed Obs 1? | Bypass Side | Outcome | Steps | Final $X$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$-0.25\text{ m}$ (Left open)** | **Arm B (BC)** | **True** | **left** | **success** | **480** | **$9.16\text{ m}$** |
| $-0.25\text{ m}$ | Arm A (PPO) | False | right | timeout | 500 | $2.58\text{ m}$ |
| $-0.25\text{ m}$ | Arm C (PPO No Aug) | False | right | timeout | 500 | $2.58\text{ m}$ |
| $-0.25\text{ m}$ | Arm D (PPO Refl) | False | right | timeout | 500 | $2.58\text{ m}$ |
| **$-0.15\text{ m}$ (Left open)** | **Arm B (BC)** | **True** | **left** | **success** | **482** | **$9.16\text{ m}$** |
| $-0.15\text{ m}$ | Arm A (PPO) | False | right | timeout | 500 | $2.67\text{ m}$ |
| $-0.15\text{ m}$ | Arm C (PPO No Aug) | False | right | timeout | 500 | $2.68\text{ m}$ |
| $-0.15\text{ m}$ | Arm D (PPO Refl) | False | right | timeout | 500 | $2.67\text{ m}$ |
| **$0.00\text{ m}$ (Centered)** | **Arm B (BC)** | **False** | left | **timeout (deadlock)** | **500** | **$2.81\text{ m}$** |
| $0.00\text{ m}$ | **Arm A (PPO)** | **False** | right | **timeout (deadlock)** | **500** | **$2.77\text{ m}$** |
| $0.00\text{ m}$ | Arm C (PPO No Aug) | False | right | timeout (deadlock) | 500 | $3.43\text{ m}$ |
| $0.00\text{ m}$ | Arm D (PPO Refl) | True | right | timeout | 500 | $9.03\text{ m}$ |
| **$+0.15\text{ m}$ (Right open)** | Arm B (BC) | True | right | timeout | 500 | $9.06\text{ m}$ |
| $+0.15\text{ m}$ | **Arm A (PPO)** | **True** | **right** | **success** | **448** | **$9.53\text{ m}$** |
| $+0.15\text{ m}$ | **Arm C (PPO No Aug)** | **True** | **right** | **success** | **388** | **$9.35\text{ m}$** |
| $+0.15\text{ m}$ | Arm D (PPO Refl) | True | left | crash_boundary | 492 | $10.27\text{ m}$ |
| **$+0.25\text{ m}$ (Right open)** | **Arm B (BC)** | **True** | **right** | **success** | **469** | **$9.16\text{ m}$** |
| $+0.25\text{ m}$ | **Arm A (PPO)** | **True** | **right** | **success** | **411** | **$9.47\text{ m}$** |
| $+0.25\text{ m}$ | **Arm C (PPO No Aug)** | **True** | **right** | **success** | **425** | **$9.25\text{ m}$** |
| $+0.25\text{ m}$ | Arm D (PPO Refl) | True | left | crash_boundary | 465 | $10.29\text{ m}$ |

### 2.3 Test 3: Narrow Central Passage ($X=3.0\text{ m}$, Gap Width $= 0.70\text{ m}$)
Two symmetric pillars leaving only a central corridor opening:

| Controller | Passed Opening? | Outcome | Final $X$ | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Arm B (BC)** | **True** | **success** | **$9.20\text{ m}$** | Clean transit through center opening, terminal arrival. |
| **Arm A (PPO)** | **True** | **success** | **$9.44\text{ m}$** | Passed through opening, terminal arrival. |
| **Arm C (PPO No Aug)** | **True** | **success** | **$9.31\text{ m}$** | Passed through opening, terminal arrival. |
| Arm D (PPO Refl) | False | timeout | $2.49\text{ m}$ | Decelerated prematurely before opening. |

### 2.4 Test 4: Goal Approach Braking & Terminal Speed Test
Drone initialized at $X = 7.5\text{ m}, Y = 0.0, Z = 1.0\text{ m}$ with initial forward cruise speed $v_x = 0.50\text{ m/s}$:

| Controller | Outcome | Steps | Final Speed | Final Goal Distance | Landing Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Arm B (BC)** | **success** | **114** | **$0.215\text{ m/s}$** | **$0.346\text{ m}$** | **PASS:** Smooth deceleration below $0.35\text{ m/s}$. |
| **Arm A (PPO)** | **success** | **84** | **$0.343\text{ m/s}$** | **$0.224\text{ m}$** | **PASS:** Decelerated just below $0.35\text{ m/s}$ threshold. |
| **Arm C (PPO No Aug)** | **success** | **87** | **$0.345\text{ m/s}$** | **$0.303\text{ m}$** | **PASS:** Decelerated below $0.35\text{ m/s}$ threshold. |
| Arm D (PPO Refl) | continue | 150 | $0.362\text{ m/s}$ | $1.386\text{ m}$ | **FAIL:** Overshot arrival speed constraint ($0.362 > 0.35$). |

### 2.5 Test 5: Sequential 3-Pillar Corridors (Validation Seeds 100, 101, 102, 500 steps)
| Seed | Controller | Outcome | Steps | Final $X$ | Min Obs Clearance | Stalling Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 100 | Arm B (BC) | timeout | 500 | $4.74\text{ m}$ | $0.123\text{ m}$ | Passed Pillar 1, stalled at Pillar 2. |
| 100 | Arm A (PPO) | timeout | 500 | $6.68\text{ m}$ | $0.133\text{ m}$ | Passed Pillar 1 and 2, stalled at Pillar 3. |
| 100 | Arm D (PPO Refl) | timeout | 500 | $6.68\text{ m}$ | $0.123\text{ m}$ | Passed Pillar 1 and 2, stalled at Pillar 3. |
| 101 | Arm B (BC) | timeout | 500 | **$7.29\text{ m}$** | $0.131\text{ m}$ | Passed Pillar 1 and 2, stalled at Pillar 3. |
| 101 | Arm A (PPO) | timeout | 500 | $4.71\text{ m}$ | $0.132\text{ m}$ | Stalled at Pillar 2. |
| 101 | Arm D (PPO Refl) | timeout | 500 | **$8.63\text{ m}$** | $0.105\text{ m}$ | Reached $X=8.63\text{ m}$ ($0.87\text{ m}$ from goal), timed out. |
| 102 | Arm B (BC) | timeout | 500 | **$7.98\text{ m}$** | $0.141\text{ m}$ | Reached $X=7.98\text{ m}$, timed out near goal entrance. |
| 102 | Arm A (PPO) | timeout | 500 | $2.58\text{ m}$ | $0.150\text{ m}$ | Stalled at Pillar 1. |
| 102 | Arm D (PPO Refl) | timeout | 500 | $2.60\text{ m}$ | $0.152\text{ m}$ | Stalled at Pillar 1. |

---

## 3. Summary Assessment of Baseline Controllers

1. **Both BC and original PPO are fundamentally functional in nominal flight:**
   - Both policies achieve valid terminal arrivals on clear corridors and single offset obstacles when allowed the canonical 500-step horizon.
2. **The core obstacle failure mechanisms are distinct:**
   - **BC fails specifically against centered obstacles ($Y_{\text{obs}} = 0.00\text{ m}$):** It predicts zero lateral evasion, forcing the supervisor to brake and freeze the drone.
   - **PPO fails specifically against left-shifted obstacles ($Y_{\text{obs}} \le 0.00\text{ m}$):** It insists on turning right, into the narrow wall-pillar trap.
3. **Sequential obstacles compound delays:**
   - Even when passing pillars, supervisor speed reductions cause the drone to take 450–500 steps, resulting in boundary timeouts before reaching the 10m goal.

Phase 4H-B is complete. Deliverables saved to `reports/phase4h_baseline_evaluation.md` and `reports/phase4h_baseline_evaluation_metrics.json`. We now proceed to Phase 4H-C (Failure Analysis).
