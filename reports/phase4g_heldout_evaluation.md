# Phase 4G-F: Final Held-Out Evaluation Report

**Date:** 2026-10-29  
**Status:** COMPLETED & OBJECTIVELY ASSESSED  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Deliverable Files:**
- `reports/phase4g_heldout_evaluation.md`
- `reports/phase4g_heldout_metrics.json`
- Evaluated Checkpoint: `experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`  
  (`SHA256: 7d739ea8ba58ef3bb8ef198a2eb839c4a75e24a4692fc3f86eb16fcb6703b4e3`)
- Baseline Checkpoint: `models/checkpoints/ppo_policy_best.pt`  
  (`SHA256: 33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0`)

---

## 1. Executive Summary

Phase 4G-F executed the pre-registered, strictly isolated 100-seed held-out evaluation on the designated untouched seed pool (`80000` to `80099`).

This evaluation directly compares:
- **Arm A — Frozen PPO Baseline** (`models/checkpoints/ppo_policy_best.pt`)
- **Arm D — Phase 4G Candidate (PPO with Reflection Augmentation)** (`experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`)
- **Arm B — Frozen BC Reference** (`models/checkpoints/bc_policy_best.pt`)

### Core Findings
1. **Primary Metric (Goal Success Rate): 0.0% (0/100) for Both PPO Controllers**
   - Neither Arm A (0/100, 0.0%, 95% CI: [0.0, 3.7]%) nor Arm D (0/100, 0.0%, 95% CI: [0.0, 3.7]%) achieved terminal arrival at the 10m goal.
   - In accordance with Section 8 Acceptance Criteria: **"If success remains zero, the phase cannot claim successful navigation improvement even if reward, displacement, or action symmetry improves."**
   - The primary navigation improvement gate is therefore classified as **NOT MET / FAIL**.
2. **Secondary Metric (Policy Symmetry): Substantial, Statistically Meaningful Improvement**
   - Open-space yaw-rate rotational bias was reduced by **59.8%** (from $-0.1537\text{ rad/s}$ in Arm A to $\mathbf{-0.0618\text{ rad/s}}$ in Arm D).
   - In training iteration comparisons, Arm D reduced yaw-rate bias by **70.0%** relative to Arm C (standard PPO).
   - Open-space lateral drift improved from $v_y = -0.2266\text{ m/s}$ to $v_y = -0.1886\text{ m/s}$.
3. **Safety & Collision Integrity:**
   - Collision rate remained **0.0% (0/100)** across all 100 held-out episodes for all controllers, confirming that the canonical `SafetySupervisor` prevents crashes even during deadlock.
   - Mean obstacle clearance remained safe: $0.152\text{ m}$ (Arm A) and $0.146\text{ m}$ (Arm D).
4. **Root Cause of Persistent Zero Success:**
   - Drones stall in front of the first pillar obstacle ($X \approx 2.3 - 3.2\text{ m}$).
   - While reflection data augmentation successfully attenuated the policy's rotational drift in open space, the policy has not learned an active lateral circumnavigation maneuver to clear obstacles. When confronted with an obstacle, the policy pushes forward, the safety supervisor intervenes with an obstacle brake (reducing forward speed to $< 0.05\text{ m/s}$), and the drone remains locked in a stationary deadlock until the 200-step horizon expires.

---

## 2. Statistical Held-Out Benchmark Results (100 Paired Seeds)

| Metric | Arm A (Frozen PPO) | Arm D (Phase 4G Candidate) | Arm B (Frozen BC) | Difference (D vs A) |
| :--- | :---: | :---: | :---: | :---: |
| **End-to-End Success Rate** | **0.0% (0/100)** | **0.0% (0/100)** | **0.0% (0/100)** | $0.0\text{ pp}$ |
| **Wilson 95% Score CI** | $[0.0, 3.7]\%$ | $[0.0, 3.7]\%$ | $[0.0, 3.7]\%$ | — |
| **Collision Rate** | $0.0\% (0/100)$ | $0.0\% (0/100)$ | $0.0\% (0/100)$ | $0.0\text{ pp}$ |
| **Wall Collision Rate** | $0.0\% (0/100)$ | $0.0\% (0/100)$ | $0.0\% (0/100)$ | $0.0\text{ pp}$ |
| **Horizon Timeout Rate** | $100.0\% (100/100)$ | $100.0\% (100/100)$ | $100.0\% (100/100)$ | $0.0\text{ pp}$ |
| **Obstacle Deadlock Rate** | $100.0\% (100/100)$ | $100.0\% (100/100)$ | $100.0\% (100/100)$ | $0.0\text{ pp}$ |
| **Mean Forward Displacement** | $2.852 \pm 0.085\text{ m}$ | $2.694 \pm 0.081\text{ m}$ | $3.246 \pm 0.098\text{ m}$ | $-0.158\text{ m}$ |
| **Median Forward Displacement**| $2.270\text{ m}$ | $2.246\text{ m}$ | $3.553\text{ m}$ | $-0.024\text{ m}$ |
| **Mean Final Goal Distance** | $6.178\text{ m}$ | $6.336\text{ m}$ | $5.782\text{ m}$ | $+0.158\text{ m}$ |
| **Mean Min Obstacle Clearance**| $0.152\text{ m}$ | $0.146\text{ m}$ | $0.189\text{ m}$ | $-0.006\text{ m}$ |
| **Mean Min Wall Clearance** | $0.354\text{ m}$ | $0.358\text{ m}$ | $0.345\text{ m}$ | $+0.004\text{ m}$ |
| **Supervisor Intervention Rate**| $89.56\%$ | $89.07\%$ | $49.98\%$ | $-0.49\text{ pp}$ |
| **Total Intervention Steps** | $17,912 / 20,000$ | $17,814 / 20,000$ | $9,997 / 20,000$ | $-98\text{ steps}$ |
| **Open-Space Center $v_y$** | $-0.2266\text{ m/s}$ | $\mathbf{-0.1886\text{ m/s}}$ | $-0.0002\text{ m/s}$ | **+16.8% center** |
| **Open-Space Center $\dot{\psi}$**| $-0.1537\text{ rad/s}$ | $\mathbf{-0.0618\text{ rad/s}}$ | $-0.0003\text{ rad/s}$ | **-59.8% bias** |

---

## 3. Subgroup Performance Analysis

Episodes across the 100 held-out seeds were partitioned by the lateral position of Obstacle 1 ($X = 3.0\text{ m}$):

### 3.1 Subgroup 1: Left-Shifted Obstacles ($Y_{\text{obs}} \in [-0.40, -0.10]\text{ m}$, $N = 34$)
- Wide gap on Right, constrained gap on Left.
- Arm A Mean Displacement: $2.84\text{ m}$
- Arm D Mean Displacement: $2.68\text{ m}$
- Arm B Mean Displacement: $3.31\text{ m}$
- Finding: Both PPO policies approach the obstacle and stall in the caution buffer ($0.14 - 0.17\text{ m}$ clearance).

### 3.2 Subgroup 2: Center Obstacles ($Y_{\text{obs}} \in [-0.10, +0.10]\text{ m}$, $N = 31$)
- Symmetric corridor geometry.
- Arm A Mean Displacement: $2.79\text{ m}$
- Arm D Mean Displacement: $2.65\text{ m}$
- Arm B Mean Displacement: $3.15\text{ m}$
- Finding: Direct forward approach triggers supervisor emergency brake at $X \approx 2.2\text{ m}$. Neither policy generates sufficient lateral velocity setpoint to bypass without colliding.

### 3.3 Subgroup 3: Right-Shifted Obstacles ($Y_{\text{obs}} \in [+0.10, +0.40]\text{ m}$, $N = 35$)
- Wide gap on Left, constrained gap on Right.
- Arm A Mean Displacement: $2.92\text{ m}$
- Arm D Mean Displacement: $2.75\text{ m}$
- Arm B Mean Displacement: $3.28\text{ m}$
- Finding: PPO's residual rightward bias leads it into the narrow right passage, where the supervisor decelerates it to maintain $0.15\text{ m}$ clearance.

---

## 4. Failure Mode Diagnosis

The held-out evaluation confirms that **policy asymmetry is only one contributor to navigation failure**:
1. **Symmetry vs. Active Avoidance:**
   Bilateral reflection augmentation successfully halved the yaw-rate asymmetry of the policy. However, reducing bias does not inherently provide the drone with a positive bypass vector. In open space, the drone is more centered; in front of an obstacle, the drone still lacks a policy gradient driving it laterally around obstacles before entering the supervisor caution buffer.
2. **Supervisor Deadlock Interaction:**
   As established in Phase 4D and Phase 4E, when a forward obstacle is detected within $0.80\text{ m}$, the canonical `SafetySupervisor` caps forward velocity to $0.49\text{ m/s}$ (or lower). If the policy does not output a strong lateral velocity command ($|v_y| > 0.3\text{ m/s}$), forward progress stops, resulting in a 200-step timeout.
3. **Training Budget and Intervention Limits:**
   In accordance with the non-negotiable rules of Phase 4G, only a single intervention (reflection augmentation) was permitted, and the reward function was kept strictly unmodified. A 10-iteration budget was sufficient to reshape rotational weights, but insufficient to discover non-zero reward paths through multi-pillar corridors.

---

## 5. Acceptance Criteria Evaluation

| Criteria Category | Requirement | Result | Evaluation |
| :--- | :--- | :---: | :---: |
| **A. Reflection Implementation** | Mathematical correctness, ray permutation verified, invariants pass, contract unchanged | 7/7 tests pass | **PASS** |
| **B. Training Integrity** | Arms C & D identical budgets, checkpoints isolated, baseline untouched | 20,480 steps/arm | **PASS** |
| **C. Policy Improvement** | Evidence of improved bypass selection and symmetry | Yaw bias -60% | **PARTIAL** |
| **D. Final Evaluation** | Statistically significant increase in end-to-end goal success | 0/100 (0.0%) | **FAIL** |
| **E. System Integrity** | No hardware calls, no supervisor edits, baseline checkpoints immutable | Fully compliant | **PASS** |

### Final Engineering Recommendation
The candidate checkpoint `experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt` successfully achieves the secondary objective of reducing open-space rotational bias by 60%, but fails the primary objective of improving end-to-end navigation success (success remains 0.0%).

**Recommendation:** Do NOT integrate `ppo_arm_d_reflection.pt` into the canonical runtime. Retain reflection augmentation as a verified component of future training pipelines, but recognize that end-to-end navigation success requires addressing obstacle-bypass guidance (e.g. gap-clearance rewards or supervisor hysteresis release) before physical deployment can be considered.
