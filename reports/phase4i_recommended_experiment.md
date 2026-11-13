# Phase 4I Prioritized Intervention Plan & Experiment Proposal

## Executive Summary

Phase 4I isolated the root cause behind zero goal successes across the three-pillar corridor:
1. **The Structural Horizon Bottleneck:** With three pillars spaced $2.0\text{ m}$ apart and an $0.80\text{ m}$ forward caution zone throttling forward speed to $\approx 0.15\text{ m/s}$, the minimum duration required to traverse the corridor collision-free is **$35.1\text{ seconds}$ ($702\text{ steps}$)**. The canonical $500\text{ step}$ ($25.0\text{ s}$) horizon is mathematically insufficient.
2. **The Counterfactual Proof:** In the absence of the supervisor's forward velocity throttling, the identical APF expert controller completes the multi-pillar corridor in **$384.8\text{ steps}$ ($19.24\text{ s}$)** with **100% success** and zero collisions.
3. **The Policy-Specific Defects:**
   - **PPO** suffers from Class 2 Unilateral Mode Lock, stalling in front of right-shifted pillars.
   - **BC Candidate** achieves steady transit and clears obstacles, but requires tighter chicane smoothing to prevent Class 4 corner clips on narrow alternating pairs.

Based on this evidence, this report presents a disciplined, single-variable prioritized intervention plan for Phase 4J.

---

## 1. Candidate Intervention Comparison

| Intervention Candidate | Target Failure Class | Core Mechanism | Expected Effect | Risk Level | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1: Isolated Supervisor Velocity-Profile Optimization** | **Class 1 (Structural Throttle Timeout)** | Tighten caution zone from $0.80\text{ m}$ to $0.55\text{ m}$; keep critical stop ($0.30\text{ m}$) untouched | Allows cruise speed closer to obstacle, cutting transit time by $8 - 10\text{ s}$ | Very Low (critical brake unchanged) | **Highest (Primary)** |
| **Option 2: Horizon Sensitivity Benchmark (750 Steps)** | **Class 1 (Verification Probe)** | Diagnostic test evaluating existing policies under $750\text{ steps}$ ($37.5\text{ s}$) | Falsifies whether existing policies reach goal given adequate time | Zero (pure diagnostic) | **High (Diagnostic)** |
| **Option 3: Multi-Obstacle Chicane BC Demonstrations** | **Class 4 (Chicane Corner Clip)** | Train BC on explicit alternating S-turns ($Y_1 = +0.25, Y_2 = -0.25$) | Eliminates the 7% corner collision rate on tight pillar pairs | Low | **Medium (Secondary)** |
| **Option 4: PPO Bilateral Exploration Penalty** | **Class 2 (Unilateral Mode Lock)** | Add entropy regularization or asymmetric wall penalties during PPO training | Breaks the remaining rightward bias on right-shifted obstacles | Medium | **Medium (Secondary)** |

---

## 2. Recommended Primary Intervention for Phase 4J

### Proposed Experiment:
**Phase 4J-A: Isolated Evaluation of Dynamic Supervisor Caution Throttling & Horizon Sensitivity**

Do **not** modify the canonical `SafetySupervisor` directly. Implement an isolated test module: `experiments/phase4j/isolated_supervisor_opt.py`.

### 2.1 The Observed Failure It Targets
- **Failure Class 1 (Cumulative Throttle Timeout):** The UAV spends $70.7\%$ of its time budget throttled to $0.15\text{ m/s}$ because the caution threshold ($0.80\text{ m}$) overlaps across adjacent pillars in a $2.0\text{ m}$ tunnel.

### 2.2 Evidence Supporting the Intervention
- In Phase 4I, the APF expert achieved **10/10 (100%) goal success** in $384.8\text{ steps}$ without the supervisor, maintaining a minimum obstacle clearance of $0.169\text{ m}$ (well above the $0.08\text{ m}$ safety requirement).
- Under the canonical supervisor, the identical expert achieved **0/10 goal success**, spending only $0.6\%$ of its time stopped, but running out of time at $X = 5.68\text{ m}$.
- The $0.80\text{ m}$ caution distance was designed for high-speed open environments. In a $2.0\text{ m}$ corridor, an obstacle at $X = 3.0\text{ m}$ triggers throttling at $X = 2.20\text{ m}$, leaving only $0.85\text{ m}$ of free space before Pillar 2.

### 2.3 Proposed Parameter Adjustments (Tested in Isolation)
| Supervisor Parameter | Canonical Value | Proposed Isolated Value | Rationale |
| :--- | :--- | :--- | :--- |
| `obstacle_caution_dist_m` | `0.80 m` | **`0.55 m`** | Eliminates inter-pillar caution overlap; drone cruises until $0.55\text{ m}$ from obstacle |
| `critical_obstacle_dist_m` | `0.30 m` | **`0.30 m`** | **UNTOUCHED.** Full emergency braking distance preserved |
| `emergency_reverse_dist_m` | `0.15 m` | **`0.15 m`** | **UNTOUCHED.** Emergency reverse threshold preserved |
| `boundary_buffer_m` | `0.25 m` | **`0.25 m`** | **UNTOUCHED.** Wall safety envelope preserved |

### 2.4 Expected Measurable Effect
- Reduces cumulative throttled transit time through three obstacles from **$19.2\text{ seconds}$** down to **$11.5\text{ seconds}$**.
- Reduces total ideal corridor flight duration from $35.1\text{ s}$ down to **$23.8\text{ seconds}$ ($476\text{ steps}$)**.
- Brings collision-free multi-pillar navigation within the canonical $500\text{ step}$ ($25.0\text{ s}$) budget.

### 2.5 Safety Risks & Mitigations
- **Risk:** Bringing the caution zone closer to the obstacle might increase the risk of collision if the policy fails to steer laterally.
- **Mitigation:** The critical brake threshold (`critical_obstacle_dist_m = 0.30m`) and emergency reverse (`0.15m`) remain strictly unchanged. At max speed ($0.60\text{ m/s}$) and deceleration ($1.0\text{ m/s}^2$), braking from $0.60\text{ m/s}$ to $0.0\text{ m/s}$ requires $\Delta x = v^2 / (2a) = 0.36 / 2.0 = 0.18\text{ m}$, which is comfortably within the $0.30\text{ m}$ critical buffer.

### 2.6 Falsification Criteria
The hypothesis will be **falsified** if:
1. Tightening `obstacle_caution_dist_m` to $0.55\text{ m}$ causes any collision on the 10 development seeds.
2. The expert controller or BC candidate fails to clear all three pillars within 500 steps under the optimized caution profile.

---

## 3. Secondary Experiment Plan (Phase 4J-B)

If Phase 4J-A validates the supervisor velocity profile, execute:
- **BC Multi-Chicane Retraining:** Augment `bc_train_phase4h.npz` with 30 explicit alternating chicane demonstrations ($Y_1 = +0.25, Y_2 = -0.25, Y_3 = +0.25$) to eliminate the 7% corner clipping rate.
- **PPO Bilateral Regularization:** Fine-tune PPO with symmetric left/right reward balancing to eliminate Class 2 rightward constriction stalling.
- **Unified Held-Out Re-Evaluation:** Evaluate all updated models on the reserved held-out seeds (`90000..90099`).
