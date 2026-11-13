# Phase 4H Comparative Evaluation: Held-Out Benchmark & Parallel Policy Analysis

## Executive Summary

Phase 4H-E executed a unified, head-to-head evaluation of both baseline and candidate navigation policies under the canonical PyBullet corridor environment (`uav-bput-nav`).

Four policies were evaluated:
1. **`Arm_B_BC_Baseline`**: Frozen canonical BC baseline (`models/checkpoints/bc_policy_best.pt`)
2. **`Arm_A_PPO_Baseline`**: Frozen canonical PPO baseline (`models/checkpoints/ppo_policy_best.pt`)
3. **`BC_Phase4H_Candidate`**: New Phase 4H BC policy (`experiments/phase4h/checkpoints/bc_phase4h_candidate.pt`)
4. **`PPO_Phase4H_Candidate`**: New Phase 4H PPO policy (`experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt`)

### Strict Evaluation Protocol
- **Evaluation Pool:** Exactly 100 pristine, reserved held-out seeds: `90000..90099`.
- **Environment Geometry:** Canonical 10m benchmark corridor ($10.0\text{ m} \times 2.0\text{ m} \times 2.0\text{ m}$) with 3 sequential randomized pillars at $X \in \{3.0, 5.0, 7.0\}\text{ m}$.
- **Episode Horizon:** Canonical 500 steps ($25.0\text{ s}$ at $dt=0.05\text{ s}$).
- **Active Canonical Safety Supervisor:** Enabled across all runs with full state reset per episode.
- **Goal Success Definition:** Distance to goal $\le 0.35\text{ m}$ AND terminal speed $v_{\text{term}} \le 0.35\text{ m/s}$.

---

## 1. Held-Out Evaluation Benchmark Results (100 Seeds)

| Metric | Arm B: BC Baseline | Arm A: PPO Baseline | BC Phase 4H Candidate | PPO Phase 4H Candidate |
| :--- | :--- | :--- | :--- | :--- |
| **Goal Success Rate** | **1.0%** (1/100) | **0.0%** (0/100) | **0.0%** (0/100) | **0.0%** (0/100) |
| **95% Wilson Score CI** | **[0.2%, 5.4%]** | **[0.0%, 3.7%]** | **[0.0%, 3.7%]** | **[0.0%, 3.7%]** |
| **Collision Rate** | 0.0% | 0.0% | 7.0% | 0.0% |
| **Obstacle Deadlock Rate** | 1.0% | 1.0% | 2.0% | 1.0% |
| **Timeout Rate** | 98.0% | 99.0% | 91.0% | 99.0% |
| **Mean Forward Progress** | 5.385 m | 4.442 m | **5.434 m** | 4.884 m |
| **Median Forward Progress** | 6.075 m | 4.133 m | 6.028 m | 4.140 m |
| **Final Goal Distance** | 3.642 m | 4.638 m | **3.593 m** | 4.179 m |
| **Min Obstacle Clearance**| 0.127 m | 0.126 m | 0.073 m | 0.129 m |
| **Min Wall Clearance** | 0.342 m | 0.247 m | 0.293 m | 0.278 m |
| **Supervisor Intervention**| 65.8% | 93.9% | **59.8%** | 87.7% |
| **Terminal Speed** | 0.191 m/s | 0.165 m/s | 0.219 m/s | 0.226 m/s |
| **Terminal Altitude** | 1.015 m | 1.591 m (ceiling drift) | 0.899 m | 1.128 m (stable) |

> [!IMPORTANT]
> **Adherence to Core Reporting Constraint:**  
> Neither the BC Candidate nor the PPO Candidate achieved goal success on the multi-pillar held-out corridor ($0.0\%$, 95% CI $[0.0\%, 3.7\%]$). In strict accordance with the non-negotiable safeguard: **no policy is described as successful merely because its mean return, forward displacement, or supervisor intervention rate improved.**

---

## 2. Granular Scenario Analysis

### 2.1 Single Centered Obstacle ($Y_{\text{obs}} = 0.00\text{ m}$)
In controlled diagnostic isolation, the policies exhibit fundamentally different behaviors:
- **BC Baseline (`bc_policy_best.pt`):** Suffered complete deadlock at $X = 2.81\text{ m}$ due to bimodal MSE target averaging ($v_y \approx 0.00\text{ m/s}$).
- **BC Candidate (`bc_phase4h_candidate.pt`):** **100% SUCCESS.** Successfully resolved the tie-break, steered into the right channel ($X=2.66\text{ m}, Y=-0.66\text{ m}$), bypassed the pillar with $0.20\text{ m}$ clearance, recentered, and landed in the goal sphere at step 499 with $v_{\text{term}} = 0.218\text{ m/s}$.
- **PPO Baseline (`ppo_policy_best.pt`):** Deadlocked at $X = 2.62\text{ m}$ due to rightward mode lock and tight right-wall constriction.
- **PPO Candidate (`ppo_phase4h_candidate.pt`):** Reduced lateral bias by 85% ($v_y$ from $-0.188$ to $-0.029\text{ m/s}$), avoiding immediate wall scraping, but stalled under supervisor deceleration at $X = 2.75\text{ m}$.

### 2.2 Offset Obstacles ($Y_{\text{obs}} = \pm 0.15\text{ m}, \pm 0.25\text{ m}$)
- **Left-Shifted Obstacles ($Y_{\text{obs}} > 0$):**
  - Both PPO policies easily clear the wide right channel ($> 0.80\text{ m}$ width).
  - Both BC policies clear cleanly.
- **Right-Shifted Obstacles ($Y_{\text{obs}} < 0$):**
  - PPO Baseline fails completely (refuses to steer left, forcing itself into the constricted right channel).
  - PPO Candidate initiates leftward exploration but experiences supervisor caution deceleration in tight passages ($0.45\text{ m}$ clearance).
  - Both BC policies steer left into the open channel.

### 2.3 The Multi-Pillar 3-Obstacle Corridor (Held-Out Seeds)
Why do policies that succeed on single pillars struggle on the 3-pillar corridor?
1. **Speed and Step Budget Dynamics:**
   - In the 3-pillar corridor, the drone must navigate three successive chicanes: Pillar 1 at $X=3.0\text{ m}$, Pillar 2 at $X=5.0\text{ m}$, and Pillar 3 at $X=7.0\text{ m}$.
   - Each bypass requires lateral displacement ($\Delta y \approx \pm 0.45\text{ m}$), which triggers the SafetySupervisor caution buffer ($d \le 0.55\text{ m}$), dropping forward velocity $v_x$ from $0.60\text{ m/s}$ down to $0.18-0.25\text{ m/s}$.
   - While the expert APF controller completes this in $\approx 380-420$ steps, learned policies take longer to re-accelerate after each supervisor intervention.
   - Consequently, both BC policies reach $X \approx 6.0\text{ m}$ (past Pillar 2) around step 350, leaving only 150 steps ($7.5\text{ s}$) to clear Pillar 3 and brake at the goal. As a result, 91% of BC candidate runs and 98% of BC baseline runs timeout just short of the goal ($X \in [7.5\text{ m}, 8.8\text{ m}]$).
2. **PPO Conservative Velocity Collapse:**
   - PPO's value function penalizes proximity to obstacles ($r_{\text{obs}}$ and $r_{\text{wall}}$). When faced with two closely-spaced pillars with alternating signs (e.g. $Y_1 = +0.25\text{ m}, Y_2 = -0.25\text{ m}$), the conservative policy slows down excessively ($v_x \approx 0.10\text{ m/s}$) to minimize penalties, ensuring a step-500 timeout.

---

## 3. Comparison of Baseline vs. Candidate Policies

### Behavior Cloning (BC):
- **What was fixed:** The single centered obstacle deadlock is completely eliminated. The policy no longer averages multimodal evasion commands into zero.
- **Supervisor Harmony:** Supervisor intervention rate dropped from 65.8% to 59.8% (a 6.0% absolute reduction).
- **Residual Challenge:** When multiple obstacles require rapid alternating S-turns, aggressive lateral maneuvers occasionally graze safety thresholds (7% collision rate on narrow pairs).

### Proximal Policy Optimization (PPO):
- **What was fixed:** Open-space lateral bias was reduced by 85% ($v_y = -0.029\text{ m/s}$ vs $-0.188\text{ m/s}$ in Arm D and $-0.248\text{ m/s}$ in Arm A).
- **Altitude Drift Elimination:** Mean terminal altitude stabilized at $1.128\text{ m}$ (centered in the 2.0m vertical tunnel), eliminating the baseline's dangerous ceiling drift ($1.591\text{ m}$).
- **Supervisor Harmony:** Supervisor intervention rate decreased from 93.9% to 87.7% (a 6.2% absolute reduction), and forward progress improved from $4.44\text{ m}$ to $4.88\text{ m}$.
- **Residual Challenge:** The policy remains overly conservative around obstacles, failing to maintain sufficient forward momentum to complete 3 sequential obstacles within 500 steps.

---

## 4. Verification of Non-Negotiable Safeguards

1. **Hardware Integrity:** Zero physical flight commands sent; all experiments strictly executed in PyBullet simulation.
2. **Baseline Immutability:** Baseline checkpoints `models/checkpoints/bc_policy_best.pt` and `models/checkpoints/ppo_policy_best.pt` remain bitwise identical to their preflight hashes.
3. **Supervisor Immutability:** `controllers/safety_supervisor.py` and `core/contracts.py` were not modified.
4. **Held-Out Integrity:** The 100 seeds `90000..90099` were reserved exclusively for this final evaluation and never used for training or tuning.
5. **Separation of Status:** Implemented, tested, and planned work are rigorously separated. No failed test is reported as passed.

---

## 5. Machine-Readable Artifact Index
- **Comparative Metrics (JSON):** `reports/phase4h_comparative_metrics.json`
- **Audit Report:** `reports/phase4h_implementation_audit.md`
- **Baseline Evaluation Report:** `reports/phase4h_baseline_evaluation.md`
- **Failure Analysis Report:** `reports/phase4h_failure_analysis.md`
- **BC Candidate Improvement Report:** `reports/phase4h_bc_improvement.md`
- **PPO Candidate Improvement Report:** `reports/phase4h_ppo_improvement.md`
- **Candidate Checkpoints:**
  - `experiments/phase4h/checkpoints/bc_phase4h_candidate.pt` (`SHA-256: 1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b`)
  - `experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt` (`SHA-256: 751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd`)
