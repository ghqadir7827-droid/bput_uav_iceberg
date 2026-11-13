# Phase 4J-B-D — Paired Caution-Distance Robustness Evaluation Report

## 1. Executive Summary

This report delivers the comprehensive empirical results of the **Phase 4J-B** safety-aware robustness evaluation.

To rigorously resolve the uncertainties identified in Phase 4J-A and test statistical reproducibility, this evaluation executed **1,550 simulation episodes** across:
- **50 Paired Development Seeds:** `2000..2049` (strictly disjoint from the Phase 4J-A dev set `1000..1009` and the reserved held-out seed pool `90000..90099`).
- **6 Caution Distance Settings:** $d_{\text{caut}} \in [0.80\text{ m (canonical)}, 0.70\text{ m}, 0.60\text{ m}, 0.55\text{ m}, 0.50\text{ m}, 0.45\text{ m}]$.
- **5 Navigation Controllers:**
  1. `Expert_APF_Reference` (Privileged Analytical Expert)
  2. `Arm_B_BC_Baseline` (`models/checkpoints/bc_policy_best.pt`)
  3. `BC_Phase4H_Candidate` (`experiments/phase4h/checkpoints/bc_phase4h_candidate.pt`)
  4. `Arm_A_PPO_Baseline` (`models/checkpoints/ppo_policy_best.pt`)
  5. `PPO_Phase4H_Candidate` (`experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt`)
- **Diagnostic Counterfactual:** `Expert_APF_NoSupervisor_Counterfactual` across all 50 seeds.

Every primary comparison strictly preserved the canonical 500-step ($25.0\text{ s}$) horizon, identical initial positions and corridor geometries, identical sensor noise models, and invariant supervisor parameters ($d_{\text{crit}} = 0.30\text{ m}$, $d_{\text{rev}} = 0.15\text{ m}$, $\text{buf} = 0.25\text{ m}$).

---

## 2. Master Robustness Table (50 Paired Seeds per Configuration)

The table below compiles performance across all 31 evaluated configurations:

| $d_{\text{caut}}$ | Controller ID | Success Rate (95% CI) | Obs Collisions | Wall Collisions | Obs Envelope Violations (<8cm) | Global Min Obs Clearance (m) | Mean Fwd Progress (m) | Mean Duration (s) | Throttled Time % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.80 m** | Expert_APF_Reference | 0/50 (0.0% [0.0–7.1%]) | 0 | 0 | 0 | 0.124 | 5.58 | 25.0 | 78.9% |
| 0.80 m | Arm_B_BC_Baseline | 0/50 (0.0% [0.0–7.1%]) | 0 | 0 | 0 | 0.108 | 5.46 | 25.0 | 51.1% |
| 0.80 m | BC_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 3 | 0 | 22 (44.0%) | -0.001 | 5.50 | 24.6 | 55.5% |
| 0.80 m | Arm_A_PPO_Baseline | 0/50 (0.0% [0.0–7.1%]) | 0 | 0 | 0 | 0.096 | 4.35 | 25.0 | 70.4% |
| 0.80 m | PPO_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 0 | 1 | 0 | 0.099 | 4.99 | 25.0 | 70.8% |
| **0.70 m** | Expert_APF_Reference | 0/50 (0.0% [0.0–7.1%]) | 0 | 0 | 0 | 0.124 | 6.25 | 25.0 | 74.8% |
| 0.70 m | Arm_B_BC_Baseline | 4/50 (8.0% [3.2–18.8%]) | 0 | 0 | 0 | 0.103 | 6.02 | 24.9 | 43.4% |
| 0.70 m | BC_Phase4H_Candidate | 1/50 (2.0% [0.4–10.5%]) | 1 | 0 | 22 (44.0%) | -0.001 | 6.04 | 24.7 | 49.6% |
| 0.70 m | Arm_A_PPO_Baseline | 4/50 (8.0% [3.2–18.8%]) | 0 | 0 | 0 | 0.105 | 4.21 | 24.9 | 61.5% |
| 0.70 m | PPO_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 0 | 4 | 0 | 0.105 | 5.29 | 24.9 | 65.3% |
| **0.60 m** | Expert_APF_Reference | 2/50 (4.0% [1.1–13.5%]) | 0 | 0 | 0 | 0.121 | 7.26 | 25.0 | 66.1% |
| 0.60 m | Arm_B_BC_Baseline | 11/50 (22.0% [12.8–35.2%]) | 0 | 0 | 0 | 0.104 | 6.09 | 24.6 | 39.1% |
| 0.60 m | BC_Phase4H_Candidate | 2/50 (4.0% [1.1–13.5%]) | 1 | 0 | 23 (46.0%) | -0.001 | 6.16 | 24.7 | 42.3% |
| 0.60 m | Arm_A_PPO_Baseline | 5/50 (10.0% [4.3–21.4%]) | 0 | 0 | 0 | 0.100 | 4.13 | 24.6 | 50.2% |
| 0.60 m | PPO_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 0 | 8 | 0 | 0.109 | 5.40 | 24.5 | 55.6% |
| **0.55 m** | Expert_APF_Reference | 11/50 (22.0% [12.8–35.2%]) | 0 | 0 | 0 | 0.124 | 7.88 | 24.7 | 56.7% |
| 0.55 m | Arm_B_BC_Baseline | 17/50 (34.0% [22.4–47.8%]) | 0 | 0 | 0 | 0.115 | 6.37 | 23.8 | 34.9% |
| 0.55 m | BC_Phase4H_Candidate | 12/50 (24.0% [14.3–37.4%]) | 1 | 0 | 16 (32.0%) | -0.001 | 7.07 | 24.2 | 33.8% |
| 0.55 m | Arm_A_PPO_Baseline | 5/50 (10.0% [4.3–21.4%]) | 0 | 0 | 0 | 0.093 | 4.18 | 24.5 | 42.7% |
| 0.55 m | PPO_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 0 | 12 | 0 | 0.102 | 5.52 | 24.1 | 49.7% |
| **0.50 m** | Expert_APF_Reference | 20/50 (40.0% [27.6–53.8%]) | 0 | 0 | 0 | 0.124 | 8.31 | 24.2 | 45.4% |
| 0.50 m | **Arm_B_BC_Baseline** | **24/50 (48.0% [34.8–61.5%])** | **0** | **0** | **0** | **0.107** | **6.53** | **23.0** | **27.6%** |
| 0.50 m | BC_Phase4H_Candidate | 17/50 (34.0% [22.4–47.8%]) | 0 | 0 | 7 (14.0%) | 0.009 | 6.93 | 23.8 | 29.3% |
| 0.50 m | Arm_A_PPO_Baseline | 6/50 (12.0% [5.6–23.8%]) | 0 | 0 | 0 | 0.087 | 4.28 | 24.3 | 36.1% |
| 0.50 m | PPO_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 0 | 13 | 0 | 0.114 | 5.70 | 23.8 | 41.7% |
| **0.45 m** | **Expert_APF_Reference** | **36/50 (72.0% [58.3–82.5%])** | **0** | **0** | **0** | **0.118** | **8.48** | **23.4** | **34.8%** |
| 0.45 m | Arm_B_BC_Baseline | 19/50 (38.0% [25.9–51.8%]) | 0 | 0 | 0 | 0.092 | 6.06 | 22.9 | 23.1% |
| 0.45 m | BC_Phase4H_Candidate | 25/50 (50.0% [36.6–63.4%]) | 0 | 0 | 6 (12.0%) | 0.037 | 7.11 | 23.1 | 21.7% |
| 0.45 m | Arm_A_PPO_Baseline | 6/50 (12.0% [5.6–23.8%]) | 0 | 0 | 0 | 0.087 | 4.44 | 24.2 | 29.2% |
| 0.45 m | PPO_Phase4H_Candidate | 0/50 (0.0% [0.0–7.1%]) | 0 | 17 | 0 | 0.101 | 6.18 | 23.2 | 31.3% |
| **No Sup** | **Expert Counterfactual** | **50/50 (100.0% [92.9–100.0%])** | **0** | **0** | **1 (2.0%)** | **0.000** | **8.66** | **19.9** | **0.0%** |

---

## 3. Critical Findings and Deep Analysis

### 3.1 The $0.50\text{ m}$ vs. $0.45\text{ m}$ Comparison
A central question from Phase 4J-A was whether $0.50\text{ m}$ or $0.45\text{ m}$ is the superior configuration:

1. **For the Analytical Expert:**
   - Success rate jumps from **$40.0\%$ (20/50) at $0.50\text{ m}$** to **$72.0\%$ (36/50) at $0.45\text{ m}$**.
   - Obstacle collisions remain **0**, boundary collisions remain **0**, and envelope violations remain **0**.
   - Minimum obstacle clearance at $0.45\text{ m}$ remains comfortably safe at **$0.118\text{ m}$** ($38\text{ mm}$ above the required $80\text{ mm}$ envelope).
   - Because the expert has mathematically optimal potential-field steering, it fully leverages the reduced throttling to reach the goal within the 25.0 s budget without any loss of safety margin.

2. **For the Canonical BC Baseline (`bc_policy_best.pt`):**
   - Success rate is **higher at $0.50\text{ m}$ (48.0%, 24/50)** than at **$0.45\text{ m}$ (38.0%, 19/50)**!
   - At $0.50\text{ m}$, the baseline controller has **0 obstacle collisions, 0 boundary collisions, and 0 envelope violations** across all 50 seeds, maintaining a global minimum clearance of **$0.107\text{ m}$**.
   - At $0.45\text{ m}$, entering obstacle gaps at higher forward velocity increases lateral overshoot during S-turns. This triggers more active boundary repulsions ($v_y \mathrel{-}= \text{repulsion}$), slowing the vehicle down and causing 5 additional episodes to time out.
   - **Conclusion:** For the deployable neural baseline, **$0.50\text{ m}$ is the empirical peak**.

3. **For the BC Phase 4H Candidate:**
   - Although its success rate increases ($34.0\% \to 50.0\%$), it suffers **severe envelope violations**:
     - At $0.50\text{ m}$: 7 episodes violated the $80\text{ mm}$ envelope, reaching down to $9\text{ mm}$ air gap.
     - At $0.45\text{ m}$: 6 episodes violated the $80\text{ mm}$ envelope, reaching down to $37\text{ mm}$ air gap.
   - Across $0.80\text{ m}\dots 0.55\text{ m}$, the Phase 4H candidate suffered actual physical obstacle collisions ($1\dots 3$ collisions) and $16\dots 23$ envelope violations!
   - **Diagnosis:** The Phase 4H BC fine-tuning prioritized forward velocity at the cost of cutting corners tightly around obstacles. The canonical baseline `bc_policy_best.pt` is **vastly superior in obstacle safety compliance**.

4. **For PPO Policies:**
   - Arm A Baseline achieves modest progress ($4.28\dots 4.44\text{ m}$), but remains prone to deadlocks.
   - PPO Phase 4H Candidate suffers an **explosion in boundary collisions**:
     - $0.80\text{ m}$: 1 collision (2%)
     - $0.70\text{ m}$: 4 collisions (8%)
     - $0.60\text{ m}$: 8 collisions (16%)
     - $0.55\text{ m}$: 12 collisions (24%)
     - $0.50\text{ m}$: 13 collisions (26%)
     - $0.45\text{ m}$: 17 collisions (34%)
   - When the forward throttle is relaxed, PPO accelerates forward into obstacles, but its severe lateral instability and total absence of goal deceleration cause catastrophic boundary impacts against side walls and the terminal back wall.

---

## 4. Diagnostic Counterfactual Reproduction (Phase 4J-B-F)

The unconstrained counterfactual (supervisor completely disabled) was evaluated across all 50 paired seeds:
- **Success Rate:** **50/50 (100.0% [92.9–100.0%])**, mean completion time **$19.92\text{ s}$**.
- **Collisions:** **0 collisions**.
- **Crucial Safety Finding (Seed 2033):**
  - In Seed 2033, without the supervisor, the expert vehicle passed within **$0.0003\text{ m}$ ($0.3\text{ mm}$)** of Pillar 1—a hair's breadth from physical collision.
  - When the supervisor was active ($d_{\text{caut}} = 0.50\text{ m}$ or $0.45\text{ m}$), the supervisor's lateral barrier and forward caution intervened, maintaining minimum clearance at **$0.124\text{ m}$**.
- **Takeaway:** The supervisor is actively preventing near-miss grazing events that even the privileged analytical expert controller cannot avoid in edge-case seed geometries. Safety protection must never be removed from the operational architecture.

---

## 5. Summary of Saved Artifacts
- Master episode traces and per-step telemetry: `experiments/phase4j/phase4jb_episode_metrics.json`
- Aggregated JSON summary matrix: `experiments/phase4j/phase4jb_summary.json`
- Step-by-step failure and envelope violation traces: `experiments/phase4j/phase4jb_traces/`

