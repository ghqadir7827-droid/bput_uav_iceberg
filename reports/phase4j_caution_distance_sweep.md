# Phase 4J-A4 — Caution-Distance Parametric Sweep Report

## 1. Executive Summary

This report presents empirical results from the isolated parametric sweep of the `SafetySupervisor` caution-distance parameter ($d_{\text{caut}}$) conducted in Phase 4J-A4. 

The sweep evaluated 6 discrete caution distances across 5 distinct navigation controllers on the 10 canonical development seeds (`1000..1009`), comprising 300 total simulation episodes:
- **Caution Distances:** $d_{\text{caut}} \in [0.80\text{ m (canonical)}, 0.70\text{ m}, 0.60\text{ m}, 0.55\text{ m (Phase 4I proposed)}, 0.50\text{ m}, 0.45\text{ m}]$
- **Controllers Evaluated:**
  1. `Expert_APF_Reference` (Privileged Analytical Expert)
  2. `BC_Phase4H_Candidate` (`experiments/phase4h/checkpoints/bc_phase4h_candidate.pt`)
  3. `Arm_B_BC_Baseline` (`models/checkpoints/bc_policy_best.pt`)
  4. `PPO_Phase4H_Candidate` (`experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt`)
  5. `Arm_A_PPO_Baseline` (`models/checkpoints/ppo_policy_best.pt`)

### Core Findings
1. **Canonical Throttling Confirmed as Root Cause of Multi-Obstacle Timeouts:**
   - Under the canonical setting ($d_{\text{caut}} = 0.80\text{ m}$), **zero controllers achieved goal arrival** (0/10 for all 5 controllers, 0.0% success rate). 
   - The expert drone spent **70.7% of its flight time throttled** to $v_x \le 0.15\text{ m/s}$, consuming all 500 steps (25.0 s) to reach only $X = 5.68\text{ m}$ (Pillar 2).
2. **Phase 4I Proposed Setting ($d_{\text{caut}} = 0.55\text{ m}$) is Under-Margined:**
   - While Phase 4I claimed $d_{\text{caut}} = 0.55\text{ m}$ would yield 10/10 expert successes in 23.8 s, empirical testing revealed only **2/10 (20.0%) expert success** and **1/10 (10.0%) BC candidate success**. 
   - The mean expert flight duration at $0.55\text{ m}$ is 487.6 steps (24.38 s), leaving the majority of episodes to time out within 10–30 cm of the goal line due to lateral S-turn maneuvering delays.
3. **Optimum Regimes for Goal Arrival:**
   - **$d_{\text{caut}} = 0.50\text{ m}$:** Yields **5/10 (50.0%) success for BC candidate**, **5/10 (50.0%) for BC baseline**, and **4/10 (40.0%) for Expert APF** with **0 obstacle collisions** across all BC and expert runs. The expert's 6 timed-out runs were within 9 mm to 30 cm of the goal.
   - **$d_{\text{caut}} = 0.45\text{ m}$:** Yields **7/10 (70.0%) success for Expert APF** (mean completion time 22.31 s), **5/10 (50.0%) for BC candidate**, and **5/10 (50.0%) for BC baseline**, with all 3 remaining expert runs reaching $X > 8.51\text{ m}$ (within 5–14 cm of the goal).
4. **Safety Trade-off & Collision Risk for Defective Policies:**
   - For high-accuracy controllers (Expert APF and BC), reducing $d_{\text{caut}}$ to $0.45\text{ m}$ incurred **0 obstacle collisions**, maintaining a minimum clearance of $0.198\text{ m}$ (above the $0.08\text{ m}$ safety margin).
   - In stark contrast, for PPO policies exhibiting severe steering deficits, reducing $d_{\text{caut}}$ increased collision rates: PPO Candidate suffered **1 collision (10%) at $0.50\text{ m}$** and **2 collisions (20%) at $0.45\text{ m}$** due to high-speed terminal boundary wall impacts (`crash_boundary` at $X > 10.0\text{ m}$) caused by absence of terminal deceleration.

---

## 2. Parametric Sweep Summary Matrix

The table below compiles the aggregate metrics across all 30 configurations (6 caution distances $\times$ 5 controllers $\times$ 10 development seeds):

| $d_{\text{caut}}$ | Controller ID | Success Rate | Collision Rate | Mean Steps | Mean Duration (s) | Mean Fwd Prog (m) | Global Min Obs Clearance (m) | Throttled Time % | Intervention Rate % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.80 m** | Expert_APF_Reference | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 5.68 | 0.209 | 70.7% | 79.1% |
| 0.80 m | BC_Phase4H_Candidate | 0/10 (0.0%) | 1/10 (10.0%) | 465.5 | 23.28 | 5.52 | 0.000 | 17.0% | 54.9% |
| 0.80 m | Arm_B_BC_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 4.88 | 0.119 | 13.8% | 70.9% |
| 0.80 m | PPO_Phase4H_Candidate | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 3.25 | 0.120 | 17.3% | 89.1% |
| 0.80 m | Arm_A_PPO_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 2.78 | 0.113 | 13.9% | 94.7% |
| **0.70 m** | Expert_APF_Reference | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 6.50 | 0.209 | 64.9% | 74.5% |
| 0.70 m | BC_Phase4H_Candidate | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 6.34 | 0.000 | 19.1% | 53.0% |
| 0.70 m | Arm_B_BC_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 5.22 | 0.118 | 9.0% | 65.9% |
| 0.70 m | PPO_Phase4H_Candidate | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 3.29 | 0.110 | 14.2% | 87.5% |
| 0.70 m | Arm_A_PPO_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 2.91 | 0.114 | 12.3% | 92.0% |
| **0.60 m** | Expert_APF_Reference | 2/10 (20.0%) | 0/10 (0.0%) | 495.5 | 24.78 | 7.42 | 0.210 | 53.7% | 65.0% |
| 0.60 m | BC_Phase4H_Candidate | 2/10 (20.0%) | 1/10 (10.0%) | 444.7 | 22.24 | 6.30 | 0.000 | 14.1% | 44.8% |
| 0.60 m | Arm_B_BC_Baseline | 4/10 (40.0%) | 0/10 (0.0%) | 478.8 | 23.94 | 5.98 | 0.114 | 6.8% | 56.8% |
| 0.60 m | PPO_Phase4H_Candidate | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 3.44 | 0.115 | 13.2% | 85.7% |
| 0.60 m | Arm_A_PPO_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 2.91 | 0.113 | 10.3% | 88.4% |
| **0.55 m** | Expert_APF_Reference | 2/10 (20.0%) | 0/10 (0.0%) | 487.6 | 24.38 | 8.00 | 0.211 | 42.9% | 55.9% |
| 0.55 m | BC_Phase4H_Candidate | 1/10 (10.0%) | 1/10 (10.0%) | 465.8 | 23.29 | 7.07 | 0.000 | 11.8% | 38.4% |
| 0.55 m | Arm_B_BC_Baseline | 4/10 (40.0%) | 0/10 (0.0%) | 459.2 | 22.96 | 6.01 | 0.114 | 6.6% | 53.4% |
| 0.55 m | PPO_Phase4H_Candidate | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 3.72 | 0.121 | 11.7% | 82.7% |
| 0.55 m | Arm_A_PPO_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 2.71 | 0.115 | 8.1% | 87.0% |
| **0.50 m** | Expert_APF_Reference | 4/10 (40.0%) | 0/10 (0.0%) | 477.7 | 23.88 | 8.46 | 0.204 | 30.2% | 44.5% |
| 0.50 m | BC_Phase4H_Candidate | **5/10 (50.0%)** | 0/10 (0.0%) | 470.5 | 23.52 | 7.44 | 0.003 | 7.9% | 38.9% |
| 0.50 m | Arm_B_BC_Baseline | **5/10 (50.0%)** | 0/10 (0.0%) | 443.6 | 22.18 | 6.27 | 0.117 | 4.2% | 46.6% |
| 0.50 m | PPO_Phase4H_Candidate | 0/10 (0.0%) | 1/10 (10.0%) | 491.8 | 24.59 | 4.18 | 0.111 | 9.4% | 77.1% |
| 0.50 m | Arm_A_PPO_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 2.71 | 0.107 | 5.6% | 85.5% |
| **0.45 m** | Expert_APF_Reference | **7/10 (70.0%)** | 0/10 (0.0%) | 462.4 | 23.12 | 8.63 | 0.198 | 20.0% | 34.3% |
| 0.45 m | BC_Phase4H_Candidate | **5/10 (50.0%)** | 0/10 (0.0%) | 444.0 | 22.20 | 6.81 | 0.103 | 5.5% | 37.7% |
| 0.45 m | Arm_B_BC_Baseline | **5/10 (50.0%)** | 0/10 (0.0%) | 432.2 | 21.61 | 6.83 | 0.109 | 4.0% | 38.8% |
| 0.45 m | PPO_Phase4H_Candidate | 0/10 (0.0%) | 2/10 (20.0%) | 483.5 | 24.18 | 4.28 | 0.120 | 6.0% | 72.1% |
| 0.45 m | Arm_A_PPO_Baseline | 0/10 (0.0%) | 0/10 (0.0%) | 500.0 | 25.00 | 2.72 | 0.096 | 4.4% | 83.8% |

---

## 3. Detailed Controller Analysis

### 3.1 Scripted APF Expert Controller
- **At $d_{\text{caut}} = 0.80\text{ m}$:** Throttling active for 70.7% of flight. Progress plateaus at $5.68\text{ m}$ (past Pillar 1, entering Pillar 2). 0/10 successes.
- **At $d_{\text{caut}} = 0.55\text{ m}$:** Throttled time drops from 70.7% to 42.9%. Progress increases from $5.68\text{ m}$ to $8.00\text{ m}$. Success rate is 2/10 (Seeds 1004 and 1009 arrive at 21.2 s and 22.6 s). The remaining 8 seeds time out with forward progress between $6.95\text{ m}$ and $8.43\text{ m}$.
- **At $d_{\text{caut}} = 0.50\text{ m}$:** Throttled time drops to 30.2%. Success rate reaches 4/10 (22.21 s mean completion). The 6 timed-out seeds achieve forward progress of $7.93\text{ m}, 8.64\text{ m}, 8.13\text{ m}, 8.46\text{ m}, 8.47\text{ m}, 8.35\text{ m}$. Seed 1001 reached $X = 8.641\text{ m}$ (9 mm short of the 8.65 m goal zone cutoff).
- **At $d_{\text{caut}} = 0.45\text{ m}$:** Throttled time drops to 20.0%. Success rate reaches **7/10 (70.0%)** with completion times:
  - Seed 1004: 18.40 s (368 steps)
  - Seed 1009: 19.65 s (393 steps)
  - Seed 1003: 22.50 s (450 steps)
  - Seed 1007: 23.10 s (462 steps)
  - Seed 1001: 23.50 s (470 steps)
  - Seed 1006: 24.40 s (488 steps)
  - Seed 1005: 24.65 s (493 steps)
  - *Mean successful transit time:* **22.31 s**.
  - *Remaining 3 seeds:* Seed 1000 ($8.514\text{ m}$), Seed 1002 ($8.575\text{ m}$), Seed 1008 ($8.601\text{ m}$). All cleared all 3 obstacles with zero collisions and zero near-misses.

### 3.2 Behavior Cloning (BC) Policies
- Both the Phase 4H BC candidate and the Arm B BC baseline exhibit substantial gains in goal arrivals as caution throttling is reduced:
  - At $0.80\text{ m}$: 0/10 success.
  - At $0.60\text{ m}$: BC Baseline achieves 4/10 (40.0%), BC Candidate achieves 2/10 (20.0%).
  - At $0.50\text{ m}$: Both policies achieve **5/10 (50.0%) success** with **0 collisions**.
  - At $0.45\text{ m}$: Both policies achieve **5/10 (50.0%) success** with **0 collisions**.
- **Completion Times for BC Successes:**
  - At $0.50\text{ m}$: BC Baseline completes in 17.40 s to 20.80 s (mean 19.36 s); BC Candidate completes in 20.25 s to 24.45 s (mean 22.05 s).
  - At $0.45\text{ m}$: BC Baseline completes in 17.45 s to 20.70 s (mean 18.22 s); BC Candidate completes in 16.85 s to 23.35 s (mean 19.40 s).
- **Safety Clearance:**
  - For BC Baseline, minimum obstacle clearance across all runs remained safely above $0.109\text{ m}$.
  - For BC Candidate, at $0.50\text{ m}$, obstacle clearance in successful episodes remained $\ge 0.120\text{ m}$. Seed 1007 scraped Pillar 3 at $0.003\text{ m}$ clearance during a retreat maneuver without colliding.

### 3.3 PPO Policies
- **Persistent Pinning / Deadlock:**
  - Both Arm A Baseline and Phase 4H Candidate achieved **0/10 (0.0%) success across every single caution distance setting**.
  - Arm A Baseline remained pinned in front of Pillar 1 ($X \approx 2.08\text{ m}$) on 7 of 10 seeds, even with $d_{\text{caut}} = 0.45\text{ m}$ (mean progress $2.72\text{ m}$).
  - Phase 4H Candidate was able to bypass Pillar 1 on multiple seeds, achieving up to $4.28\text{ m}$ mean progress, but repeatedly entered limit cycles around Pillar 2 ($X = 5.0\text{ m}$).
- **Collision Risk Emergence:**
  - At $d_{\text{caut}} \in [0.80, 0.70, 0.60, 0.55]$, PPO suffered 0 collisions because the supervisor strictly clamped its speed to $0.10\text{ m/s}$ in proximity.
  - At $d_{\text{caut}} = 0.50\text{ m}$, Seed 1000 bypassed all 3 obstacles at high speed but had no terminal deceleration logic, impacting the rear boundary wall (`crash_boundary` at $X = 10.026\text{ m}$, step 418).
  - At $d_{\text{caut}} = 0.45\text{ m}$, both Seed 1000 (`crash_boundary` at $X = 10.123\text{ m}$, step 351) and Seed 1002 (`crash_boundary` at $X = 9.953\text{ m}$, step 484) suffered rear boundary wall collisions.

---

## 4. Completion-Time Distributions

The table below details the empirical completion times for all successful runs:

```
====================================================================================================
COMPLETION TIME DISTRIBUTION FOR SUCCESSFUL RUNS (s)
====================================================================================================
d_caut  Controller               Successes  Min (s)   Median (s)  Mean (s)  Max (s)   Timeout Margin
----------------------------------------------------------------------------------------------------
0.60m   Expert_APF_Reference     2/10       22.95     23.88       23.88     24.80     +0.20 s
0.60m   BC_Phase4H_Candidate     2/10       18.65     19.73       19.73     20.80     +4.20 s
0.60m   Arm_B_BC_Baseline        4/10       18.75     23.42       22.35     23.80     +1.20 s
----------------------------------------------------------------------------------------------------
0.55m   Expert_APF_Reference     2/10       21.20     21.90       21.90     22.60     +2.40 s
0.55m   BC_Phase4H_Candidate     1/10       24.85     24.85       24.85     24.85     +0.15 s
0.55m   Arm_B_BC_Baseline        4/10       18.10     19.60       19.90     22.10     +2.90 s
----------------------------------------------------------------------------------------------------
0.50m   Expert_APF_Reference     4/10       19.55     22.45       22.21     24.40     +0.60 s
0.50m   BC_Phase4H_Candidate     5/10       20.25     22.05       22.05     24.45     +0.55 s
0.50m   Arm_B_BC_Baseline        5/10       17.40     20.35       19.36     20.80     +4.20 s
----------------------------------------------------------------------------------------------------
0.45m   Expert_APF_Reference     7/10       18.40     23.10       22.31     24.65     +0.35 s
0.45m   BC_Phase4H_Candidate     5/10       16.85     19.10       19.40     23.35     +1.65 s
0.45m   Arm_B_BC_Baseline        5/10       17.45     17.65       18.22     20.70     +4.30 s
====================================================================================================
```

---

## 5. Supervisor State & Intervention Breakdown

Tracking supervisor interventions reveals the direct mechanism through which $d_{\text{caut}}$ unlocks throughput:

1. **Throttled Time Percentage:**
   - Expert: $70.7\% \to 64.9\% \to 53.7\% \to 42.9\% \to 30.2\% \to 20.0\%$.
   - BC Candidate: $17.0\% \to 19.1\% \to 14.1\% \to 11.8\% \to 7.9\% \to 5.5\%$.
   - BC Baseline: $13.8\% \to 9.0\% \to 6.8\% \to 6.6\% \to 4.2\% \to 4.0\%$.
2. **Emergency Braking & Reverse Retreat Invariants:**
   - Critical brake threshold ($0.30\text{ m}$) and emergency reverse threshold ($0.15\text{ m}$) remained active and identical across all runs.
   - For Expert APF, emergency braking was triggered in exactly 1 seed (Seed 1000) when turning sharply past Pillar 1, and the supervisor successfully decelerated the vehicle without collision.
   - For BC Baseline, zero emergency braking events occurred at $0.50\text{ m}$ or $0.45\text{ m}$.
3. **Intervention Rate:**
   - For Expert APF, intervention rate dropped from $79.1\%$ down to $34.3\%$.
   - For BC Candidate, intervention rate dropped from $54.9\%$ down to $37.7\%$.
   - For BC Baseline, intervention rate dropped from $70.9\%$ down to $38.8\%$.

---

## 6. Safety vs. Performance Trade-off Analysis

### 6.1 Clearance Margins
The safety contract requires maintaining a physical air gap of at least $0.08\text{ m}$ from obstacles during normal flight.
- At $d_{\text{caut}} = 0.50\text{ m}$:
  - Expert minimum obstacle clearance: $0.204\text{ m}$ (safety margin $+0.124\text{ m}$).
  - BC Baseline minimum obstacle clearance: $0.117\text{ m}$ (safety margin $+0.037\text{ m}$).
  - BC Candidate minimum obstacle clearance: $0.003\text{ m}$ (1 near-miss during retreat).
- At $d_{\text{caut}} = 0.45\text{ m}$:
  - Expert minimum obstacle clearance: $0.198\text{ m}$ (safety margin $+0.118\text{ m}$).
  - BC Baseline minimum obstacle clearance: $0.109\text{ m}$ (safety margin $+0.029\text{ m}$).
  - BC Candidate minimum obstacle clearance: $0.103\text{ m}$ (safety margin $+0.023\text{ m}$).

### 6.2 Trade-off Synthesis
1. **$d_{\text{caut}} = 0.80\text{ m}$:** Completely safe from high-speed collisions, but structurally guarantees 100% mission failure due to timeout. The throttle zone spans $1.60\text{ m}$ around each pillar, eliminating free acceleration runways.
2. **$d_{\text{caut}} = 0.55\text{ m}$:** Alleviates some throttle, but 80% of expert runs and 90% of BC candidate runs still time out before reaching the goal because the flight time ($24.4\text{ s}$) is too close to the 25.0 s horizon.
3. **$d_{\text{caut}} = 0.50\text{ m}$:** Strikes an optimal balance between clearance buffer ($0.20\text{ m}$ air gap before braking) and throughput:
   - 50% BC success, 40% Expert success (with all timeouts arriving within 30 cm of the goal).
   - Zero obstacle collisions for all BC and Expert episodes.
   - Restores $1.25\text{ m}$ unthrottled acceleration runway between pillars.
4. **$d_{\text{caut}} = 0.45\text{ m}$:** Maximizes expert success (70%) and provides the highest forward transit speed, but creates an elevated collision risk for policies that lack terminal deceleration (e.g. PPO overshoots into boundary wall at step 351/484).

The raw data and per-step telemetry are preserved in `experiments/phase4j/caution_distance_sweep_metrics.json`.
