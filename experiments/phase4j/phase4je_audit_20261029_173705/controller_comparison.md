# Phase 4J-E: Controlled Controller Comparison Report

**Audit Directory:** `experiments/phase4j/phase4je_audit_20261029_173705`  
**Evaluation Seeds:** 50 Development Seeds (`2000..2049`)  
**Conditions:** 10m Corridor Benchmark, Horizon 500 steps (25.0s), Isolated Supervisor $d_{\text{caut}} = 0.50\text{ m}$  
**Date:** 2026-10-29  

---

## 1. Primary Benchmark Results Table

| Metric | Arm B: BC Baseline (Primary) | Arm A: PPO Baseline (Primary) | BC Phase 4H Candidate (Secondary) | PPO Phase 4H Candidate (Secondary) |
|:---|:---:|:---:|:---:|:---:|
| **Goal Success Count** | **24 / 50** | **6 / 50** | **17 / 50** | **0 / 50** |
| **Goal Success Rate** | **48.0%** | **12.0%** | **34.0%** | **0.0%** |
| **95% Wilson Score CI** | `[34.8%, 61.5%]` | `[5.6%, 23.8%]` | `[22.4%, 47.8%]` | `[0.0%, 7.1%]` |
| **Obstacle Collisions** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** |
| **Boundary Collisions** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **13 (26.0%)** |
| **Obstacle Envelope Violations (< 8cm)** | **0 (0.0%)** | **0 (0.0%)** | **7 (14.0%)** | **0 (0.0%)** |
| **Boundary Envelope Violations (< 8cm)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **13 (26.0%)** |
| **Episode Timeouts (500 steps)** | 26 (52.0%) | 44 (88.0%) | 33 (66.0%) | 37 (74.0%) |
| **Mean Forward Progress** | **6.53 m** | 4.28 m | 6.93 m | 5.70 m |
| **Median Forward Progress** | **6.32 m** | 4.14 m | 8.19 m | 4.24 m |
| **Global Min Obstacle Clearance** | 0.1070 m | 0.0874 m | 0.0095 m | 0.1138 m |
| **Global Min Boundary Clearance** | 0.2537 m | 0.2434 m | 0.1784 m | **-0.0146 m** |
| **Mean Episode Duration** | 22.99 s | 24.34 s | 23.82 s | 23.78 s |
| **Mean Throttled Fraction** | 27.6% | 36.1% | 29.3% | 41.7% |
| **Mean Intervention Rate** | 0.477 | 0.739 | 0.427 | 0.670 |
| **Mean Raw Lateral Velocity ($v_y$)** | -0.0035 m/s | **+0.0212 m/s** | +0.0060 m/s | +0.0193 m/s |
| **Policy Saturation ($v_x$)** | 56.0% | 53.4% | 48.7% | 57.9% |

---

## 2. Paired Statistical Comparison: BC Baseline vs. PPO Baseline

- **Sample Size:** 50 paired development episodes (Seeds `2000..2049`)
- **BC Successes:** 24 / 50 (48.0%)
- **PPO Successes:** 6 / 50 (12.0%)
- **Absolute Success Difference:** **+36.0 percentage points** in favor of BC Baseline

### Paired Contingency Table (50 Seeds):
| Outcome Combination | Seed Count | Percentage |
|:---|:---:|:---:|
| **Both BC and PPO Succeed** | **2** | **4.0%** |
| **BC Succeeds, PPO Fails** | **22** | **44.0%** |
| **PPO Succeeds, BC Fails** | **4** | **8.0%** |
| **Both BC and PPO Fail** | **22** | **44.0%** |

### McNemar's Exact Test:
- Disagreeing Pairs: $b = 22$ (BC wins), $c = 4$ (PPO wins)
- **Two-Tailed Exact $p$-Value:** **0.000534**
- **Statistical Significance:** **STATISTICALLY SIGNIFICANT ($p < 0.05$)**

---

## 3. Failure Mode Distribution Across Controllers

| Failure Mode | BC Baseline | PPO Baseline | BC Phase 4H Candidate | PPO Phase 4H Candidate |
|:---|:---:|:---:|:---:|:---:|
| `SUCCESS` | 24 | 6 | 17 | 0 |
| `WALL_COLLISION` | 0 | 0 | 0 | **13** |
| `OBSTACLE_COLLISION` | 0 | 0 | 0 | 0 |
| `PROGRESS_STALL_PILLAR_1` | 6 | 19 | 1 | 13 |
| `PROGRESS_STALL_PILLAR_2` | 10 | 17 | 12 | 13 |
| `PROGRESS_STALL_PILLAR_3` | 10 | 8 | 10 | 10 |
| `PROGRESS_LATERAL_OSCILLATION` | 0 | 0 | 4 | 1 |
| `TIMEOUT_EXCESSIVE_THROTTLING` | 0 | 0 | 0 | 0 |
