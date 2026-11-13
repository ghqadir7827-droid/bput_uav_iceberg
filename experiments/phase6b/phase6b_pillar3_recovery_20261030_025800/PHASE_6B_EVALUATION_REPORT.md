# Phase 6B Evaluation & Benchmark Comparison Report

**Phase:** Phase 6B — Targeted Pillar 3 Recovery Experiment  
**Directory:** `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/`  
**Protocol:** Paired Development Benchmark across Seeds 2000–2049 & Diagnostic Scenarios  
**Controllers Evaluated:**
1. **Phase 6A Baseline:** Frozen BC + Stateless Clearance Guidance Controller (`StatelessClearanceGuidanceController`)
2. **Phase 6B Candidate:** Frozen BC + Stateless Guidance + Post-Evasion Centerline Recovery (`PostEvasionRecoveryController`)

---

## 1. Executive Summary & Stop-Condition Ruling

In accordance with Phase 6B Section 9 (Stop Conditions) and the Completion Standard:
- **Baseline Reproduction:** 42 / 50 successes (84.0%) replicated exactly.
- **Root Cause Proof:** Pillar 3 standoff mechanisms empirically confirmed through per-step telemetry.
- **Candidate Pillar 3 Performance:** The candidate successfully resolved Pillar 3 traversal across **all 5 target failure seeds** (`p_cleared = 3` on Seeds 2001, 2005, 2006, 2022, 2042). Pillar 3 passage rate increased from 92.0% (46/50) to 94.0% (47/50).
- **Benchmark Net Outcome:** The candidate achieved **39 / 50 successes (78.0%)**, representing a **net degradation of -3 successes (-6.0%)** relative to baseline.
- **Scenario C Regression:** The candidate suffered a material timeout regression on the canonical diagnostic scenarios (Scenario C Wide Right: 388 steps $\to$ 500 steps timeout; Scenario C Wide Left: 351 steps $\to$ 500 steps timeout).
- **RULING: STOP CONDITION TRIGGERED. Candidate is REJECTED. Phase 6A baseline is RETAINED as the production standard.**

---

## 2. Quantitative Benchmark Results (50 Development Seeds 2000–2049)

| Evaluation Metric | Phase 6A Baseline (Current Production) | Phase 6B Candidate (Post-Evasion Recovery) | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Completions (Success / 50)** | **42 / 50** | **39 / 50** | **-3 episodes** | **Regressed (-6.0%)** |
| **Success Rate (%)** | **84.0%** | **78.0%** | **-6.0%** | **Regressed** |
| **Paired Contingency: Both Succeed** | 35 | 35 | — | Stable Core |
| **Paired Contingency: Candidate-Only** | — | **4** | +4 | Improved Seeds (2001, 2005, 2019, 2049) |
| **Paired Contingency: Baseline-Only** | **7** | — | -7 | Regressed Seeds (2000, 2015, 2016, 2021, 2025, 2032, 2038) |
| **Paired Contingency: Both Fail** | 4 | 4 | — | Residual Failures (2006, 2022, 2036, 2042) |
| **Simulated Collisions** | **0** | **0** | 0 | Strict Zero Preserved |
| **Clear-Envelope Violations ($<0.080\text{m}$)** | **0** | **0** | 0 | Strict Zero Preserved |
| **Min Obstacle Clearance (m)** | $0.0957\text{ m}$ (Seed 2042) | $0.0865\text{ m}$ (Seed 2016) | $-0.0092\text{ m}$ | Margin Reduced |
| **Min Boundary Clearance (m)** | $0.1942\text{ m}$ (Seed 2042) | $0.1891\text{ m}$ (Seed 2042) | $-0.0051\text{ m}$ | Margin Reduced |
| **Mean Goal Progress (m)** | $8.475\text{ m}$ | $8.476\text{ m}$ | $+0.001\text{ m}$ | Identical |
| **Mean Flight Duration (s)** | $20.30\text{ s}$ | $21.28\text{ s}$ | $+0.98\text{ s}$ | Slower Transit |
| **Supervisor Throttled Fraction (%)** | $17.5\%$ | $16.8\%$ | $-0.7\%$ | Marginal Reduction |

---

## 3. Pillar Clearance & Passage Rates

| Corridor Stage | Phase 6A Baseline | Phase 6B Candidate | Outcome |
| :--- | :---: | :---: | :---: |
| **Pillar 1 Clearance ($x \ge 3.2\text{ m}$)** | 50 / 50 (100.0%) | 50 / 50 (100.0%) | Unchanged |
| **Pillar 2 Clearance ($x \ge 5.2\text{ m}$)** | 50 / 50 (100.0%) | 50 / 50 (100.0%) | Unchanged |
| **Pillar 3 Clearance ($x \ge 7.2\text{ m}$)** | **46 / 50 (92.0%)** | **47 / 50 (94.0%)** | **+1 Seed (+2.0%)** |

---

## 4. Analysis of the Five Target Failure Seeds

| Seed | Baseline Outcome | Baseline Progress | Candidate Outcome | Candidate Progress | Pillar 3 Passed? | Forensic Analysis |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **2001** | `TIMEOUT_SLOW_TRANSIT` | $7.05\text{ m}$ | **`SUCCESS`** | **$8.66\text{ m}$** | **YES** | Converted to clean success (steps: $406$, min obs: $0.171\text{ m}$) |
| **2005** | `PROGRESS_STALL_PILLAR_3` | $6.14\text{ m}$ | **`SUCCESS`** | **$8.65\text{ m}$** | **YES** | Converted to clean success (steps: $407$, min obs: $0.173\text{ m}$) |
| **2006** | `PROGRESS_STALL_PILLAR_3` | $6.28\text{ m}$ | `GOAL_ALIGNMENT_TIMEOUT` | $8.52\text{ m}$ | **YES** | Standoff eliminated; cleared P3; step 500 timeout at goal entrance ($d=0.48\text{ m}$) |
| **2022** | `PROGRESS_STALL_PILLAR_3` | $6.34\text{ m}$ | `GOAL_ALIGNMENT_TIMEOUT` | $8.64\text{ m}$ | **YES** | Standoff eliminated; cleared P3; step 500 timeout at goal entrance ($d=0.37\text{ m}$) |
| **2042** | `PROGRESS_STALL_PILLAR_3` | $6.16\text{ m}$ | `TIMEOUT_SLOW_TRANSIT` | $7.82\text{ m}$ | **YES** | Standoff eliminated; cleared P3; step 500 timeout at $x = 7.82\text{ m}$ |

**Result:** On the targeted failures, the recovery candidate succeeded in preventing lateral standoff in 5 out of 5 cases, converting 2 directly into successes and advancing the other 3 past Pillar 3 into the terminal goal approach zone.

---

## 5. Diagnostic Scenarios & Forensic Root Cause of Regression

### Diagnostic Scenario Results
| Scenario | Baseline Success | Baseline Steps | Baseline Min Obs | Candidate Success | Candidate Steps | Candidate Min Obs | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario A Centered** | 1 | 370 | $0.265\text{ m}$ | 1 | 435 | $0.119\text{ m}$ | Success (slower) |
| **Scenario B Left Mirrored** | 1 | 392 | $0.151\text{ m}$ | 1 | 420 | $0.133\text{ m}$ | Success (slower) |
| **Scenario B Right Mirrored**| 1 | 344 | $0.309\text{ m}$ | 1 | 402 | $0.229\text{ m}$ | Success (slower) |
| **Scenario C Wide Right** | **1** | **388** | $0.125\text{ m}$ | **0** | **500 (Timeout)** | $0.132\text{ m}$ | **REGRESSION** |
| **Scenario C Wide Left** | **1** | **351** | $0.174\text{ m}$ | **0** | **500 (Timeout)** | $0.137\text{ m}$ | **REGRESSION** |

### Why Did the Candidate Regress?
A rigorous audit of the 7 regressed seeds (2000, 2015, 2016, 2021, 2025, 2032, 2038) and the Scenario C benchmarks revealed a fundamental control trade-off:

1. **Velocity Clamping and Time-Budget Exhaustion:**
   - In open corridors, the frozen BC policy outputs a nominal cruising speed $v_x \approx 0.25\text{ m/s}$ when moving along the centerline, whereas it outputs $v_x \ge 0.50\text{ m/s}$ when maneuvering along corridor walls.
   - Forcing the UAV back toward the centerline between obstacles prolonged flight time by $1.0\text{ s}$ to $3.0\text{ s}$ per episode.
   - In Seeds 2000, 2015, 2021, and 2038, the vehicle reached the terminal goal sphere ($x \ge 9.0\text{ m}$) but ran out of steps (step 500 timeout) within centimeters of the destination radius.

2. **The Spatial Coupling Conflict with Subsequent Obstacles:**
   - In Seeds 2016, 2025, and 2032, the baseline policy cleared Pillar 2 by flying wide along one wall ($y \approx \pm 0.55\text{ m}$), and because Pillar 3 happened to be situated in the corridor center or opposite half, the baseline's wide trajectory naturally bypassed Pillar 3 without ever needing to evade it.
   - The candidate's post-evasion recovery actively pulled the UAV back toward the centerline ($y \to 0$) in the short $1.2\text{ m}$ gap between Pillar 2 and Pillar 3.
   - By centering the UAV, the candidate steered the drone directly into the frontal caution zone ($d < 0.50\text{ m}$) of Pillar 3, transforming what would have been a wide, unobstructed bypass into a frontal standoff!

3. **Information-Theoretic Conclusion:**
   - A reactive, stateless controller lacks look-ahead visibility beyond its rangefinder horizon ($4.0\text{ m}$) and has zero multi-step path planning capability.
   - Centerline restoration is a **blind spatial prior**: it assumes the centerline is always safer than the corridor perimeter.
   - In a multi-obstacle environment with staggered pillars, centering between obstacles is not globally optimal and directly conflicts with subsequent obstacle evasion.

---

## 6. Official Stop-Condition Determination

In compliance with Phase 6B Section 9:
1. **Material Regression on Scenario C:** Candidate failed Scenario C Wide Left and Wide Right (both timed out at 500 steps, whereas baseline completed cleanly in 351 and 388 steps).
2. **Net Development Benchmark Degradation:** 39/50 (78.0%) vs. 42/50 (84.0%) baseline ($p = 0.375$ with 7 regressions).
3. **Verdict:**
   - **Candidate `PostEvasionRecoveryController` is NOT promoted.**
   - **Phase 6A `StatelessClearanceGuidanceController` is retained as the verified production candidate.**
   - **The held-out evaluation seeds (`90000–90099`) remain strictly SEALED.**

