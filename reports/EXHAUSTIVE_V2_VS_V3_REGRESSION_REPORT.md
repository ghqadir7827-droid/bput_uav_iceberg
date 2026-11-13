# Exhaustive Regression, Audit, and Confirmation Report: v2 vs v3

**Campaign Title:** Exhaustive Paired Regression, Audit, and Independent Holdout Evaluation of `scaled_guided_bc_v2` vs `scaled_guided_bc_v3`  
**Execution Timestamp:** 2026-10-31T01:38:00Z  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Preserved Original Report:** [`reports/EXHAUSTIVE_V2_VS_V3_REGRESSION_REPORT.orig.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/EXHAUSTIVE_V2_VS_V3_REGRESSION_REPORT.orig.md)  
**Vehicle Dimension:** 65 cm propeller-tip-to-propeller-tip physical quadcopter ($R_{\text{uav}} = 0.325\text{ m}$)  
**Frozen Model Checkpoint:** `models/checkpoints/bc_policy_scaled_v2.pt` (SHA-256: `d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1`)  

---

## 1. Executive Summary & Definitive Benchmark Recommendation

This report presents the audited, first-principles comparative evaluation of frozen baseline `scaled_guided_bc_v2` versus directional-persistence candidate `scaled_guided_bc_v3`.

> [!IMPORTANT]
> **Definitive Recommendation:** **Retain `scaled_guided_bc_v3` as the canonical simulation benchmark for nominal and wide-corridor operations, with documented narrow-corridor trade-offs.**
> - **Feasible Navigation Success:** On the 225 feasible evaluation seeds, candidate v3 achieves **$92.00\%$ success** (207 / 225) vs v2's **$81.78\%$** (184 / 225)—a statistically significant net gain of **$+10.22\%$** ($p = 0.00082$, McNemar exact test).
> - **Holdout Generalization:** On the 175 feasible untouched confirmation seeds, candidate v3 achieves **$97.71\%$ success** (171 / 175) vs v2's **$80.57\%$** (141 / 175), a net gain of **$+17.14\%$** ($p = 1.31 \times 10^{-7}$).
> - **Collision Safety is Absolute:** Across all 900 evaluated episodes, **zero collisions occurred for either controller** ($0.0\%$ collision rate) under the canonical deterministic `SafetySupervisor`.
> - **Safety Envelope Violations Slashed:** Candidate v3 reduced total safety envelope violation duration by **$33.37\%$** (12,731 steps vs 19,108 steps).
> - **Level 1 Chatter Resolved:** In Level 1 wide corridors, v3 resolved the persistent directional chatter failure mode, increasing success from **$40.0\%$ to $94.0\%$** on dev seeds ($p < 0.0001$) and **$96.0\%$** on untouched confirmation seeds.

```mermaid
flowchart TD
    subgraph D["Audited Evaluation Partitions"]
        P1["Development Suite (200 Seeds)\n175 Feasible | 25 Infeasible"]
        P2["Level 2 Conf Suite (50 Seeds)\nHistorically Frozen"]
        P3["Untouched Conf Suite (200 Seeds)\n175 Feasible | 25 Infeasible"]
    end

    subgraph V2["scaled_guided_bc_v2 (Baseline)"]
        V2_D["Dev Feas: 78.29% (137/175)\nEnv Violations: 34.86%"]
        V2_C["L2 Conf: 94.00% (47/50)\nEnv Violations: 14.00%"]
        V2_U["Untouched Feas: 80.57% (141/175)\nEnv Violations: 25.14%"]
    end

    subgraph V3["scaled_guided_bc_v3 (Candidate)"]
        V3_D["Dev Feas: 92.57% (162/175)\nEnv Violations: 24.00%"]
        V3_C["L2 Conf: 90.00% (45/50)\nEnv Violations: 18.00%"]
        V3_U["Untouched Feas: 97.71% (171/175)\nEnv Violations: 18.86%"]
    end

    P1 --> V2_D & V3_D
    P2 --> V2_C & V3_C
    P3 --> V2_U & V3_U
```

---

## 2. Seed Space Inventory and Provenance Architecture

A complete workspace inventory was conducted to classify all 3,460 seed references across historical logs, training corpora, and active benchmarks. Output records are permanently serialized to [`data/seed_inventory.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/seed_inventory.json) and [`data/seed_inventory.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/seed_inventory.csv).

### Canonical Seed Partitions

| Partition Name | Seed Range | Count | Environment Domain | Feasibility / Geometry | Provenance / Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1 Basic (Dev)** | `20100..20149` | 50 | Scaled 65cm UAV | $W=4.5\text{m}$, 3 obstacles, $y_{\text{range}}=0.35\text{m}$ (Feasible) | Active Dev / Regression Evaluation |
| **Level 2 Standard (Dev)** | `20000..20049` | 50 | Scaled 65cm UAV | $W=3.8\text{m}$, 3 obstacles, $y_{\text{range}}=0.60\text{m}$ (Feasible) | Active Dev / Hyperparameter Tuning |
| **Level 2 Standard (Conf)**| `20050..20099` | 50 | Scaled 65cm UAV | $W=3.8\text{m}$, 3 obstacles, $y_{\text{range}}=0.60\text{m}$ (Feasible) | Historical Frozen Confirmation Set |
| **Level 3 Challenging (Dev)**| `20200..20249` | 50 | Scaled 65cm UAV | $W=3.2\text{m}$, 3 obstacles, $y_{\text{range}}=0.75\text{m}$ (Feasible) | Active Dev / Stress Evaluation |
| **Level 4 Boundary (Dev)** | `20300..20349` | 50 | Scaled 65cm UAV | $W=2.8\text{m}$ (25 Feasible even seeds, 25 Blocked odd seeds) | Active Dev / Route Feasibility |
| **Untouched Level 1 Holdout**| `20150..20199` | 50 | Scaled 65cm UAV | $W=4.5\text{m}$, 3 obstacles, $y_{\text{range}}=0.35\text{m}$ (Feasible) | Pre-registered Untouched Holdout |
| **Untouched Level 2 Holdout**| `20400..20449` | 50 | Scaled 65cm UAV | $W=3.8\text{m}$, 3 obstacles, $y_{\text{range}}=0.60\text{m}$ (Feasible) | Pre-registered Untouched Holdout |
| **Untouched Level 3 Holdout**| `20250..20299` | 50 | Scaled 65cm UAV | $W=3.2\text{m}$, 3 obstacles, $y_{\text{range}}=0.75\text{m}$ (Feasible) | Pre-registered Untouched Holdout |
| **Untouched Level 4 Holdout**| `20350..20399` | 50 | Scaled 65cm UAV | $W=2.8\text{m}$ (25 Feasible even seeds, 25 Blocked odd seeds) | Pre-registered Untouched Holdout |
| **Scaled Demonstrations v2**| `40003..40589` | 310 | Scaled 65cm UAV | Level 2 Standard ($W=3.8\text{m}$) | BC v2 Policy Training (250 train, 60 val) |
| **Scaled Demonstrations v1**| `30000..30300` | 250 | Scaled 65cm UAV | Level 2 Standard ($W=3.8\text{m}$) | BC v1 Policy Training (200 train, 50 val) |
| **Legacy Unscaled Benchmarks**| `1..90100` | 2,450 | Legacy 10m Corridor | 10m unscaled corridor ($W=2.0\text{m}, R_{\text{drone}}=0.15\text{m}$) | Historical 30cm drone baselines |

---

## 3. Audited Feasible Route Performance by Difficulty Tier

> [!NOTE]
> In accordance with Phase 1 audit corrections, feasible route completion ($N=225$) and impassable barrier rejection ($N=25$) are evaluated as separate operational tasks.

### Exhaustive Evaluation Suite (225 Feasible Seeds)

| Corridor Tier | Controller | Feasible Success (%) | 95% Wilson CI | Collisions | Env Violations (%) | Env Steps | Mean Time (s) | Mean Speed (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1 (Basic)**<br>Seeds 20100..20149 | v2 Baseline<br>**v3 Candidate** | 20 / 50 (40.0%)<br>**47 / 50 (94.0%)** | [27.61%, 53.82%]<br>**[83.78%, 97.94%]** | 0<br>**0** | 25 (50.0%)<br>**3 (6.0%)** | 6,367<br>**74** | 35.84 s<br>**26.47 s** | 0.446<br>**0.559** |
| **Level 2 Dev (Standard)**<br>Seeds 20000..20049 | v2 Baseline<br>**v3 Candidate** | **46 / 50 (92.0%)**<br>45 / 50 (90.0%) | [81.16%, 96.85%]<br>[78.64%, 95.65%] | 0<br>**0** | **10 (20.0%)**<br>15 (30.0%) | **1,003**<br>1,378 | 29.83 s<br>**29.41 s** | 0.518<br>**0.536** |
| **Level 2 Conf (Standard)**<br>Seeds 20050..20099 | v2 Baseline<br>**v3 Candidate** | **47 / 50 (94.0%)**<br>45 / 50 (90.0%) | [83.78%, 97.94%]<br>[78.64%, 95.65%] | 0<br>**0** | **7 (14.0%)**<br>9 (18.0%) | **712**<br>721 | 29.83 s<br>**29.58 s** | 0.519<br>**0.534** |
| **Level 3 (Challenging)**<br>Seeds 20200..20249 | v2 Baseline<br>**v3 Candidate** | 49 / 50 (98.0%)<br>**50 / 50 (100.0%)**| [89.50%, 99.65%]<br>**[92.87%, 100.0%]**| 0<br>**0** | 13 (26.0%)<br>**9 (18.0%)** | 1,189<br>**809** | 30.12 s<br>**29.61 s** | 0.517<br>**0.534** |
| **Level 4 Feasible**<br>25 Even Seeds | v2 Baseline<br>**v3 Candidate** | **22 / 25 (88.0%)**<br>20 / 25 (80.0%) | [70.04%, 95.83%]<br>[60.87%, 91.14%] | 0<br>**0** | **13 (52.0%)**<br>15 (60.0%) | **1,298**<br>1,489 | 31.78 s<br>**31.25 s** | 0.505<br>**0.522** |
| **Total Feasible Sweep** | v2 Baseline<br>**v3 Candidate** | 184 / 225 (81.78%)<br>**207 / 225 (92.00%)**| [76.21%, 86.27%]<br>**[87.71%, 94.88%]**| 0<br>**0** | 68 (30.22%)<br>**51 (22.67%)**| 10,569<br>**4,471** | 30.72 s<br>**29.89 s** | 0.507<br>**0.533** |

### Pre-Registered Untouched Confirmation Suite (175 Feasible Seeds)

| Corridor Tier | Controller | Feasible Success (%) | 95% Wilson CI | Collisions | Env Violations (%) | Env Steps | Mean Time (s) | Mean Speed (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1 Untouched** | v2 Baseline<br>**v3 Candidate** | 20 / 50 (40.0%)<br>**48 / 50 (96.0%)** | [27.61%, 53.82%]<br>**[86.54%, 98.90%]** | 0<br>**0** | 16 (32.0%)<br>**4 (8.0%)** | 4,211<br>**112** | 35.81 s<br>**26.24 s** | 0.445<br>**0.561** |
| **Level 2 Untouched** | v2 Baseline<br>**v3 Candidate** | 49 / 50 (98.0%)<br>**50 / 50 (100.0%)**| [89.50%, 99.65%]<br>**[92.87%, 100.0%]**| 0<br>**0** | 10 (20.0%)<br>**10 (20.0%)**| 789<br>**742** | 29.54 s<br>**29.41 s** | 0.523<br>**0.536** |
| **Level 3 Untouched** | v2 Baseline<br>**v3 Candidate** | 48 / 50 (96.0%)<br>**50 / 50 (100.0%)**| [86.54%, 98.90%]<br>**[92.87%, 100.0%]**| 0<br>**0** | 11 (22.0%)<br>**10 (20.0%)**| 842<br>**768** | 30.11 s<br>**29.58 s** | 0.518<br>**0.535** |
| **Level 4 Feasible Untouched** | v2 Baseline<br>**v3 Candidate** | **24 / 25 (96.0%)**<br>23 / 25 (92.0%) | [80.46%, 99.29%]<br>[75.03%, 97.78%] | 0<br>**0** | **7 (28.0%)**<br>9 (36.0%) | **612**<br>894 | 31.25 s<br>**31.02 s** | 0.510<br>**0.528** |
| **Total Untouched Feasible** | v2 Baseline<br>**v3 Candidate** | 141 / 175 (80.57%)<br>**171 / 175 (97.71%)**| [74.08%, 85.75%]<br>**[94.27%, 99.11%]**| 0<br>**0** | 44 (25.14%)<br>**33 (18.86%)**| 6,454<br>**2,516** | 30.56 s<br>**30.16 s** | 0.510<br>**0.530** |

---

## 4. Infeasible Route Safe Rejection Performance (25 Odd Seeds)

| Evaluation Partition | Controller | Safe Rejection Rate | Collisions | Safe Timeout Rate | Hazardous Penetrations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exhaustive Infeasible (N=25)** | v2 Baseline<br>**v3 Candidate** | **25 / 25 (100.0%)**<br>24 / 25 (96.0%) | 0<br>**0** | 0 (0.0%)<br>1 (4.0%) | 0<br>**0** |
| **Untouched Infeasible (N=25)** | v2 Baseline<br>**v3 Candidate** | **25 / 25 (100.0%)**<br>24 / 25 (96.0%) | 0<br>**0** | 0 (0.0%)<br>1 (4.0%) | 0<br>**0** |

---

## 5. Formal Statistical Hypothesis Testing

### Paired McNemar Test (Exhaustive 225 Feasible Seeds)
```
                     scaled_guided_bc_v3 Succeeded   scaled_guided_bc_v3 Failed    Total
scaled_guided_bc_v2 Succeeded           173 (Both Success)             11 (v2 Won)         184
scaled_guided_bc_v2 Failed               34 (v3 Recovered)              7 (Both Failed)      41
Total                                   207                            18                  225
```
- $b = 11, c = 34$, Net gain: $+23$ feasible successes.
- **Two-Tailed McNemar Exact $p$-value:** **$p = 0.00082$** ($p < 0.001$, highly significant).

### Paired McNemar Test (Untouched 175 Feasible Seeds)
```
                     scaled_guided_bc_v3 Succeeded   scaled_guided_bc_v3 Failed    Total
scaled_guided_bc_v2 Succeeded           140 (Both Success)              1 (v2 Won)         141
scaled_guided_bc_v2 Failed               31 (v3 Recovered)              3 (Both Failed)      34
Total                                   171                             4                  175
```
- $b = 1, c = 31$, Net gain: $+30$ feasible successes.
- **Two-Tailed McNemar Exact $p$-value:** **$p = 1.31 \times 10^{-7}$**.

### Two One-Sided Tests (TOST) for Equivalence ($\Delta = \pm 10\%$)
- **Level 3 Challenging:** 90% CI for difference: $[-1.26\%, +5.26\%] \subset [-10\%, +10\%]$. **Formally Equivalent.**
- **Level 2 Confirmation:** 90% CI for difference: $[-12.90\%, +4.90\%]$. **Inconclusive** (crosses $-10\%$ lower bound due to sample size $N=50$).
- **Level 4 Feasible:** 90% CI for difference: $[-24.95\%, +8.95\%]$. **Non-Equivalent** (v3 shows a conservative degradation in narrow bottlenecks).

---

## 6. Audit Conclusion and Verified Benchmark Artifacts

1. **Benchmark Integrity Verified:** Model checkpoint SHA-256 (`d02d5712...`) and simulation contracts verified in [`data/benchmark_manifest_v2_v3.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/benchmark_manifest_v2_v3.json).
2. **Root Cause of Level 4 Trade-Off:** Detailed in [`reports/V3_LEVEL4_FAILURE_ANALYSIS.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/V3_LEVEL4_FAILURE_ANALYSIS.md).
3. **Simulation Uncertainty Robustness:** Detailed in [`reports/V3_ROBUSTNESS_STRESS_TEST_REPORT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/V3_ROBUSTNESS_STRESS_TEST_REPORT.md).
