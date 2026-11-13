# Phase 4G-E: Validation Evaluation Report

**Date:** 2026-10-29  
**Status:** COMPLETED & CANDIDATE FROZEN  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Deliverable Files:**
- `reports/phase4g_validation_report.md`
- `experiments/phase4g/logs/phase4g_validation_results.json`
- Frozen Candidate Checkpoint: `experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`

---

## 1. Executive Summary

Phase 4G-E conducted a rigorous, paired validation evaluation across four experimental controllers:
1. **Arm A — Frozen PPO Baseline** (`models/checkpoints/ppo_policy_best.pt`)
2. **Arm B — Frozen BC Baseline** (`models/checkpoints/bc_policy_best.pt`)
3. **Arm C — PPO without Reflection Augmentation** (`experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt`)
4. **Arm D — PPO with Reflection Augmentation** (`experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`)

Evaluations were performed across:
- **Controlled Open-Space Symmetry Probes:** Lateral sweep $Y \in [-0.40, +0.40]\text{ m}$ at $X=1.0\text{ m}$, $Z=1.0\text{ m}$ under forward cruise ($v_x = 0.60\text{ m/s}$).
- **Mirrored Obstacle Pair Bypass Tests:** Obstacle at $X=3.0\text{ m}$ with lateral shifts $Y_{\text{obs}} \in \{-0.25, -0.15, 0.00, +0.15, +0.25\}\text{ m}$.
- **10-Seed Fixed Validation Pool:** Seeds `100` to `109` under identical starting conditions and supervisor configurations.

### Key Validation Findings
1. **Marked Yaw-Rate Rotational Bias Reduction in Arm D:**
   - In open space at centerline ($Y=0.00\text{ m}$), Arm A commands a rightward yaw rate of $\dot{\psi} = -0.1537\text{ rad/s}$.
   - Arm C commands $\dot{\psi} = -0.1364\text{ rad/s}$.
   - Arm D achieves $\dot{\psi} = \mathbf{-0.0618\text{ rad/s}}$, representing a **59.8% reduction in rotational bias** relative to Arm A.
2. **Obstacle Bypass Success at $Y_{\text{obs}} = +0.15\text{ m}$:**
   - At $Y_{\text{obs}} = +0.15\text{ m}$ (right gap $= 0.95\text{ m}$, feasible), Arm A and Arm C stalled and failed to pass the obstacle (`passed = False`).
   - Arm D successfully passed the obstacle (`passed = True`, final $X = 3.82\text{ m}$).
3. **Obstacle Route Selection Persistence:**
   - When the obstacle was shifted left ($Y_{\text{obs}} = -0.25\text{ m}$ and $-0.15\text{ m}$, where the left gap is wide and feasible while the right is constricted), Arm D improved obstacle clearance ($0.170\text{ m}$ vs $0.140\text{ m}$), but continued to attempt the constricted right side.
   - BC remained the only controller that cleanly chose the wider left passage.
4. **Validation Benchmark Success Rate:**
   - On the 10-seed validation benchmark, all three PPO arms (Arm A, Arm C, Arm D) achieved $0.0\%$ goal completion within the 200-step ($10\text{ s}$) horizon. Drones consistently reached $X \approx 3.2 - 4.1\text{ m}$, where supervisor deceleration near the first pillar caused timeouts before reaching the 10m terminal goal.

---

## 2. Checkpoint Selection Decision

In accordance with Section 6, candidate selection followed the documented criteria:
1. **Primary Criterion:** Validation benchmark end-to-end goal success rate. (Tied at $0.0\%$).
2. **Secondary Criterion:** Minimization of open-space lateral velocity and yaw-rate bias ($|v_y| + |\dot{\psi}|$).
   - Arm A: $0.2266 + 0.1537 = 0.3803$
   - Arm C: $0.2165 + 0.1364 = 0.3529$
   - Arm D: $0.1886 + 0.0618 = \mathbf{0.2504}$ (Lowest combined error, **34.2% lower than Arm A**).
3. **Tertiary Criterion:** Obstacle passage and clearance in mirrored pairs:
   - Arm D passed 2/5 obstacle configurations (vs 1/5 for Arm A and Arm C).

**Selected Candidate for Final Held-Out Evaluation:**
`experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`  
**Candidate Hash (Frozen):**
`SHA256: 7d739ea8ba58ef3bb8ef198a2eb839c4a75e24a4692fc3f86eb16fcb6703b4e3`

---

## 3. Detailed Experimental Comparisons

### 3.1 Controlled Open-Space Probe ($v_x = 0.60\text{ m/s}$)
| Lateral Pos $Y$ | Arm A Raw $v_y$ | Arm A Raw $\dot{\psi}$ | Arm C Raw $v_y$ | Arm C Raw $\dot{\psi}$ | Arm D Raw $v_y$ | Arm D Raw $\dot{\psi}$ | Arm B (BC) $v_y$ | Arm B (BC) $\dot{\psi}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $-0.40\text{ m}$ | $+0.021\text{ m/s}$ | $+0.038\text{ rad/s}$ | $+0.040\text{ m/s}$ | $+0.052\text{ rad/s}$ | $+0.065\text{ m/s}$ | $+0.076\text{ rad/s}$ | $+0.008\text{ m/s}$ | $+0.013\text{ rad/s}$ |
| $-0.30\text{ m}$ | $-0.061\text{ m/s}$ | $-0.018\text{ rad/s}$ | $-0.045\text{ m/s}$ | $-0.010\text{ rad/s}$ | $-0.018\text{ m/s}$ | $+0.027\text{ rad/s}$ | $+0.006\text{ m/s}$ | $+0.009\text{ rad/s}$ |
| $-0.20\text{ m}$ | $-0.137\text{ m/s}$ | $-0.077\text{ rad/s}$ | $-0.123\text{ m/s}$ | $-0.065\text{ rad/s}$ | $-0.093\text{ m/s}$ | $-0.015\text{ rad/s}$ | $+0.004\text{ m/s}$ | $+0.006\text{ rad/s}$ |
| $-0.10\text{ m}$ | $-0.193\text{ m/s}$ | $-0.124\text{ rad/s}$ | $-0.181\text{ m/s}$ | $-0.108\text{ rad/s}$ | $-0.149\text{ m/s}$ | $-0.043\text{ rad/s}$ | $+0.002\text{ m/s}$ | $+0.003\text{ rad/s}$ |
| $\mathbf{0.00\text{ m}}$ | $\mathbf{-0.227\text{ m/s}}$ | $\mathbf{-0.154\text{ rad/s}}$ | $\mathbf{-0.217\text{ m/s}}$ | $\mathbf{-0.136\text{ rad/s}}$ | $\mathbf{-0.189\text{ m/s}}$ | $\mathbf{-0.062\text{ rad/s}}$ | $\mathbf{0.000\text{ m/s}}$ | $\mathbf{0.000\text{ rad/s}}$ |
| $+0.10\text{ m}$ | $-0.244\text{ m/s}$ | $-0.170\text{ rad/s}$ | $-0.236\text{ m/s}$ | $-0.151\text{ rad/s}$ | $-0.212\text{ m/s}$ | $-0.070\text{ rad/s}$ | $-0.003\text{ m/s}$ | $-0.004\text{ rad/s}$ |
| $+0.20\text{ m}$ | $-0.252\text{ m/s}$ | $-0.179\text{ rad/s}$ | $-0.244\text{ m/s}$ | $-0.159\text{ rad/s}$ | $-0.223\text{ m/s}$ | $-0.071\text{ rad/s}$ | $-0.005\text{ m/s}$ | $-0.007\text{ rad/s}$ |
| $+0.30\text{ m}$ | $-0.253\text{ m/s}$ | $-0.183\text{ rad/s}$ | $-0.245\text{ m/s}$ | $-0.162\text{ rad/s}$ | $-0.226\text{ m/s}$ | $-0.068\text{ rad/s}$ | $-0.007\text{ m/s}$ | $-0.009\text{ rad/s}$ |
| $+0.40\text{ m}$ | $-0.249\text{ m/s}$ | $-0.185\text{ rad/s}$ | $-0.240\text{ m/s}$ | $-0.160\text{ rad/s}$ | $-0.222\text{ m/s}$ | $-0.061\text{ rad/s}$ | $-0.008\text{ m/s}$ | $-0.011\text{ rad/s}$ |

### 3.2 Paired Mirrored Obstacle Tests ($X_{\text{obs}} = 3.0\text{ m}$)
| $Y_{\text{obs}}$ | Wider Side | Arm A (PPO Frozen) | Arm B (BC Frozen) | Arm C (PPO No Aug) | Arm D (PPO Reflection) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $-0.25\text{ m}$ | Left ($1.05\text{ m}$) | Right, Fail (0.140m) | **Left, Pass (0.199m)** | Right, Fail (0.137m) | Right, Fail (0.170m) |
| $-0.15\text{ m}$ | Left ($0.95\text{ m}$) | Right, Fail (0.128m) | **Left, Pass (0.139m)** | Right, Fail (0.137m) | Right, Fail (0.142m) |
| $0.00\text{ m}$ | Symmetric | Right, Fail (0.131m) | Left, Fail (0.132m) | Right, Fail (0.129m) | Right, Fail (0.136m) |
| $+0.15\text{ m}$ | Right ($0.95\text{ m}$) | Right, Fail (0.134m) | **Right, Pass (0.191m)** | Right, Fail (0.135m) | **Right, Pass (0.133m)** |
| $+0.25\text{ m}$ | Right ($1.05\text{ m}$) | **Right, Pass (0.185m)** | **Right, Pass (0.271m)** | **Right, Pass (0.250m)** | **Right, Pass (0.137m)** |

### 3.3 10-Seed Fixed Validation Pool Summary
| Metric | Arm A (Frozen PPO) | Arm B (Frozen BC) | Arm C (PPO No Aug) | Arm D (PPO Reflection) |
| :--- | :---: | :---: | :---: | :---: |
| Success Rate (%) | 0.0% (0/10) | 0.0% (0/10) | 0.0% (0/10) | 0.0% (0/10) |
| Mean Forward Displacement | $3.350\text{ m}$ | $3.457\text{ m}$ | $3.289\text{ m}$ | $3.123\text{ m}$ |
| Mean Min Obstacle Clearance | $0.154\text{ m}$ | $0.195\text{ m}$ | $0.150\text{ m}$ | $0.162\text{ m}$ |
| Mean Min Wall Clearance | $0.210\text{ m}$ | $0.345\text{ m}$ | $0.220\text{ m}$ | $0.228\text{ m}$ |
| Supervisor Intervention Rate | $88.6\%$ | $44.0\%$ | $90.9\%$ | $88.4\%$ |

---

## 4. Validation Gate Assessment

- **Candidate Checkpoint Status:** **FROZEN** (`ppo_arm_d_reflection.pt`).
- Candidate demonstrates substantial, measurable reduction in rotational asymmetry (yaw-rate bias reduced by 60%).
- Candidate demonstrates improved passage at $Y_{\text{obs}} = +0.15\text{ m}$.
- Candidate preserves safe obstacle clearances without crashing.
- Validation gate cleared for final evaluation on reserved held-out seeds `80000..80099`.
