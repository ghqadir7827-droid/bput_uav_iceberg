# Phase 4G-D: Isolated Experimental Training Report

**Date:** 2026-10-29  
**Status:** COMPLETED & VERIFIED  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Deliverable Files:**
- `reports/phase4g_training_report.md`
- `reports/phase4g_training_metrics.json`
- `experiments/phase4g/logs/arm_c_training_log.json`
- `experiments/phase4g/logs/arm_d_training_log.json`
- `experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt`
- `experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`

---

## 1. Executive Summary

Phase 4G-D conducted a strictly isolated, fair comparison between two experimental training arms:
- **Arm C — PPO without reflection augmentation:** Fine-tuning from `ppo_policy_best.pt` using standard environment rollouts (replicates existing training pipeline).
- **Arm D — PPO with bilateral reflection augmentation:** Fine-tuning from the identical `ppo_policy_best.pt` checkpoint with alternating 50% bilateral reflection rollouts ($\mathcal{T}_{\text{obs}}, \mathcal{T}_{\text{act}}$).

Both arms operated under identical step budgets (10 iterations $\times$ 2048 steps = 20,480 environment steps), identical learning rates ($3 \times 10^{-4}$), identical batch sizes (128), identical PPO epochs (4), identical reward functions, and identical safety supervisor configurations.

### Key Empirical Findings
1. **Yaw-Rate Bias Reduced by 70.0% in Arm D:**
   - Under forward cruise ($v_x = 0.60\text{ m/s}$) at corridor center ($Y = 0.00\text{ m}$), pre-training PPO had a rightward yaw rate of $\dot{\psi} = -0.1513\text{ rad/s}$.
   - In Arm C (standard PPO), yaw-rate bias worsened to $\dot{\psi} = -0.2527\text{ rad/s}$ (a 67% increase in rightward rotational bias).
   - In Arm D (reflection augmentation), yaw-rate bias dropped to $\dot{\psi} = -0.0758\text{ rad/s}$ (a **70.0% reduction** in rotational bias compared to Arm C).
2. **Lateral Velocity Bias Moderation:**
   - Pre-training lateral velocity: $v_y = -0.2263\text{ m/s}$.
   - Both Arm C and Arm D reduced lateral drift to $v_y \approx -0.177\text{ m/s}$.
3. **Supervisor Intervention Rate in Validation:**
   - In Arm C, validation intervention rate remained high at $90.9\%$.
   - In Arm D, validation intervention rate dropped to $82.9\%$.
4. **Validation Success Rate:**
   - Both Arm C and Arm D achieved 0.0% end-to-end goal success on the 10-seed validation benchmark during this 10-iteration budget (drones continue to stall or collide near obstacles, as analyzed in Phase 4D/4F).
   - In accordance with Rule 11 and Section 8, this result is reported transparently: **symmetry improved markedly in yaw rate, but 10 iterations of reflection fine-tuning did not resolve obstacle deadlock on validation seeds.**

---

## 2. Checkpoint and Configuration Manifest

### 2.1 Checkpoint SHA-256 Hashes
| Checkpoint | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| Starting Warmstart | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | UNCHANGED (Immutable) |
| Arm C Candidate | `experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt` | `3a67286c9433ed7d776c1cda110fd203edd74ae60eb4f28d852bc3a2dee397f5` | ISOLATED CANDIDATE |
| Arm D Candidate | `experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt` | `7d739ea8ba58ef3bb8ef198a2eb839c4a75e24a4692fc3f86eb16fcb6703b4e3` | ISOLATED CANDIDATE |

### 2.2 Controlled Training Hyperparameters
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| Environment Scenario | `10m` | Headless PyBullet 240Hz, $dt=0.05\text{ s}$ |
| Initialization Mode | `ppo` | Loaded actor & critic from `ppo_policy_best.pt` |
| Total Step Budget | 20,480 steps | 10 iterations $\times$ 2048 steps per arm |
| Batch Size / Epochs | 128 / 4 epochs | Minibatch sampling with shuffle |
| Optimizer | AdamW | $\text{lr} = 3 \times 10^{-4}$, weight decay $= 10^{-4}$ |
| PPO Clipping $\epsilon$ | 0.20 | Standard clipped surrogate |
| GAE $\lambda$ / Discount $\gamma$ | 0.95 / 0.99 | Standard value bootstrapping on truncation |
| Value Loss Coef / Entropy Coef | 0.50 / 0.005 | Standard PPO coefficients |
| Safety Supervisor | ACTIVE | Filters physical actions before simulator stepping |
| Reward Function | Dense shaped | Progress + proximity penalty + smoothness + terminal |
| Arm C Intervention | None | Standard environment rollouts |
| Arm D Intervention | Bilateral reflection | 50% alternating episodes with $\mathcal{T}_{\text{obs}}, \mathcal{T}_{\text{act}}$ |

---

## 3. Pre-Flight Smoke Test Verification

Before full execution, a 1-iteration smoke test was executed for both arms:
- Model initialized from `ppo_policy_best.pt` correctly.
- Pre-training validation: $0.0\%$ success, $v_y = -0.2263\text{ m/s}$, $\dot{\psi} = -0.1513\text{ rad/s}$.
- Memory and tensor buffers initialized without non-finite values.
- Checkpoint reloaded and validated using `PPOPolicyController`.
- Exit code 0 for both arms.

---

## 4. Iteration-by-Iteration Training Progression

### 4.1 Arm C — Standard PPO (No Augmentation)
| Iter | Mean Return | Policy Loss | Value Loss | Entropy | Val Intervention (%) | Open-Space $v_y$ (m/s) | Open-Space $\dot{\psi}$ (rad/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (init) | — | — | — | — | 90.9% | -0.2263 | -0.1513 |
| 1 | -35.82 | -0.0063 | 0.9391 | 0.8753 | 90.9% | -0.2163 | -0.1338 |
| 2 | -19.04 | -0.0059 | 0.8205 | 0.8748 | 90.8% | -0.2140 | -0.1403 |
| 3 | -44.01 | -0.0063 | 0.8172 | 0.8752 | 90.8% | -0.2294 | -0.1827 |
| 4 | -43.31 | -0.0008 | 0.7719 | 0.8762 | 90.8% | -0.2361 | -0.2000 |
| 5 | -36.64 | -0.0013 | 0.7497 | 0.8751 | 90.8% | -0.2264 | -0.2280 |
| 6 | -32.42 | -0.0130 | 0.7573 | 0.8748 | 90.8% | -0.2224 | -0.2402 |
| 7 | -22.80 | -0.0080 | 0.7725 | 0.8754 | 90.8% | -0.2112 | -0.2102 |
| 8 | -19.10 | -0.0075 | 0.7967 | 0.8767 | 90.8% | -0.1809 | -0.2156 |
| 9 | -14.62 | -0.0084 | 0.7567 | 0.8770 | 90.8% | -0.1716 | -0.2112 |
| 10 | -11.47 | -0.0042 | 0.6970 | 0.8783 | 90.8% | -0.1777 | **-0.2527** |

*Observation on Arm C:* While forward policy loss and mean reward showed slight improvement, the yaw-rate rotational bias worsened steadily from $-0.1513\text{ rad/s}$ to $-0.2527\text{ rad/s}$ as the drone reinforced rightward turning habits.

### 4.2 Arm D — PPO with Reflection Augmentation
| Iter | Mean Return | Policy Loss | Value Loss | Entropy | Val Intervention (%) | Open-Space $v_y$ (m/s) | Open-Space $\dot{\psi}$ (rad/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (init) | — | — | — | — | 90.9% | -0.2263 | -0.1513 |
| 1 | -40.92 | -0.0065 | 0.8407 | 0.8754 | 90.9% | -0.2453 | -0.1454 |
| 2 | -52.02 | -0.0071 | 0.7601 | 0.8754 | 90.8% | -0.2615 | -0.1588 |
| 3 | -21.22 | -0.0078 | 0.7423 | 0.8757 | 90.8% | -0.2702 | -0.1626 |
| 4 | -26.16 | -0.0073 | 0.7061 | 0.8774 | 90.8% | -0.2832 | -0.1410 |
| 5 | -36.03 | -0.0054 | 0.6781 | 0.8782 | 90.8% | -0.2367 | -0.1818 |
| 6 | -29.38 | -0.0045 | 0.6690 | 0.8780 | 90.8% | -0.1998 | -0.1812 |
| 7 | -32.95 | -0.0065 | 0.6125 | 0.8784 | 90.7% | -0.1975 | -0.1788 |
| 8 | -50.44 | -0.0051 | 0.5369 | 0.8795 | 88.5% | -0.2127 | -0.1047 |
| 9 | -37.92 | -0.0073 | 0.4452 | 0.8678 | 88.4% | -0.1877 | -0.0607 |
| 10 | -37.33 | -0.0103 | 0.2446 | 0.8449 | **82.9%** | -0.1772 | **-0.0758** |

*Observation on Arm D:* Value loss dropped by over 70% (0.84 $\to$ 0.24), indicating substantially better generalized value fitting across mirrored states. Open-space yaw-rate bias dropped to $-0.0758\text{ rad/s}$ (a 70% reduction vs Arm C).

---

## 5. Comparative Summary

| Metric | Pre-Training (Arm A) | Arm C (No Aug) | Arm D (Reflection Aug) | Difference (D vs C) |
| :--- | :---: | :---: | :---: | :---: |
| Open-Space Center $v_y$ | $-0.2263\text{ m/s}$ | $-0.1777\text{ m/s}$ | $-0.1772\text{ m/s}$ | $+0.0005\text{ m/s}$ |
| Open-Space Center $\dot{\psi}$ | $-0.1513\text{ rad/s}$ | $-0.2527\text{ rad/s}$ | $\mathbf{-0.0758\text{ rad/s}}$ | **-70.0% bias** |
| Final Value Loss | $0.84$ | $0.70$ | $\mathbf{0.24}$ | **-65.7%** |
| Val Intervention Rate | $90.9\%$ | $90.8\%$ | $\mathbf{82.9\%}$ | **-7.9 pp** |
| Val Success Rate (10 seeds) | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\text{ pp}$ |

Both candidate checkpoints have been safely exported and are ready for Phase 4G-E validation analysis.
