# Phase 6B Pre-Run Audit & Baseline Verification Report

**Phase:** Phase 6B — Targeted Pillar 3 Recovery Experiment  
**Timestamp:** 2026-10-30T02:58:00+05:30  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Root:** `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/`  
**Git Commit SHA:** `f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f`  

---

## 1. Executive Summary

Prior to implementing any candidate recovery algorithms or modifying configuration boundaries, a strict preflight audit was conducted to establish the exact source-of-truth baseline for the UAV corridor navigation system.

The audit verified:
1. Exact reproduction of the Phase 6A baseline across the 50 canonical development seeds (`2000–2049`): **42 / 50 successes (84.0%)**, zero simulated collisions, zero safety envelope violations, minimum obstacle clearance of $0.0957\text{ m}$, and minimum wall clearance of $0.1942\text{ m}$.
2. Precise replication of all 8 residual failure seeds, specifically the five targeted Pillar 3 lateral traps (Seeds 2001, 2005, 2006, 2022, 2042), two slow-transit timeouts (Seeds 2019, 2049), and one goal-alignment timeout (Seed 2036).
3. Exact SHA-256 integrity verification of all protected assets (canonical safety supervisor, baseline BC model checkpoint, legacy PPO model checkpoint).
4. Physical hardware bus lockdown and held-out seeds (`90000–90099`) sealing.

---

## 2. Protected Asset Cryptographic Manifest

| File / Component | Expected SHA-256 Hash | Pre-Run Measured SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :--- |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **Verified Match (Zero Mod)** |
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **Verified Match (Zero Mod)** |
| `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **Verified Match (Zero Mod)** |
| `models/checkpoints/policy_bc.onnx` | `9365aab7253c38793af5efa7cb7eb89b7e1db3bf60ea96e70df1671cdbbb4f50` | `9365aab7253c38793af5efa7cb7eb89b7e1db3bf60ea96e70df1671cdbbb4f50` | **Verified Match** |
| `core/guidance.py` | `c69b7ca069e1c43668718f940b0dcc3ea0f2437b99412d89c88d146b1032c5d9` | `c69b7ca069e1c43668718f940b0dcc3ea0f2437b99412d89c88d146b1032c5d9` | **Verified Match** |
| `controllers/guidance_controller.py` | `2818fa497cd9aa1657114a895ce031edfc0687e1211cbead4003caffdd47dc21` | `2818fa497cd9aa1657114a895ce031edfc0687e1211cbead4003caffdd47dc21` | **Verified Match** |
| `hardware/onnx_runner.py` | `d80e4d8f2cfbbd9e7184ddfc138ffeb2614d0c2bb4861c1413a583d417cc30cd` | `d80e4d8f2cfbbd9e7184ddfc138ffeb2614d0c2bb4861c1413a583d417cc30cd` | **Verified Match** |

---

## 3. Physical Invariants and Operating Parameters

From inspection of `core/geometry.py`, `core/contracts.py`, `sim/corridor_env.py`, and `controllers/safety_supervisor.py`:
- **Corridor Dimensions:** Length $L = 10.0\text{ m}$ ($x \in [0.0, 10.0]$), Width $W = 2.0\text{ m}$ ($y \in [-1.0, 1.0]$), Height $H = 2.0\text{ m}$ ($z \in [0.0, 2.0]$).
- **Obstacle Placement:** 3 cylindrical pillars at nominal X positions $x \in [3.0, 5.0, 7.0]\text{ m}$. Pillar radius $r_{\text{obs}} = 0.20\text{ m}$, randomized lateral offset $y \in [-0.35, 0.35]\text{ m}$.
- **Drone Radius:** $r_{\text{drone}} = 0.15\text{ m}$, mass $m = 1.2\text{ kg}$.
- **Start Position:** $(0.5, y_{\text{start}}, 1.0)$ with initial perturbation $y_{\text{start}} \in [-0.10, 0.10]\text{ m}$.
- **Goal Position & Sphere:** $(9.5, 0.0, 1.0)$ with capture radius $r_{\text{goal}} = 0.35\text{ m}$.
- **Success Criteria:** Termination reason == `"success"` AND final speed $|v| \le 0.35\text{ m/s}$ at step $\le 500$.
- **Action Limits:** $v_x \in [0.0, 0.6]\text{ m/s}$, $v_y \in [-0.4, 0.4]\text{ m/s}$, $v_z \in [-0.3, 0.3]\text{ m/s}$, $\dot{\psi} \in [-0.6, 0.6]\text{ rad/s}$.
- **Observation Space:** 23 continuous floats:
  - `obs[0:12]`: 12 normalized rangefinder rays ($0.0 \sim 4.0\text{ m}$).
  - `obs[12:15]`: Relative goal vector in body FLU, scaled by $\min(1.0, d_{\text{goal}} / 2.0)$.
  - `obs[15:18]`: Body velocity vector $[v_x, v_y, v_z]\text{ m/s}$.
  - `obs[18:22]`: Attitude trigonometric components $[\sin\phi, \cos\phi, \sin\theta, \cos\theta]$.
  - `obs[22]`: Yaw rate $\dot{\psi}\text{ rad/s}$.
- **Supervisor Caution Distance:** $d_{\text{caut}} = 0.50\text{ m}$ (verified canonical override), critical brake $d_{\text{crit}} = 0.30\text{ m}$, emergency reverse $d_{\text{rev}} = 0.15\text{ m}$, boundary buffer $0.25\text{ m}$, critical lateral clamp $0.32\text{ m}$.

---

## 4. Baseline Reproduction Outcome (Development Seeds 2000–2049)

The Phase 6A baseline controller was executed using `diagnose_phase6a_failures.py`:
- **Completions:** Exactly **42 / 50 successes (84.0%)**.
- **Mean Progress:** $8.475\text{ m}$.
- **Simulated Collisions:** 0 (0 obstacle, 0 boundary).
- **Safety Envelope Violations:** 0 (minimum obstacle clearance $0.0957\text{ m} > 0.080\text{ m}$, minimum wall clearance $0.1942\text{ m} > 0.080\text{ m}$).
- **Pre-Run Test Suite Baseline:** **153 / 153 tests passed** in $2.50\text{ s}$.

### Failure Seed Audit
| Seed | Termination Reason | Progress | Final Outcome Classification |
| :---: | :---: | :---: | :--- |
| **2001** | `timeout` | $7.051\text{ m}$ | Pillar 3 Slow Crawl / Step Runout |
| **2005** | `timeout` | $6.145\text{ m}$ | Pillar 3 Lateral Pinch Standoff |
| **2006** | `timeout` | $6.283\text{ m}$ | Pillar 3 Lateral Pinch Standoff |
| **2019** | `timeout` | $9.493\text{ m}$ | Timeout Slow Transit (cleared all 3 pillars) |
| **2022** | `timeout` | $6.339\text{ m}$ | Pillar 3 Lateral Pinch Standoff |
| **2036** | `timeout` | $8.673\text{ m}$ | Goal Alignment Timeout (cleared all 3 pillars) |
| **2042** | `timeout` | $6.160\text{ m}$ | Pillar 3 Lateral Pinch Standoff |
| **2049** | `timeout` | $9.527\text{ m}$ | Timeout Slow Transit (cleared all 3 pillars) |

The pre-run baseline reproduction was verified with 100% mathematical fidelity.

