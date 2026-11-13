# Phase 4G-A: Pre-Training Audit and Baseline Reproduction Report

**Date:** 2026-10-29  
**Status:** COMPLETED & VERIFIED  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Deliverable Files:**
- `reports/phase4g_baseline_reproduction.md`
- `reports/phase4g_baseline_metrics.json`

---

## 1. Executive Summary

In accordance with Phase 4G-A instructions, an independent pre-training audit and empirical baseline reproduction were conducted prior to modifying any learning pipeline or executing training.

### Key Findings
1. **PPO Rightward Lateral Bias Confirmed & Independently Reproduced:**
   - Under forward cruise ($v_x = 0.60\text{ m/s}$) at corridor centerline ($Y = 0.00\text{ m}$), the frozen canonical PPO checkpoint (`ppo_policy_best.pt`) commands a strong rightward lateral velocity $v_y = -0.2267\text{ m/s}$ and clockwise yaw rate $\dot{\psi} = -0.1530\text{ rad/s}$.
   - Under identical conditions, the Behavior Cloning baseline (`bc_policy_best.pt`) commands near-perfect lateral centering: $v_y = -0.0005\text{ m/s}$ and $\dot{\psi} = -0.0006\text{ rad/s}$.
2. **Constrained Right-Hand Passage Selection Confirmed:**
   - In paired mirrored obstacle tests at $X=3.0\text{ m}$, when the obstacle is shifted left ($Y_{\text{obs}} = -0.25\text{ m}$ or $-0.15\text{ m}$), the left bypass gap is wide and feasible ($1.05\text{ m}$ and $0.95\text{ m}$), while the right passage is constricted ($0.55\text{ m}$ and $0.65\text{ m}$).
   - PPO systematically attempts the constricted right-hand passage in 100% of cases, suffering severe supervisor interventions and failing to pass the obstacle.
   - BC correctly identifies and chooses the wider left-hand passage, passing the obstacle cleanly ($100\%$ pass rate).
3. **All Checkpoint Hashes Match Canonical Records:**
   - `ppo_policy_best.pt`, `bc_policy_best.pt`, and `ppo_phase4b_candidate.pt` remain identical to immutable baselines.

---

## 2. Checkpoint Verification Manifest

All checkpoint SHA-256 hashes were independently computed and verified against canonical records:

| Checkpoint File | SHA-256 Hash | Status |
| :--- | :--- | :--- |
| `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | VERIFIED |
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | VERIFIED |
| `models/checkpoints/ppo_phase4b_candidate.pt` | `5b9381188058ee8e690db3ec9b929a1b99a74469f7647bc527d2e2e76e5a8583` | VERIFIED |
| `models/checkpoints/policy_bc.onnx` | `9365aab7253c38793af5efa7cb7eb89b7e1db3bf60ea96e70df1671cdbbb4f50` | VERIFIED |
| `models/checkpoints/policy_ppo.onnx` | `38dd6575d9828b4616d740b296b067c7ee964fc7de1ff5e8f671c4ecb82a0ecf` | VERIFIED |

---

## 3. System Architecture & Baseline Pipeline Audit

### 3.1 Policy Architecture (`NavigationPolicyMLP`)
- **Inputs:** 23 continuous features.
- **Backbone:** 3-layer MLP: `Linear(23, 128)` -> `LayerNorm(128)` -> `GELU()` -> `Linear(128, 128)` -> `LayerNorm(128)` -> `GELU()` -> `Linear(128, 64)` -> `LayerNorm(64)` -> `GELU()`.
- **Output Head:** `Linear(64, 4)` predicting raw logits $\mathbf{z} = [z_0, z_1, z_2, z_3]$.
- **Action Bounding:** Scaled hyperbolic tangent enforces physical UAV velocity bounds:
  $$\mathbf{a} = \mathbf{a}_{\text{center}} + \mathbf{a}_{\text{half\_range}} \odot \tanh(\mathbf{z})$$
  - Forward velocity $v_x = 0.30 + 0.50 \tanh(z_0) \in [-0.20, +0.80]\text{ m/s}$
  - Lateral velocity $v_y = 0.00 + 0.60 \tanh(z_1) \in [-0.60, +0.60]\text{ m/s}$ (+Left, -Right)
  - Vertical velocity $v_z = 0.00 + 0.40 \tanh(z_2) \in [-0.40, +0.40]\text{ m/s}$ (+Up, -Down)
  - Yaw rate $\dot{\psi} = 0.00 + 0.60 \tanh(z_3) \in [-0.60, +0.60]\text{ rad/s}$ (+CCW/Left, -CW/Right)

### 3.2 Observation Contract (23 Dimensions)
Exact feature ordering verified in `core/contracts.py`:
- `0..11`: 12-channel rangefinder distances normalized by $d_{\text{max}} = 4.0\text{ m} \in [0.0, 1.0]$:
  - `Ray 0`: Front center ($0^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 1`: Front-left $30^\circ$ ($+30^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 2`: Front-right $30^\circ$ ($-30^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 3`: Side-front-left $60^\circ$ ($+60^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 4`: Side-front-right $60^\circ$ ($-60^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 5`: Left $90^\circ$ ($+90^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 6`: Right $90^\circ$ ($-90^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 7`: Rear $180^\circ$ ($180^\circ$ azimuth, $0^\circ$ pitch)
  - `Ray 8`: Front pitch up $45^\circ$ ($0^\circ$ azimuth, $+45^\circ$ pitch)
  - `Ray 9`: Front pitch down $45^\circ$ ($0^\circ$ azimuth, $-45^\circ$ pitch)
  - `Ray 10`: Ceiling zenith ($0^\circ$ azimuth, $+90^\circ$ pitch)
  - `Ray 11`: Floor nadir ($0^\circ$ azimuth, $-90^\circ$ pitch)
- `12..14`: Relative goal unit vector in body FLU frame $[g_x, g_y, g_z]$
- `15..17`: Estimated body velocity in body FLU frame $[v_x, v_y, v_z]\text{ m/s}$
- `18..21`: Attitude trigonometric vector $[\sin(\phi), \cos(\phi), \sin(\theta), \cos(\theta)]$
- `22`: Yaw angular rate $\dot{\psi}\text{ rad/s}$

### 3.3 Inference Behavior
- **Evaluation / Deployment:** Deterministic mode ($a = \mu(s)$ via bounded $\tanh$ head).
- **Training (PPO Exploration):** Diagonal Gaussian distribution $\mathcal{N}(\mu(s), \sigma^2)$ where $\sigma = \exp(\text{clamp}(\log \sigma, -2.5, 0.2))$. Default $\log \sigma = -1.2$.

### 3.4 Training Reward Definition (`models/train_ppo.py`)
Dense shaped reward function preserved across all phases:
$$R_t = R_{\text{progress}} + R_{\text{obs\_proximity}} + R_{\text{wall\_proximity}} + R_{\text{smoothness}} + R_{\text{terminal}}$$
1. $R_{\text{progress}} = 4.0 \cdot (d_{\text{prev\_goal}} - d_{\text{curr\_goal}})$
2. $R_{\text{obs\_proximity}} = -2.5 \cdot (0.40 - d_{\text{obs}})^2$ if $d_{\text{obs}} < 0.40\text{ m}$ else $0$
3. $R_{\text{wall\_proximity}} = -2.5 \cdot (0.35 - d_{\text{wall}})^2$ if $d_{\text{wall}} < 0.35\text{ m}$ else $0$
4. $R_{\text{smoothness}} = -0.015 \sum_{i} (a_i - a_{i, \text{prev}})^2$
5. $R_{\text{terminal}} = +20.0$ on success, $-20.0$ on crash.

### 3.5 Termination Conditions
- Goal reached: $|x - x_{\text{goal}}| < 0.25\text{ m}$, $|y| < 0.35\text{ m}$, $|z - z_{\text{goal}}| < 0.30\text{ m}$, terminal speed $\le 0.35\text{ m/s}$.
- Crash: Obstacle clearance $\le 0.05\text{ m}$ or wall clearance $\le 0.05\text{ m}$.
- Timeout: Maximum 200 steps ($10.0\text{ s}$ simulated time).

### 3.6 Safety Supervisor Integration
- In training rollouts, `SafetySupervisor` actively filters proposed policy commands before simulator stepping (`use_supervisor=True`).
- Rollout buffer stores proposed policy action and its log probability to preserve valid on-policy likelihood ratios for PPO surrogate objective.

### 3.7 Checkpoint Loading & State Restoration
- `ppo_policy_best.pt` stores: `model_state_dict`, `critic_state_dict`, `iteration`, `eval_success_rate`, `mean_reward`, `obs_dim`, `act_dim`.
- Does **not** contain optimizer state or `log_std`.
- PPO initialization from `ppo_policy_best.pt` restores actor and critic weights; optimizer state and `log_std` are initialized to defaults (`lr=3e-4`, `log_std=-1.2`). This is recorded as an intentional training warm-start, not an exact checkpoint resume.

---

## 4. Empirical Reproduction of Baseline Bias

### 4.1 Controlled Open-Space Probe
Conducted at $X = 1.0\text{ m}$, $Z = 1.0\text{ m}$ across lateral sweep $Y \in [-0.40, +0.40]\text{ m}$:

#### Static Hover Initial Condition ($v_x = 0.00\text{ m/s}$)
| Lateral Pos $Y$ | PPO Raw $v_y$ | PPO Raw $\dot{\psi}$ | BC Raw $v_y$ | BC Raw $\dot{\psi}$ | Supervisor State (PPO) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $-0.40\text{ m}$ | $+0.0876\text{ m/s}$ | $+0.0315\text{ rad/s}$ | $+0.0043\text{ m/s}$ | $+0.0067\text{ rad/s}$ | `obstacle_brake` |
| $-0.30\text{ m}$ | $+0.0468\text{ m/s}$ | $+0.0118\text{ rad/s}$ | $+0.0034\text{ m/s}$ | $+0.0066\text{ rad/s}$ | `obstacle_brake` |
| $-0.20\text{ m}$ | $+0.0079\text{ m/s}$ | $-0.0048\text{ rad/s}$ | $+0.0024\text{ m/s}$ | $+0.0065\text{ rad/s}$ | `obstacle_brake` |
| $-0.10\text{ m}$ | $-0.0157\text{ m/s}$ | $-0.0187\text{ rad/s}$ | $+0.0013\text{ m/s}$ | $+0.0061\text{ rad/s}$ | `obstacle_brake` |
| $\mathbf{0.00\text{ m}}$ | $\mathbf{-0.0346\text{ m/s}}$ | $\mathbf{-0.0299\text{ rad/s}}$ | $\mathbf{0.0000\text{ m/s}}$ | $\mathbf{+0.0053\text{ rad/s}}$ | `obstacle_brake` |
| $+0.10\text{ m}$ | $-0.0470\text{ m/s}$ | $-0.0378\text{ rad/s}$ | $-0.0017\text{ m/s}$ | $+0.0038\text{ rad/s}$ | `obstacle_brake` |
| $+0.20\text{ m}$ | $-0.0519\text{ m/s}$ | $-0.0416\text{ rad/s}$ | $-0.0038\text{ m/s}$ | $+0.0014\text{ rad/s}$ | `obstacle_brake` |
| $+0.30\text{ m}$ | $-0.0487\text{ m/s}$ | $-0.0401\text{ rad/s}$ | $-0.0055\text{ m/s}$ | $-0.0018\text{ rad/s}$ | `obstacle_brake` |
| $+0.40\text{ m}$ | $-0.0381\text{ m/s}$ | $-0.0334\text{ rad/s}$ | $-0.0062\text{ m/s}$ | $-0.0053\text{ rad/s}$ | `obstacle_brake` |

#### Forward Cruise Initial Condition ($v_x = 0.60\text{ m/s}$)
| Lateral Pos $Y$ | PPO Raw $v_y$ | PPO Raw $\dot{\psi}$ | BC Raw $v_y$ | BC Raw $\dot{\psi}$ | PPO Equilibrium Shift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $-0.40\text{ m}$ | $+0.0210\text{ m/s}$ | $+0.0382\text{ rad/s}$ | $+0.0084\text{ m/s}$ | $+0.0125\text{ rad/s}$ | PPO zero crossing at |
| $-0.30\text{ m}$ | $-0.0612\text{ m/s}$ | $-0.0183\text{ rad/s}$ | $+0.0060\text{ m/s}$ | $+0.0090\text{ rad/s}$ | $Y \approx -0.38\text{ m}$ |
| $-0.20\text{ m}$ | $-0.1369\text{ m/s}$ | $-0.0766\text{ rad/s}$ | $+0.0039\text{ m/s}$ | $+0.0058\text{ rad/s}$ | (strong rightward offset) |
| $-0.10\text{ m}$ | $-0.1931\text{ m/s}$ | $-0.1235\text{ rad/s}$ | $+0.0018\text{ m/s}$ | $+0.0026\text{ rad/s}$ | |
| $\mathbf{0.00\text{ m}}$ | $\mathbf{-0.2267\text{ m/s}}$ | $\mathbf{-0.1530\text{ rad/s}}$ | $\mathbf{-0.0005\text{ m/s}}$ | $\mathbf{-0.0006\text{ rad/s}}$ | BC zero crossing at |
| $+0.10\text{ m}$ | $-0.2443\text{ m/s}$ | $-0.1698\text{ rad/s}$ | $-0.0028\text{ m/s}$ | $-0.0037\text{ rad/s}$ | $Y \approx 0.00\text{ m}$ |
| $+0.20\text{ m}$ | $-0.2520\text{ m/s}$ | $-0.1788\text{ rad/s}$ | $-0.0048\text{ m/s}$ | $-0.0066\text{ rad/s}$ | (perfect corridor center) |
| $+0.30\text{ m}$ | $-0.2530\text{ m/s}$ | $-0.1834\text{ rad/s}$ | $-0.0065\text{ m/s}$ | $-0.0090\text{ rad/s}$ | |
| $+0.40\text{ m}$ | $-0.2486\text{ m/s}$ | $-0.1849\text{ rad/s}$ | $-0.0076\text{ m/s}$ | $-0.0107\text{ rad/s}$ | |

### 4.2 Mirrored Obstacle Pair Bypass Tests
Obstacle placed at $X = 3.0\text{ m}$ with varying lateral offsets:

| Obstacle $Y_{\text{obs}}$ | Left Gap | Right Gap | Feasible Side | PPO Bypass | PPO Passed? | BC Bypass | BC Passed? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $-0.25\text{ m}$ | $1.05\text{ m}$ | $0.55\text{ m}$ | Left only | Right | **False (Crash/Stall)** | Left | **True** |
| $-0.15\text{ m}$ | $0.95\text{ m}$ | $0.65\text{ m}$ | Left only | Right | **False (Crash/Stall)** | Left | **True** |
| $0.00\text{ m}$ | $0.80\text{ m}$ | $0.80\text{ m}$ | Symmetric | Right | False (Stall) | Left | False (Stall) |
| $+0.15\text{ m}$ | $0.65\text{ m}$ | $0.95\text{ m}$ | Right only | Right | False (Stall) | Right | **True** |
| $+0.25\text{ m}$ | $0.55\text{ m}$ | $1.05\text{ m}$ | Right only | Right | **True** | Right | **True** |

**Conclusion of Audit & Reproduction:**
The baseline rightward bias is conclusively reproduced in both open space and obstacle interactions. PPO has an asymmetric lateral equilibrium near $Y = -0.38\text{ m}$ and refuses to turn left around obstacles even when the left side is wide and feasible.

We are authorized to proceed to Phase 4G-B (Verify Reflection Transformation).
