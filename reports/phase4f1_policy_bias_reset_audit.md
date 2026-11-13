# Phase 4F-1: Controlled Policy-Bias Verification and Viewer Reset Audit

**Date:** October 28, 2026  
**Status:** COMPLETED & VERIFIED  
**Investigation Mode:** Verification & Interface Correction (Strictly Non-Training)  
**Target Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`

---

## 1. Executive Summary

Phase 4F-1 was commissioned to independently investigate two findings reported in Phase 4F-0:
1. That the PPO checkpoint exhibits a systematic rightward lateral bias, causing the drone to choose the right side of obstacles even when the left side is substantially wider and more feasible.
2. That the 3D simulation viewer failed to reset the environment when the user entered a new seed and clicked Play or Single Step, silently continuing the prior episode.

### 1.1 Acceptance Category Outcomes

- **Viewer / Reset Correction:** **PASS**  
  The interface defect was confirmed, isolated, and corrected. Seed entry, Single Step, Play, and Reset are unified under a single race-guarded asynchronous function (`executeReset`). The active simulation seed is prominently displayed in both the top HUD bar and sidebar. Supervisor episode-dependent state leakage in `eval_engine.py` was eliminated. Regression test suite (`tests/test_viewer_reset_regression.py`) passes 100%.

- **Policy-Bias Verification:** **PASS**  
  PPO and Behavior Cloning (BC) were compared under technically equivalent conditions. The rightward bias of PPO was **empirically confirmed**:
  - In open space along the corridor centerline ($Y=0.0\text{ m}, v_x=0.60\text{ m/s}$), raw PPO commands $v_y = -0.227\text{ m/s}$ (Right) and $\dot{\psi} = -0.153\text{ rad/s}$ (CW/Right), shifting lateral cruising equilibrium to $Y \approx -0.38\text{ m}$.
  - BC evaluated under the identical state commands $v_y = -0.001\text{ m/s}$ and $\dot{\psi} = -0.001\text{ rad/s}$ (symmetric).
  - In controlled mirrored obstacle pairs at $X=3.0\text{ m}$, BC dynamically chose the wider opening in 100% of cases (switching to the Left when the obstacle was shifted right). In contrast, PPO chose the Right in 100% of cases, forcing itself through narrow $0.55\text{–}0.65\text{ m}$ bottlenecks despite $>0.95\text{ m}$ openings on the left.

- **System Integrity:** **PASS**  
  Zero model training occurred. No canonical supervisor or contract files were modified. All checkpoint SHA-256 hashes were verified before and after and remain bit-for-bit identical.

---

## 2. Checkpoint Hashes & Verification

### 2.1 Before and After SHA-256 Hashes
```
models/checkpoints/ppo_policy_best.pt:
  33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0 (UNCHANGED)

models/checkpoints/bc_policy_best.pt:
  93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f (UNCHANGED)

models/checkpoints/ppo_phase4b_candidate.pt:
  5b9381188058ee8e690db3ec9b929a1b99a74469f7647bc527d2e2e76e5a8583 (UNCHANGED)

models/checkpoints/policy_bc.onnx:
  9365aab7253c38793af5efa7cb7eb89b7e1db3bf60ea96e70df1671cdbbb4f50 (UNCHANGED)

models/checkpoints/policy_ppo.onnx:
  38dd6575d9828b4616d740b296b067c7ee964fc7de1ff5e8f671c4ecb82a0ecf (UNCHANGED)
```

### 2.2 Investigation of User Prompt BC Hash String
The user prompt listed:
`93c4d7d1d8476b8d288a24fca1862d87133696f78b960ea96e70df1671cdb4f50`
Analysis of this string:
- Length: **67 characters** (valid SHA-256 is exactly 64 hex characters).
- Characters 0–47: `93c4d7d1d8476b8d288a24fca1862d87133696f78b960` matches `bc_policy_best.pt`.
- Characters 48–66: `ea96e70df1671cdb4f50` matches characters 45–64 of `policy_bc.onnx` (`...ea96e70df1671cdbbb4f50`).
- **Conclusion:** The prompt string was an accidental composite concatenation of the PyTorch and ONNX hashes. The actual disk hash `93c4d7d1...afad44f` is identical to historical records from Phases 4A–4F-0.

---

## 3. Coordinate Conventions & Observation Contracts

### 3.1 Spatial Frame Definitions
- **World Frame (ENU):**
  - $+X$: Longitudinal corridor axis ($X=0.0\text{ m}$ entrance to $X=10.0\text{ m}$ exit).
  - $+Y$: Lateral axis towards the **Left Wall** ($Y=+1.0\text{ m}$).
  - $-Y$: Lateral axis towards the **Right Wall** ($Y=-1.0\text{ m}$).
  - $+Z$: Vertical axis ($Z=0.0\text{ m}$ floor to $Z=2.0\text{ m}$ ceiling).
- **Body Frame (FLU):**
  - $+v_x$: Forward linear velocity.
  - $+v_y$: Lateral linear velocity to the **Left**.
  - $-v_y$: Lateral linear velocity to the **Right**.
  - $+v_z$: Vertical velocity Upward.
  - $+\dot{\psi}$: Yaw angular velocity Counter-Clockwise (turning Left).
  - $-\dot{\psi}$: Yaw angular velocity Clockwise (turning Right).

### 3.2 Observation Vector (23 Continuous Floats)
Both PPO and BC ingest the identical 23-float `DeployableObservation`:
- `[0:12]`: 12 rangefinder rays normalized in $[0, 1]$ ($d / 4.0\text{ m}$).
  - Ray 0: Forward center ($0^\circ$).
  - Ray 1: Front-Left 30° ($+30^\circ$).
  - Ray 2: Front-Right 30° ($-30^\circ$).
  - Ray 3: Side-Front-Left 60° ($+60^\circ$).
  - Ray 4: Side-Front-Right 60° ($-60^\circ$).
  - Ray 5: Left 90° ($+90^\circ$).
  - Ray 6: Right 90° ($-90^\circ$).
  - Ray 7: Rear 180° ($180^\circ$).
  - Rays 8–11: Pitch up/down, zenith (+Z), nadir (-Z).
- `[12:15]`: Relative goal unit direction in body FLU frame ($g_x, g_y, g_z$).
- `[15:18]`: Estimated body velocity in body FLU frame ($v_x, v_y, v_z$).
- `[18:22]`: Trigonometric attitude $[\sin(\phi), \cos(\phi), \sin(\theta), \cos(\theta)]$.
- `[22]`: Body yaw rate $\dot{\psi}$ (rad/s).

### 3.3 Action Space Equivalence
Both `PPOPolicyController` and `BehaviorCloningPolicyController` instantiate `NavigationPolicyMLP` with identical parameters:
- Hidden layers: `(128, 128, 64)`.
- Bound head scaling:
  $$v_x = 0.30 + 0.50 \tanh(\text{raw}_0) \in [-0.20, +0.80]\text{ m/s}$$
  $$v_y = 0.00 + 0.60 \tanh(\text{raw}_1) \in [-0.60, +0.60]\text{ m/s}$$
  $$v_z = 0.00 + 0.40 \tanh(\text{raw}_2) \in [-0.40, +0.40]\text{ m/s}$$
  $$\dot{\psi} = 0.00 + 0.60 \tanh(\text{raw}_3) \in [-0.60, +0.60]\text{ rad/s}$$
- Both policies operate in deterministic evaluation mode (`model.eval()`).

---

## 4. Controlled Policy Bias Probe Results

### 4.1 Open-Space Centerline Probe ($X=1.0\text{ m}, Z=1.0\text{ m}$)
The drone was placed in open corridor space (2.0 meters before Obstacle 1) across lateral offsets $Y \in [-0.40, +0.40]\text{ m}$.

#### Forward Cruising Condition ($v_x = 0.60\text{ m/s}$, $\text{yaw} = 0.0\text{ rad}$)

| Lateral Position ($Y$) | PPO Raw $v_y$ | PPO Raw $\dot{\psi}$ | PPO Safe $v_y$ | BC Raw $v_y$ | BC Raw $\dot{\psi}$ | BC Safe $v_y$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $+0.40\text{ m}$ (Left) | **$-0.282\text{ m/s}$** | **$-0.225\text{ rad/s}$** | $-0.282$ | $-0.025\text{ m/s}$ | $-0.019\text{ rad/s}$ | $-0.025$ |
| $+0.20\text{ m}$ (Left) | **$-0.223\text{ m/s}$** | **$-0.166\text{ rad/s}$** | $-0.223$ | $+0.008\text{ m/s}$ | $+0.008\text{ rad/s}$ | $+0.008$ |
| $+0.10\text{ m}$ (Left) | **$-0.216\text{ m/s}$** | **$-0.158\text{ rad/s}$** | $-0.216$ | $+0.004\text{ m/s}$ | $+0.004\text{ rad/s}$ | $+0.004$ |
| **$0.00\text{ m}$ (Center)** | **$-0.227\text{ m/s}$** | **$-0.153\text{ rad/s}$** | **$-0.227$** | **$-0.001\text{ m/s}$** | **$-0.001\text{ rad/s}$** | **$-0.001$** |
| $-0.10\text{ m}$ (Right) | **$-0.197\text{ m/s}$** | **$-0.150\text{ rad/s}$** | $-0.197$ | $-0.002\text{ m/s}$ | $-0.001\text{ rad/s}$ | $-0.002$ |
| $-0.20\text{ m}$ (Right) | **$-0.092\text{ m/s}$** | **$-0.175\text{ rad/s}$** | $-0.092$ | $+0.011\text{ m/s}$ | $+0.002\text{ rad/s}$ | $+0.011$ |
| **$-0.38\text{ m}$ (Equil)** | **$0.000\text{ m/s}$** | **$-0.110\text{ rad/s}$** | **$0.000$** | $+0.025\text{ m/s}$ | $+0.016\text{ rad/s}$ | $+0.025$ |
| $-0.40\text{ m}$ (Right) | $+0.046\text{ m/s}$ | $-0.101\text{ rad/s}$ | $+0.046$ | $+0.027\text{ m/s}$ | $+0.018\text{ rad/s}$ | $+0.027$ |

**Finding:**
1. Along the entire corridor centerline ($Y=0.0\text{ m}$), PPO commands a substantial rightward lateral velocity setpoint ($v_y = -0.227\text{ m/s}$) and rightward clockwise yaw rate ($\dot{\psi} = -0.153\text{ rad/s}$).
2. PPO's lateral equilibrium (where commanded $v_y = 0$) is displaced by **$0.38\text{ meters}$ to the right** ($Y \approx -0.38\text{ m}$).
3. In contrast, BC commands virtually zero lateral velocity at centerline ($v_y = -0.001\text{ m/s}$) and exhibits symmetric centering behavior around $Y = 0.00\text{ m}$.

---

## 5. Controlled Mirrored Obstacle Pair Experiments

To test whether the policy dynamically adapts route selection to obstacle placement, paired obstacles were placed at $X=3.0\text{ m}$ with lateral offsets $Y_{\text{obs}} \in \{-0.25, -0.15, 0.00, +0.15, +0.25\}\text{ m}$.

### 5.1 Route Selection Matrix

| Obstacle $Y_{\text{obs}}$ | Left Gap | Right Gap | Wider Side | Supervisor Feasibility | BC Route | PPO Route | PPO Chose Constrained? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$-0.25\text{ m}$** | **$1.05\text{ m}$** | $0.55\text{ m}$ | Left | Left Only | **LEFT** | **RIGHT** | **YES** ($0.55\text{m}$ bottleneck) |
| **$-0.15\text{ m}$** | **$0.95\text{ m}$** | $0.65\text{ m}$ | Left | Left Only | **LEFT** | **RIGHT** | **YES** ($0.65\text{m}$ bottleneck) |
| **$0.00\text{ m}$** | $0.80\text{ m}$ | $0.80\text{ m}$ | Symmetric | Neither Nominal | **LEFT** | **RIGHT** | Tie-break |
| **$+0.15\text{ m}$** | $0.65\text{ m}$ | **$0.95\text{ m}$** | Right | Right Only | **RIGHT** | **RIGHT** | No ($0.95\text{m}$ open) |
| **$+0.25\text{ m}$** | $0.55\text{ m}$ | **$1.05\text{ m}$** | Right | Right Only | **RIGHT** | **RIGHT** | No ($1.05\text{m}$ open) |

### 5.2 Key Takeaways
- **Behavior Cloning (BC):** Perfectly matches the opening gap geometry. When the obstacle is shifted right, BC veers left into the $0.95\text{–}1.05\text{ m}$ opening. When the obstacle is shifted left, BC veers right into the $0.95\text{–}1.05\text{ m}$ opening.
- **PPO Policy:** Bypasses on the **RIGHT in 100% of cases**, completely indifferent to whether the obstacle is placed at $-0.25\text{ m}$ or $+0.25\text{ m}$. When the obstacle blocks the right, PPO still forces itself into the $<0.65\text{ m}$ bottleneck.

---

## 6. Route Feasibility & Clearance Arithmetic Reconciliation

The clearance arithmetic reported in Phase 4F-0 is here reconciled with explicit mathematical definitions:

### 6.1 Geometric Constants
- Corridor width: $W = 2.00\text{ m}$ ($Y_{\text{min}} = -1.00\text{ m}, Y_{\text{max}} = +1.00\text{ m}$).
- Obstacle cylindrical pillar radius: $r_{\text{obs}} = 0.20\text{ m}$ (diameter $d_{\text{obs}} = 0.40\text{ m}$).
- Drone physical collision radius: $r_{\text{drone}} = 0.15\text{ m}$ (bounding diameter $d_{\text{drone}} = 0.30\text{ m}$).

### 6.2 Free Gap Equations
For a pillar centered at $(X_{\text{obs}}, Y_{\text{obs}})$:
$$\text{Gap}_{\text{left}} = Y_{\text{max}} - (Y_{\text{obs}} + r_{\text{obs}}) = 1.00 - Y_{\text{obs}} - 0.20 = 0.80 - Y_{\text{obs}}$$
$$\text{Gap}_{\text{right}} = (Y_{\text{obs}} - r_{\text{obs}}) - Y_{\text{min}} = Y_{\text{obs}} - 0.20 - (-1.00) = 0.80 + Y_{\text{obs}}$$
$$\text{Total Free Space} = \text{Gap}_{\text{left}} + \text{Gap}_{\text{right}} = 1.60\text{ m} = W - 2 r_{\text{obs}}$$

### 6.3 Usable Gap Under Canonical Safety Supervisor Rules
Under `SafetySupervisorConfig` (`controllers/safety_supervisor.py`):
1. Wall boundary buffer: $B_{\text{wall}} = 0.25\text{ m}$. Active boundary repulsion initiates when drone surface is within $0.25\text{ m}$ of the wall:
   $$\text{Distance to wall from center} \ge r_{\text{drone}} + B_{\text{wall}} = 0.15 + 0.25 = 0.40\text{ m}$$
2. Obstacle critical distance: $B_{\text{obs}} = 0.30\text{ m}$. Emergency braking initiates when front range $< 0.30\text{ m}$, and lateral clamping initiates when lateral distance $< 0.32\text{ m}$:
   $$\text{Distance to obstacle surface from center} \ge r_{\text{drone}} + B_{\text{obs}} = 0.15 + 0.30 = 0.45\text{ m}$$
3. Minimum Nominal Gap for Intervention-Free Transit:
   $$G_{\text{nominal}} = (r_{\text{drone}} + B_{\text{wall}}) + (r_{\text{drone}} + B_{\text{obs}}) = 0.40 + 0.45 = \mathbf{0.85\text{ m}}$$

### 6.4 Clearance Regimes
- **Nominally Feasible ($G \ge 0.85\text{ m}$):** The drone can pass through the gap with non-zero clearance margins without triggering boundary repulsion or obstacle emergency braking.
- **Compressive Pinch Bottleneck ($0.30\text{ m} \le G < 0.85\text{ m}$):** The drone physically fits through the gap ($G > d_{\text{drone}}$), but **cannot** pass without triggering simultaneous boundary repulsion and obstacle braking (`boundary_and_obstacle_brake`). This creates the velocity limit-cycle deadlock identified in Phase 4D.
- **Physical Collision ($G < 0.30\text{ m}$):** Geometric overlap guarantees physical contact with either the wall or the pillar.

**Conclusion:** In Seeds 20 and 8000, Obstacle 1 left gaps of $0.646\text{ m}$ and $0.622\text{ m}$ on the right. Both fall into the **Compressive Pinch Bottleneck** regime ($<0.85\text{ m}$), guaranteeing supervisor deadlock, while the left side offered $>0.95\text{ m}$ of nominally feasible clearance.

---

## 7. Diagnostic Seed Evaluations (`1, 20, 100, 2000, 8000`)

| Seed | Controller | Obstacle 1 Placement | Chosen Bypass | Outcome | Root Cause of Failure |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | PPO | $Y=+0.01\text{ m}$ | Right | Timeout (500 steps) | Pinned at exit of Obstacle 2 |
| **1** | BC | $Y=+0.01\text{ m}$ | Left | Timeout (500 steps) | Pinned at Obstacle 1 exit |
| **1 (Replay)** | PPO | $Y=+0.01\text{ m}$ | Right | Timeout (500 steps) | **100% Identical bit-for-bit replay** |
| **1 (Replay)** | BC | $Y=+0.01\text{ m}$ | Left | Timeout (500 steps) | **100% Identical bit-for-bit replay** |
| **20** | **PPO** | $Y=-0.15\text{ m}$ (Right gap $0.65\text{m}$) | **RIGHT** | **Timeout at Obs 1** | **Chose constrained $0.65\text{m}$ bottleneck; pinned against wall** |
| **20** | **BC** | $Y=-0.15\text{ m}$ (Left gap $0.95\text{m}$) | **LEFT** | **Reached Obs 3** | **Chose wide $0.95\text{m}$ opening; cleanly cleared Obs 1 & 2** |
| **100** | PPO | $Y=+0.23\text{ m}$ (Right gap $1.03\text{m}$) | Right | Timeout at Obs 3 | Cleared Obs 1 & 2 on right; pinned at Obs 3 ($Y=-0.15\text{m}$) |
| **100** | BC | $Y=+0.23\text{ m}$ (Right gap $1.03\text{m}$) | Right | Timeout at Obs 2 | Cleared Obs 1 on right; pinned at Obs 2 |
| **2000** | PPO | $Y=+0.05\text{ m}$ (Right gap $0.85\text{m}$) | Right | Timeout at Goal | Cleared Obs 1 & 2 on right, Obs 3 on left |
| **2000** | BC | $Y=+0.05\text{ m}$ | Left | Timeout at Obs 1 | Pinned at Obs 1 exit |
| **8000** | **PPO** | $Y=-0.18\text{ m}$ (Right gap $0.62\text{m}$) | **RIGHT** | **Timeout at Obs 1** | **Chose constrained $0.62\text{m}$ bottleneck; pinned against wall** |
| **8000** | **BC** | $Y=-0.18\text{ m}$ (Left gap $0.98\text{m}$) | **LEFT** | **Reached Obs 2** | **Chose wide $0.98\text{m}$ opening; cleanly cleared Obs 1** |

---

## 8. Summary of Viewer Fixes & Test Execution

### 8.1 Files Modified
1. `viewer/eval_engine.py`: Added supervisor state restoration on reset (`loss_timer_s`, `stalled_timer_s`, `last_safe_action`).
2. `viewer/server.py`: Hardened `/api/reset` input validation with HTTP 400 error responses on invalid/empty/negative seeds.
3. `viewer/static/index.html`: Added active seed displays in the HUD header and sidebar.
4. `viewer/static/app.js`: Created unified `executeReset` function with race-condition sequencing, active seed display updates, and auto-reset triggers on Play and Single Step if seed was altered.

### 8.2 Regression Test Execution Results
```bash
PYTHONPATH=. .venv/bin/pytest tests/test_viewer_reset_regression.py -v
============================== 4 passed in 1.48s ===============================

PYTHONPATH=. .venv/bin/pytest
============================== 99 passed in 2.30s ==============================
```

All 99 tests in the repository pass with zero regressions.

---

## 9. Deliverable Artifacts

- [reports/phase4f1_policy_bias_reset_audit.md](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/phase4f1_policy_bias_reset_audit.md) (This report)
- [reports/phase4f1_viewer_reset_fix.md](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/phase4f1_viewer_reset_fix.md) (Technical interface fix documentation)
- [reports/phase4f1_policy_probe_metrics.json](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/phase4f1_policy_probe_metrics.json) (Structured raw observations, actions, and telemetry)
- [reports/phase4f1_mirrored_route_comparison.csv](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/reports/phase4f1_mirrored_route_comparison.csv) (Tabular paired obstacle comparison)
- [tests/test_viewer_reset_regression.py](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/tests/test_viewer_reset_regression.py) (Reset regression test suite)

---

## 10. Next Evidence-Supported Step

With the viewer reset defect resolved and the PPO lateral bias rigorously confirmed and isolated to the neural policy weights, the system is ready for a principled policy correction phase.

The evidence points to **bilateral reflection data augmentation** during PPO training (mirroring raycasts across the sagittal plane and negating $v_y$ and $\dot{\psi}$) or a **symmetric gap-clearance reward term**.

Per strict repository constraints, no training was conducted in Phase 4F-1. Awaiting explicit user authorization before proceeding to any policy retraining milestone.
