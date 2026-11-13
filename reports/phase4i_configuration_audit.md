# Phase 4I Configuration and Baseline Verification Audit

## Executive Summary

Phase 4I-A provides a rigorous, code-level verification of the canonical corridor environment, evaluation horizons, success and termination criteria, action contracts, safety supervisor dynamics, and policy checkpoint integrity.

This audit establishes the baseline facts required to investigate why both BC and PPO fail to achieve reliable goal arrival across multi-obstacle corridors, despite localized single-obstacle improvements and reduced supervisor intervention rates.

---

## 1. Verified System Configuration & Mathematical Parameters

| Parameter | Code Location | Verified Value | Physical Meaning / Units |
| :--- | :--- | :--- | :--- |
| **Physics Timestep ($dt$)** | `sim/corridor_env.py` | `0.05 s` | $20.0\text{ Hz}$ simulation and control loop rate |
| **Canonical Horizon (`max_steps`)** | `core/geometry.py:69` | `500 steps` | **$25.00\text{ s}$** maximum flight duration per episode |
| **Corridor Dimensions** | `core/geometry.py:56-60` | $10.0\text{ m} \times 2.0\text{ m} \times 2.0\text{ m}$ | Length ($X$), Width ($Y$), Height ($Z$) |
| **Start Position** | `core/geometry.py:93` | $(0.5, 0.0, 1.0)\text{ m}$ | UAV launch position on corridor centerline |
| **Goal Position** | `core/geometry.py:94` | $(9.5, 0.0, 1.0)\text{ m}$ | Center of terminal arrival target |
| **Goal Acceptance Radius** | `core/geometry.py:95` | `0.35 m` | Euclidean sphere at target: $\|p - p_{\text{goal}}\| \le 0.35\text{ m}$ |
| **Terminal Arrival Speed Gate** | Evaluation harnesses | $\le 0.35\text{ m/s}$ | Safe touchdown / hover braking requirement |
| **Drone Bounding Radius** | `core/geometry.py:96` | `0.15 m` | Collision sphere of the UAV airframe |
| **Pillar Nominal Positions** | `core/geometry.py:97` | $X \in \{3.0, 5.0, 7.0\}\text{ m}$ | Longitudinal positions of the 3 internal obstacles |
| **Pillar Radius** | `core/geometry.py:99` | `0.20 m` | Radius of vertical cylindrical obstacles |
| **Pillar Lateral Offset Range** | `core/geometry.py:98` | $\pm 0.35\text{ m}$ | Uniform random lateral sampling range: $Y \in [-0.35, 0.35]$ |

---

## 2. Action Limits, Coordinate Frames, and Transformations

The policy outputs actions in the **Body FLU frame** (Forward, Left, Up, Yaw CCW) according to `core/contracts.py`:
- **Forward Velocity ($v_x$):** $[-0.20, +0.60]\text{ m/s}$ (allows $-0.20\text{ m/s}$ for emergency retreat)
- **Lateral Velocity ($v_y$):** $[-0.40, +0.40]\text{ m/s}$ (+Left, -Right)
- **Vertical Velocity ($v_z$):** $[-0.30, +0.30]\text{ m/s}$ (+Up, -Down)
- **Yaw Angular Rate ($\dot{\psi}$):** $[-0.50, +0.50]\text{ rad/s}$ (+CCW, -CW)
- **Maximum Linear Acceleration:** $1.00\text{ m/s}^2$

---

## 3. Canonical Safety Supervisor Operational Envelope

Inspecting `controllers/safety_supervisor.py` and `SafetySupervisorConfig` reveals the exact thresholds governing autonomous intervention:

### 3.1 Forward Envelopes
- **Obstacle Caution Distance (`obstacle_caution_dist_m`):** `0.80 m`
  - When effective front clearance $d_{\text{front}} \in [0.30\text{ m}, 0.80\text{ m}]$, the supervisor linearly throttles forward velocity:
    $$\text{max\_caut\_vx} = \max\left(0.10, \frac{d_{\text{front}} - 0.30}{0.80 - 0.30} \cdot 0.60\right)\text{ m/s}$$
- **Critical Forward Brake (`critical_obstacle_dist_m`):** `0.30 m`
  - When $d_{\text{front}} \in [0.15\text{ m}, 0.30\text{ m}]$, forward velocity is clamped to **$v_x = 0.0\text{ m/s}$**.
- **Emergency Reverse Distance (`emergency_reverse_dist_m`):** `0.15 m`
  - When $d_{\text{front}} < 0.15\text{ m}$, the supervisor forces reverse thrust: **$v_x = -0.10\text{ m/s}$**.

### 3.2 Lateral & Boundary Envelopes
- **Boundary Buffer (`boundary_buffer_m`):** `0.25 m`
  - Distance from wall to initiate active proportional boundary repulsion: $v_y \leftarrow v_y \mp 1.5 \cdot (0.25 - d_{\text{wall}})$.
- **Lateral Caution Distance (`caution_lateral_dist_m`):** `0.55 m`
  - Throttles lateral velocity toward an obstacle or wall.
- **Lateral Critical Distance (`critical_lateral_dist_m`):** `0.32 m`
  - Strictly zeroes lateral velocity toward the constrained side ($v_y \le 0$ if left constrained, $v_y \ge 0$ if right constrained).
- **Stalled Recovery (`stalled_duration_s`):** `0.35 s`
  - If $v_x \le 0.05\text{ m/s}$ and $d_{\text{front}} < 0.38\text{ m}$ for $\ge 0.35\text{ s}$ ($7\text{ steps}$), triggers reverse retreat setpoint $v_x = -0.10\text{ m/s}$ (provided rear clearance $d_{\text{rear}} > 0.45\text{ m}$).

---

## 4. Policy Checkpoint Verification & Bitwise Integrity

All four policies were audited and confirmed compatible:

| Policy Identifier | Model Architecture | Checkpoint File Path | Verified SHA-256 Checksum | Compatibility Status |
| :--- | :--- | :--- | :--- | :--- |
| **`Arm_B_BC_Baseline`** | `NavigationPolicyMLP` (28,740 params) | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **Verified & Bitwise Immutable** |
| **`Arm_A_PPO_Baseline`** | `ActorCriticPPO` (actor + critic) | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **Verified & Bitwise Immutable** |
| **`BC_Phase4H_Candidate`** | `NavigationPolicyMLP` (28,740 params) | `experiments/phase4h/checkpoints/bc_phase4h_candidate.pt` | `1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b` | **Verified Candidate** |
| **`PPO_Phase4H_Candidate`** | `ActorCriticPPO` (actor + critic) | `experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt` | `751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd` | **Verified Candidate** |

---

## 5. Horizon Discrepancy Resolution

The audit confirms:
- **Canonical Simulation Horizon:** $500\text{ steps} \times 0.05\text{ s} = 25.00\text{ s}$.
- **Phase 4G Truncated Probe:** In Phase 4G, validation loops used `while not done and steps < 200:`. This truncated flight to $10.0\text{ s}$, under which even collision-free expert flight was marked as failed timeout.
- **Phase 4H Resolution:** All Phase 4H comparative benchmarks were confirmed to use the canonical $500\text{ steps}$ ($25.00\text{ s}$).
- Under $500\text{ steps}$, unconstrained flight at cruise speed ($0.60\text{ m/s}$) covers $9.0\text{ m}$ in $300\text{ steps}$ ($15.0\text{ s}$), leaving $200\text{ steps}$ ($10.0\text{ s}$) of time margin for obstacle bypasses.

---

## 6. Audit Conclusion
The environment geometry, contracts, safety supervisor parameters, and policy checkpoints are fully established from source code. The diagnostic suite in Phase 4I-B and time-budget instrumentation in Phase 4I-C can proceed on a verified foundation.
