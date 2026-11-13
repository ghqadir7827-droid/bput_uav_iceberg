# Phase 4F: Continuous Clearance-Aware Lateral Guidance Design

**Document ID:** `reports/PHASE4F_CLEARANCE_GUIDANCE_DESIGN.md`  
**Date:** 2026-10-31  
**Target Platform:** 65 cm Physical Quadcopter Model in Scaled Simulation  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Git Branch:** `feature/v4-radial-adaptive-guidance`  
**Active Checkpoint SHA-256:** `d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1` (`bc_policy_scaled_v2.pt`)  
**Controller Identifier:** `scaled_guided_bc_v5`  

---

## 1. Executive Summary and Motivation

Previous iterations of the clearance guidance layer introduced discrete velocity setpoints:
- **v2 (Harmonized):** Constant $v_{y,\text{override}} = 0.35\text{ m/s}$ across all corridors.
- **v3 (Directional Persistence):** Constant $v_{y,\text{override}} = 0.40\text{ m/s}$ with latching tie-breaks.
- **v4 (Corridor-Adaptive):** Bimodal switching between $0.40\text{ m/s}$ (wide corridors) and $0.30\text{ m/s}$ (narrow corridors).

While v4 reduced wall chattering in narrow corridors, paired evaluation revealed that discrete bimodal switching yielded an improvement of only 1 additional success out of 175 paired scenarios. In tight spaces (e.g. Level 4, corridor width $2.80\text{ m}$), commanding a fixed $0.30\text{ m/s}$ override still drove the 65 cm vehicle ($R_{\text{uav}} = 0.325\text{ m}$) into safety buffer zones, causing opposing supervisor interventions and lateral stalling.

**Guidance Candidate v5 (`scaled_guided_bc_v5`)** introduces continuous, clearance-aware lateral velocity scaling:
- Continuous scaling between $v_{\text{override\_min}} = 0.15\text{ m/s}$ and $v_{\text{override\_max}} = 0.40\text{ m/s}$.
- Proportional response to available maneuvering headroom on the evasion side.
- Directional persistence and hysteresis to suppress 20 Hz chattering.
- Tight integration with the safe-arrest state machine to suppress lateral scanning sweeps at impassable barriers.
- Absolute preservation of the frozen behavior-cloning policy interface (23 floats).

---

## 2. Mathematical Formulation

### 2.1 Available Maneuvering Space (Headroom)

When guidance selects an evasion direction (e.g., Left or Right), it evaluates the available physical clearance on the target side:
$$C_{\text{target}} = \begin{cases}
\min(d_{\text{left\_wall}}, d_{\text{left\_obs}}), & \text{if direction is LEFT} \\
\min(d_{\text{right\_wall}}, d_{\text{right\_obs}}), & \text{if direction is RIGHT}
\end{cases}$$

For the 65 cm quadcopter ($R_{\text{uav}} = 0.325\text{ m}$) operating with a required $0.150\text{ m}$ safety envelope, the critical lateral safety boundary is:
$$d_{\text{crit\_lat}} = R_{\text{uav}} + \text{margin} = 0.325\text{ m} + 0.150\text{ m} = 0.475\text{ m}$$

The net maneuvering headroom available on the target side is defined as:
$$\Delta_{\text{headroom}} = \max(0.0, C_{\text{target}} - d_{\text{crit\_lat}})$$

### 2.2 Continuous Headroom Ratio and Velocity Scaling

Available headroom is normalized against a reference comfortable maneuvering margin ($H_{\text{ref}} = 1.00\text{ m}$):
$$\rho = \text{clamp}\left(\frac{\Delta_{\text{headroom}}}{H_{\text{ref}}}, 0.0, 1.0\right) = \min\left(1.0, \frac{\max(0.0, C_{\text{target}} - 0.475)}{1.00}\right)$$

The continuous commanded lateral override speed $v_{y,\text{override}}$ is given by:
$$v_{y,\text{override}} = v_{\text{override\_min}} + \rho \cdot (v_{\text{override\_max}} - v_{\text{override\_min}}) = 0.15 + 0.25 \cdot \rho \quad [\text{m/s}]$$

### 2.3 Physical Operating Regimes

| Operating Regime | Target Clearance $C_{\text{target}}$ | Headroom $\Delta_{\text{headroom}}$ | Headroom Ratio $\rho$ | Commanded Override $v_{y,\text{override}}$ | Controller Behavior |
|---|---|---|---|---|---|
| **Wide Open Corridor** | $\ge 1.475\text{ m}$ | $\ge 1.00\text{ m}$ | $1.00$ | **$0.400\text{ m/s}$** | Maximum evasive speed; rapid symmetry-breaking without forward stall. |
| **Standard Corridor** | $1.000\text{ m}$ | $0.525\text{ m}$ | $0.525$ | **$0.281\text{ m/s}$** | Smooth evasion scaled to corridor dimensions. |
| **Narrow Boundary** | $0.675\text{ m}$ | $0.200\text{ m}$ | $0.200$ | **$0.200\text{ m/s}$** | Gentle evasive nudge; avoids penetrating wall buffer zones. |
| **Constrained Margin** | $\le 0.475\text{ m}$ | $0.000\text{ m}$ | $0.000$ | **$0.150\text{ m/s}$** | Minimal override; prevents driving airframe toward constrained boundaries. |
| **Safe Arrest / Infeasible** | Immaterial | $0.000\text{ m}$ | N/A | **$0.000\text{ m/s}$** | Guidance suppression; lateral exploration halted to allow stable hover. |

---

## 3. Directional Persistence and Hysteresis

To eliminate 20 Hz lateral tie-break chattering in symmetric corridors:
1. **Clearance Differential:** $\Delta_{\text{clear}} = c_{\text{left}} - c_{\text{right}}$.
2. **Hysteresis Offsets:** When previously turning Left or Right, a hysteresis band $h = 0.06\text{ m}$ and deadband $\delta = 0.08\text{ m}$ are applied to the switching threshold.
3. **Persistent Tie-Breaking:** If $\Delta_{\text{clear}}$ lies within $[-\delta, +\delta]$, the controller maintains its previously latched direction (`TIE_BREAK_LEFT` or `TIE_BREAK_RIGHT`), unless the wall on that side becomes tight ($< 0.675\text{ m}$).
4. **Soft Terminal Release:** Within the destination zone ($\|\mathbf{p}_{\text{goal}} - \mathbf{p}\| < 1.0\text{ m}$), guidance overrides release control back to nominal policy for soft touchdown.

---

## 4. Architectural Separation and Invariants

1. **Policy Independence:** The frozen BC policy (`bc_policy_scaled_v2.pt`, SHA-256 `d02d5712...`) is never modified and receives strictly 23 floats.
2. **Guidance Layer Responsibility:** Proposes smooth, continuous lateral setpoints. It does not weaken safety constraints.
3. **Supervisor Precedence:** The canonical `SafetySupervisor` remains downstream and authoritative. If a guidance-commanded $v_y$ approaches an obstacle or wall closer than $d_{\text{crit\_lat}}$, the supervisor clamps or repulses the action independently.
4. **Safe-Arrest Coupling:** When `SafeArrestStateMachine` verifies an impassable barrier, it sets `suppress_guidance=True`, which sets $v_y = 0.0\text{ m/s}$, terminating oscillatory sweeps.

