# Phase 4J-A1 Source and Configuration Audit

## Executive Summary

Phase 4J-A1 provides a code-level audit of the canonical safety supervisor implementation (`controllers/safety_supervisor.py`), the rangefinder sensor physics (`sim/corridor_env.py`), the spatial geometry (`core/geometry.py`), and the action contract (`core/contracts.py`).

The audit verifies the mathematical definitions of the obstacle caution zone, the velocity clamp equation, the exact measurement reference frame, and the conditions governing activation and release.

---

## 1. Safety Supervisor Parameters & Measurement Reference

### 1.1 Distance Measurement Reference
- **Source Code Verification (`sim/corridor_env.py:257-284`):**
  PyBullet distance rays are cast outward from the drone's collision sphere offset ($r_{\text{drone}} + 0.03 = 0.18\text{ m}$).
  The calculated range is:
  $$\text{dist} = \text{offset} + \text{hit\_fraction} \cdot (\text{max\_range} - \text{offset})$$
  This establishes that **`eff_front` is measured from the DRONE'S CENTER OF MASS to the OBSTACLE'S OUTER COLLISION SURFACE**.

- **Physical Air-Gap Clearance vs. `eff_front`:**
  Since the drone airframe has a bounding collision radius $r_{\text{drone}} = 0.15\text{ m}$, the physical air gap between the drone hull and obstacle surface is:
  $$\text{Clearance}_{\text{air-gap}} = \text{eff\_front} - r_{\text{drone}} = \text{eff\_front} - 0.15\text{ m}$$

- **Center-to-Center Distance:**
  With pillar radius $r_{\text{pillar}} = 0.20\text{ m}$:
  $$\Delta X_{\text{centers}} = \text{eff\_front} + r_{\text{pillar}} = \text{eff\_front} + 0.20\text{ m}$$

| Supervisor Threshold | Parameter Name | Measured Value (`eff_front`) | Air-Gap Clearance | Center-to-Center Distance |
| :--- | :--- | :--- | :--- | :--- |
| **Caution Zone Boundary** | `obstacle_caution_dist_m` | **`0.80 m`** | **`0.65 m`** | **`1.00 m`** |
| **Critical Forward Brake** | `critical_obstacle_dist_m` | **`0.30 m`** | **`0.15 m`** | **`0.50 m`** |
| **Emergency Reverse** | `emergency_reverse_dist_m` | **`0.15 m`** | **`0.00 m`** (Contact envelope)| **`0.35 m`** |
| **Proposed Caution Zone** | (Phase 4J Experimental) | **`0.55 m`** | **`0.40 m`** | **`0.75 m`** |

---

## 2. Forward Velocity Clamp Equation & Throttling Dynamics

### 2.1 Clamp Formulation (`controllers/safety_supervisor.py:170-176`)
When $d_{\text{crit}} \le \text{eff\_front} < d_{\text{caut}}$, the forward setpoint $v_x$ is clamped to:
$$v_{x,\text{max}}^{\text{caut}} = \max\left(0.10, \frac{\text{eff\_front} - d_{\text{crit}}}{d_{\text{caut}} - d_{\text{crit}}} \cdot v_{x,\text{max}}\right)$$
Where:
- $v_{x,\text{max}} = 0.60\text{ m/s}$ (ActionLimits max forward speed)
- $d_{\text{crit}} = 0.30\text{ m}$
- $d_{\text{caut}} = 0.80\text{ m}$ (canonical) or $0.55\text{ m}$ (proposed)

### 2.2 Numerical Comparison of Canonical vs. Proposed Throttling Profiles
| Effective Front Range (`eff_front`) | Air-Gap Clearance | Canonical Clamp ($d_{\text{caut}}=0.80\text{ m}$) | Proposed Clamp ($d_{\text{caut}}=0.55\text{ m}$) | Velocity Difference |
| :--- | :--- | :--- | :--- | :--- |
| **$\ge 0.80\text{ m}$** | $\ge 0.65\text{ m}$ | $0.60\text{ m/s}$ (Unconstrained) | $0.60\text{ m/s}$ (Unconstrained) | $0.00\text{ m/s}$ |
| **$0.70\text{ m}$** | $0.55\text{ m}$ | **$0.48\text{ m/s}$** | **$0.60\text{ m/s}$ (Unconstrained)**| **$+0.12\text{ m/s}$** |
| **$0.60\text{ m/s}$** | $0.45\text{ m}$ | **$0.36\text{ m/s}$** | **$0.60\text{ m/s}$ (Unconstrained)**| **$+0.24\text{ m/s}$** |
| **$0.55\text{ m}$** | $0.40\text{ m}$ | **$0.30\text{ m/s}$** | **$0.60\text{ m/s}$ (Caution onset)**| **$+0.30\text{ m/s}$** |
| **$0.50\text{ m}$** | $0.35\text{ m}$ | **$0.24\text{ m/s}$** | **$0.48\text{ m/s}$** | **$+0.24\text{ m/s}$** |
| **$0.45\text{ m}$** | $0.30\text{ m}$ | **$0.18\text{ m/s}$** | **$0.36\text{ m/s}$** | **$+0.18\text{ m/s}$** |
| **$0.40\text{ m}$** | $0.25\text{ m}$ | **$0.12\text{ m/s}$** | **$0.24\text{ m/s}$** | **$+0.12\text{ m/s}$** |
| **$0.35\text{ m}$** | $0.20\text{ m}$ | **$0.10\text{ m/s}$** | **$0.12\text{ m/s}$** | $+0.02\text{ m/s}$ |
| **$\le 0.30\text{ m}$** | $\le 0.15\text{ m}$ | **$0.00\text{ m/s}$ (Critical Brake)** | **$0.00\text{ m/s}$ (Critical Brake)**| **$0.00\text{ m/s}$ (Identical)** |
| **$< 0.15\text{ m}$** | $< 0.00\text{ m}$ | **$-0.10\text{ m/s}$ (Reverse)** | **$-0.10\text{ m/s}$ (Reverse)** | **$0.00\text{ m/s}$ (Identical)** |

---

## 3. Activation, Release, and Recovery Logic

1. **Throttling Activation:** Instantly triggered when $\text{eff\_front} < d_{\text{caut}}$. There is **no activation delay or hysteresis** in the canonical supervisor.
2. **Throttling Release:** Instantly released when $\text{eff\_front} \ge d_{\text{caut}}$.
3. **Critical Braking:** When $\text{eff\_front} < 0.30\text{ m}$, forward speed is strictly clamped to $0.0\text{ m/s}$. Lateral motion toward the obstacle is zeroed if within $0.32\text{ m}$.
4. **Stalled Recovery State Machine:**
   If $v_x \le 0.05\text{ m/s}$ and $\text{eff\_front} < 0.38\text{ m}$ persist for $\ge 0.35\text{ s}$ ($7\text{ steps}$), the supervisor initiates an autonomous backward retreat ($v_x = -0.10\text{ m/s}$) provided rear rangefinder clearance $d_{\text{rear}} > 0.45\text{ m}$.

---

## 4. Corridor Geometry and Spatial Overlap Analysis

- **Corridor Dimensions:** Length = $10.0\text{ m}$, Width = $2.0\text{ m}$, Height = $2.0\text{ m}$.
- **Pillar Longitudinal Positions:** $X_1 = 3.0\text{ m}$, $X_2 = 5.0\text{ m}$, $X_3 = 7.0\text{ m}$.
- **Pillar Spacing:** $\Delta X = 2.00\text{ m}$ between adjacent pillars.
- **Pillar Radii:** $r_{\text{pillar}} = 0.20\text{ m}$.
- **Spatial Caution Span per Pillar (under $d_{\text{caut}} = 0.80\text{ m}$):**
  - Upstream entry: $X_{\text{entry}} = X_{\text{pillar}} - r_{\text{pillar}} - d_{\text{caut}} = X_{\text{pillar}} - 1.00\text{ m}$.
  - Downstream clearance: The drone clears the obstacle when $X_{\text{drone}} > X_{\text{pillar}} + r_{\text{pillar}} + r_{\text{drone}} \approx X_{\text{pillar}} + 0.35\text{ m}$.
  - For Pillar 1 ($X_1 = 3.0\text{ m}$): Caution entry at **$X = 2.00\text{ m}$**, clears at **$X = 3.35\text{ m}$**.
  - For Pillar 2 ($X_2 = 5.0\text{ m}$): Caution entry at **$X = 4.00\text{ m}$**, clears at **$X = 5.35\text{ m}$**.
  - **Inter-Pillar Free Gap:** Between clearing Pillar 1 ($X = 3.35\text{ m}$) and entering Pillar 2 caution ($X = 4.00\text{ m}$), there is **only $0.65\text{ m}$ of unconstrained space**!
  - Under proposed $d_{\text{caut}} = 0.55\text{ m}$:
    - Pillar 1 caution entry shifts from $X=2.00\text{ m}$ to **$X=2.25\text{ m}$**.
    - Pillar 2 caution entry shifts from $X=4.00\text{ m}$ to **$X=4.25\text{ m}$**.
    - Inter-pillar unconstrained gap expands from $0.65\text{ m}$ to **$0.90\text{ m}$** (+38% increase in free acceleration runway).

---

## 5. Audit Conclusions
The source audit confirms:
1. `eff_front` measures distance from the drone center to the obstacle surface.
2. The emergency braking ($0.30\text{ m}$) and retreat ($0.15\text{ m}$) logic are completely decoupled from $d_{\text{caut}}$ and remain intact.
3. The canonical $0.80\text{ m}$ caution zone severely constricts the inter-pillar transit corridor to just $0.65\text{ m}$ of unthrottled runway, providing the theoretical basis for Phase 4J-A2's time-budget recalculation.
