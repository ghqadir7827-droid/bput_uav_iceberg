# Phase 6B Architecture & Algorithm Design Document

**Phase:** Phase 6B — Targeted Pillar 3 Recovery Experiment  
**Module:** `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/recovery_guidance.py`  
**Candidate Controller:** `PostEvasionRecoveryController`  

---

## 1. Architectural Principles & Isolation

In strict accordance with Phase 6B requirements:
- The candidate recovery logic is implemented in an **isolated module** (`recovery_guidance.py` and `pillar3_recovery_controller.py`) inside the experiment root.
- The production Phase 6A controller (`controllers/guidance_controller.py`) and shared guidance layer (`core/guidance.py`) remain completely untouched as the operational baseline.
- Zero privileged world-coordinate state is utilized: all state determinations rely exclusively on observable rangefinder beams and deployable kinematics.
- The candidate adheres strictly to a **two-tier priority hierarchy**:
  1. **Tier 1 (Strict Priority): Obstacle Clearance Guidance.** Whenever an obstacle is detected within the frontal activation envelope ($c_{\text{front}} < 0.85\text{ m}$), clearance guidance has absolute control.
  2. **Tier 2: Post-Evasion Centerline Recovery.** Active only when an obstacle has been cleared and the corridor ahead is open ($c_{\text{front}} \ge 0.90\text{ m}$).

---

## 2. Observable State Estimation

In a straight corridor of nominal width $W = 2.0\text{ m}$ ($y \in [-1.0, 1.0]$), the lateral 90-degree rangefinder beams (Ray 5 at $+90^\circ$ azimuth and Ray 6 at $-90^\circ$ azimuth) provide an observable estimate of the UAV's lateral position:

$$\text{wall\_l} = r_5 \cdot d_{\max}, \quad \text{wall\_r} = r_6 \cdot d_{\max}$$

When both beams reflect off the smooth corridor walls:

$$W_{\text{obs}} = \text{wall\_l} + \text{wall\_r} \approx 2.0\text{ m}$$

The signed observable lateral displacement from the corridor centerline ($y = 0$) is:

$$y_{\text{obs}} = \frac{\text{wall\_r} - \text{wall\_l}}{2}$$

- If $y_{\text{obs}} > 0$: The UAV is displaced toward the **LEFT** wall ($+Y$).
- If $y_{\text{obs}} < 0$: The UAV is displaced toward the **RIGHT** wall ($-Y$).

### Boundary Validation & Fail-Closed Guard
To prevent corrupt readings from adjacent obstacles flanking the lateral beams or specular dropouts:
$$W_{\min} (1.60\text{ m}) \le W_{\text{obs}} \le W_{\max} (2.40\text{ m})$$
If $W_{\text{obs}}$ falls outside this band, or if any ray contains non-finite values (NaN / Inf), the recovery state immediately **fails closed** and yields to the nominal policy.

---

## 3. Post-Evasion Recovery State Machine

```
      +-------------------------------------------------------------+
      |                Episode Reset (reset())                      |
      |   obstacle_encountered = False, recovery_active = False     |
      +-------------------------------------------------------------+
                                     |
                                     v
               +------------------------------------------+
               |         c_front < 0.85 m ?               |
               +------------------------------------------+
                     /                             \
             YES    /                               \   NO
                   v                                 v
   +---------------------------------+   +------------------------------------+
   | Tier 1: Clearance Guidance      |   | In Terminal Zone (goal_norm<0.5)   |
   | - obstacle_encountered = True   |   | OR obstacle_encountered == False?  |
   | - recovery_active = False       |   +------------------------------------+
   | - Evasion (LEFT / RIGHT)        |          /                      \
   +---------------------------------+  YES    /                        \  NO
                                              v                          v
                              +-------------------------+  +-------------------------+
                              | Yield to Nominal Policy |  | W_obs in [1.6, 2.4] &   |
                              | recovery_active = False |  | c_front >= 0.90 m ?     |
                              +-------------------------+  +-------------------------+
                                                                  /            \
                                                          YES    /              \  NO
                                                                v                v
                                               +-------------------+   +-------------------------+
                                               | Check Hysteresis  |   | Fail-Closed Guard       |
                                               +-------------------+   | recovery_active = False |
                                                                       +-------------------------+
```

### Activation and Release Hysteresis
To prevent high-frequency toggling around spatial boundaries:
- **Activation Threshold:** $|y_{\text{obs}}| \ge 0.22\text{ m}$ (UAV is significantly displaced).
- **Release Threshold:** $|y_{\text{obs}}| \le 0.08\text{ m}$ (Centerline is restored).
- **Hysteresis Band:** $\Delta y = 0.14\text{ m}$ of deadband prevents state chattering.

### Clearance Validation in Recovery Direction
Before commanding lateral velocity toward the centerline:
- If $y_{\text{obs}} > 0$ (recovering rightward, $v_y < 0$): Requires $c_{\text{right}} \ge 0.55\text{ m}$.
- If $y_{\text{obs}} < 0$ (recovering leftward, $v_y > 0$): Requires $c_{\text{left}} \ge 0.55\text{ m}$.
If the intended recovery path is obstructed by an unseen obstacle flank, recovery aborts immediately.

### Bounded Proportional Velocity Control
Recovery commands are strictly bounded to maintain smooth, non-disruptive flight:

$$v_{y,\text{cmd}} = -\text{sign}(y_{\text{obs}}) \cdot \min\left(v_{y,\max}^{\text{rec}}, k_{\text{gain}} \cdot |y_{\text{obs}}|\right)$$

where:
- $v_{y,\max}^{\text{rec}} = 0.15\text{ m/s}$ (bounded gradual command).
- $k_{\text{gain}} = 0.50\text{ s}^{-1}$ (proportional restoration gain).
- $v_{x,\text{cmd}} = \max\left(\text{raw\_action.vx}, 0.35\text{ m/s}\right)$ (prevents time-budget starvation in clear corridors).

---

## 4. Interaction with the Deterministic Safety Supervisor

The recovery candidate outputs a standard `VelocityAction(vx, vy, vz, yaw_rate)`. 
This proposed action is passed directly into the canonical `SafetySupervisor.filter_action(...)`.
- The recovery controller has **zero authority to bypass or relax** supervisor constraints.
- If a recovery command approaches a corridor boundary ($C_{\text{boundary}} < 0.25\text{ m}$), the supervisor repels or clamps $v_y$.
- If an obstacle appears within $0.50\text{ m}$, the supervisor throttles forward speed regardless of recovery commands.
- The supervisor's parameter configuration ($d_{\text{caut}} = 0.50\text{ m}$) remains identical across all benchmark comparisons.

