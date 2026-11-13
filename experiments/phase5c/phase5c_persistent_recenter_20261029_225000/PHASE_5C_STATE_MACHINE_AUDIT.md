# PHASE 5C — STATE MACHINE AND ARTIFACT AUDIT REPORT
## Formal State Machine Specification & Comparative Design Audit

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Isolation Directory:** `experiments/phase5c/phase5c_persistent_recenter_20261029_225000/`

---

## 1. Resolution of Reporting Inconsistencies from Phase 5B

An audit of the Phase 5B data files clarifies two apparent reporting discrepancies:

### 1. Progress Metric Discrepancy (7.936 m vs 8.155 m)
- **7.936 m** is the exact mean progress achieved by Controller B (Stateless Guidance) across the complete **50-episode canonical development set (`seeds 2000–2049`)** in `paired_development_results.csv`.
- **8.155 m** is the mean progress achieved by Controller B across the **10-seed subset (`seeds 2000–2009`)** evaluated during the threshold sensitivity sweep in `threshold_sensitivity.csv`.
- Evaluating seeds `2000–2009` from `paired_development_results.csv` yields an identical mean of **8.155 m**. The difference reflects sample size ($N=50$ vs $N=10$) rather than code, parameter, or environment changes.

### 2. Distinction Between Progress Levels and True Goal Success
In accordance with project safety standards, forward progress is strictly separated from mission completion:
- **Pillar 3 Clearance:** Drone reaches $X > 8.0\text{ m}$.
- **Goal Region Arrival:** Drone enters within Euclidean radius $R_{\text{goal}} = 0.50\text{ m}$ of the target $[9.0, 0.0, 1.0]\text{ m}$.
- **Terminal Goal Success:** Requires satisfying two simultaneous conditions:
  1. Position inside the goal sphere ($d_{\text{goal}} \le 0.50\text{ m}$).
  2. Terminal velocity bounded at landing speed: $\|\mathbf{v}_{\text{terminal}}\| \le 0.35\text{ m/s}$.
- **Timeout after Substantial Progress:** Episodes reaching $X > 8.1\text{ m}$ (clearing all obstacles) but expiring at step 500 without satisfying the speed bound are classified as `TIMEOUT_SLOW_TRANSIT` or `GOAL_ALIGNMENT_TIMEOUT`, never as successes.

---

## 2. Specification of the Three Evaluated Controllers

```
+----------------------------------------------------------------------------------------------------+
|                                    CONTROLLER SPECIFICATION MATRIX                                 |
+--------------------------+-----------------------+-------------------------+-----------------------+
| Attribute                | Controller A          | Controller B            | Controller C          |
|                          | (Frozen Baseline)     | (Stateless Guidance)    | (Persistent Guidance) |
+--------------------------+-----------------------+-------------------------+-----------------------+
| Checkpoint               | bc_policy_best.pt     | bc_policy_best.pt       | bc_policy_best.pt     |
| Guidance Memory          | None (Feedforward)    | Stateless (Instantaneous| 4-State Machine       |
| Activation Threshold     | N/A                   | C_front < 0.85 m        | C_front < 0.85 m      |
| Lateral Bias Magnitude   | None                  | +/- 0.25 m/s            | +/- 0.25 m/s          |
| Forward Speed Override   | None                  | None                    | min 0.30 m/s in avoid |
| Direction Persistence    | None                  | 0 steps (Drops if front | Latched until exit    |
|                          |                       | clearance opens)        | condition or timeout  |
| Re-Centering State       | None                  | None (UAV remains near  | SETTLING state        |
|                          |                       | wall after bypass)      | (vy = -0.20*sign(y))  |
+--------------------------+-----------------------+-------------------------+-----------------------+
```

---

## 3. Formal Definition of the Controller C State Machine

Controller C implements an explicit finite state machine with four discrete states:
$$\mathcal{S} = \{\text{NOMINAL}, \text{AVOID\_LEFT}, \text{AVOID\_RIGHT}, \text{SETTLING}\}$$

```
             ┌────────────────────────┐
             │        NOMINAL         │◄───────────────────────┐
             └────────────────────────┘                        │
                          │                                    │
          C_front < 0.85m │ C_front < 0.85m                    │ |y| < 0.12m
          and diff > 0.08 │ and diff < -0.08                   │ or timeout (1.5s)
          (or tie-break)  │                                    │
                 ▼        ▼                                    │
      ┌──────────────┐ ┌───────────────┐                       │
      │  AVOID_LEFT  │ │  AVOID_RIGHT  │                       │
      └──────────────┘ └───────────────┘                       │
                 │             │                               │
                 │ C_front > 1.25m and dx > 0.65m              │
                 │ or timeout (2.5s)                           │
                 ▼             ▼                               │
             ┌────────────────────────┐                        │
             │        SETTLING        │────────────────────────┘
             │  (Re-centering vy)     │
             └────────────────────────┘
                          │
                          │ C_front < 0.85m (New Obstacle)
                          └───────────────────► [AVOID_LEFT / AVOID_RIGHT]
```

### Detailed State Specifications

#### State 1: `NOMINAL` (Goal-Seeking Cruise)
- **Entry Condition:** Initial state at episode start, or transition from `SETTLING` upon reaching $|y| < 0.12\text{ m}$ or after 30 settling steps.
- **Command Output:** Exact raw action from `NavigationPolicyMLP`:
  $$v_{x, \text{cmd}} = v_{x, \text{BC}}, \quad v_{y, \text{cmd}} = v_{y, \text{BC}}, \quad v_{z, \text{cmd}} = v_{z, \text{BC}}, \quad \dot{\psi}_{\text{cmd}} = \dot{\psi}_{\text{BC}}$$
- **Exit Condition:** Front clearance drops below activation threshold ($C_{\text{front}} < 0.85\text{ m}$).
  - If $\Delta = C_{\text{left}} - C_{\text{right}} > +0.08\text{ m} \implies \text{AVOID\_LEFT}$.
  - If $\Delta < -0.08\text{ m} \implies \text{AVOID\_RIGHT}$.
  - If $|\Delta| \le 0.08\text{ m} \implies \text{AVOID\_LEFT}$ (Deterministic Tie-Breaker).
- **Recorded Variables:** Latch longitudinal coordinate $x_{\text{trigger}} = x_{\text{drone}}$, reset `state_step_count = 0`.

#### State 2: `AVOID_LEFT` (Latched Left Bypass)
- **Entry Condition:** Transition from `NOMINAL` or `SETTLING` with left clearance preference or tie-breaker.
- **Command Output:**
  $$v_{y, \text{cmd}} = \max(v_{y, \text{BC}}, +0.25\text{ m/s})$$
  $$v_{x, \text{cmd}} = \max(v_{x, \text{BC}}, +0.30\text{ m/s})$$
- **Exit Condition:**
  $$\left(C_{\text{front}} > 1.25\text{ m} \quad \text{AND} \quad (x_{\text{drone}} - x_{\text{trigger}}) \ge 0.65\text{ m}\right) \quad \text{OR} \quad \text{state\_step\_count} \ge 50 \text{ (2.5 s)}$$
  $\implies \text{Transition to } \text{SETTLING}$.
- **Safety Invariant:** Latched setpoints are bounded by `ActionLimits` ($|v_y| \le 0.40\text{ m/s}$) and filtered by `SafetySupervisor` (left wall repulsion active at $d_{\text{wall}} < 0.25\text{ m}$).

#### State 3: `AVOID_RIGHT` (Latched Right Bypass)
- **Entry Condition:** Transition from `NOMINAL` or `SETTLING` with right clearance preference.
- **Command Output:**
  $$v_{y, \text{cmd}} = \min(v_{y, \text{BC}}, -0.25\text{ m/s})$$
  $$v_{x, \text{cmd}} = \max(v_{x, \text{BC}}, +0.30\text{ m/s})$$
- **Exit Condition:**
  $$\left(C_{\text{front}} > 1.25\text{ m} \quad \text{AND} \quad (x_{\text{drone}} - x_{\text{trigger}}) \ge 0.65\text{ m}\right) \quad \text{OR} \quad \text{state\_step\_count} \ge 50 \text{ (2.5 s)}$$
  $\implies \text{Transition to } \text{SETTLING}$.

#### State 4: `SETTLING` (Active Corridor Re-centering)
- **Entry Condition:** Transition from `AVOID_LEFT` or `AVOID_RIGHT` upon clearing the active obstacle.
- **Command Output:** Enforces gentle, proportional re-centering towards the centerline:
  $$v_{y, \text{cmd}} = -0.20 \cdot \text{sign}(y_{\text{drone}}) \quad \text{if } |y_{\text{drone}}| > 0.08\text{ m}, \quad \text{else } 0.00\text{ m/s}$$
  $$v_{x, \text{cmd}} = v_{x, \text{BC}}$$
- **Exit Conditions:**
  1. **New Obstacle Encounter:** If $C_{\text{front}} < 0.85\text{ m} \implies$ immediately transition to `AVOID_LEFT` or `AVOID_RIGHT` based on new local clearance $\Delta$.
  2. **Re-centering Complete:** If $|y_{\text{drone}}| < 0.12\text{ m} \implies$ transition to `NOMINAL`.
  3. **Settling Timeout:** If `state_step_count` $\ge 30$ ($1.5\text{ s}$) $\implies$ transition to `NOMINAL`.

---

## 4. Key Hypotheses for the Paired Benchmark

1. **The Corner-Pinch Hypothesis:** In Controller B (Stateless), bypassing Pillar 1 and Pillar 2 pushes the drone into the lateral margin ($Y \approx \pm 0.60\text{ m}$). In Controller C, the `SETTLING` state restores the drone towards $Y \approx 0.0\text{ m}$ between pillars, eliminating the corner-pinch at Pillar 3 and increasing the Pillar 3 clearance rate from $78\%$ toward $> 85\%$.
2. **Transition Interference Risk:** In tightly spaced obstacle pairs, initiating re-centering too aggressively could steer the drone toward the path of a downstream obstacle. The immediate preemption condition in `SETTLING` ($C_{\text{front}} < 0.85\text{ m}$) is designed to mitigate this risk.

