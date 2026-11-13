# Phase 4F: Safe-Arrest & Infeasibility Classification State Machine Design

**Document ID:** `reports/PHASE4F_SAFE_ARREST_DESIGN.md`  
**Date:** 2026-10-31  
**Target Platform:** 65 cm Physical Quadcopter Model in Scaled Simulation  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Git Branch:** `feature/v4-radial-adaptive-guidance`  
**Module Implementation:** `controllers/safe_arrest.py`  

---

## 1. Problem Statement: Lateral Scanning Defect at Infeasible Barriers

During the evaluation of 25 narrow-barrier scenarios (Level 4, gate opening $w_{\text{gap}} \approx 0.60\text{ m}$ vs 65 cm UAV diameter):
- **Observed Behavior:** In 23 of 25 scenarios, the `SafetySupervisor` safely halted forward speed at $x \approx 6.5\text{ m}$, preventing any collision with the impassable barrier at $x = 7.0\text{ m}$.
- **Defect:** However, the guidance layer continually attempted exploratory lateral evasions, oscillating back and forth ($v_y \approx \pm 0.30\text{ m/s}$ to $\pm 0.40\text{ m/s}$) for $> 500$ steps until step 1000 timeout.
- **Classification Failure:** Because the vehicle maintained lateral velocity ($0.20\text{--}0.30\text{ m/s}$), the legacy evaluation heuristic (`speed_2d < 0.05 m/s`) never triggered. These safe halts were misclassified as `TIMEOUT` rather than `SAFE_REJECTION`.

This was a **liveness and classification defect**, not a safety breach.

---

## 2. Finite State Machine Architecture

The `SafeArrestStateMachine` manages stopping behavior and traversability assessment through 8 explicit states:

```
                  +-----------------------------------------------------+
                  |                     APPROACHING                     |
                  +-----------------------------------------------------+
                                       |              ^
                        c_front < 0.85m |              | c_front >= 0.85m
                                       v              |
                  +-----------------------------------------------------+
                  |                       BLOCKED                       |
                  +-----------------------------------------------------+
                                       |
                                       v
                  +-----------------------------------------------------+
                  |               ASSESSING_TRAVERSABILITY              |
                  +-----------------------------------------------------+
                                       |
           Both sides < 0.65m          |
           for 10 consecutive steps    | (Lateral space > 0.65m exists)
                                       v
                  +-----------------------------------------------------+
                  |                 VERIFIED_INFEASIBLE                 |
                  +-----------------------------------------------------+
                                       |
                                       v (Lock anchor, suppress guidance vy=0)
                  +-----------------------------------------------------+
                  |                     SAFE_ARREST                     |
                  +-----------------------------------------------------+
                      |                                       |
                      | 40 steps (2.0s)                       | Clearance < 0.05m
                      | ||v_xy|| <= 0.05 m/s                  | or Drift > 0.20m
                      | Drift <= 0.10 m                       v
                      v                         +---------------------------+
        +---------------------------+           |       UNSAFE_ARREST       |
        |      SAFE_REJECTION       |           +---------------------------+
        |   (Terminal Evaluation)   |
        +---------------------------+
```

### State Descriptions

1. **`APPROACHING`:** Nominal forward navigation. Free flight under policy and guidance.
2. **`BLOCKED`:** Forward clearance dropped below $0.85\text{ m}$ (or analytical obstacle clearance $< 0.85\text{ m}$). Forward velocity is throttled/braked by supervisor.
3. **`ASSESSING_TRAVERSABILITY`:** Rangefinder and boundary geometry evaluate whether feasible lateral openings exist ($w_{\text{passage}} \ge 0.65\text{ m}$).
4. **`VERIFIED_INFEASIBLE`:** Forward passage and both lateral flanks are confirmed impassable ($< 0.65\text{ m}$) for 10 consecutive steps ($0.5\text{ s}$). The state machine records the current position as the **arrest anchor** and commands immediate lateral guidance suppression (`suppress_lateral_guidance = True`, $v_y = 0.0\text{ m/s}$).
5. **`SAFE_ARREST`:** Drone decelerates to hover at the anchor position. Dwell timer counts while verifying stability invariants.
6. **`SAFE_REJECTION`:** Terminal outcome achieved after 40 consecutive steps ($2.0\text{ s}$) of verified stable hover satisfying all kinematic and clearance invariants.
7. **`UNRESOLVED`:** Forward passage is blocked, but evidence is noisy or insufficient to confirm total infeasibility. Drone halts safely while logging diagnostic state.
8. **`UNSAFE_ARREST`:** Halting behavior violated stability (drift $> 0.20\text{ m}$) or positive surface clearance ($d_{\text{surf}} \le 0.05\text{ m}$).

---

## 3. Strict Quantitative Thresholds and Invariants

All thresholds are grounded in the physical dynamics and geometry of the 65 cm quadcopter:

| Parameter | Symbol | Authoritative Value | Physical Rationale |
|---|---|---|---|
| **Max Residual Horizontal Speed** | $v_{xy,\text{max}}$ | **$0.05\text{ m/s}$** | Authoritative definition of stationary quadcopter hover in PyBullet. |
| **Max Residual Vertical Speed** | $v_{z,\text{max}}$ | **$0.05\text{ m/s}$** | Enforces zero vertical altitude drift during hover. |
| **Max Position Drift** | $\Delta p_{\text{max}}$ | **$0.10\text{ m}$** | Bounded Euclidean drift from arrest anchor position $\|\mathbf{p} - \mathbf{p}_{\text{anchor}}\|$. |
| **Min Physical Clearance** | $d_{\text{min\_clear}}$ | **$0.05\text{ m}$** | Ensures positive airframe surface clearance to walls and obstacles at all times. |
| **Required Dwell Window** | $T_{\text{dwell}}$ | **$2.0\text{ s}$ (40 steps)** | Sufficient duration at 20 Hz to eliminate transient settling effects. |
| **Min Traversable Width** | $w_{\text{min\_pass}}$ | **$0.65\text{ m}$** | Exact physical tip-to-tip diameter of the 65 cm quadcopter airframe. |
| **Confirmation Horizon** | $N_{\text{confirm}}$ | **$10\text{ steps}$ ($0.5\text{ s}$)** | Filter window eliminating transient single-frame sensor noise or dropouts. |

---

## 4. Elimination of Lateral Oscillation

By asserting `suppress_lateral_guidance = True` upon entering `VERIFIED_INFEASIBLE`:
1. The guidance arbitrator zeroes its lateral command ($v_y = 0.0\text{ m/s}$).
2. The vehicle is not driven into alternating wall buffers.
3. The aerodynamic drag and flight controller bring horizontal body speed below $0.05\text{ m/s}$ within $\sim 15$ steps ($0.75\text{ s}$).
4. After 40 steps of stable hover, the episode terminates cleanly as `SAFE_REJECTION` in $\sim 70\text{ steps}$ ($3.5\text{ s}$ total), eliminating 930 steps of useless simulation time per scenario.

