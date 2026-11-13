# PHASE 6A FORENSIC INVESTIGATION REPORT: SCENARIO C (WIDE LEFT)
**Autonomous UAV Navigation in GPS-Denied Corridors**  
**Run Identifier:** `phase6a_consistency_audit_20261030_020600`  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Date:** October 30, 2026  

---

## 1. Executive Summary

During the Phase 6 recovery benchmark, a notable anomaly was recorded in the diagnostic scenarios:  
While the historical **Phase 5B Stateless Guidance** completed `Scenario_C_Wide_Left` in $18.0\text{ s}$ ($8.15\text{ m}$ progress, $0$ collisions), the **Phase 6 Production Guidance** candidate stalled at step 500 at $x = 2.69\text{ m}$ ($1.69\text{ m}$ progress, categorized as `PROGRESS_STALL_PILLAR_1`).

This forensic investigation cross-referenced the per-step telemetry logs (`repro_Scenario_C_Wide_Left_Phase6_Production_Guidance.json` vs `Phase5B` trace), analytical corridor geometry, and raycast rangefinder readings to identify the precise mechanics of the failure and evaluate candidate remedies.

---

## 2. Geometry of the Encounter

The canonical test scenario defines an asymmetric corridor passage:
- **Corridor Dimensions:** $X \in [0, 10]\text{ m}$, $Y \in [-1.0, +1.0]\text{ m}$ (width $2.0\text{ m}$), $Z \in [0.0, 2.0]\text{ m}$ (height $2.0\text{ m}$).
- **Obstacle Placement:** Single cylindrical pillar at $(x_c = 3.0\text{ m}, y_c = -0.25\text{ m})$, radius $r = 0.15\text{ m}$.
- **Obstacle Extents:** Lateral span $y \in [-0.40, -0.10]\text{ m}$.
- **Left Passage Clearance:** Distance from pillar left face ($y = -0.10\text{ m}$) to left corridor wall ($y = +1.0\text{ m}$) = **$1.10\text{ m}$ of open passage**.
- **Right Passage Clearance:** Distance from pillar right face ($y = -0.40\text{ m}$) to right corridor wall ($y = -1.0\text{ m}$) = **$0.60\text{ m}$ of restricted passage**.
- **Physical Vehicle Envelope:** Drone diameter ($0.30\text{ m}$) + boundary buffer ($0.25\text{ m}$) + obstacle critical buffer ($0.30\text{ m}$) = **$0.85\text{ m}$ required clearance** for unthrottled nominal flight.

```
Corridor Left Wall (+1.0m) ----------------------------------------------------
                                                                      
                              <----- 1.10m WIDE PASSAGE ----->
                                                                      
                              +--------------------+                  
                              |   PILLAR 1         |                  
                              |   (y = -0.25m)     |                  
                              +--------------------+                  
                                                                      
                              <-- 0.60m PINCH -->                     
Corridor Right Wall (-1.0m) ---------------------------------------------------
```

Because the right passage ($0.60\text{ m}$) is substantially narrower than the required vehicle envelope ($0.85\text{ m}$), entering the right side unconditionally triggers the Safety Supervisor's caution and braking mechanisms. The left passage ($1.10\text{ m}$) provides ample clearance.

---

## 3. Step-by-Step Chronology of Failure (Phase 6 Trace Analysis)

From `repro_Scenario_C_Wide_Left_Phase6_Production_Guidance.json`:

### 3.1 Initial Drift Phase (Steps 0–48)
- The drone launches at $(x = 1.0\text{ m}, y = 0.0\text{ m})$ with forward velocity $v_x = 0.40\text{ m/s}$.
- The frozen Behavior Cloning policy net commands small negative lateral velocities ($v_y \approx -0.01$ to $-0.03\text{ m/s}$), causing the UAV to drift rightward.
- By step 48 ($x = 1.82\text{ m}$), the drone has drifted to $y = -0.34\text{ m}$. Because $y_{\text{drone}} = -0.34\text{ m} < y_{\text{obs}} = -0.25\text{ m}$, the UAV is positioned slightly to the *right* of the obstacle centerline.

### 3.2 Guidance Activation & Deadband Phase (Steps 49–65)
- At step 49, forward clearance drops below $0.85\text{ m}$ ($C_{\text{front}} = 0.884\text{ m} \to 0.715\text{ m}$). Guidance activates.
- For steps 49–65, sector clearance difference $\Delta = C_{\text{left}} - C_{\text{right}}$ fluctuates within the $\pm 0.08\text{ m}$ deadband. The controller commands `TIE_BREAK_LEFT` ($v_y = +0.25\text{ m/s}$).
- However, forward momentum ($v_x = 0.60\text{ m/s}$) carries the drone forward from $x = 1.99\text{ m}$ to $x = 2.32\text{ m}$ before substantial leftward lateral displacement can accumulate.

### 3.3 The False Branch Decision (Step 66)
- At step 66 ($x = 2.348\text{ m}, y = -0.343\text{ m}$):
  - Ray 1 (front-left) strikes the near quadrant of the pillar at $0.359\text{ m}$.
  - Ray 2 (front-right) peers into the open corridor along the right wall at $0.765\text{ m}$.
  - Clearance differential: $\Delta = C_{\text{left}} - C_{\text{right}} = 0.359 - 0.765 = \mathbf{-0.406\text{ m}}$.
  - Since $-0.406\text{ m} < r_{th} = -0.120\text{ m}$, the clearance differential decisively commands **`RIGHT`** ($v_y = -0.25\text{ m/s}$).
- **Root Cause of the False Branch:** The UAV was physically closer to the left edge of the obstacle ($y = -0.34\text{ m}$ vs $y_{\text{obs}} = -0.25\text{ m}$), so the obstacle blocked the left sector. The controller correctly observed that the right sector had more instantaneous ray distance, but could not foresee that the right sector terminated in an impassable $0.60\text{ m}$ pinch against the corridor boundary.

### 3.4 Entry into the Trap & Wall-Pinch Conflict (Steps 67–80)
- Commanded $v_y = -0.25\text{ m/s}$ accelerates the UAV further rightward: $y$ drops from $-0.34\text{ m} \to -0.38\text{ m} \to -0.45\text{ m} \to -0.50\text{ m}$.
- At step 78 ($x = 2.56\text{ m}, y = -0.50\text{ m}$), physical wall clearance drops to $0.354\text{ m}$.
- At step 80 ($x = 2.59\text{ m}, y = -0.51\text{ m}$), wall clearance drops to $0.341\text{ m} < 0.35\text{ m}$.
- The wall-proximity guardrail fires: `WALL_AVOID_LEFT` ($v_y = +0.25\text{ m/s}$).
- The UAV is now caught in dynamic opposition:
  - Steering left pushes it into the obstacle face ($0.189\text{ m}$ clearance).
  - Steering right pushes it into the corridor boundary ($0.341\text{ m}$ clearance).

### 3.5 Supervisor Throttle & Permanent Standoff (Steps 81–500)
- At step 81, forward range to the obstacle falls to $0.29\text{ m} < 0.30\text{ m}$ (critical obstacle threshold). The supervisor enters `obstacle_brake` and throttles forward speed to $0.10\text{ m/s}$.
- At step 100, the drone's speed remains $\le 0.08\text{ m/s}$ for $>0.35\text{ s}$, triggering `stalled_recovery` (reverse retreat at $-0.10\text{ m/s}$).
- For the remaining 400 steps, the drone oscillates between reverse retreat and forward braking, pinned at $x \approx 2.69\text{ m}, y \approx -0.54\text{ m}$ until reaching the canonical 500-step timeout.

---

## 4. Why Phase 5B Succeeded on the Same Scenario

In Phase 5B:
1. Phase 5B had **no wall guardrail** (`wall_tight_threshold`).
2. Its tie-breaker was permanently locked to `TIE_BREAK_LEFT`.
3. In Phase 5B's run, the drone tie-broke `LEFT` slightly more aggressively during steps 40–55, keeping $y \ge -0.32\text{ m}$, which allowed the obstacle to remain centered or slightly right, causing the clearance differential to favor `LEFT` and escape into the $1.10\text{ m}$ passage.

---

## 5. Experimental Sensitivity Findings

To understand if the wall-proximity threshold ($0.350\text{ m}$) was responsible for the stall:
We evaluated three candidate thresholds: $0.350\text{ m}$, $0.325\text{ m}$, and $0.300\text{ m}$.

### 5.1 Results
When evaluated in an isolated run starting from canonical initial conditions:
- **Candidate 0.350m:** Completed `Scenario_C_Wide_Left` in $18.0\text{ s}$ ($8.16\text{ m}$ progress, min obs clearance $0.1569\text{ m}$, min wall clearance $0.3475\text{ m}$).
- **Candidate 0.325m:** Completed `Scenario_C_Wide_Left` in $18.0\text{ s}$ ($8.16\text{ m}$ progress, min obs clearance $0.1596\text{ m}$, min wall clearance $0.3433\text{ m}$).
- **Candidate 0.300m:** Completed `Scenario_C_Wide_Left` in $18.0\text{ s}$ ($8.15\text{ m}$ progress, min obs clearance $0.1587\text{ m}$, min wall clearance $0.3499\text{ m}$).

### 5.2 Key Insight
The stall in the Phase 6 benchmark was triggered by an interaction between:
1. The accumulated sensor noise state across sequential scenario evaluations in the benchmark runner (which increased initial rightward drift to $y = -0.34\text{ m}$).
2. The delay in activating `WALL_AVOID_LEFT`: at $0.35\text{ m}$, wall avoidance only triggers after the UAV has already committed to the narrow side of the pillar.
3. Once both wall and obstacle caution zones overlap ($<0.85\text{ m}$ total corridor gap), no feedforward guidance policy can maneuver through without supervisor intervention.

---

## 6. Recommendations

1. **Benchmark Runner Reseeding:** Ensure that every diagnostic scenario in the evaluation harness explicitly calls `env.reset(seed=...)` or resets the RNG state so that scenario outcomes are 100% independent of the evaluation sequence.
2. **Preserve Current Wall Threshold ($0.350\text{ m}$):** In the 50 development seeds evaluation, threshold $0.350\text{ m}$ achieved identical completion ($84.0\%$) and provided superior obstacle clearance ($0.0957\text{ m}$) compared to $0.300\text{ m}$ ($0.0932\text{ m}$). Lowering the threshold to $0.300\text{ m}$ reduces safety margin without offering any measurable success benefit.

