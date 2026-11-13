# Phase 4J-C — Diagnostic Investigation of the Rear-Lateral Obstacle Blind Spot

**Incident Context:** Phase 4J-A / Phase 4J-B Near-Contact Anomaly  
**Subject Scenario:** Seed `1007`, Steps 160–172  
**Controller:** `BC_Phase4H_Candidate`  
**Location:** Pillar 0 at $X = 3.000\text{ m}, Y = -0.300\text{ m}$  
**Event:** Air gap reached an extreme minimum of **$d_{\text{obs}} = 0.00294\text{ m}$ ($2.94\text{ mm}$)** without registering collision or supervisor intervention.

---

## 1. Sensor Geometry and Angular Coverage Layout

In `sim/corridor_env.py` (`SensorConfig`), the drone is equipped with a 12-beam rangefinder array. The physical ray orientations in the vehicle's body frame are:

```
                          Ray 0 (0°, Front)
                                |
             Ray 1 (+30°)       |       Ray 2 (-30°)
                    \           |           /
           Ray 3 (+60°) \       |       / Ray 4 (-60°)
                        \       |       /
Ray 5 (+90°, Left) <------------+------------> Ray 6 (-90°, Right)
                   .                         .
                     .   [90° BLIND SPOT]  .
                       .                 .
                                |
                          Ray 7 (180°, Rear)
```

### The $90^\circ$ Angular Blind Zones
- **Rear-Right Quadrant:** Span between Ray 6 ($-90^\circ$, pure right) and Ray 7 ($180^\circ$, pure rear). There are **zero rays between $-90^\circ$ and $-180^\circ$**—a full $90^\circ$ blind zone!
- **Rear-Left Quadrant:** Span between Ray 5 ($+90^\circ$, pure left) and Ray 7 ($180^\circ$, pure rear). There are **zero rays between $+90^\circ$ and $+180^\circ$**—a full $90^\circ$ blind zone!

---

## 2. Step-by-Step Geometry and Telemetry Evolution

The table below reconstructs the exact geometric coordinates, drone attitude, body-frame obstacle azimuth, and raw sensor readings during the approach to minimum clearance (Steps 160–172):

| Step | Drone Position $[X, Y]$ | Yaw Angle | Obstacle Center Azimuth (Body Frame) | True Air Gap $d_{\text{obs}}$ (m) | Ray 5 ($+90^\circ$, Left) | Ray 6 ($-90^\circ$, Right) | Ray 7 ($180^\circ$, Rear) | Policy Action $[v_x, v_y]$ | Supervisor State |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 160 | $(3.400, -0.184)$ | $-28.6^\circ$ | **$-135.1^\circ$** | **$0.0667\text{ m}$** | $1.32\text{ m}$ | $0.93\text{ m}$ (wall) | $2.48\text{ m}$ (open) | $[0.08, -0.40]$ | Nominal |
| 162 | $(3.388, -0.223)$ | $-31.3^\circ$ | **$-137.4^\circ$** | **$0.0456\text{ m}$** | $1.44\text{ m}$ | $0.90\text{ m}$ (wall) | $2.36\text{ m}$ (open) | $[0.09, -0.40]$ | Nominal |
| 164 | $(3.375, -0.262)$ | $-34.1^\circ$ | **$-140.1^\circ$** | **$0.0273\text{ m}$** | $1.52\text{ m}$ | $0.87\text{ m}$ (wall) | $2.27\text{ m}$ (open) | $[0.12, -0.40]$ | Nominal |
| 166 | $(3.363, -0.302)$ | $-36.8^\circ$ | **$-143.5^\circ$** | **$0.0129\text{ m}$** | $1.63\text{ m}$ | $0.90\text{ m}$ (wall) | $2.16\text{ m}$ (open) | $[0.13, -0.40]$ | Nominal |
| 167 | $(3.356, -0.322)$ | $-38.2^\circ$ | **$-145.3^\circ$** | **$0.0070\text{ m}$** | $1.69\text{ m}$ | $0.87\text{ m}$ (wall) | $2.11\text{ m}$ (open) | $[0.14, -0.36]$ | Nominal |
| **168** | **$(3.351, -0.340)$** | **$-39.6^\circ$** | **$-147.0^\circ$** | **$0.0029\text{ m}$** | $1.73\text{ m}$ | $0.86\text{ m}$ (wall) | **$0.25\text{ m}$** (edge) | $[0.24, -0.33]$ | Nominal |
| 169 | $(3.349, -0.361)$ | $-40.9^\circ$ | **$-148.9^\circ$** | **$0.0046\text{ m}$** | $1.82\text{ m}$ | $0.82\text{ m}$ (wall) | $0.23\text{ m}$ (edge) | $[0.26, -0.29]$ | Nominal |
| 170 | $(3.350, -0.380)$ | $-42.2^\circ$ | **$-150.7^\circ$** | **$0.0089\text{ m}$** | $1.91\text{ m}$ | $0.82\text{ m}$ (wall) | $0.20\text{ m}$ (edge) | $[0.29, -0.26]$ | Nominal |
| 172 | $(3.356, -0.419)$ | $-44.6^\circ$ | **$-153.9^\circ$** | **$0.0256\text{ m}$** | $2.05\text{ m}$ | $0.79\text{ m}$ (wall) | $0.27\text{ m}$ (edge) | $[0.38, -0.19]$ | Nominal |

---

## 3. Findings on the Six Investigation Questions

### 3.1 Was the obstacle outside coverage, occluded, or visible?
**Finding:** The obstacle was **completely outside sensor coverage** from steps 160 through 167.
- During this window, the relative azimuth of the pillar center was between **$-135.1^\circ$ and $-145.3^\circ$**.
- Because the sensor only has rays at $-90^\circ$ (Ray 6) and $180^\circ$ (Ray 7), the obstacle fell directly into the empty $90^\circ$ gap. Ray 6 shot past the obstacle toward the right corridor wall ($0.90\text{ m}$), while Ray 7 shot straight rearward into open corridor space ($2.1\text{–}2.5\text{ m}$).
- The physical air gap deteriorated from $66.7\text{ mm}$ down to $7.0\text{ mm}$ with **zero sensor rays intercepting the cylinder**.

### 3.2 Did the supervisor have sufficient information to identify the risk?
**Finding:** **NO.**
- The supervisor relies on rangefinder observations. At step 167 (when the drone was only $7.0\text{ mm}$ from impact), the forward rays saw open corridor toward Pillar 1, and the lateral rays saw the corridor walls.
- The supervisor's observation vector contained no evidence of an obstacle within $0.08\text{ m}$. It is physically and mathematically impossible for a purely reactive rangefinder-based supervisor to intervene when the obstacle does not intersect any active sensor beam.

### 3.3 Is the clearance metric calculated consistently?
**Finding:** **YES.**
- `obs.distance_to_point(pos)` calculates Euclidean distance from drone center $(3.351, -0.340)$ to cylinder surface ($0.15294\text{ m}$).
- Subtracting `drone_radius` ($0.150\text{ m}$) yields exactly $0.00294\text{ m}$ ($2.94\text{ mm}$).
- This formula is identical to the ground-truth collision check in `check_termination`.

### 3.4 Earliest time a robust monitor could have detected the danger
- **With Ground-Truth Geometry (Oracle / Simulator):**
  At **Step 160** ($X = 3.400\text{ m}, Y = -0.184\text{ m}$), ground-truth clearance dropped to $0.0667\text{ m}$, breaching the $0.080\text{ m}$ envelope. A geometric monitor tracking obstacle cylinders would have detected the danger at step 160—a full 8 steps ($0.40\text{ s}$) before the minimum clearance point!
- **With Sensor-Only Architecture:**
  Would require adding diagonal rear rays at $\pm 135^\circ$ azimuth to eliminate the rear-quarter blind spot.

### 3.5 Root-Cause Attribution: Layout, Policy, or Supervisor?
**Finding:** The near-contact was caused by a **compound system interaction**:
1. **Policy Defect (Primary Trigger):** The BC candidate policy executed an overly aggressive lateral cut-in ($v_y = -0.40\text{ m/s}$) immediately after passing $X = 3.0\text{ m}$ without allowing sufficient longitudinal clearance before turning.
2. **Sensor Layout Limitation (Perceptual Blind Spot):** The 12-beam ray arrangement lacks diagonal rear-quarter rays ($\pm 135^\circ$), allowing obstacles behind the lateral plane to be invisible during forward-diagonal turns.
3. **Supervisor Logic Limitation (Structural Assumption):** The canonical supervisor assumes obstacles approaching from the rear are irrelevant during forward flight, checking `d_rear` only when the vehicle has already stalled for $> 0.35\text{ s}$.

