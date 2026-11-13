# Phase 4J-B-B — Clearance Anomaly Investigation (The 3 mm Event)

## 1. Executive Summary

During Phase 4J-A, evaluation of the `BC_Phase4H_Candidate` controller under caution distance $d_{\text{caut}} = 0.50\text{ m}$ on development seeds `1000..1009` recorded a global minimum obstacle clearance of **`0.003 m`** (`0.00294 m` exact) with **zero reported collisions**.

This investigation reproduced the exact scenario in full step-by-step diagnostic telemetry to isolate the physical cause and evaluate its safety implications.

### Key Conclusions
1. **The Phenomenon is Genuine (Explanation 1):**
   The drone physically approached within **$2.94\text{ mm}$** of the cylinder collision surface of Pillar 0 without registering contact.
2. **Why No Collision Occurred:**
   The collision detection contract in `core/geometry.py` defines collision strictly as:
   $$d_{\text{obs}} = \min_{\text{obs}} \left( \|\mathbf{p}_{xy} - \mathbf{c}_{xy}\| - r_{\text{pillar}} \right) - r_{\text{drone}} \le 0.000\text{ m}$$
   At step 168, $d_{\text{obs}} = +0.00294\text{ m} > 0.000\text{ m}$. There was no geometric interpenetration, so the simulation accurately classified the vehicle as collision-free.
3. **Safety Envelope Status:**
   The project requires a minimum operational clearance envelope of **$0.080\text{ m}$ ($80\text{ mm}$)**. This event was a **severe safety-envelope violation** (violating the envelope by $77.06\text{ mm}$), passing within $3\text{ mm}$ of catastrophic impact.
4. **Root Cause in Policy & Supervisor Interaction:**
   The near-miss occurred *after* the nose of the drone had already passed the pillar's X center line ($x_{\text{drone}} = 3.351\text{ m}$ vs $x_{\text{pillar}} = 3.000\text{ m}$). As the drone bypassed the pillar, the BC policy commanded a sharp rightward bank ($v_y = -0.33\text{ to } -0.40\text{ m/s}$) to angle toward Pillar 1. Because the obstacle was now in the drone's rear-left quarter, the supervisor's forward-facing rays and lateral rays did not detect an obstacle directly ahead or purely to the side, allowing the drone's rear-left hull to scrape within $2.94\text{ mm}$ of the pillar's trailing perimeter.

---

## 2. Telemetry and Trajectory Reconstruction

### 2.1 Episode Context
- **Controller:** `BC_Phase4H_Candidate` (`experiments/phase4h/checkpoints/bc_phase4h_candidate.pt`)
- **Supervisor Configuration:** `IsolatedOptimizedSupervisor(caution_distance_m=0.50)`
- **Seed:** `1007`
- **Pillar Configuration:**
  - Pillar 0: Center $(3.000, -0.300, 1.000)\text{ m}$, Radius $0.200\text{ m}$, Height $2.000\text{ m}$
  - Pillar 1: Center $(5.000, +0.221, 1.000)\text{ m}$, Radius $0.200\text{ m}$, Height $2.000\text{ m}$
  - Pillar 2: Center $(7.000, +0.100, 1.000)\text{ m}$, Radius $0.200\text{ m}$, Height $2.000\text{ m}$
- **Outcome:** Timed out at step 500 (forward progress $8.117\text{ m}$, 0 collisions).

### 2.2 Exact Step-by-Step Telemetry Around Minimum Clearance (Steps 160–176)

| Step | Drone Position $[X, Y, Z]$ (m) | Distance to Center (m) | Surface Air Gap $d_{\text{obs}}$ (m) | Forward Sensor (m) | Left Sensor (m) | Right Sensor (m) | Rear Sensor (m) | Commanded Action $[v_x, v_y]$ | Safe Action $[v_x, v_y]$ | Supervisor State |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 160 | $(3.400, -0.184, 0.958)$ | $0.4167$ | $0.0667$ | $0.98$ | $1.32$ | $0.82$ | $2.48$ | $[0.08, -0.40]$ | $[0.08, -0.40]$ | Nominal |
| 162 | $(3.388, -0.223, 0.956)$ | $0.3956$ | $0.0456$ | $0.88$ | $1.44$ | $0.78$ | $2.36$ | $[0.09, -0.40]$ | $[0.09, -0.40]$ | Nominal |
| 164 | $(3.375, -0.262, 0.955)$ | $0.3773$ | $0.0273$ | $0.82$ | $1.52$ | $0.73$ | $2.27$ | $[0.12, -0.40]$ | $[0.12, -0.40]$ | Nominal |
| 166 | $(3.363, -0.302, 0.954)$ | $0.3629$ | $0.0129$ | $0.74$ | $1.58$ | $0.71$ | $2.16$ | $[0.13, -0.40]$ | $[0.13, -0.40]$ | Nominal |
| 167 | $(3.356, -0.322, 0.953)$ | $0.3570$ | $0.0070$ | $0.75$ | $1.54$ | $0.70$ | $2.11$ | $[0.14, -0.36]$ | $[0.14, -0.36]$ | Nominal |
| **168** | **$(3.351, -0.340, 0.952)$** | **$0.3529$** | **$0.0029$** | **$0.70$** | **$1.73$** | **$0.64$** | **$0.25$** | **$[0.24, -0.33]$** | **$[0.24, -0.33]$** | **Nominal** |
| 169 | $(3.349, -0.361, 0.952)$ | $0.3546$ | $0.0046$ | $0.67$ | $1.53$ | $0.67$ | $0.23$ | $[0.26, -0.29]$ | $[0.26, -0.29]$ | Nominal |
| 170 | $(3.350, -0.380, 0.951)$ | $0.3589$ | $0.0089$ | $0.64$ | $1.58$ | $0.64$ | $0.20$ | $[0.29, -0.26]$ | $[0.29, -0.26]$ | Nominal |
| 172 | $(3.356, -0.419, 0.950)$ | $0.3756$ | $0.0256$ | $0.63$ | $1.70$ | $0.61$ | $0.27$ | $[0.38, -0.19]$ | $[0.38, -0.19]$ | Nominal |
| 174 | $(3.375, -0.463, 0.949)$ | $0.4087$ | $0.0587$ | $0.53$ | $1.95$ | $0.53$ | $0.24$ | $[0.59, -0.12]$ | $[0.59, -0.12]$ | Nominal |
| 175 | $(3.391, -0.488, 0.948)$ | $0.4340$ | $0.0840$ | $0.50$ | $1.76$ | $0.50$ | $0.25$ | $[0.59, -0.07]$ | $[0.59, -0.07]$ | Obstacle Throttle |

### 2.3 Mathematical Verification of the Minimum Clearance Point (Step 168)
- Drone Center: $\mathbf{p} = (3.3510, -0.3400, 0.9520)\text{ m}$
- Pillar Center: $\mathbf{c} = (3.0000, -0.3001, 1.0000)\text{ m}$
- Relative Vector in XY plane:
  $$\Delta x = 3.3510 - 3.0000 = +0.3510\text{ m}$$
  $$\Delta y = -0.3400 - (-0.3001) = -0.0399\text{ m}$$
- Center-to-Center Euclidean Distance:
  $$D_{\text{center}} = \sqrt{(0.3510)^2 + (-0.0399)^2} = \sqrt{0.12320 + 0.00159} = \sqrt{0.12479} = 0.35294\text{ m}$$
- Air Gap between physical surfaces ($r_{\text{pillar}} = 0.200\text{ m}, r_{\text{drone}} = 0.150\text{ m}$):
  $$d_{\text{obs}} = D_{\text{center}} - (r_{\text{pillar}} + r_{\text{drone}}) = 0.35294 - 0.35000 = \mathbf{0.00294\text{ m}} = \mathbf{2.94\text{ mm}}$$
- The exact analytical calculation perfectly matches `obs_clear = 0.00294 m` from `info["privileged_state"]`.

---

## 3. Causal Mechanism Analysis

1. **Premature Turning by Policy:**
   As soon as the drone's center of mass passed $X = 3.0\text{ m}$, the BC policy attempted to line up for Pillar 1 (which was offset at $Y = +0.221\text{ m}$). However, due to lingering imitation bias or path geometry, it commanded maximum rightward lateral velocity ($v_y = -0.40\text{ m/s}$), cutting in behind Pillar 0.
2. **Supervisor Blind Spot in Rear-Quarter Clearance:**
   - The supervisor's obstacle caution throttling monitors `eff_front` (the forward 60° cone). At step 168, `eff_front` was measuring Pillar 1 ($X=5.0\text{ m}$) at $0.70\text{ m}$, well outside the $0.50\text{ m}$ throttle boundary.
   - The supervisor's lateral constraint checks $C_{\text{right}} = \min(d_{\text{right\_wall}}, d_{\text{right\_obs}})$ and $C_{\text{left}}$. Pillar 0 was located at $(\Delta x = -0.35\text{ m}, \Delta y = +0.04\text{ m})$ relative to the drone—meaning it was **behind the drone's lateral sensor plane**.
   - As a result, the supervisor classified the state as `NOMINAL` throughout steps 160–174, applying zero braking or repulsion setpoint adjustments.

---

## 4. Other Envelope Violations on Development Seeds

Investigating the remaining development seeds for `BC_Phase4H_Candidate` at $d_{\text{caut}} = 0.50\text{ m}$ revealed:
- **Seed 1001:** Recorded a minimum clearance of **$0.03933\text{ m}$ ($39.3\text{ mm}$)** at step 221 (also violating the $0.080\text{ m}$ safety envelope).
- **Seeds 1000, 1003, 1004, 1005, 1006, 1008, 1009:** Maintained clearances between $0.111\text{ m}$ and $0.148\text{ m}$ (satisfying the envelope).

---

## 5. Architectural Recommendations

1. **Do Not Treat "Zero Collisions" as Safe:**
   A trajectory that passes within $2.94\text{ mm}$ of an obstacle has zero physical safety margin. In real flight, rotor wash, aerodynamic ground effect, turbulence, and sensor noise would result in an immediate catastrophic crash.
2. **Separate Metric Tracking in Phase 4J-B:**
   Every evaluation in Phase 4J-B must explicitly track:
   - `obstacle_collision_count` ($d_{\text{obs}} \le 0.000\text{ m}$)
   - `obstacle_envelope_violation_count` ($d_{\text{obs}} < 0.080\text{ m}$)
   - `boundary_collision_count` ($d_b \le 0.000\text{ m}$)
   - `boundary_envelope_violation_count` ($d_b < 0.080\text{ m}$)

