# Phase 6B Failure Diagnosis Report: Forensic Telemetry Reconstruction

**Phase:** Phase 6B — Targeted Pillar 3 Recovery Experiment  
**Directory:** `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/`  
**Target Failure Episodes:** Seeds 2001, 2005, 2006, 2022, 2042  

---

## 1. Overview of Forensic Investigation

The development benchmark identified five episodes (Seeds 2001, 2005, 2006, 2022, 2042) where the UAV successfully traversed Pillars 1 and 2, but failed at or immediately following Pillar 3 ($x \approx 7.0\text{ m}$). 

Using full per-step telemetry saved in `traces/seed_<seed>_phase6a_telemetry.json`, each failure was mathematically reconstructed across physical position, body velocities, rangefinder sector clearances, guidance commands, supervisor overrides, and corridor geometry.

---

## 2. Forensic Reconstruction of the Five Failure Episodes

### A. Seed 2001 (Centered Pillar 3 Directional Chattering)
- **Obstacle Layout:**
  - Pillar 1: $x = 3.0\text{ m}, y = -0.053\text{ m}, r = 0.20\text{ m}$
  - Pillar 2: $x = 5.0\text{ m}, y = -0.207\text{ m}, r = 0.20\text{ m}$
  - Pillar 3: $x = 7.0\text{ m}, y = -0.017\text{ m}, r = 0.20\text{ m}$
  - Pillar 3 Clearances: Left gap $= 0.817\text{ m}$, Right gap $= 0.783\text{ m}$ (nearly symmetric).
- **Trajectory Timeline:**
  - Drone clears Pillar 2 on the left at step 305 ($x = 5.21\text{ m}, y = +0.298\text{ m}$).
  - In the inter-pillar gap, the UAV drifts toward the centerline, arriving at $x = 6.23\text{ m}, y = +0.03\text{ m}$ at step 380.
  - At step 380, front clearance drops below $0.85\text{ m}$ ($c_{\text{front}} = 0.58\text{ m}$) directly in front of the centered obstacle ($y_{\text{obs}} = -0.017\text{ m}$).
  - Clearance difference $\Delta = c_{\text{left}} - c_{\text{right}}$ hovers inside the $\pm 0.08\text{ m}$ deadband ($\Delta \in [-0.06\text{ m}, +0.04\text{ m}]$).
  - Between steps 363 and 440 (77 steps = $3.85\text{ s}$), guidance rapidly alternates between `RIGHT` and `TIE_BREAK_LEFT` with **13 direction switches**.
  - Because of this rapid lateral toggling, forward velocity is continuously throttled by the supervisor's front caution envelope ($v_x \le 0.16\text{ m/s}$ to $0.00\text{ m/s}$).
  - At step 440, the UAV finally commits rightward and slips past Pillar 3 at step 480.
  - **Terminal State:** Step 500 timeout at $x = 7.521\text{ m}, y = -0.404\text{ m}$, goal distance $d = 2.02\text{ m}$. All 3 pillars were cleared (`p_cleared = [1, 1, 1]`), but the 77-step chattering depleted the episode horizon.
- **Root Cause Classification:**
  - **Primary: B. Guidance repeatedly switching directions (chattering)**.
  - **Secondary: D. Supervisor throttling preventing escape from local standoff**.
  - **Evidence:** 13 direction switches during steps 363–440; forward speed throttled below $0.16\text{ m/s}$ across 77 steps; obstacle cleared at step 480 but steps expired before reaching goal sphere.

---

### B. Seed 2005 (Left Pinch Squeeze Standoff)
- **Obstacle Layout:**
  - Pillar 1: $x = 3.0\text{ m}, y = +0.152\text{ m}, r = 0.20\text{ m}$
  - Pillar 2: $x = 5.0\text{ m}, y = -0.011\text{ m}, r = 0.20\text{ m}$
  - Pillar 3: $x = 7.0\text{ m}, y = +0.214\text{ m}, r = 0.20\text{ m}$
  - Pillar 3 Clearances: Left gap $= 0.586\text{ m}$ (NARROW PINCH), Right gap $= 1.014\text{ m}$ (WIDE PASSAGE).
- **Trajectory Timeline:**
  - Drone clears Pillar 2 on the right at step 188 ($x = 5.20\text{ m}, y = -0.512\text{ m}$).
  - In the inter-pillar gap ($x \in [5.20, 5.85]\text{ m}$), $c_{\text{front}} > 0.85\text{ m}$, leaving guidance dormant (`NONE`).
  - The nominal BC policy exhibits autoregressive lateral velocity persistence, outputting $v_y \in [+0.27, +0.35]\text{ m/s}$, crossing the corridor from $y = -0.512\text{ m}$ all the way to $y = +0.350\text{ m}$ at $x = 6.29\text{ m}$ (step 248).
  - Approaching Pillar 3 from $y = +0.350\text{ m}$, the obstacle ($y = +0.214\text{ m}$) appears on the UAV's **right**.
  - Local clearance on the right is $c_{\text{right}} \approx 0.35\text{ m}$ (the obstacle surface); clearance on the left is $c_{\text{left}} \approx 0.65\text{ m}$ (the left corridor wall).
  - Guidance computes $\Delta = c_{\text{left}} - c_{\text{right}} > 0$ and commands `LEFT` ($v_y = +0.25\text{ m/s}$), steering the drone directly into the narrow $0.586\text{ m}$ left pinch!
  - By step 278 ($x = 6.63\text{ m}, y = +0.56\text{ m}$), the distance to the left wall is $1.00 - 0.56 - 0.15 = 0.29\text{ m} < d_{\text{crit\_lat}} (0.32\text{ m})$.
  - The safety supervisor activates its lateral barrier and **clamps lateral velocity to zero ($v_y = 0.00\text{ m/s}$)** across 176 consecutive steps!
  - Concurrently, distance to Pillar 3 front is $< 0.35\text{ m}$, throttling $v_x$ to $0.10\text{ m/s}$ / $0.00\text{ m/s}$.
  - **Terminal State:** Step 500 timeout at $x = 6.640\text{ m}, y = +0.569\text{ m}$ ($v_x = 0.0\text{ m/s}, v_y = 0.0\text{ m/s}$). Pillar 3 was not passed (`p_cleared = [1, 1, 0]`).
- **Root Cause Classification:**
  - **Primary: C. Wall guardrail and obstacle guidance opposing one another**.
  - **Contributing: Post-Pillar 2 lateral overshoot into the narrow pinch**.
  - **Evidence:** Guidance commanded `LEFT` into the $0.586\text{ m}$ pinch; wall guardrail clamped $v_y = 0.0\text{ m/s}$ for 176 of 178 stall steps; obstacle caution throttled $v_x \le 0.10\text{ m/s}$, producing zero forward displacement for $8.9\text{ s}$.

---

### C. Seed 2006 (Right Pinch Squeeze Standoff)
- **Obstacle Layout:**
  - Pillar 1: $x = 3.0\text{ m}, y = +0.260\text{ m}, r = 0.20\text{ m}$
  - Pillar 2: $x = 5.0\text{ m}, y = +0.041\text{ m}, r = 0.20\text{ m}$
  - Pillar 3: $x = 7.0\text{ m}, y = -0.138\text{ m}, r = 0.20\text{ m}$
  - Pillar 3 Clearances: Left gap $= 0.938\text{ m}$ (WIDE PASSAGE), Right gap $= 0.662\text{ m}$ (NARROW PINCH).
- **Trajectory Timeline:**
  - Drone clears Pillar 2 on the left at step 202 ($x = 5.21\text{ m}, y = +0.517\text{ m}$).
  - Between $x = 5.21\text{ m}$ and $x = 5.80\text{ m}$, the BC policy drives rightward with $v_y \in [-0.20, -0.40]\text{ m/s}$, overshooting past the centerline ($y = +0.01\text{ m}$ at step 247) to $y = -0.262\text{ m}$ at step 262.
  - Pillar 3 is at $y = -0.138\text{ m}$. Because the drone is at $y = -0.262\text{ m}$, the obstacle is on its **left**.
  - The right gap is only $0.662\text{ m}$. Guidance commands `RIGHT` ($v_y = -0.25\text{ m/s}$) into the right wall pinch.
  - At $x = 6.74\text{ m}, y = -0.59\text{ m}$, clearance to the right wall is $1.00 - 0.59 - 0.15 = 0.26\text{ m} < 0.32\text{ m}$.
  - The supervisor clamps $v_y = 0.00\text{ m/s}$ (137 clamped steps) and obstacle braking clamps $v_x = 0.00\text{ m/s}$.
  - **Terminal State:** Step 500 timeout at $x = 6.783\text{ m}, y = -0.592\text{ m}$. Pillar 3 was not passed (`p_cleared = [1, 1, 0]`).
- **Root Cause Classification:**
  - **Primary: C. Wall guardrail and obstacle guidance opposing one another**.
  - **Contributing: Post-Pillar 2 lateral overshoot across centerline into narrow pinch**.
  - **Evidence:** Guidance commanded `RIGHT` into the $0.662\text{ m}$ pinch; supervisor clamped $v_y = 0.00\text{ m/s}$ across 137 steps; obstacle front brake held $v_x = 0.00\text{ m/s}$.

---

### D. Seed 2022 (Left Displacement Persistence Standoff)
- **Obstacle Layout:**
  - Pillar 1: $x = 3.0\text{ m}, y = -0.177\text{ m}, r = 0.20\text{ m}$
  - Pillar 2: $x = 5.0\text{ m}, y = -0.285\text{ m}, r = 0.20\text{ m}$
  - Pillar 3: $x = 7.0\text{ m}, y = +0.078\text{ m}, r = 0.20\text{ m}$
  - Pillar 3 Clearances: Left gap $= 0.722\text{ m}$ (NARROW PINCH), Right gap $= 0.878\text{ m}$ (WIDER PASSAGE).
- **Trajectory Timeline:**
  - Drone clears Pillar 2 on the left at step 164 ($x = 5.21\text{ m}, y = +0.310\text{ m}$).
  - In the inter-pillar gap, the nominal policy exhibits zero restorative centering command ($v_y \approx +0.03\text{ m/s}$), cruising at a constant displacement of $y \approx +0.30\text{ m}$.
  - At $x = 6.48\text{ m}$ (step 209), the UAV reaches Pillar 3 ($y = +0.078\text{ m}$) from the left.
  - Guidance commands `LEFT` ($v_y = +0.29\text{ m/s}$), pushing the vehicle further left to $y = +0.55\text{ m}$.
  - At $y = +0.55\text{ m}$, clearance to the left wall is $0.30\text{ m} < 0.32\text{ m}$. The supervisor clamps $v_y = 0.00\text{ m/s}$ across 164 steps.
  - Concurrently, front clearance to Pillar 3 triggers continuous obstacle braking ($v_x \le 0.10\text{ m/s}$).
  - **Terminal State:** Step 500 timeout at $x = 6.834\text{ m}, y = +0.536\text{ m}$ ($212$ standoff steps). Pillar 3 was not passed (`p_cleared = [1, 1, 0]`).
- **Root Cause Classification:**
  - **Primary: C. Wall guardrail and obstacle guidance opposing one another**.
  - **Contributing: Lack of restorative centering after Pillar 2**.
  - **Evidence:** 164 clamped $v_y$ steps; $v_x \le 0.10\text{ m/s}$ across 212 steps in the $0.722\text{ m}$ pinch.

---

### E. Seed 2042 (Left Pinch Squeeze Standoff)
- **Obstacle Layout:**
  - Pillar 1: $x = 3.0\text{ m}, y = +0.023\text{ m}, r = 0.20\text{ m}$
  - Pillar 2: $x = 5.0\text{ m}, y = -0.145\text{ m}, r = 0.20\text{ m}$
  - Pillar 3: $x = 7.0\text{ m}, y = +0.205\text{ m}, r = 0.20\text{ m}$
  - Pillar 3 Clearances: Left gap $= 0.595\text{ m}$ (NARROW PINCH), Right gap $= 1.005\text{ m}$ (WIDE PASSAGE).
- **Trajectory Timeline:**
  - Drone clears Pillar 2 on the right at step 280 ($x = 5.21\text{ m}, y = -0.571\text{ m}$).
  - In the inter-pillar gap, the nominal policy overshoots from $y = -0.571\text{ m}$ across the centerline to $y = +0.340\text{ m}$ at $x = 6.36\text{ m}$.
  - Approaching Pillar 3 from $y = +0.340\text{ m}$, the obstacle is on the right; guidance commands `LEFT` into the $0.595\text{ m}$ pinch.
  - The UAV reaches $y = +0.57\text{ m}$ at $x = 6.66\text{ m}$. Distance to the left wall is $0.28\text{ m} < 0.32\text{ m}$.
  - The supervisor clamps $v_y = 0.00\text{ m/s}$ (85/85 stall steps clamped), and front distance $< 0.30\text{ m}$ engages the emergency forward brake ($v_x = 0.00\text{ m/s}$).
  - **Terminal State:** Step 500 timeout at $x = 6.658\text{ m}, y = +0.571\text{ m}$. Pillar 3 was not passed (`p_cleared = [1, 1, 0]`).
- **Root Cause Classification:**
  - **Primary: C. Wall guardrail and obstacle guidance opposing one another**.
  - **Contributing: Post-Pillar 2 lateral overshoot across centerline into narrow pinch**.
  - **Evidence:** 85/85 stall steps clamped $v_y = 0.00\text{ m/s}$; forward brake engaged $v_x = 0.00\text{ m/s}$.

---

## 3. Synthesis of Failure Root Causes

| Failure Category | Description | Affected Seeds | Mathematical / Telemetric Proof |
| :--- | :--- | :--- | :--- |
| **Category B: Directional Chattering** | Symmetric returns cause rapid left/right oscillation | Seed 2001 | 13 direction switches in steps 363–440; forward speed throttled below $0.16\text{ m/s}$ |
| **Category C: Wall Guardrail vs. Guidance Opposition** | Vehicle enters narrow pinch ($<0.70\text{m}$); wall guardrail zeroes $v_y$ while obstacle brake zeroes $v_x$ | Seeds 2005, 2006, 2022, 2042 | $v_y$ clamped to $0.00\text{ m/s}$ for $85$ to $176$ consecutive steps; $v_x \le 0.10\text{ m/s}$ |

### The Upstream Origin of Category C Failures
In all four Category C failures (2005, 2006, 2022, 2042), the failure at Pillar 3 is caused by **upstream lateral state upon arrival at $x \approx 6.3\text{ m}$**. 
Because the distance between Pillar 2 ($x = 5.0\text{ m}$) and Pillar 3 ($x = 7.0\text{ m}$) is only $2.0\text{ m}$, the UAV spends $\approx 1.2\text{ m}$ in an open corridor ($c_{\text{front}} > 0.85\text{ m}$).
During this open window:
1. The nominal BC policy's autoregressive velocity shortcut either persists an evasive velocity (causing a cross-corridor overshoot from $-0.55\text{ m}$ to $+0.35\text{ m}$, as in Seeds 2005 and 2042) or lacks centering authority (maintaining $+0.30\text{ m}$, as in Seed 2022).
2. The UAV arrives at Pillar 3 displaced toward the **narrow pinch** side of the obstacle.
3. Once displaced toward the narrow side, purely local rangefinder clearances dictate steering toward the adjacent wall, where the supervisor's boundary guardrail and obstacle caution zone permanently arrest motion.

