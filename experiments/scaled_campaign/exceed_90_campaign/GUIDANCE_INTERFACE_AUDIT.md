# Comprehensive Interface & Guidance Audit: Scaled 65 cm UAV

**Date:** October 30, 2026  
**Artifact Target:** `core/guidance.py`, `controllers/safety_supervisor.py`, `models/policy_net.py`  

---

## 1. Coordinate Frames, Units & Normalization Audit

| Interface Component | Convention | Simulated Units | Verification Status |
|---|---|---|---|
| **World Frame** | ISO ENU (East-North-Up) | Meters, m/s | **Verified Consistent** |
| **Body Frame** | Forward-Left-Up (FLU) | Meters, m/s, rad/s | **Verified Consistent** |
| **Sensor Rays** | Multi-beam rangefinders (12 rays) | Normalized to $[0, 1]$ ($d = \text{ray} \times 4.0$ m) | **Verified Consistent** |
| **Goal Direction Vector** | Relative vector in FLU frame | Unit vector when $d > 2.0$ m; scaled by $d/2.0$ when $d \le 2.0$ m | **Verified Consistent** |
| **Attitude Trig** | $[\sin\phi, \cos\phi, \sin\theta, \cos\theta]$ | Dimensionless continuous floats in $[-1, 1]$ | **Verified Consistent** |
| **Action Bounds** | High-level velocity setpoints | $v_x \in [-0.2, 0.8], v_y \in [-0.6, 0.6], v_z \in [-0.4, 0.4], \dot{\psi} \in [-0.6, 0.6]$ | **Verified Consistent** |

---

## 2. Root Cause Analysis: The Guidance-Supervisor Discrepancy & Deadlock

### 2.1 The Forward Ray Discrepancy
- **Safety Supervisor:** Uses `FORWARD_RAY_INDICES = (0, 1, 2, 3, 4)`:
  - Ray 0: $0^\circ$ (Direct Front)
  - Ray 1: $+30^\circ$ (Front-Left 30)
  - Ray 2: $-30^\circ$ (Front-Right 30)
  - Ray 3: $+60^\circ$ (Front-Left 60)
  - Ray 4: $-60^\circ$ (Front-Right 60)
- **Guidance Arbitrator (`core/guidance.py`):**
  - Line 92: `c_front = float(min(rays[0], rays[1], rays[2]))`
  - **Severe Omission:** Rays 3 and 4 were completely omitted from the forward clearance check.

### 2.2 Deadlock Mechanism
1. When approaching a cylindrical obstacle with lateral offset (e.g. at azimuth $-37^\circ$ to $-55^\circ$):
   - Rays 0, 1, and 2 read open corridor space ($> 3.0$ m).
   - Guidance Arbitrator calculates `c_front > 1.225` m and concludes `selected_dir = "NONE"`.
   - The learned BC policy, observing open forward rays, outputs nominal forward velocity $v_x = 0.60$ m/s, $v_y \approx 0.00$ m/s.
2. Simultaneously, Ray 4 detects the obstacle perimeter at $0.50$ m.
   - The Safety Supervisor evaluates `FORWARD_RAY_INDICES` and registers an imminent collision ($< 0.20$ m clearance).
   - The Supervisor intervenes with `obstacle_brake`, clamping forward velocity to $v_x = 0.00$ m/s.
3. Neither subsystem yields:
   - Guidance refuses to evade because Rays 0–2 are clear.
   - The Supervisor refuses to accelerate because Ray 4 is blocked.
   - The BC policy outputs forward thrust without lateral intent.
   - **Result:** The drone is paralyzed indefinitely, racking up 800+ intervention steps until episode timeout at step 1,000.

---

## 3. Systematic Corrective Actions

1. **Harmonize Forward Ray Coverage:**
   Update Guidance Arbitrator to compute `c_front = float(min(rays[0], rays[1], rays[2], rays[3], rays[4]))` in strict mathematical agreement with the Safety Supervisor.
2. **Calibrate Activation & Deceleration Envelope:**
   Increase activation threshold to $d_{\text{act}} = R_{\text{drone}} + 0.95 = 1.275$ m and lateral override speed to $0.35$ m/s to guarantee lateral evasion clearance before the supervisor's $0.20$ m emergency braking envelope is reached.
3. **Active Evasion Enforcement in Stalled States:**
   When forward clearance is restricted, the guidance arbitrator must enforce a persistent lateral velocity setpoint ($v_y = \pm 0.35$ m/s) until the 5 forward rays are completely clear.

