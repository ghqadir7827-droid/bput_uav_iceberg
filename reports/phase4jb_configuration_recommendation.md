# Phase 4J-B-E — Configuration Recommendation and Architectural Decision Report

## 1. Executive Summary

This report delivers the final synthesis, formal configuration decision, and architectural safety assessment for **Phase 4J-B**.

Every finding and recommendation in this report is backed by:
- Mathematical source-code audits of `controllers/safety_supervisor.py`, `sim/corridor_env.py`, `core/geometry.py`, and `core/contracts.py`.
- Empirical step-by-step telemetry reproduction of the 3 mm near-miss anomaly.
- A 1,550-episode paired evaluation across 50 fresh development seeds (`2000..2049`) covering 6 caution distances ($0.80\text{ m}$ to $0.45\text{ m}$) across 5 controllers and an unconstrained counterfactual.
- 119/119 passing unit tests verifying parameter propagation, distance conventions, and envelope metrics.

---

## 2. Synthesis of Verified Facts vs. Uncertainties

### 2.1 Verified Directly from Source Code
1. **Sensor Measurement Convention:** Rangefinder distance `dist` and `eff_front` measure Euclidean distance from the **drone's center of mass to the obstacle's outer cylinder surface**.
2. **Physical Air Gap:** True surface-to-surface clearance is $\text{Air Gap} = \text{eff\_front} - r_{\text{drone}} = \text{eff\_front} - 0.15\text{ m}$.
3. **Threshold vs. Air Gap:**
   - Canonical caution distance $d_{\text{caut}} = 0.80\text{ m}$ triggers throttling at a physical air gap of $0.65\text{ m}$ ($1.00\text{ m}$ center-to-center).
   - Critical brake $d_{\text{crit}} = 0.30\text{ m}$ triggers at an air gap of $0.15\text{ m}$ ($0.50\text{ m}$ center-to-center).
   - Reverse threshold $d_{\text{rev}} = 0.15\text{ m}$ triggers at an air gap of $0.00\text{ m}$ (physical surface contact).
4. **Collision Detection Mechanics:** Analytical Euclidean formulas (`obstacle_clearance <= 0.0` and `boundary_distance <= 0.0`) govern collision termination. Contact points from PyBullet dynamics are not used for termination checks.
5. **Deceleration & Stopping Mechanics:** In simulation, velocity changes occur via instantaneous velocity resets (`resetBaseVelocity`), yielding an artificial $5.0\text{ mm}$ stopping distance from $0.10\text{ m/s}$. In physical flight, quadrotor pitch dynamics and latency require $11.25\text{ mm}$ to stop from $0.10\text{ m/s}$, and $25.5\text{ cm}$ to stop from $0.60\text{ m/s}$.

### 2.2 Reproduced Experimentally Across 50 Paired Seeds
1. **Root Cause of Multi-Obstacle Timeouts:** Canonical $d_{\text{caut}} = 0.80\text{ m}$ constricts the unthrottled acceleration runway between pillars to only $0.40\text{ m}$, forcing the drone to spend $78.9\%$ of flight throttled to $\approx 0.14\text{ m/s}$ and producing a $0.0\%$ success rate across all 5 controllers.
2. **Rejection of Phase 4I Claims:** Phase 4I's claim that $d_{\text{caut}} = 0.55\text{ m}$ guarantees 100% expert arrival in $23.8\text{ s}$ was completely disproven. Across 50 seeds, $0.55\text{ m}$ yielded only **22.0% expert success** and **34.0% BC baseline success** because lateral S-turn maneuvering delays push the majority of flights past the 500-step horizon.
3. **The 3 mm Clearance Anomaly Was Genuine:** Step-by-step reproduction of Seed 1007 confirmed that the drone reached an exact physical air gap of $2.94\text{ mm}$ at step 168 without penetrating the obstacle. This was a severe safety-envelope violation ($< 0.080\text{ m}$) caused by the policy cutting corners behind the obstacle outside the supervisor's forward sensor cone.
4. **BC Baseline is Vastly Safer than BC Candidate:** The canonical `bc_policy_best.pt` achieved **zero envelope violations and zero collisions across all 300 runs** (minimum clearance $\ge 0.092\text{ m}$). The Phase 4H BC candidate suffered up to 46% envelope violations and multiple obstacle collisions.
5. **Supervisor is Indispensable:** In the unconstrained counterfactual (supervisor disabled), the expert achieved 100% success, but suffered an extreme near-miss in Seed 2033 ($0.3\text{ mm}$ clearance). Active supervision is required to prevent catastrophic edge-case grazing.

### 2.3 Remaining Uncertainties & Simulation Limitations
1. **Sensor Occlusion in Tight Turns:** The 12-beam rangefinder configuration has angular gaps between rays. When an obstacle falls in the rear-lateral quadrant ($90^\circ\dots 135^\circ$ azimuth), it is not detected by forward rays or pure lateral rays.
2. **PyBullet Aerodynamic Simplicity:** Simulation assumes rigid-body velocity control without rotor downwash aerodynamic interaction, propeller ground effect, or wall cushion effects.
3. **Zero Flight Readiness:** Simulation success under ideal conditions provides zero guarantee of physical stability on a real drone with IMU vibration, optical flow drift, or battery voltage sag.

---

## 3. Configuration Decision

In accordance with Phase 4J-B specifications, we define three explicit tiers:

### Outcome A: Development Candidate
- **Selected Parameter:** **$d_{\text{caut}} = 0.50\text{ m}$**
- **Justification:**
  - For the primary deployable neural controller (`Arm_B_BC_Baseline`), $d_{\text{caut}} = 0.50\text{ m}$ achieves the **highest success rate (48.0% [34.8–61.5%])**, outperforming $0.45\text{ m}$ (38.0%).
  - Provides a physical deceleration buffer of $\Delta s = 0.20\text{ m}$ ($20\text{ cm}$) between caution throttling onset ($0.50\text{ m}$) and critical braking ($0.30\text{ m}$), allowing a smooth, physically realizable deceleration of $0.875\text{ m/s}^2$.
  - Preserves zero obstacle collisions, zero boundary collisions, and zero envelope violations for the baseline controller.

### Outcome B: Safety-Compliant Candidate
- **Selected Controller & Configuration:** **`Arm_B_BC_Baseline` with $d_{\text{caut}} = 0.50\text{ m}$**
- **Simulator Compliance Status:** **FULLY COMPLIANT** across all 50 evaluated development seeds.
  - Obstacle collisions: **0/50 (0.0%)**
  - Boundary collisions: **0/50 (0.0%)**
  - Obstacle safety-envelope violations ($< 0.080\text{ m}$): **0/50 (0.0%)**
  - Global minimum obstacle clearance: **$0.107\text{ m}$** ($> 0.080\text{ m}$)
  - Global minimum boundary clearance: **$0.187\text{ m}$** ($> 0.080\text{ m}$)
- **Limitations of Sample:** Validated on 50 development seeds (`2000..2049`). Generalization to unseen held-out environments (`90000..90099`) remains to be established in formal held-out validation.
- *Notice on Controller Capability:* The Analytical Expert is capable of safely operating at $d_{\text{caut}} = 0.45\text{ m}$ ($72.0\%$ success, $0.118\text{ m}$ min clearance, 0 violations). However, among learned neural policies, only `Arm_B_BC_Baseline` at $0.50\text{ m}$ satisfies all safety constraints.

### Outcome C: Flight-Ready Configuration
- **Status:** **NONE DECLARED**.
- **Reason:** Simulation-only results cannot establish physical flight readiness. The current velocity-reset dynamics model does not capture quadrotor tilt dynamics, thrust lag, aerodynamic ground/wall interactions, localization drift, or communication jitter. Declaring flight readiness prior to hardware-in-the-loop (HIL) testing would violate core aviation engineering safety standards.

---

## 4. Why $d_{\text{caut}} = 0.45\text{ m}$ Is Not Recommended for Neural Policies

While $0.45\text{ m}$ boosts Expert APF success to $72.0\%$, empirical evaluation across 50 seeds demonstrates that it degrades neural policy safety and performance:
1. **Lower Success for BC Baseline:** Success drops from $48.0\%$ at $0.50\text{ m}$ down to $38.0\%$ at $0.45\text{ m}$. Entering obstacle passages faster induces lateral overshoot, triggering strong boundary repulsion that cancels forward momentum.
2. **Severe PPO Boundary Crashes:** PPO Candidate boundary collisions escalate to **34.0% (17/50 crashes)** at $0.45\text{ m}$.
3. **Narrow Deceleration Buffer:** Deceleration distance shrinks to only $0.15\text{ m}$ ($15\text{ cm}$), demanding $1.17\text{ m/s}^2$ deceleration which exceeds physical flight controller limits.

---

## 5. Artifact and Verification Manifest

### 5.1 Source Files Used
- `controllers/safety_supervisor.py` (Canonical supervisor, strictly unmodified)
- `sim/corridor_env.py` (Gymnasium PyBullet environment)
- `core/geometry.py` (Corridor, obstacle, and termination geometry)
- `core/contracts.py` (Sensor and action data contracts)
- `experiments/phase4j/isolated_supervisor_opt.py` (Isolated experimental wrapper)
- `experiments/phase4j/phase4jb_robustness_eval.py` (Phase 4J-B evaluation engine)
- `tests/test_phase4jb_safety_eval.py` (Phase 4J-B test suite)

### 5.2 Checkpoint Identifiers (SHA-256 Hashes)
- Canonical BC Baseline (`models/checkpoints/bc_policy_best.pt`):  
  `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f`
- Canonical PPO Baseline (`models/checkpoints/ppo_policy_best.pt`):  
  `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0`
- Phase 4H BC Candidate (`experiments/phase4h/checkpoints/bc_phase4h_candidate.pt`):  
  `1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b`
- Phase 4H PPO Candidate (`experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt`):  
  `751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd`

### 5.3 Evaluation Seed Record
- **Phase 4J-B Paired Seed Pool:** `2000, 2001, ..., 2049` (50 distinct seeds).
- **Reserved Held-Out Pool:** `90000..90099` (Preserved strictly untouched).

