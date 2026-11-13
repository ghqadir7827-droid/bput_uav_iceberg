# AUDIT LIMITATIONS, SIMULATION FIDELITY & UNANSWERED QUESTIONS
**Audit Artifact ID:** `reports/master_forensic_audit_20261030_130812/AUDIT_LIMITATIONS.md`  
**Timestamp:** 2026-10-30T13:30:00+05:30  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Verification Level:** VERIFIED (Forensic Boundary Analysis)

---

## 1. Overview

An audit is only as defensible as its explicit boundary conditions. This document details:
1. What could not be verified from the repository artifacts alone.
2. Unanswered engineering and hardware specifications.
3. The deliberate sealing of the held-out evaluation seed pool (`90000–90099`).
4. Physical aerodynamic phenomena absent from the simulation environment.

---

## 2. Unverified Systems & Missing Hardware Artifacts

### 2.1 Physical Flight Controller Firmware & Protocol Parameters
- **Status:** **UNKNOWN / NOT DOCUMENTED IN REPO**
- **Details:** The repository specifies the target flight controller as the *RadioLink CrossFlight*. However, the repository contains no configuration files (`.param` or `params.pck`), no documentation on whether it runs ArduPilot Copter (v4.3+) or PX4 Autopilot, and no documentation on the serial baud rate (e.g., 57600 vs 921600 baud) or configured stream rates (`SRx_POSITION`, `SRx_EXTRA1`).
- **Impact:** Communication timing and GUIDED mode arming protocols cannot be independently verified without the physical flight controller.

### 2.2 Concrete Physical Sensor Specifications
- **Status:** **UNKNOWN / ABSTRACTED**
- **Details:** The simulation models an ideal 32-beam 2D rangefinder with $5.0$m range. The repository does not specify which physical sensor will be mounted on the drone:
  - Is it a spinning mechanical 2D LIDAR (e.g., Slamtec RPLIDAR C1 / A1)?
  - Is it a solid-state multi-zone ToF array (e.g., STMicroelectronics VL53L5CX)?
  - Is it an array of discrete single-point 1D rangefinders (e.g., TFmini Plus)?
- **Impact:** Sensor update frequency (5–10 Hz mechanical vs 20 Hz sim), beam divergence, multi-path reflections, and sunlight blinding vulnerabilities cannot be evaluated.

---

## 3. The Held-Out Seed Benchmark (`90000–90099`) Boundary

### 3.1 Verification State
- **BC Baseline (`bc_policy_best.pt`):** Evaluated on `90000–90099` ($N=100$) in Phase 4J-D. Result: **47.0% success (47/100)**, 0 collisions, 53 timeouts.
- **Phase 5 & Phase 6 Guidance Candidates:** **NEVER EVALUATED** on `90000–90099`.

### 3.2 Methodological Justification
During Phases 5A, 5B, 5C, 6A, and 6B, the audit protocol strictly forbade executing candidates against the held-out pool `90000–90099` to prevent:
1. Adaptive overfitting (tweaking thresholds until the test set passes).
2. P-hacking and metric contamination.
3. Degradation of the test set's utility as a final blind verification benchmark.

### 3.3 Audit Caveat
The reported **84.0% success rate (42/50)** for the Phase 6A Production Guidance candidate is **strictly a Development Benchmark result** across seeds `2000–2049`. While the candidate is completely uncontaminated, its generalizability to the 100 held-out seeds remains theoretically unconfirmed until authorized for unsealing.

---

## 4. Simulation Fidelity & Aerodynamic Sim-to-Real Gaps

The simulation environment (`simulation/environment.py`) utilizes a simplified point-mass kinematic model with first-order linear drag. The following critical real-world physical dynamics are completely unmodeled:

### 4.1 Aerodynamic Wall Attraction (Bernoulli / Venturi Effect)
- **Physics:** When a multirotor operates near a solid vertical surface (e.g., a corridor wall), the high-velocity rotor downwash creates a localized low-pressure zone between the vehicle and the wall. This generates a lateral aerodynamic suction force pulling the UAV toward the wall.
- **Simulation Reality:** In Phase 6A, the minimum recorded wall clearance was **$0.1942$m ($19.4$ cm)**.
- **Real-Flight Risk:** In physical flight, entering within $20$ cm of a wall at speed almost certainly triggers Bernoulli wall suction, potentially overpowering the lateral roll authority and causing a wall strike.

### 4.2 Rotor Downwash & Ground Effect
- **Physics:** Within 1 rotor diameter of the floor ($z < 0.4$m), ground effect creates high pressure and turbulence, requiring significant throttle reduction to maintain altitude.
- **Simulation Reality:** The simulation models altitude kinematics identically at all heights ($z \in [0.2, 2.5]$m).

### 4.3 Sensor Blind-Spots & Vertical Obstacles
- **Physics:** Real obstacles may be hung from ceilings, protrude from walls at varying heights, or have non-vertical geometries.
- **Simulation Reality:** The 32-beam planar sensor casts rays strictly in the horizontal plane at the drone's current altitude ($z_{uav}$). Any obstacle that does not intersect the exact horizontal slicing plane is completely invisible to the drone.

