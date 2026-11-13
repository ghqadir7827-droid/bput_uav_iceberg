# Phase 4I Failure Taxonomy: Multi-Obstacle Autonomous Navigation

## Executive Summary

Phase 4I-D identifies, categorizes, and provides evidence for the failure mechanisms observed across the diagnostic suite and development seeds.

Rather than treating "timeout" or "low return" as an undifferentiated failure, the diagnostic traces isolate five distinct failure classes with specific spatial points of onset, controller-supervisor interactions, and recovery dynamics.

---

## 1. Complete Failure Mode Taxonomy

```
+-----------------------------------------------------------------------------------------+
|                                    FAILURE TAXONOMY                                     |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [Class 1: Cumulative Throttle Timeout]  --> Continuous forward progress, but           |
|                                              supervisor caution zones (d < 0.80m)       |
|                                              throttle speed to ~0.15 m/s.               |
|                                              Onset: X in [4.5, 6.5]m (Past Pillar 2).   |
|                                                                                         |
|  [Class 2: Unilateral Mode Lock Stalling] --> Policy insists on rightward bypass even   |
|                                              when right channel is blocked.             |
|                                              Onset: X in [2.0, 2.5]m (Before Pillar 1). |
|                                                                                         |
|  [Class 3: Symmetric Bimodal Collapse]    --> Centered obstacle produces vy = 0.00 m/s; |
|                                              drone flies straight into front stop.      |
|                                              Onset: X in [2.6, 2.8]m.                   |
|                                                                                         |
|  [Class 4: Dynamic Chicane Corner Clip]   --> Rapid alternating S-turns between pillars |
|                                              graze safety buffer or wall perimeter.     |
|                                              Onset: X in [3.8, 5.2]m.                   |
|                                                                                         |
|  [Class 5: Terminal Braking Overshoot]   --> Drone reaches destination sphere but fails |
|                                              v_term <= 0.35 m/s arrival gate.           |
|                                              Onset: X >= 9.15m.                         |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Granular Class Descriptions and Physical Traces

### Class 1: Cumulative Throttle Timeout (The Dominant Multi-Obstacle Failure)
- **Primary Mechanism:** The drone makes continuous, collision-free forward progress and clears internal obstacles, but forward velocity $v_x$ is throttled by the supervisor's forward caution envelope ($d_{\text{front}} < 0.80\text{ m}$) down to $0.12 - 0.20\text{ m/s}$.
- **Point of Onset:** Begins at $X = 2.20\text{ m}$ (Pillar 1 caution entry) and compounds through Pillar 2 ($X = 4.20\text{ m}$) and Pillar 3 ($X = 6.20\text{ m}$).
- **Representative Trace (Seed 1000, `Expert_APF_Reference`):**
  - Steps 0–45 ($X = 0.5 \to 2.2\text{ m}$): Cruise at $0.55\text{ m/s}$. Mode: `unconstrained_forward`.
  - Steps 46–180 ($X = 2.2 \to 3.4\text{ m}$): Bypasses Pillar 1. Throttled to $v_x = 0.16\text{ m/s}$. Mode: `supervisor_constrained_forward`.
  - Steps 181–225 ($X = 3.4 \to 4.2\text{ m}$): Inter-pillar gap. Accelerates to $0.38\text{ m/s}$.
  - Steps 226–370 ($X = 4.2 \to 5.4\text{ m}$): Bypasses Pillar 2. Throttled to $v_x = 0.14\text{ m/s}$.
  - Steps 371–420 ($X = 5.4 \to 6.2\text{ m}$): Inter-pillar gap.
  - Steps 421–500 ($X = 6.2 \to 7.05\text{ m}$): Enters Pillar 3 caution. Episode terminates at step 500 timeout at $X = 7.05\text{ m}$.
- **Controllers Affected:**
  - `Expert_APF_Reference`: 10/10 multi-pillar runs (100%).
  - `BC_Phase4H_Candidate`: 8/10 multi-pillar runs (80%).
  - `Arm_B_BC_Baseline`: 6/10 multi-pillar runs (60%).
- **Verification:** This is **not** a navigation failure, but an operational mismatch between the 500-step horizon budget and the consecutive caution zones.

---

### Class 2: Unilateral Mode Lock Stalling (PPO Asymmetry)
- **Primary Mechanism:** PPO broke symmetry during early RL training, converging to a rigid rightward avoidance bias ($v_y < 0$). When an obstacle is shifted to the right ($Y_{\text{obs}} < 0$), the right channel is impassable ($< 0.50\text{ m}$). The policy refuses to steer left into the open $1.2\text{ m}$ channel, gets wedged between the pillar and right wall, and triggers supervisor critical braking ($v_x = 0, v_y = 0$).
- **Point of Onset:** $X \approx 2.05 - 2.15\text{ m}$ (in front of Pillar 1).
- **Representative Trace (Seed 1004, `Arm_A_PPO_Baseline`):**
  - Steps 0–35 ($X = 0.5 \to 2.02\text{ m}$): Drone drifts toward right wall ($Y = -0.32\text{ m}$).
  - Step 36 ($X = 2.02\text{ m}$): Pillar 1 is at $Y = -0.28\text{ m}$. Gap to right wall is $0.42\text{ m}$. Policy commands $v_y = -0.35\text{ m/s}$.
  - Step 40: Supervisor activates lateral critical clamp (`right_obs = 0.28m < 0.32m`). Clamps $v_y = 0.0$.
  - Steps 41–500: Drone spends **24.8% of the episode completely stopped** ($v_x = 0.0, v_y = 0.0$). Stalled timer triggers reverse retreat, but policy immediately commands forward-right into the pillar again. Deadlocks at $X = 2.02\text{ m}$.
- **Controllers Affected:**
  - `Arm_A_PPO_Baseline`: 8/10 multi-pillar runs (80%), 100% of right-shifted single pillar runs.
  - `PPO_Phase4H_Candidate`: 6/10 multi-pillar runs (60%).

---

### Class 3: Centered Obstacle Multimodal Symmetry Collapse
- **Primary Mechanism:** Supervised MSE regression on symmetric observations with conflicting left and right evasion demonstrations causes the policy to output the expected mean: $v_y \approx 0.00\text{ m/s}$. The drone commands zero lateral velocity and flies straight into the obstacle until halted by supervisor critical braking at $X = 2.81\text{ m}$.
- **Point of Onset:** $X \approx 2.60 - 2.81\text{ m}$.
- **Representative Trace (Single Centered Obstacle, `Arm_B_BC_Baseline`):**
  - Step 1 to 55 ($X = 0.50 \to 2.20\text{ m}$): $v_y = +0.002\text{ m/s}$. Flying straight along centerline.
  - Step 60 ($X = 2.50\text{ m}$): Front clearance drops below $0.55\text{ m}$. Supervisor slows $v_x$ to $0.20\text{ m/s}$. Policy still commands $v_y = -0.0007\text{ m/s}$.
  - Step 85 ($X = 2.81\text{ m}$): Front clearance reaches critical threshold ($0.25\text{ m}$). Supervisor commands $v_x = 0.0\text{ m/s}$.
  - Steps 86–500: Drone remains motionless at $(2.81, 0.00, 1.00)\text{ m}$ until timeout.
- **Controllers Affected:**
  - `Arm_B_BC_Baseline`: 100% on centered single obstacle.
- **Status in Phase 4H/4I:** **Fully resolved** in `BC_Phase4H_Candidate` via consistent tie-breaking demonstrations.

---

### Class 4: Dynamic Chicane Corner Clip (Collision Failure)
- **Primary Mechanism:** When negotiating tightly spaced obstacles with alternating lateral offsets (e.g., Pillar 1 at $Y = +0.25\text{ m}$, Pillar 2 at $Y = -0.25\text{ m}$), the drone must transition rapidly across the corridor width ($\Delta y = 0.90\text{ m}$). If the policy initiates the return maneuver late, the UAV airframe ($r = 0.15\text{ m}$) clips the corner of the obstacle or outer wall.
- **Point of Onset:** $X \approx 3.8 - 4.5\text{ m}$ (entrance to Pillar 2).
- **Representative Trace (Seed 1005, `BC_Phase4H_Candidate`):**
  - Step 0–80 ($X = 0.5 \to 2.5\text{ m}$): Clean rightward bypass of Pillar 1 ($Y = +0.28\text{ m}$).
  - Step 90 ($X = 2.81\text{ m}$): Lateral velocity $v_y = -0.38\text{ m/s}$ reaches peak right offset ($Y = -0.68\text{ m}$).
  - Step 115 ($X = 3.65\text{ m}$): Pillar 2 is positioned at $Y = -0.32\text{ m}$. The drone is currently on the right wall and must swing back to the left channel. The policy commands $v_y = +0.35\text{ m/s}$, but linear momentum carries the drone within the collision boundary of Pillar 2 at step 154 ($X = 2.81\text{ m}$, obstacle clearance $< 0.00\text{ m}$).
- **Controllers Affected:**
  - `BC_Phase4H_Candidate`: 1/10 dev seeds (10%), 7/100 held-out seeds (7%).

---

### Class 5: Terminal Approach Braking Overshoot
- **Primary Mechanism:** Drone reaches the terminal destination region ($X \ge 9.15\text{ m}$), but fails the terminal velocity condition ($v_{\text{term}} \le 0.35\text{ m/s}$), crashing into the end-of-corridor boundary or overshooting.
- **Point of Onset:** $X \ge 9.0\text{ m}$.
- **Representative Trace (Clear Corridor, `PPO_Phase4H_Candidate`):**
  - The policy races forward at $v_x = 0.58\text{ m/s}$. When entering the goal region at step 370 ($X = 9.20\text{ m}$), $v_x = 0.52\text{ m/s}$. It fails to decelerate before passing the exit plane, terminating with `crash_boundary` at $X = 9.77\text{ m}$ ($v_{\text{term}} = 0.521\text{ m/s}$).
- **Controllers Affected:**
  - `PPO_Phase4H_Candidate`: Clear corridor and goal-approach probe.

---

## 3. Summary Distribution Across Diagnostic Evaluations

| Failure Class | Primary Mechanism | Expert APF | BC Baseline | BC Phase 4H Cand | PPO Baseline | PPO Phase 4H Cand |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Class 1 (Cumulative Throttle)** | Successive caution zone slowdowns | **10/10 (100%)** | 6/10 (60%) | **8/10 (80%)** | 1/10 (10%) | 3/10 (30%) |
| **Class 2 (Unilateral Mode Lock)**| Rightward drift into constricted wall | 0/10 (0%) | 2/10 (20%) | 0/10 (0%) | **8/10 (80%)** | **6/10 (60%)** |
| **Class 3 (Multimodal Averaging)**| Symmetrical centered obstacle deadlock| 0/10 (0%) | 2/10 (20%) | **0/10 (0%)** | 0/10 (0%) | 0/10 (0%) |
| **Class 4 (Chicane Corner Clip)** | Rapid S-turn wall/obstacle contact | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) | 0/10 (0%) | 0/10 (0%) |
| **Class 5 (Terminal Overshoot)** | Arrival speed $> 0.35$ m/s | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |
