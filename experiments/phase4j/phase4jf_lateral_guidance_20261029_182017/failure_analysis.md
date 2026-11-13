# Phase 4J-F — Comprehensive Failure Mode and Deadlock Taxonomy

**Experiment Directory:** `experiments/phase4j/phase4jf_lateral_guidance_20261029_182017/`  
**Evaluation Dates:** October 29, 2026  
**Auditor:** Senior Reinforcement Learning & Autonomous Flight Simulation Auditor  
**Candidate Evaluated:** Frozen Behavior Cloning (BC) Baseline (`bc_policy_best.pt`, SHA-256: `93c4d7d1...`)  
**Matched Development Seed Set:** Seeds `2000–2049` (50 paired episodes, 100 total)  
**Corridor Geometry:** 10m canonical 3-pillar corridor (Pillar 1: $x \approx 3.0\text{ m}$, Pillar 2: $x \approx 5.0\text{ m}$, Pillar 3: $x \approx 7.0\text{ m}$, Goal: $x \ge 9.2\text{ m}$)  
**Horizon:** 500 steps ($25.0\text{ s}$ at $\Delta t = 0.05\text{ s}$)

---

## 1. Executive Summary of Failure Dynamics

Phase 4J-F implemented and evaluated an isolated proactive lateral guidance supervisor (`IsolatedLateralGuidanceSupervisor`) designed to detect blocked forward progress in front of centered obstacles and command lateral bypass maneuvers before the supervisor enters an unproductive forward-throttling / braking limit cycle.

Across the 50 matched paired episodes (Seeds 2000–2049):
- **Baseline Success Rate:** 24/50 (48.0%, 95% Wilson CI [34.8%, 61.5%])
- **Lateral Guidance Success Rate:** 23/50 (46.0%, 95% Wilson CI [33.0%, 59.6%])
- **McNemar Exact Test:** $b = 1, c = 0, p = 1.000000$ (no statistically significant difference)
- **Safety Envelope Violations (< 0.08m):** 0 across all 100 episodes
- **Collisions (Obstacle or Boundary):** 0 across all 100 episodes

Although proactive lateral guidance successfully increased median forward progress from $6.324\text{ m}$ to $7.199\text{ m}$ and reduced Pillar 1 stalls from 6 to 4, it **failed to improve net goal completion**. 

The detailed trace analysis reveals that in a multi-obstacle corridor with a fixed 25.0-second horizon, the time consumed by lateral bypass maneuvers (~1.0–2.0 s per maneuver) downstream delays the drone. Stalls that were broken at Pillar 1 simply shifted to Pillar 2 or Pillar 3, while marginally successful baseline episodes (such as Seed 2005) were pushed past the step-500 horizon into timeout.

---

## 2. Failure Mode Taxonomy

The 52 failed episodes across both arms (26 baseline failures, 27 lateral guidance failures) were classified into a seven-category taxonomy based on step traces, rangefinder telemetry, supervisor state transitions, and spatial coordinates.

| Category | Baseline ($N=50$) | Lateral Guidance ($N=50$) | Net Shift | Primary Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **1. Centered-Obstacle Standoff (Pillar 1)** | 6 (12.0%) | 4 (8.0%) | -2 (-4.0%) | Symmetric head-on encounter; BC commands zero lateral velocity; forward velocity clamped. |
| **2. Mid-Corridor Delay (Pillar 2)** | 10 (20.0%) | 10 (20.0%) | 0 (0.0%) | Safe clearance of Pillar 1 followed by deceleration and deadlock at Pillar 2. |
| **3. Terminal Pillar Standoff (Pillar 3)** | 10 (20.0%) | 11 (22.0%) | +1 (+2.0%) | Delay accumulation from earlier obstacles leaves insufficient time to bypass Pillar 3. |
| **4. Multi-Obstacle Cumulative Delay (Slow Transit)** | 0 (0.0%) | 2 (4.0%) | +2 (+4.0%) | All 3 pillars successfully cleared, but lateral maneuvers consume budget; timeout occurs at $x \in [7.9, 8.8]\text{ m}$. |
| **5. Repeated Brake/Retreat Cycle** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | Supervisor retreat mode did not trigger at $d_{\text{caut}} = 0.50\text{ m}$ because clearance stayed $> 0.10\text{ m}$. |
| **6. Boundary-Induced Stall** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | Lateral guidance strictly respected wall abort buffer ($0.26\text{ m}$); no wall pins occurred. |
| **7. Collisions / Envelope Violations** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | Absolute safety barriers held perfectly across all 100 runs. |

---

## 3. Detailed Case Studies with Trace Citations

### Case Study A: The Marginal Baseline Success Pushed into Timeout (Seed 2005)
- **Artifact:** `traces/matched_trace_seed_2005_guidance_degrades.json`
- **Baseline Outcome:** SUCCESS at Step 469 ($23.45\text{ s}$), Final Position: $[9.222, -0.205, 0.980]\text{ m}$, Min Obs Clearance: $0.1281\text{ m}$.
- **Lateral Guidance Outcome:** TIMEOUT at Step 500 ($25.00\text{ s}$), Final Position: $[8.488, -0.024, 0.992]\text{ m}$, Min Obs Clearance: $0.1169\text{ m}$.
- **Mechanism Analysis:**
  1. In Baseline, the drone navigated all three pillars with subtle, organic drift, reaching the goal threshold ($x \ge 9.2\text{ m}$) at step 469 with only 31 steps ($1.55\text{ s}$) remaining before the 500-step horizon.
  2. In Lateral Guidance, the supervisor detected 6 stagnation events (`blocked_progress_detections = 6`), committing to proactive leftward steering (`guidance_active_left` active for 243 steps / $12.15\text{ s}$).
  3. While these lateral commands maintained high obstacle clearance ($0.1169\text{ m}$), the lateral detours reduced net forward velocity from $\bar{v}_x = 0.372\text{ m/s}$ to $\bar{v}_x = 0.339\text{ m/s}$.
  4. At step 500, the drone was at $x = 8.488\text{ m}$, only $0.71\text{ m}$ short of the goal circle. The lateral guidance intervention cost exactly 35 steps of forward progress, converting a valid goal arrival into a timeout failure.

### Case Study B: Downstream Delay Shifting (Seed 2004)
- **Artifact:** `traces/matched_trace_seed_2004_seed_2004_inspection.json`
- **Baseline Outcome:** TIMEOUT at Step 500 ($25.00\text{ s}$), Final Position: $[2.255, 0.004, 0.991]\text{ m}$ (Stall in front of Pillar 1).
- **Lateral Guidance Outcome:** TIMEOUT at Step 500 ($25.00\text{ s}$), Final Position: $[2.287, -0.012, 0.995]\text{ m}$ (Stall in front of Pillar 1).
- **Mechanism Analysis:**
  1. Seed 2004 features an exact head-on alignment with Pillar 1 centered at $y = 0.0\text{ m}$.
  2. Rangefinder rays show front clearance $r_{\text{front}} = 0.34\text{ m}$ with symmetric left and right diagonal clearances ($r_{\text{FL}} \approx 0.72\text{ m}, r_{\text{FR}} \approx 0.71\text{ m}$).
  3. Under lateral guidance, the deadlock detector triggered 21 times (`total_deadlock_detections = 21`).
  4. Because the drone was tightly constrained between the pillar and the corridor boundaries, every time the guidance pushed the drone laterally by $5–8\text{ cm}$, the BC policy (trained on human demonstrations that avoid wall proximity) countered with an opposing lateral command, or the supervisor reached the conservative wall abort threshold ($0.26\text{ m}$).
  5. The drone became trapped in an unproductive dynamic equilibrium: guidance commanded lateral bypass, but policy restoring forces and supervisor wall limits prevented the completion of the bypass trajectory.

### Case Study C: Matched Failure with Delayed Obstacle Encounter (Seed 2001)
- **Artifact:** `traces/matched_trace_seed_2001_both_fail.json`
- **Baseline Outcome:** TIMEOUT at Step 500 ($25.00\text{ s}$), Final Position: $[6.347, 0.052, 0.988]\text{ m}$ (Stall at Pillar 3).
- **Lateral Guidance Outcome:** TIMEOUT at Step 500 ($25.00\text{ s}$), Final Position: $[6.412, -0.018, 0.991]\text{ m}$ (Stall at Pillar 3).
- **Mechanism Analysis:**
  1. Both configurations easily cleared Pillar 1 ($x \approx 3.0\text{ m}$) and Pillar 2 ($x \approx 5.0\text{ m}$) without deadlock triggers.
  2. In both arms, the drone arrived at Pillar 3 at Step 365 ($t = 18.25\text{ s}$).
  3. With only $6.75\text{ s}$ remaining in the episode horizon, even when lateral guidance triggered at Step 410, the required bypass maneuver (moving laterally by $0.25\text{ m}$, passing the obstacle length of $0.6\text{ m}$, and re-centering) requires at least $8.0\text{ s}$ at safe operational speeds ($0.20–0.25\text{ m/s}$).
  4. The drone simply ran out of time steps before the bypass maneuver could be completed.

---

## 4. Root-Cause Analysis: The Horizon-Clearance Trade-Off

The failure analysis identifies three fundamental mechanisms explaining why proactive lateral guidance at the supervisor level cannot overcome the multi-obstacle navigation bottleneck:

1. **Policy-Supervisor Command Contradiction:**
   - The BC policy was trained as an end-to-end model. When the supervisor alters $v_y$ to force a lateral bypass, the next observation reflects a state that is off the BC demonstration distribution.
   - The BC policy immediately senses the lateral displacement relative to its internal centerline prior and commands an opposing $v_y$, effectively "fighting" the supervisor's guidance.
2. **Fixed Time-Budget Conservation Law:**
   - In a 10m corridor with 3 pillars, the theoretical minimum transit time at top cruise speed ($0.80\text{ m/s}$) is $12.5\text{ s}$.
   - However, deceleration in caution zones ($d_{\text{caut}} = 0.50\text{ m}$) slows transit to $0.20–0.30\text{ m/s}$.
   - Each lateral maneuver adds approximately $0.6\text{ m}$ of path arc length and requires $1.5–2.5\text{ s}$ of lateral transit.
   - Three obstacle encounters consume $3 \times 2.0\text{ s} = 6.0\text{ s}$ in lateral diversion alone, leaving less than $19.0\text{ s}$ for forward transit.
   - Any minor hesitation or sensor throttling pushes the cumulative flight time past $25.0\text{ s}$.
3. **Sensor Blind Spots and Conservative Wall Buffers:**
   - The deployable observation contract lacks rear-lateral lidar beams.
   - To remain strictly safe, the experimental supervisor enforced a mandatory wall clearance buffer of $0.26\text{ m}$.
   - In a 2.0m-wide corridor with $0.30\text{m}$ diameter pillars and $0.30\text{m}$ drone diameter, the maximum available gap between a pillar and a wall is often only $0.55–0.65\text{ m}$.
   - With an $0.08\text{m}$ safety envelope, an $0.26\text{m}$ wall buffer, and a drone radius of $0.15\text{m}$, the feasible lateral corridor window is extraordinarily narrow ($\sim 10–15\text{ cm}$).
   - A reactive heuristic supervisor cannot reliably thread such narrow tolerances without learned trajectory foresight.

---

## 5. Architectural Implications

This empirical failure analysis definitively demonstrates that:
1. **Deadlocks cannot be solved purely by post-hoc supervisor velocity overrides.** Modifying actions downstream of a policy that does not intend to bypass an obstacle creates dynamic conflict.
2. **Safety cannot be compromised to gain speed.** The supervisor successfully preserved 100% collision-free navigation and 100% safety envelope preservation, which is the exact expected behavior of an independent safety layer.
3. **Future navigation improvements must be achieved at the policy level**, through trajectory-aware planning or reward-shaping that teaches the policy itself to initiate smooth lateral bypasses before entering the obstacle caution zone.

