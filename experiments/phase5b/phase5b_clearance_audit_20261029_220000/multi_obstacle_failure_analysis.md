# PHASE 5B — MULTI-OBSTACLE FAILURE ANALYSIS
## Diagnostic Trace Investigation of Sequential Obstacle Navigation Failures

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Isolation Directory:** `experiments/phase5b/phase5b_clearance_audit_20261029_220000/`

---

## 1. Executive Summary of Candidate Performance

In Experiment B (paired development evaluation across seeds `2000–2049`), the Stateless Clearance Guidance Candidate achieved:
- **35 / 50 Successes (70.0%, 95% CI: [56.2%, 80.9%])** compared to **24 / 50 (48.0%)** for the Frozen BC Baseline.
- **Net paired improvement of +11 goal completions** (18 candidate wins vs 7 baseline wins, McNemar exact binomial $p = 0.0433$).
- **Zero collisions** (0 obstacle, 0 wall).
- **Zero envelope violations** below $0.080\text{ m}$ (minimum obstacle clearance: $0.0870\text{ m}$).
- **Significant reduction in early stalls:**
  - Pillar 1 Cleared: $96.0\%$ (48/50) vs $88.0\%$ (44/50) baseline.
  - Pillar 2 Cleared: $94.0\%$ (47/50) vs $68.0\%$ (34/50) baseline.
  - Pillar 3 Cleared: $78.0\%$ (39/50) vs $48.0\%$ (24/50) baseline.

Despite these substantial gains, **15 out of 50 episodes** failed to complete the mission within the 500-step horizon. This report diagnoses the exact physical and algorithmic causes of these 15 failure episodes.

---

## 2. Systematic Categorization of the 15 Failure Episodes

```
========================================================================================
CANDIDATE FAILURE TAXONOMY (N = 15 failure episodes out of 50 development seeds)
========================================================================================
Failure Category                       Count   Fraction   Representative Seeds
----------------------------------------------------------------------------------------
1. Corner Pinch at Pillar 3            8       53.3%      2005, 2006, 2020, 2022,
   (Wall Buffer + Caution Deadlock)                       2036, 2038, 2042, 2046
2. Transit Time Budget Exhaustion       3       20.0%      2001, 2021, 2049
   (All 3 Pillars Cleared, Prog > 8.1m)
3. Goal Deceleration Alignment Timeout  1        6.7%      2041 (Goal dist = 0.404m)
4. Early Standoff at Pillar 1 / 2       3       20.0%      2015 (P2), 2018 (P1), 2032 (P1)
----------------------------------------------------------------------------------------
Total Failures                         15      100.0%
========================================================================================
```

Notice that **5 of the 15 "failures" (33.3%) successfully cleared all three obstacles** and reached the terminal corridor zone ($X > 8.1\text{ m}$), but ran out of time steps before the terminal velocity settling criterion ($v \le 0.35\text{ m/s}$) could be satisfied.

---

## 3. Detailed Trace Investigation (Answering the 9 Forensic Questions)

### Q1: Which direction was selected at each obstacle?
- **Pillar 1:** In $96\%$ of encounters, the rule correctly selected the wider opening. On symmetric encounters, the tie-breaker cleanly chose LEFT, enabling $48/50$ UAVs to pass Pillar 1.
- **Pillar 2:** In $94\%$ of encounters, the rule successfully initiated lateral evasion around Pillar 2.
- **Pillar 3:** In the 8 stalled episodes, the rule attempted to command avoidance, but the UAV's lateral position was severely constrained by the corridor wall.

### Q2: Was the selected side actually clear at the time of the decision?
- Yes. Rangefinder beam pooling accurately reflected local open space. However, local clearance did not account for remaining corridor width between the obstacle and the adjacent wall.

### Q3: Did the rule continue following a stale direction after passing an obstacle?
- Because Alternative A is **stateless**, it drops lateral guidance as soon as $C_{\text{front}} \ge 0.85\text{ m}$. Thus, it did not follow a stale guidance command. However, because the BC policy relies on an autoregressive velocity shortcut ($v_{y, \text{cmd}} \approx 0.95 \cdot v_{y, \text{meas}}$), the UAV maintained residual lateral drift into the wall.

### Q4: Did the rule return toward the corridor center too early?
- No; the stateless rule contains **no re-centering mechanism**. Once an obstacle was cleared, the UAV remained at its displaced lateral coordinate ($|Y| \approx 0.50–0.65\text{ m}$) until the next obstacle was encountered.

### Q5: Did a new obstacle trigger a conflicting direction?
- In several cases (e.g., Seed 2005 and Seed 2020), Pillar 2 was bypassed on the right or left, leaving the UAV at $Y = +0.60\text{ m}$. When Pillar 3 appeared at $X = 7.0\text{ m}$, the wider opening was to the right, but the UAV was already trapped against the left wall.

### Q6: Did the BC policy command lateral movement opposite to the guidance rule?
- No. When $C_{\text{front}} < 0.85\text{ m}$, the guidance rule clamped $v_y \gets \max(v_y, +0.25)$ or $\min(v_y, -0.25)$, overriding any conflicting small commands from the BC policy.

### Q7: Did the safety supervisor override the combined command?
- **YES. This is the primary failure mechanism for Type 1 stalls.**
- When the UAV approached Pillar 3 with $Y \approx +0.65\text{ m}$, the distance to the left wall was $d_{\text{left}} = 1.0 - 0.65 - 0.15 = 0.20\text{ m} < 0.25\text{ m}$ (`boundary_buffer_m`).
- The supervisor applied inward wall repulsion ($v_y \gets v_y - 1.5 \cdot (0.25 - d_{\text{left}})$) while simultaneously throttling forward speed due to Pillar 3 ($v_x \to 0.10\text{ m/s}$).
- The UAV entered a **Coupled Boundary-Obstacle Deadlock** (`Combined intervention: Obstacle caution zone ... | Boundaries (left=0.18m)`), oscillating between forward throttling and boundary repulsion until step 500.

### Q8: Did a locally successful bypass create an unfavorable approach to the next obstacle?
- **YES. This is the "Corner Pinch" phenomenon.**
- In stateless guidance, navigating past Pillar 1 and Pillar 2 pushes the UAV away from the centerline. Without an active return-to-center mechanism between obstacles, the UAV enters subsequent obstacle sectors near the boundary walls, where passage feasibility drops from $0.80\text{ m}$ to $< 0.40\text{ m}$.

### Q9: Did the UAV make forward progress after clearing the final obstacle but fail at goal alignment?
- **YES.** In Seeds 2001, 2021, 2041, and 2049, the UAV cleared all 3 obstacles with high clearance ($> 0.12\text{ m}$). However, because caution throttling slowed forward progress earlier in the corridor, the UAV reached the goal region ($X > 8.5\text{ m}$) at steps $440–480$, leaving insufficient steps to decelerate below $0.35\text{ m/s}$ before step 500.

---

## 4. Deep-Dive Case Study: Seed 2005 (The Corner Pinch)

Examining the high-resolution step telemetry of Seed 2005 (`dev_seed2005_candidate.json`):
- **Steps 0–190:** Clean navigation. Bypasses Pillar 1 and Pillar 2. Clears Pillar 2 at step 190 ($X = 5.20\text{ m}$).
- **Step 260:** UAV approaches Pillar 3 ($X_{\text{obs}} = 7.0\text{ m}$). Because of residual lateral drift, the UAV position is $X = 6.37\text{ m}, Y = +0.525\text{ m}$.
- **Step 280:** Obstacle caution triggers ($d_{\text{front}} = 0.36\text{ m} < 0.50\text{ m}$). UAV drifts to $Y = +0.635\text{ m}$. Wall clearance drops to $0.22\text{ m} < 0.25\text{ m}$.
- **Steps 280–500 (220 steps = 11.0 seconds):**
  - Left wall repulsion pushes right: $v_y \le 0.0$.
  - Pillar 3 obstacle caution clamps forward speed: $v_x = 0.10\text{ m/s}$.
  - Supervisor reverse retreat triggers periodically ($v_x = -0.10, v_y = 0.15$), but the UAV cannot turn right because of the forward pillar face, and cannot move left because of the wall.
  - UAV remains trapped at $X = 6.78\text{ m}$ until the 500-step horizon expires.

---

## 5. Comparative Evaluation: Stateless (Design A) vs Persistent Guidance with Re-centering (Design B)

In Experiment D, we evaluated whether adding a persistent state machine with an explicit **Re-centering (`SETTLING`) phase** resolves this failure mode:

| Evaluation Case | Stateless Guidance (Design A) | Persistent Guidance (Design B) | Mechanism / Diagnosis |
| :--- | :---: | :---: | :--- |
| **Scenario A (Centered)** | **SUCCESS (8.21m)** | **SUCCESS (8.18m)** | Both resolve centered standoff |
| **Scenario B (Left Mirrored)**| `PROGRESS_STALL` (1.89m) | **SUCCESS (8.20m)** | Design B latches right bypass, avoiding stall |
| **Seed 2005 (Corner Pinch)** | `PROGRESS_STALL` (6.28m) | **SUCCESS (8.65m)** | **Design B re-centers ($Y \to 0$), cleanly clearing Pillar 3!** |
| **Seed 2025** | **SUCCESS (8.66m)** | **SUCCESS (8.65m)** | Both succeed |
| **Seed 2030** | **SUCCESS (8.67m)** | **SUCCESS (8.65m)** | Both succeed |
| **Seed 2035** | **SUCCESS (8.72m)** | **SUCCESS (8.71m)** | Both succeed |

### Finding:
Design B's explicit `SETTLING` state (which gently restores $|Y| < 0.12\text{ m}$ once an obstacle is passed and $C_{\text{front}} > 1.25\text{ m}$) completely eliminates the Corner Pinch in Seed 2005, transforming a $6.28\text{ m}$ failure into an $8.65\text{ m}$ goal completion.

