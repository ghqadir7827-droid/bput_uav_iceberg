# Phase 4J-A3 Expert Comparison Reproduction Report

## Executive Summary

Phase 4J-A3 independently reproduces the comparison between the analytical expert controller (`ScriptedAPFExpertController`) operating with the canonical `SafetySupervisor` enabled versus operating with the supervisor disabled (diagnostic counterfactual).

Evaluated across the 10 development seeds (`1000..1009`) under identical PyBullet physics, geometry, action limits, and goal-success conditions, the empirical results confirm:
1. **Config A (With Canonical Supervisor):** **0/10 (0.0%) goal successes.** All 10 episodes terminate with timeout at step 500, achieving an average forward progress of $5.68\text{ m}$.
2. **Config B (Supervisor Disabled Counterfactual):** **10/10 (100.0%) goal successes.** All 10 episodes complete cleanly with an average step count of **$393.1\text{ steps}$ ($19.66\text{ s}$)**, zero collisions, safe obstacle clearances ($\ge 0.210\text{ m}$), and an average terminal arrival speed of **$0.170\text{ m/s}$**.

---

## 1. Controlled Experimental Setup

- **Controller:** `ScriptedAPFExpertController` (analytical potential fields with tangential circulation and smooth touchdown braking).
- **Seeds:** `1000..1009` (10 development seeds, strictly excluding held-out sets).
- **Corridor Geometry:** 10m benchmark ($10.0\text{ m} \times 2.0\text{ m} \times 2.0\text{ m}$, start $X=0.5\text{ m}$, goal $X=9.5\text{ m}$).
- **Obstacle Placement:** 3 procedural pillars with randomized lateral offsets $Y \in [-0.35, +0.35]\text{ m}$.
- **Success Criteria:** Euclidean distance to goal center $\le 0.35\text{ m}$ AND terminal speed $v_{\text{term}} \le 0.35\text{ m/s}$.
- **Execution Script:** `experiments/phase4j/reproduce_expert.py`.
- **Saved Machine-Readable Data:** `experiments/phase4j/expert_reproduction_metrics.json`.

---

## 2. Seed-by-Seed Empirical Comparison

| Seed | Config A: Canonical Supervisor ($d_{\text{caut}}=0.80\text{ m}$) | Config B: Supervisor Disabled (Counterfactual) | Difference ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **1000** | Steps: 500, Status: `timeout`, Fwd: 5.05m, MinObs: 0.327m | **Steps: 397 (19.85s), Status: `success`, Fwd: 8.66m, MinObs: 0.210m** | **-103 steps (-5.15s), Goal Reached** |
| **1001** | Steps: 500, Status: `timeout`, Fwd: 5.44m, MinObs: 0.366m | **Steps: 400 (20.00s), Status: `success`, Fwd: 8.66m, MinObs: 0.394m** | **-100 steps (-5.00s), Goal Reached** |
| **1002** | Steps: 500, Status: `timeout`, Fwd: 4.98m, MinObs: 0.376m | **Steps: 403 (20.15s), Status: `success`, Fwd: 8.65m, MinObs: 0.395m** | **-97 steps (-4.85s), Goal Reached** |
| **1003** | Steps: 500, Status: `timeout`, Fwd: 5.91m, MinObs: 0.440m | **Steps: 371 (18.55s), Status: `success`, Fwd: 8.66m, MinObs: 0.323m** | **-129 steps (-6.45s), Goal Reached** |
| **1004** | Steps: 500, Status: `timeout`, Fwd: 6.80m, MinObs: 0.494m | **Steps: 353 (17.65s), Status: `success`, Fwd: 8.66m, MinObs: 0.494m** | **-147 steps (-7.35s), Goal Reached** |
| **1005** | Steps: 500, Status: `timeout`, Fwd: 5.86m, MinObs: 0.270m | **Steps: 431 (21.55s), Status: `success`, Fwd: 8.65m, MinObs: 0.233m** | **-69 steps (-3.45s), Goal Reached** |
| **1006** | Steps: 500, Status: `timeout`, Fwd: 5.03m, MinObs: 0.209m | **Steps: 394 (19.70s), Status: `success`, Fwd: 8.66m, MinObs: 0.219m** | **-106 steps (-5.30s), Goal Reached** |
| **1007** | Steps: 500, Status: `timeout`, Fwd: 5.93m, MinObs: 0.356m | **Steps: 396 (19.80s), Status: `success`, Fwd: 8.66m, MinObs: 0.347m** | **-104 steps (-5.20s), Goal Reached** |
| **1008** | Steps: 500, Status: `timeout`, Fwd: 5.64m, MinObs: 0.366m | **Steps: 431 (21.55s), Status: `success`, Fwd: 8.66m, MinObs: 0.341m** | **-69 steps (-3.45s), Goal Reached** |
| **1009** | Steps: 500, Status: `timeout`, Fwd: 6.20m, MinObs: 0.405m | **Steps: 355 (17.75s), Status: `success`, Fwd: 8.65m, MinObs: 0.410m** | **-145 steps (-7.25s), Goal Reached** |

---

## 3. Aggregate Statistical Comparison

| Metric | Config A: Canonical Supervisor ($d_{\text{caut}}=0.80\text{ m}$) | Config B: Supervisor Disabled (Counterfactual) | Safety Envelope Compliance |
| :--- | :--- | :--- | :--- |
| **Goal Success Rate** | **0.0% (0/10)** | **100.0% (10/10)** | Successful goal arrival |
| **Mean Duration (Steps / s)**| 500.0 steps ($25.00\text{ s}$) | **393.1 steps ($19.66\text{ s}$)** | **-106.9 steps (-5.34 s)** |
| **Minimum Flight Duration** | 500 steps (Timeout) | **353 steps ($17.65\text{ s}$)** | Fastest clean passage |
| **Maximum Flight Duration** | 500 steps (Timeout) | **431 steps ($21.55\text{ s}$)** | Longest clean passage |
| **Mean Forward Progress** | $5.68\text{ m}$ (Stalled near Pillar 2/3) | **$8.66\text{ m}$ (Arrival sphere)** | Full corridor traversal |
| **Collision Count** | **0 (0.0%)** | **0 (0.0%)** | Zero collisions in both |
| **Mean Min Obstacle Clearance**| $0.351\text{ m}$ | **$0.337\text{ m}$** | Well above $0.08\text{ m}$ threshold |
| **Global Min Obstacle Clearance**| $0.209\text{ m}$ | **$0.210\text{ m}$** | Safe separation preserved |
| **Mean Min Wall Clearance** | $0.223\text{ m}$ | **$0.190\text{ m}$** | Safe lateral margins |
| **Mean Terminal Velocity** | $0.252\text{ m/s}$ (stalled timeout) | **$0.170\text{ m/s}$ (landing hover)** | Within $\le 0.35\text{ m/s}$ gate |

---

## 4. Key Engineering Takeaways

1. **The Counterfactual Claim is Validated:** The Phase 4I reported claim that the expert achieves 10/10 successes without the supervisor is **100% reproduced** (averaging $393.1\text{ steps} = 19.66\text{ s}$).
2. **The Expert Trajectory is Inherently Safe:** Even with the supervisor completely disabled, the expert's analytical potential field maintains a global minimum obstacle clearance of **$0.210\text{ m}$** (more than $2.5\times$ the $0.08\text{ m}$ physical threshold) and decelerates naturally to **$0.170\text{ m/s}$** at the goal.
3. **The Role of the Supervisor:** The supervisor's $0.80\text{ m}$ caution clamp is **not** preventing collisions (the expert has zero collisions either way); rather, it is enforcing a low-speed clamp ($v_x \le 0.15\text{ m/s}$) far too early ($1.0\text{ m}$ from pillar center), which adds an artificial delay of **$+5.34\text{ seconds}$ (+107 steps)**, pushing the flight past the 500-step horizon.
4. **Diagnostic Boundary:** As noted in safeguards, the supervisor-disabled run is strictly a diagnostic counterfactual to isolate the velocity penalty. It is not an acceptable operational configuration for learned neural policies, which lack formal collision guarantees and require supervisor intervention.
