# Phase 4J-G — Policy-Supervisor Compatibility and Control Conflict Audit

**Experiment:** Phase 4J-G  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Directory:** `experiments/phase4j/phase4jg_horizon_control_audit_20261029_192000/`  
**Execution Timestamp:** 2026-10-29T19:20:00+05:30  
**Auditor:** Senior Reinforcement Learning Engineer, Autonomous UAV Navigation Researcher  

---

## 1. Executive Summary

Phase 4J-F identified an adversarial dynamic tension during proactive lateral guidance: when the experimental supervisor attempted to force a lateral bypass around centered obstacles, the Behavior Cloning (BC) policy frequently commanded motion in the opposite direction.

Experiment C of Phase 4J-G quantitatively audits this phenomenon across both the canonical baseline configuration and the Phase 4J-F guidance configuration.

### Key Findings:
1. **Canonical Supervisor Compatibility is Extremely High:**
   - Across 22,986 evaluated steps on seeds `2000–2049`, lateral command sign opposition ($\text{sign}(v_{y, \text{raw}}) \neq \text{sign}(v_{y, \text{safe}})$) occurred on **only 36 steps (0.16%)**.
   - The maximum continuous conflict duration was exactly **1 step** ($0.05\text{ s}$).
   - The canonical supervisor does **not** induce lateral oscillations or fight policy steering commands. Its primary interaction is pure forward velocity throttling ($45.18\%$ of flight steps) and barrier clamping.
2. **Phase 4J-F Guidance Created Genuine Control Contradiction:**
   - Under proactive guidance, opposing lateral steps increased by **5.6x** to 202 steps ($0.88\%$), with sustained conflict bursts persisting up to **9 consecutive steps** ($0.45\text{ s}$).
   - During guidance-policy conflict intervals, mean forward progress per step collapsed by **90%** (from $0.0147\text{ m/step}$ to $0.0015\text{ m/step}$).
3. **The Root Cause is Policy Symmetry Standoff, Not Supervisor Repulsion:**
   - When confronting a centered pillar, the BC policy commands $v_y \approx 0.0\text{ m/s}$ (symmetric lidar inputs induce near-zero lateral command).
   - Because the policy refuses to steer, the canonical supervisor throttles $v_x \to 0.10\text{ m/s}$ to maintain safety.
   - The standoff is an intrinsic property of the learned policy's inability to break symmetry in near-obstacle states, not an artifact of supervisor interference.

---

## 2. Quantitative Comparison of Policy-Supervisor Arbitration

| Telemetry Metric | Arm A: Canonical Supervisor Baseline | Arm B: Phase 4J-F Lateral Guidance | Interpretation |
| :--- | :---: | :---: | :--- |
| **Total Evaluated Steps** | 22,986 | 23,021 | 50 matched episodes (500 steps max) |
| **Opposing Lateral Steps** | **36 (0.16%)** | **202 (0.88%)** | 5.6x increase under proactive guidance |
| **Clamped Lateral Steps** ($|v_y| > 0.05 \to 0.0$) | 704 (3.06%) | 1,214 (5.27%) | Barrier enforcement near corridor walls |
| **Mean Conflict Magnitude** ($|\Delta v_y|$) | $0.1858\text{ m/s}$ | $0.1064\text{ m/s}$ | Canonical repulsion vs Guidance bias |
| **Max Conflict Magnitude** | $0.2272\text{ m/s}$ | $0.2272\text{ m/s}$ | Capped by action limits |
| **Mean Conflict Duration** | **1.00 steps** ($0.05\text{ s}$) | **1.64 steps** ($0.08\text{ s}$) | Canonical resolves instantly |
| **Max Continuous Conflict Duration** | **1 step** ($0.05\text{ s}$) | **9 steps** ($0.45\text{ s}$) | Guidance causes sustained tug-of-war |
| **Forward Progress Rate During Conflict** | $0.0009\text{ m/step}$ ($0.018\text{ m/s}$) | $0.0015\text{ m/step}$ ($0.030\text{ m/s}$) | Forward transit essentially halts |
| **Forward Progress Rate During Nominal Flight**| $0.0142\text{ m/step}$ ($0.284\text{ m/s}$) | $0.0147\text{ m/step}$ ($0.294\text{ m/s}$) | 10x to 15x faster during nominal flight |

---

## 3. Detailed Case Analysis of Control Contradiction

### Case 1: The Canonical Supervisor Behavior (Seed 2004)
In Seed 2004, the drone approaches Pillar 1 centered at $y \approx 0.0\text{ m}$.
- **Policy Output:** 
  The raw policy outputs:
  $$\mathbf{a}_{\text{raw}} = [v_x = +0.63\text{ m/s}, v_y = +0.004\text{ m/s}, v_z = +0.09\text{ m/s}, \dot{\psi} = +0.016\text{ rad/s}]$$
  The policy desires to fly forward at high speed directly into the pillar, commanding virtually zero lateral deflection ($v_y \approx +0.004\text{ m/s}$).
- **Supervisor Action:**
  As front distance drops to $0.42\text{ m} < d_{\text{caut}} = 0.50\text{ m}$, the supervisor throttles $v_x$ down to $0.10\text{ m/s}$.
  Because $v_y$ is essentially zero and wall distances are symmetric ($d_{\text{left}} \approx 0.85\text{ m}, d_{\text{right}} \approx 0.85\text{ m}$), the supervisor applies **zero lateral modification**.
- **Result:**
  Safe, stable hover at $x = 2.78\text{ m}$. Zero conflict exists between policy and supervisor ($v_y$ is neither opposed nor overridden). The drone stalls solely because the policy does not output a bypass maneuver.

### Case 2: The Proactive Guidance Conflict (Seed 2004 under Phase 4J-F)
When proactive lateral guidance is introduced:
- At step 30, stagnation is detected. Guidance evaluates rangefinder rays and commands:
  $$v_{y, \text{guidance}} = +0.25\text{ m/s} \quad (\text{steer left})$$
- The drone displaces laterally from $y = 0.0\text{ m}$ to $y = +0.08\text{ m}$.
- At the next observation, the BC policy senses displacement from the centerline and rangefinder asymmetry. Trained on centered corridor trajectories, the policy commands:
  $$v_{y, \text{raw}} = -0.18\text{ m/s} \quad (\text{restore to center})$$
- The supervisor overrides this with $v_{y, \text{guidance}} = +0.25\text{ m/s}$.
- This creates an 8-to-9 step sustained control conflict where the policy and supervisor pull in opposite directions. The drone oscillates laterally while forward progress drops to $0.0015\text{ m/step}$.
- Once guidance commitment releases, the policy immediately snaps the drone back to $y \approx 0.0\text{ m}$, returning to the head-on stall.

---

## 4. The Training Distribution Hypothesis

Why does the BC baseline policy output $v_y \approx 0.0\text{ m/s}$ when facing centered obstacles?

1. **Demonstration Dataset Bias:**
   - Behavior cloning relies on state-action pairs collected from an expert (APF controller).
   - A competent expert steers around obstacles **early** (at distances of $1.5–2.5\text{ m}$).
   - Consequently, the training dataset contains virtually **zero demonstrations** where the drone is hovering $0.35–0.45\text{ m}$ directly in front of a pillar at near-zero velocity.
2. **Symmetric Blindness in Multi-Layer Perceptrons:**
   - In head-on configurations, the 12 rangefinder rays are near-symmetric ($r_{\text{FL}} \approx r_{\text{FR}}, r_{\text{SL}} \approx r_{\text{SR}}$).
   - A feedforward MLP with tanh/ReLU activations mapping symmetric inputs produces output $v_y \approx 0.0$.
   - Without an asymmetric prior, memory mechanism (RNN/LSTM), or global path planner, the MLP has no mathematical basis to prefer left over right.
3. **Supervisor Decoupling:**
   - The safety supervisor's mandate is collision avoidance, not path planning.
   - Throttling forward velocity to prevent a crash is the correct, verified safety function.
   - Expecting a reactive supervisor to solve high-level path planning around symmetric obstacles creates the exact policy-supervisor friction observed in Phase 4J-F.

---

## 5. Architectural Conclusion

1. **The canonical supervisor is fully compatible with the BC policy.** It does not fight the policy's lateral decisions; it intervenes almost exclusively on $v_x$ throttling.
2. **Supervisor-level lateral guidance is an architectural dead-end.** Forcing lateral bypasses downstream of an end-to-end neural network creates severe state-distribution mismatch and control conflict.
3. **Symmetry breaking must occur inside the policy.** The policy must be trained to anticipate obstacles and commit to a bypass trajectory before entering the supervisor's caution zone.

