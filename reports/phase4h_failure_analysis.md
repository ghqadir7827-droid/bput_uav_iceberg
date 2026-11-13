# Phase 4H Failure Analysis: BC and PPO Navigation Failure Mechanisms

## Executive Summary

Phase 4H-C conducts a rigorous, evidence-based failure diagnosis of the existing Behavior Cloning (BC) baseline, the original Proximal Policy Optimization (PPO) baseline, and the Phase 4G PPO candidates under the canonical PyBullet corridor environment (`uav-bput-nav`).

The investigation reveals two distinct failure mechanisms rooted in how each learning paradigm handles spatial symmetry and multimodal decision boundaries:
1. **Behavior Cloning (BC)** suffers from **multimodal target averaging under MSE loss** at centered obstacles: when an obstacle is dead-center ($Y_{\text{obs}} = 0.00\text{ m}$), the symmetric rangefinder observation induces a conditional expectation $\mathbb{E}[v_y \mid Y_{\text{obs}}=0] \approx 0.00\text{ m/s}$. The policy commands near-zero lateral velocity ($|v_y| < 0.003\text{ m/s}$), flying straight into the obstacle until the Safety Supervisor halts forward motion at $X \approx 2.81\text{ m}$, causing an irrecoverable deadlock.
2. **Proximal Policy Optimization (PPO)** suffers from **unilateral mode lock (symmetry breaking)**: RL policy optimization converged to a persistent rightward evasion strategy ($v_y \approx -0.25\text{ m/s}$, $\dot{\psi} \approx -0.15\text{ rad/s}$). While this strategy successfully clears left-shifted obstacles ($Y_{\text{obs}} > 0$), it fails catastrophically against centered and right-shifted obstacles ($Y_{\text{obs}} \le 0$) by forcing the UAV into the constricted right-hand gap, triggering supervisor boundary intervention or obstacle deadlock.
3. **Safety Supervisor Interaction**: The canonical `SafetySupervisor` acts as a velocity limiter (reducing $v_x$ within the $0.55\text{ m}$ caution zone and commanding zero velocity within the $0.25\text{ m}$ critical threshold). It does **not** inject lateral bypass maneuvers. Consequently, if the learned policy fails to generate lateral clearance, the supervisor correctly prevents collision but induces a terminal deadlock.

---

## 1. Behavior Cloning (BC) Failure Diagnosis

### 1.1 Dataset Distribution & Expert Policy Architecture
The expert dataset (`data/demonstrations/bc_train_10m.npz`) was audited:
- **Total Transitions:** 47,780 transitions across 120 accepted episodes.
- **Teacher Controller:** Scripted Artificial Potential Field (`controllers/apf_expert.py`).
- **Obstacle Placement During Data Generation:** Obstacles are placed with random offsets:
  $$Y_{\text{pillar}} \sim \mathcal{U}(-0.35\text{ m}, +0.35\text{ m})$$
  Because $Y_{\text{pillar}}$ is drawn from a continuous uniform distribution, the probability of sampling an exact centered obstacle ($Y = 0.000\text{ m}$) was zero.
- **Expert Circulation Logic (`controllers/apf_expert.py:187`):**
  ```python
  evade_dir = -1.0 if obs_c[1] >= 0 else 1.0
  circ_mag = float(self.k_circ * np.exp(-max(0.0, clearance) / 0.35))
  vy_des += float(rep_mag * n_rep[1] + circ_mag * evade_dir)
  ```
  The expert circulates rightwards ($evade\_dir = -1.0$) for $Y \ge 0$, and leftwards ($evade\_dir = +1.0$) for $Y < 0$.

### 1.2 The Multimodal Averaging Problem Under MSE
Behavior Cloning minimizes the Mean Squared Error (MSE) loss:
$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N}\sum_{i=1}^N \|\pi_\theta(s_i) - a_i^*\|^2$$
It is a well-known theoretical result in statistical learning that the minimizer of MSE loss for a conditional distribution $p(a \mid s)$ is the conditional expectation:
$$\pi^*(s) = \mathbb{E}[a \mid s] = \int a \, p(a \mid s) \, da$$

When the UAV approaches a centered obstacle on the centerline ($Y \approx 0.0$):
- Observations with slight positive perturbations ($Y = +\epsilon$) have expert labels with rightward evasion: $a_{\text{expert}} = (v_x \approx 0.45, v_y \approx -0.35, v_z \approx 0.0, \dot{\psi} \approx -0.40)$.
- Observations with slight negative perturbations ($Y = -\epsilon$) have expert labels with leftward evasion: $a_{\text{expert}} = (v_x \approx 0.45, v_y \approx +0.35, v_z \approx 0.0, \dot{\psi} \approx +0.40)$.
- Since the rangefinder observation at $Y=0$ is left-right symmetric, the network outputs:
  $$\mathbb{E}[v_y \mid s_{\text{center}}] = \frac{1}{2}(+0.35) + \frac{1}{2}(-0.35) \approx 0.00\text{ m/s}$$
  $$\mathbb{E}[\dot{\psi} \mid s_{\text{center}}] = \frac{1}{2}(+0.40) + \frac{1}{2}(-0.40) \approx 0.00\text{ rad/s}$$

### 1.3 Empirical Verification from Controlled Trajectory Traces
In the Phase 4H-B controlled lateral sweep at $Y_{\text{obs}} = 0.00\text{ m}$:
| Timestep | Drone X (m) | Drone Y (m) | Raw $v_x$ (m/s) | Raw $v_y$ (m/s) | Sup $v_x$ (m/s) | Sup $v_y$ (m/s) | Front Clear (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Step 1 | 0.500 | 0.000 | 0.505 | +0.0021 | 0.505 | +0.0021 | 2.50 |
| Step 20 | 1.002 | 0.000 | 0.518 | +0.0028 | 0.518 | +0.0028 | 2.00 |
| Step 40 | 1.512 | 0.000 | 0.521 | +0.0031 | 0.521 | +0.0031 | 1.49 |
| Step 60 | 2.025 | 0.000 | 0.519 | +0.0029 | 0.519 | +0.0029 | 0.98 |
| Step 75 | 2.408 | 0.000 | 0.510 | -0.0007 | 0.380 | -0.0007 | 0.59 |
| Step 90 | 2.750 | 0.000 | 0.490 | -0.0005 | 0.082 | -0.0005 | 0.25 |
| Step 105 | 2.812 | 0.000 | 0.485 | -0.0002 | 0.000 | -0.0002 | 0.188 |
| Step 500 | 2.812 | 0.000 | 0.485 | -0.0002 | 0.000 | -0.0002 | 0.188 |

**Conclusion for BC:** BC fails at centered obstacles not because the corridor is infeasible, nor because of observation ambiguity, but because the supervised regression objective averaged contradictory lateral expert actions into zero lateral output.

---

## 2. Proximal Policy Optimization (PPO) Failure Diagnosis

### 2.1 Unilateral Symmetry Breaking (Rightward Mode Lock)
During RL training, the agent explores actions randomly via Gaussian exploration $\mathcal{N}(\mu(s), \sigma^2)$.
- If early random exploration finds that turning right allows the drone to bypass an obstacle, policy gradient updates reinforce positive reward for rightward velocity ($v_y < 0$).
- Because the corridor is symmetric and both left and right bypasses yield similar task rewards, gradient updates rapidly break symmetry, collapsing the policy into a single mode.
- In `models/checkpoints/ppo_policy_best.pt`, in open space without obstacles:
  $$v_y = -0.248\text{ m/s}, \quad \dot{\psi} = -0.152\text{ rad/s}$$
  The policy continuously drifts rightwards.

### 2.2 Controlled Lateral Sweep Performance
In the controlled single-pillar sweep ($X_{\text{pillar}} = 3.0\text{ m}$):
- **$Y_{\text{obs}} = +0.25\text{ m}$ (Shifted Left):**
  - Right gap width: $1.0 - (-0.25) - 0.20 - 0.15 = 0.90\text{ m}$ (wide and clear).
  - PPO commands rightward velocity, sails cleanly through the right channel, and arrives at the goal ($X = 9.5\text{ m}$) in 379 steps (**SUCCESS**).
- **$Y_{\text{obs}} = +0.15\text{ m}$ (Shifted Left):**
  - Right gap width: $0.80\text{ m}$.
  - PPO steers right, clears the obstacle, and arrives at the goal in 383 steps (**SUCCESS**).
- **$Y_{\text{obs}} = 0.00\text{ m}$ (Dead Center):**
  - Right gap width: $0.65\text{ m}$. Left gap width: $0.65\text{ m}$.
  - PPO attempts its habitual right turn, but the lateral clearance is tighter than its learned safety margin. The right wall repulsion and obstacle repulsion compress the trajectory, triggering the supervisor caution buffer. PPO deadlocks at $X = 2.62\text{ m}$ (**DEADLOCK**).
- **$Y_{\text{obs}} = -0.15\text{ m}$ (Shifted Right):**
  - Left gap is wide ($0.80\text{ m}$), right gap is constricted ($0.50\text{ m}$).
  - PPO refuses to take the open left path. It turns right into the $0.50\text{ m}$ constriction, gets squeezed between the pillar and the right wall, and deadlocks at $X = 2.58\text{ m}$ (**DEADLOCK**).
- **$Y_{\text{obs}} = -0.25\text{ m}$ (Shifted Right):**
  - Left gap is open ($0.90\text{ m}$), right gap is impassable ($0.40\text{ m}$).
  - PPO continues commanding rightward steering, stalls against the obstacle and wall, and deadlocks at $X = 2.54\text{ m}$ (**DEADLOCK**).

### 2.3 Phase 4G Reflection Candidate Analysis
In Phase 4G, bilateral reflection augmentation was applied to PPO (`Arm_D_PPO_Reflection`):
- Reflection augmentation successfully neutralized open-space bias: initial lateral velocity dropped to $|v_y| < 0.015\text{ m/s}$.
- However, Phase 4G trained under a truncated 200-step horizon. Because reaching $X=9.5\text{ m}$ in 200 steps ($10\text{ s}$) requires sustained high speed ($v_x \ge 0.50\text{ m/s}$), the policy learned aggressive forward charging without terminal deceleration.
- In the Phase 4H-B Goal-Approach test, `Arm_D` overshot the arrival speed gate ($0.362\text{ m/s} > 0.350\text{ m/s}$ threshold), registering terminal arrival failure.

---

## 3. Shared Safety Supervisor Interaction Analysis

### 3.1 Supervisor Operational Envelope
The canonical supervisor (`controllers/safety_supervisor.py`) evaluates forward and boundary clearances:
$$\text{Caution Threshold } d_{\text{caut}} = 0.55\text{ m}, \quad \text{Critical Threshold } d_{\text{crit}} = 0.25\text{ m}$$
When $d_{\text{min}} < d_{\text{caut}}$, the supervisor scales down $v_x$:
$$v_x^{\text{filt}} = v_x^{\text{raw}} \cdot \max\left(0.0, \frac{d_{\text{min}} - d_{\text{crit}}}{d_{\text{caut}} - d_{\text{crit}}}\right)$$
When $d_{\text{min}} \le d_{\text{crit}}$, $v_x = v_y = v_z = 0$.

### 3.2 Key Finding on Supervisor Role in Deadlocks
- The supervisor is **strictly a passive protective governor**, not an active navigator.
- It does **not** generate lateral avoidance setpoints ($v_y$).
- When the policy commands $v_y \approx 0$ (BC on centered obstacles) or commands $v_y < 0$ into a blocked passage (PPO on right-shifted obstacles), the supervisor decelerates the drone to prevent collision.
- Because the policy never updates its lateral command to steer into the open free space, the drone remains halted indefinitely.
- The failure is **100% policy-driven**, with the supervisor performing its contractual duty of preserving physical integrity. Modifying the supervisor to arbitrarily steer the drone would violate the system boundary between safety monitoring and autonomous navigation.

---

## 4. Synthesis and Concrete Requirements for Improvement (Phase 4H-D)

Based on these verified failure mechanisms, the requirements for independent improvement are unambiguous:

### For Behavior Cloning (Phase 4H-D BC Candidate):
1. **Disambiguate Multimodal Centerline Targets:** When an obstacle is centered ($|Y_{\text{obs}}| \le 0.08\text{ m}$), provide a consistent, deterministic evasion demonstration (e.g. decisive left-side bypass commanding $v_y \ge +0.35\text{ m/s}$ and $\dot{\psi} \ge +0.35\text{ rad/s}$).
2. **Explicit Symmetric Counter-Demonstrations:** For obstacles offset to the right, provide clear left-side bypass demonstrations; for obstacles offset to the left, provide clear right-side bypass demonstrations.
3. **Deceleration Demonstrations at Goal:** Include explicit terminal deceleration transitions approaching $X = 9.5\text{ m}$ to ensure arrival speed $\le 0.35\text{ m/s}$.

### For Proximal Policy Optimization (Phase 4H-D PPO Candidate):
1. **Full 500-Step Episode Horizon:** Eliminate the artificial 200-step truncation during rollout collection to allow realistic approach and obstacle navigation dynamics (average expert flight duration is 398 steps).
2. **Bilateral Reflection Augmentation:** Maintain bilateral reflection invariance to prevent symmetry breaking and eliminate unilateral lateral mode lock.
3. **Targeted Multi-Scenario Rollout Mixture:** Train across diverse obstacle lateral offsets ($Y_{\text{obs}} \in [-0.35, +0.35]$) so the policy discovers and reinforces both left-bypass and right-bypass trajectories.

---

## 5. Artifact and Trace Provenance
- Baseline evaluation traces: `reports/phase4h_baseline_evaluation_metrics.json`
- Baseline evaluation summary: `reports/phase4h_baseline_evaluation.md`
- Implementation audit: `reports/phase4h_implementation_audit.md`
- Verification test suite: `pytest -q` (106/106 passing)
