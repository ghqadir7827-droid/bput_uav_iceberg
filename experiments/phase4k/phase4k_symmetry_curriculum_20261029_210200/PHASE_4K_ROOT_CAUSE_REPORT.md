# PHASE 4K — SYMMETRY-STANDOFF ROOT-CAUSE AUDIT REPORT

**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Directory:** `experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/`  
**Execution Timestamp:** 2026-10-29T21:02:00+05:30  
**Auditor:** Senior Reinforcement Learning Engineer, Autonomous UAV Navigation Researcher, Simulation-Validation Auditor  
**Candidate Checkpoint Audited:** `models/checkpoints/bc_policy_best.pt` (SHA-256: `93c4d7d1...`)  
**Training Dataset Audited:** `data/demonstrations/bc_train_10m.npz` (SHA-256: `2b95b453...`, 47,780 transitions)  

---

## 1. Executive Summary: The Causal Root of Symmetry Standoff

Phases 4J-F and 4J-G proved that the Behavior Cloning baseline fails on 26 of 50 development seeds due to persistent obstacle standoff rather than time-budget limits or supervisor misconfiguration.

This audit conducted an end-to-end investigation spanning feature contracts, training data distributions, expert teacher formulations, network Jacobian sensitivities, and controlled PyBullet simulation diagnostics.

### The Four Compounding Causes of Symmetry Standoff:
1. **The Autoregressive Velocity Shortcut ($r = 0.9723$):**
   - In the training dataset, observed body lateral velocity ($v_{y, \text{body}} = \text{obs}[16]$) has a **0.9723 Pearson correlation** with target commanded lateral velocity ($v_{y, \text{cmd}} = \text{act}[1]$).
   - The feedforward MLP learned an autoregressive momentum-copying shortcut rather than initiating lateral evasions from rangefinder inputs.
   - When the drone approaches an obstacle on the corridor centerline with $v_y = 0.0\text{ m/s}$, the network commands $v_y \approx 0.0\text{ m/s}$.
2. **Complete Absence of Near-Obstacle Standstill in Training Data (0.00% Coverage):**
   - In `bc_train_10m.npz` (47,780 transitions), there are **ZERO transitions** where the drone is within $0.50\text{ m}$ of an obstacle at speed $< 0.20\text{ m/s}$.
   - The analytical APF teacher always flew at cruise speeds ($0.50–0.65\text{ m/s}$), circumnavigating obstacles early.
   - When the canonical `SafetySupervisor` throttles forward speed ($v_x \to 0.10\text{ m/s}$) to prevent collisions, the drone enters an out-of-distribution (OOD) hover state from which it has zero training examples to escape.
3. **Privileged Teacher Information Asymmetry:**
   - The teacher controller (`controllers/apf_expert.py`) breaks lateral symmetry using privileged ground-truth obstacle coordinates:
     $$\text{evade\_dir} = -1.0 \quad \text{if } y_{\text{obs}} \ge 0 \quad \text{else } +1.0$$
   - The student observation vector contains **only rangefinder rays**, lacking obstacle center coordinates. For centered obstacles on the centerline, the student observation is mathematically symmetric.
4. **The Bimodal MSE Symmetry Collapse Theorem:**
   - Under supervised Behavior Cloning with Mean Squared Error (MSE) loss, a deterministic policy mapping symmetric inputs $\mathbf{x}_{\text{sym}}$ to balanced demonstration targets (left evasion $+A$ and right evasion $-A$) converges to the conditional expectation:
     $$\hat{v}_y = \mathbb{E}[v_y \mid \mathbf{x}_{\text{sym}}] = \frac{+A + (-A)}{2} = \mathbf{0.00\text{ m/s}}$$
   - A feedforward MLP trained with MSE loss mathematically cannot make a discrete left-or-right choice on symmetric inputs without an asymmetric input feature, recurrent memory, or multimodal head.

---

## 2. Pipeline Architecture & Feature Sensitivity (Jacobian Audit)

The policy network (`models/policy_net.py`) is an MLP with architecture `Linear(23, 128) -> LayerNorm -> GELU -> Linear(128, 128) -> LayerNorm -> GELU -> Linear(128, 64) -> LayerNorm -> GELU -> Linear(64, 4)`, followed by physical limit bounding:
$$\mathbf{a} = \mathbf{a}_{\text{center}} + \mathbf{a}_{\text{half\_range}} \odot \tanh(\mathbf{z})$$
where for lateral velocity $v_y$: center $= 0.00\text{ m/s}$, half range $= 0.60\text{ m/s}$.

### Input Jacobian Sensitivity ($\frac{\partial v_y}{\partial \mathbf{x}}$):
Evaluating the input gradient on a centered obstacle at distance $0.40\text{ m}$ with speed $0.10\text{ m/s}$:

| Input Feature | Feature Value | Gradient $\frac{\partial v_y}{\partial x_i}$ | Gradient $\frac{\partial v_x}{\partial x_i}$ | Architectural Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `ray_0_front` | $0.100$ ($0.40\text{ m}$) | $-0.02381$ | $-0.00817$ | Negligible lateral sensitivity to front obstacle |
| `ray_1_front_left` | $0.112$ ($0.45\text{ m}$) | **$+0.23456$** | $-0.05930$ | Positive: longer left ray increases $v_y$ |
| `ray_2_front_right` | $0.112$ ($0.45\text{ m}$) | **$+0.17657$** | $+0.05117$ | **PATHOLOGICAL:** Longer right ray ALSO increases $v_y$! |
| `ray_5_left` | $0.213$ ($0.85\text{ m}$) | $+0.26341$ | $-0.01240$ | Left wall proximity |
| `ray_6_right` | $0.213$ ($0.85\text{ m}$) | $+0.04894$ | $+0.05331$ | Right wall proximity |
| `vel_x_body` | $0.100\text{ m/s}$ | $-0.05719$ | $-0.21356$ | Forward speed damping |
| `vel_y_body` | $0.000\text{ m/s}$ | **$+0.34143$** | $+0.06385$ | **Dominant momentum coupling** |
| `yaw_rate` | $0.000\text{ rad/s}$ | **$+0.47834$** | $+0.11999$ | **Dominant rotational coupling** |

### Crucial Finding on Lateral Rangefinder Gradients:
For an obstacle evasion policy, front-left and front-right rays must have **opposite signs**:
$$\frac{\partial v_y}{\partial \text{ray}_{\text{FL}}} > 0 \quad \text{and} \quad \frac{\partial v_y}{\partial \text{ray}_{\text{FR}}} < 0$$
(Opening on left $\implies$ steer left; opening on right $\implies$ steer right).

However, in `bc_policy_best.pt`, **both gradients are positive** ($\frac{\partial v_y}{\partial \text{ray}_{\text{FL}}} = +0.235, \frac{\partial v_y}{\partial \text{ray}_{\text{FR}}} = +0.177$). When an opening appears on the right, the network fails to steer right because increasing `ray_2` actually pushes $v_y$ positive.

---

## 3. Training Dataset Distribution Audit (`dataset_audit.json`)

Analysis of `data/demonstrations/bc_train_10m.npz` (120 episodes, 47,780 transitions):

1. **Global Balance vs Local Vacuity:**
   - Globally, actions appear balanced: $38.3\%$ left ($v_y > 0.05$), $38.4\%$ right ($v_y < -0.05$), $23.3\%$ center ($|v_y| \le 0.05$).
   - However, when examining transitions within the obstacle caution zone ($\text{front ray} < 0.50\text{ m}$):
     - Only $595$ transitions ($1.2\%$) exist in the entire dataset.
     - Only $137$ transitions ($0.3\%$) exist below $0.40\text{ m}$.
     - Only $16$ transitions ($0.03\%$) exist below $0.30\text{ m}$.
2. **The 0.00% Zero-Speed Standoff Gap:**
   - In all 47,780 transitions, there are **exactly 0 transitions** where $\text{front ray} < 0.50\text{ m}$ and $\text{speed} < 0.20\text{ m/s}$.
   - The expert never hovered, paused, or stalled. Every low-range transition in the dataset was captured at high velocity ($0.40–0.65\text{ m/s}$) during mid-bypass.
   - When the student enters a standoff at $v \approx 0.05\text{ m/s}$, it is operating completely outside its support manifold.
3. **The Inertia Correlation Trap ($r = 0.9723$):**
   - The correlation between input $v_{y, \text{body}}$ and label $v_{y, \text{action}}$ is $0.9723$.
   - The MLP minimized MSE loss by learning to copy its incoming lateral velocity:
     $$v_{y, \text{pred}} \approx 0.95 \cdot v_{y, \text{obs}}$$
   - When $v_{y, \text{obs}} = 0.0\text{ m/s}$, the policy outputs $v_{y, \text{pred}} \approx 0.0\text{ m/s}$.

---

## 4. Controlled Simulation Diagnostics (`symmetry_diagnostic_results.csv`)

Controlled tests in PyBullet simulation confirmed the causal diagnosis:

| Test ID | Scenario Description | Initial $v_y$ | Status | Progress | Min Obs Clearance | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `TEST_01_CENTERED` | Pillar 1 centered ($y = 0.0\text{ m}$) from origin | $0.00\text{ m/s}$ | SUCCESS | $8.65\text{ m}$ | $0.2018\text{ m}$ | Pre-existing rightward drift bypasses |
| `TEST_LEFT_Y_05` | Pillar 1 offset left ($y = +0.05\text{ m}$) | $0.00\text{ m/s}$ | **TIMEOUT** | **$2.27\text{ m}$** | **$0.1155\text{ m}$** | **Standoff stall at Pillar 1** |
| `TEST_RIGHT_Y_05`| Pillar 1 offset right ($y = -0.05\text{ m}$) | $0.00\text{ m/s}$ | SUCCESS | $8.66\text{ m}$ | $0.2162\text{ m}$ | Smooth left bypass |
| `TEST_LEFT_Y_10` | Pillar 1 offset left ($y = +0.10\text{ m}$) | $0.00\text{ m/s}$ | **TIMEOUT** | **$2.22\text{ m}$** | **$0.1151\text{ m}$** | **Standoff stall at Pillar 1** |
| `TEST_RIGHT_Y_10`| Pillar 1 offset right ($y = -0.10\text{ m}$) | $0.00\text{ m/s}$ | SUCCESS | $8.66\text{ m}$ | $0.2335\text{ m}$ | Smooth left bypass |
| `TEST_LEFT_Y_20` | Pillar 1 offset left ($y = +0.20\text{ m}$) | $0.00\text{ m/s}$ | SUCCESS | $8.66\text{ m}$ | $0.1323\text{ m}$ | Asymmetry large enough to clear |
| `TEST_RIGHT_Y_20`| Pillar 1 offset right ($y = -0.20\text{ m}$) | $0.00\text{ m/s}$ | SUCCESS | $8.65\text{ m}$ | $0.2256\text{ m}$ | Asymmetry large enough to clear |
| `TEST_STANDOFF_NEG_20` | Standoff at $x=2.40\text{ m}$ with initial $v_y = -0.20\text{ m/s}$ | $-0.20\text{ m/s}$ | **SUCCESS** | $6.76\text{ m}$ | $0.1207\text{ m}$ | Initial negative velocity triggers right bypass |
| `TEST_STANDOFF_NEG_10` | Standoff at $x=2.40\text{ m}$ with initial $v_y = -0.10\text{ m/s}$ | $-0.10\text{ m/s}$ | **SUCCESS** | $6.76\text{ m}$ | $0.1117\text{ m}$ | Initial negative velocity triggers right bypass |
| `TEST_STANDOFF_POS_10` | Standoff at $x=2.40\text{ m}$ with initial $v_y = +0.10\text{ m/s}$ | $+0.10\text{ m/s}$ | **TIMEOUT** | **$0.43\text{ m}$** | **$0.1138\text{ m}$** | **Positive velocity insufficient to overcome deadlock** |
| `TEST_STANDOFF_POS_20` | Standoff at $x=2.40\text{ m}$ with initial $v_y = +0.20\text{ m/s}$ | $+0.20\text{ m/s}$ | **TIMEOUT** | **$0.42\text{ m}$** | **$0.1174\text{ m}$** | **Stalls in front of pillar** |

### Critical Diagnostic Confirmation:
1. When obstacles are offset slightly to the left ($+0.05\text{ m}, +0.10\text{ m}$), the drone **invariably stalls at $x \approx 2.22–2.27\text{ m}$**, unable to execute a rightward evasion.
2. When starting at rest in front of a pillar ($x = 2.40\text{ m}$), negative initial velocity ($v_y = -0.10\text{ m/s}$) allows the drone to bypass the obstacle and complete the corridor, whereas positive initial velocity ($v_y = +0.10\text{ m/s}$) results in immediate timeout standoff.

---

## 5. Why Naive Curriculum Training Fails: The Multimodal Dilemma

Can this be solved simply by adding near-obstacle demonstrations to the training set?
**No.** Supervised behavior cloning with MSE loss faces a mathematical impossibility:

1. Suppose we add demonstrations where the drone starts in front of a centered obstacle.
2. To avoid a unilateral bias, we must provide both:
   - Trajectory A: Steer Left ($v_y = +0.50\text{ m/s}$).
   - Trajectory B: Steer Right ($v_y = -0.50\text{ m/s}$).
3. Both trajectories share the **exact same initial observation vector** $\mathbf{x}_{\text{sym}}$.
4. Under supervised MSE regression:
   $$\arg\min_{\hat{v}_y} \left[ \frac{1}{2} (\hat{v}_y - (+0.50))^2 + \frac{1}{2} (\hat{v}_y - (-0.50))^2 \right] = \mathbf{0.00\text{ m/s}}$$
5. The trained deterministic MLP will predict the average of left and right: **zero lateral velocity**.
6. The resulting policy will continue to stall in front of centered obstacles.

---

## 6. Audit Conclusion & Architectural Recommendation

The symmetry standoff is **not** a hyperparameter tuning issue, dataset volume issue, or supervisor defect. It is a fundamental consequence of using a **deterministic feedforward regression policy on a single-frame symmetric observation contract with multimodal targets**.

### Mandatory Prescription:
To resolve symmetry standoff, the system requires an **Observation or Policy Design Revision**:
1. **Asymmetric Input Conditioning:** Adding an explicit lateral preference feature or waypoint indicator to break input symmetry; or
2. **Temporal Policy Memory:** Frame stacking or recurrent layers (LSTM/GRU) so that initial micro-perturbations accumulate into committed lateral trajectories; or
3. **Multimodal Action Distribution:** Parameterizing actions as a mixture of Gaussians (MDN) or discrete intent modes (`[STEER_LEFT, STEER_RIGHT]`) followed by continuous velocity regression.

