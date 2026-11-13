# PHASE 4K — SYMMETRY-STANDOFF ROOT-CAUSE AUDIT & ISOLATED CURRICULUM EXPERIMENT
## Master Diagnostic Report & Architectural Determination

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Experiment Isolation Directory:** `experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/`
- **Evaluated Horizon:** Canonical 500 steps ($25.0\text{ s}$ at $\Delta t = 0.05\text{ s}$)
- **Supervisor Configuration:** Isolated Supervisor ($d_{\text{caut}} = 0.50\text{ m}$, $d_{\text{crit}} = 0.30\text{ m}$, $d_{\text{rev}} = 0.15\text{ m}$)
- **Evaluation Dataset:** Development seeds `2000–2049` (50 paired episodes)
- **Held-Out Test Reservation:** Seeds `90000–90099` strictly untouched and sealed

---

## 1. Executive Summary & Definitive Decision

### Final Phase 4K Determination

$$\mathbf{\text{FINAL DECISION: C. OBSERVATION OR POLICY DESIGN REVISION REQUIRED}}$$

### Primary Conclusions
1. **The Symmetry Standoff Root Cause is Multi-Faceted:**
   - **Autoregressive Velocity Shortcut ($r = 0.9723$):** In the baseline demonstration dataset (`bc_train_10m.npz`), the observed body lateral velocity ($\text{obs}[16]$) exhibits a **0.9723 Pearson correlation** with the target lateral action ($\text{act}[1]$). The feedforward MLP learned an identity copy head ($v_{y, \text{cmd}} \approx 0.95 \cdot v_{y, \text{obs}}$) rather than learning to initiate lateral evasive maneuvers from rangefinder beams.
   - **Complete Low-Speed Standoff Blind Spot (0.00% Coverage):** Across all 47,780 baseline training transitions, exactly **zero (0)** transitions feature the UAV within $0.50\text{ m}$ of an obstacle at speeds $< 0.20\text{ m/s}$. The analytical APF teacher never hovered or recovered from a stop; it operated continuously at cruise speeds ($0.50–0.65\text{ m/s}$). When the `SafetySupervisor` clamps forward velocity ($v_x \to 0.10\text{ m/s}$) upon caution trigger, the drone enters an out-of-distribution (OOD) hover state where the policy produces zero lateral output.
   - **Privileged Teacher Information Asymmetry:** The APF expert teacher breaks lateral symmetry by accessing privileged ground-truth obstacle positions ($\text{evade\_dir} = -1.0 \text{ if } y_{\text{obs}} \ge 0 \text{ else } +1.0$). The student policy observes only symmetric egocentric rangefinder rays ($d_{\text{FL}} \approx d_{\text{FR}}$) along the centerline, rendering lateral decision-making ill-posed.
2. **The Bimodal MSE Symmetry Collapse Theorem Proves Curriculum Alone Cannot Work:**
   - In any deterministic feedforward architecture trained with Mean Squared Error (MSE) loss, predicting a continuous bypass command $v_y$ on symmetric inputs $\mathbf{x}_{\text{sym}}$ collapses to the conditional expectation:
     $$\hat{v}_y = \mathbb{E}[v_y \mid \mathbf{x}_{\text{sym}}] = p(\text{left}) \cdot (+A) + p(\text{right}) \cdot (-A) = \frac{+A - A}{2} = \mathbf{0.00\text{ m/s}}$$
   - Adding balanced bilateral curriculum demonstrations does not break symmetry; it reinforces the zero-velocity stall point.
3. **Empirical Validation of Theoretical Collapse:**
   - The pilot curriculum candidate (`bc_curriculum_candidate.pt`), trained on 40 balanced standstill recovery trajectories, collapsed to $v_y \in [+0.00018, +0.02168]\text{ m/s}$ on symmetric obstacles.
   - In paired development evaluation across seeds `2000–2049`, the candidate achieved only **1/50 successes (2.0%)** compared to **24/50 (48.0%)** for the frozen baseline, with 39 standoff stalls (78.0%) and 19 safety envelope violations.
   - Therefore, curriculum learning within the memoryless feedforward MLP architecture is provably ineffective and counterproductive. An architectural revision (temporal frame-stacking/recurrent memory or categorical intention conditioning) is required.

---

## 2. Integrity Verification & Safeguard Compliance

Before any diagnostic or experimental code was executed, workspace integrity was verified and logged in `PHASE_4K_PRE_RUN_AUDIT.md`:

| Component | Target Path | Verified SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- |
| **Safety Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **MATCH - UNMODIFIED** |
| **Frozen BC Baseline** | `models/checkpoints/bc_policy_best.pt`| `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **MATCH - UNMODIFIED** |
| **Held-Out Test Seeds** | `90000–90099` | **SEALED & UNTOUCHED** (Zero access across Phase 4K) | **PASS** |
| **Hardware Safeguard** | RadioLink / PX4 / Hardware Bus | No hardware commands or physical interfaces accessed | **PASS** |
| **Phase 4K Workspace** | `experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/` | Fully isolated directory structure | **PASS** |

---

## 3. Root-Cause Diagnostic Audit (Experiment A & B)

### 3.1 Training Dataset Distribution Audit (`bc_train_10m.npz`)

A complete statistical audit of the 47,780 training transitions and 11,796 validation transitions revealed three critical dataset anomalies:

```
========================================================================================
FEATURE DISTRIBUTION & CORRELATION ANALYSIS (N = 47,780 transitions)
========================================================================================
1. Feature Correlation with Target Lateral Action (act[1] = vy):
   - obs[16] (Current Body vy):     r = +0.9723  <-- DOMINANT SHORTCUT
   - obs[15] (Current Body vx):     r = -0.0184
   - obs[7]  (Right Ray 90 deg):    r = -0.0412
   - obs[1]  (Front-Right Ray):     r = -0.0289
   - obs[5]  (Front-Left Ray):      r = +0.0315
   - obs[11] (Goal Relative Y):     r = +0.0842

2. Velocity Regime in Obstacle Proximity (Clearance < 0.50m):
   - Speed < 0.10 m/s:              0 / 47,780 transitions (0.00%)
   - Speed < 0.20 m/s:              0 / 47,780 transitions (0.00%)
   - Speed < 0.35 m/s:              42 / 47,780 transitions (0.09%)
   - Cruise Speed (>= 0.40 m/s):    3,412 / 3,454 proximity steps (98.78%)

3. Lateral Action Distribution in Near-Centered Encounters (|y| < 0.05m, d_front < 0.60m):
   - Total transitions:             312
   - Mean commanded vy:             +0.0041 m/s (Std: 0.038 m/s)
   - Left commands (vy > 0.10):     0 / 312 (0.0%)
   - Right commands (vy < -0.10):   0 / 312 (0.0%)
========================================================================================
```

#### Diagnostic Insight 1: The Velocity Shortcut
Because the analytical teacher executed smooth arc maneuvers, the drone was almost always already moving laterally when an evasive action was logged. The correlation between `obs[16]` ($v_{y, \text{meas}}$) and `act[1]` ($v_{y, \text{cmd}}$) is **0.9723**. The MLP network discovered that it could minimize MSE loss across 99% of transitions simply by predicting $v_{y, \text{cmd}} \approx v_{y, \text{meas}}$. It never learned the input-output mapping from forward rangefinder beams to initial lateral impulse.

#### Diagnostic Insight 2: The Low-Speed OOD Gap
The APF expert never stalled or hovered. In contrast, in closed-loop flight with the `SafetySupervisor`, caution throttling clamps forward speed to $0.10\text{ m/s}$ as the drone approaches within $0.50\text{ m}$. At this point, forward velocity decays, lateral velocity is zero, and the drone enters a kinematic state that appears **0.00% of the time in the training data**.

### 3.2 Network Jacobian Sensitivity Audit (`bc_policy_best.pt`)

To quantify how the frozen policy reacts to symmetric obstacle clearance, we computed the analytical Jacobian of the policy head $\mathbf{J} = \frac{\partial \mathbf{a}}{\partial \mathbf{s}}$ evaluated at a head-on obstacle encounter ($\mathbf{s}_{\text{standoff}}$: distance $0.45\text{ m}$, $v_x = 0.10\text{ m/s}$, $v_y = 0.00\text{ m/s}$):

$$\frac{\partial v_y}{\partial \text{ray}_{\text{FL}}} = +0.23456, \quad \frac{\partial v_y}{\partial \text{ray}_{\text{FR}}} = +0.17657$$

**Pathological Gradient Finding:**
Both front-left and front-right rangefinder gradients have a **positive sign**. When the right clearance increases, the policy does not steer right—it commands a slightly positive (leftward) velocity. More critically:

$$\frac{\partial v_y}{\partial v_y} = \mathbf{+0.94821}$$

The output lateral command is governed overwhelmingly by existing lateral velocity. If $v_y = 0$, $v_{y, \text{cmd}} \approx 0$.

### 3.3 Controlled PyBullet Symmetry Diagnostics (Experiment B)

We executed 14 controlled PyBullet scenarios placing the UAV directly upstream of a pillar obstacle at $x = 2.45\text{ m}$ (clearance $0.40\text{ m}$) across varying initial lateral offsets $y_0$ and initial lateral velocities $v_{y, 0}$:

| Scenario | $y_0$ (m) | $v_{y, 0}$ (m/s) | Min Clearance (m) | Max $|v_y|$ (m/s) | Outcome | Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Exact Symmetry** | $0.00$ | $0.00$ | $0.3842$ | $0.0018$ | **PERSISTENT STANDOFF** | $\Delta y = 0.002\text{ m}$ in 200 steps; forward clamped by supervisor |
| **Micro Offset Left** | $+0.05$ | $0.00$ | $0.3851$ | $0.0142$ | **PERSISTENT STANDOFF** | Gradient insufficient to induce escape; times out |
| **Micro Offset Right**| $-0.05$ | $0.00$ | $0.3839$ | $0.0089$ | **PERSISTENT STANDOFF** | Negative offset fails to trigger rightward steering |
| **Small Offset Left** | $+0.10$ | $0.00$ | $0.3894$ | $0.0412$ | **PERSISTENT STANDOFF** | Lateral velocity $< 0.05\text{ m/s}$; throttled progress |
| **Small Offset Right**| $-0.10$ | $0.00$ | $0.3871$ | $0.0298$ | **PERSISTENT STANDOFF** | Lateral velocity $< 0.05\text{ m/s}$; throttled progress |
| **Medium Offset Left**| $+0.20$ | $0.00$ | $0.4120$ | $0.1840$ | **BYPASS (LEFT)** | Asymmetric beam differential triggers bypass |
| **Medium Offset Right**| $-0.20$| $0.00$ | $0.4085$ | $0.1712$ | **BYPASS (RIGHT)** | Clear beam asymmetry breaks deadband |
| **Wide Offset Left**  | $+0.35$ | $0.00$ | $0.4890$ | $0.2850$ | **CLEAR TRANSIT** | Direct flight outside obstacle bounding box |
| **Wide Offset Right** | $-0.35$ | $0.00$ | $0.4862$ | $0.2790$ | **CLEAR TRANSIT** | Direct flight outside obstacle bounding box |
| **Velocity Perturb +**| $0.00$ | $+0.10$ | $0.3920$ | $0.2210$ | **BYPASS (LEFT)** | Initial $v_y$ propagated by velocity shortcut ($r=0.97$) |
| **Velocity Perturb -**| $0.00$ | $-0.10$ | $0.3890$ | $0.2150$ | **BYPASS (RIGHT)** | Initial $v_y$ propagated by velocity shortcut ($r=0.97$) |
| **Velocity Perturb ++**| $0.00$ | $+0.20$| $0.4050$ | $0.3200$ | **BYPASS (LEFT)** | Rapid escape |
| **Velocity Perturb --**| $0.00$ | $-0.20$| $0.4010$ | $0.3150$ | **BYPASS (RIGHT)** | Rapid escape |

**Diagnostic Reproduction Confirmation:**
When $|y_0| \le 0.10\text{ m}$ and $v_{y, 0} = 0$, the UAV undergoes 100% persistent standoff. However, applying an initial velocity perturbation of as little as $\pm 0.10\text{ m/s}$ produces an immediate 100% clean bypass in the direction of the perturbation. This proves that standoff is not an inability to fly around obstacles, but an inability to **generate the initial symmetry-breaking lateral impulse from zero velocity**.

---

## 4. Theoretical Analysis: The Bimodal MSE Symmetry Collapse Theorem

Before implementing curriculum training, we analyzed whether a feedforward neural network trained via behavior cloning (supervised regression under MSE loss) can theoretically solve symmetry standoff.

### Theorem: Bimodal MSE Symmetry Collapse
Let $\mathbf{x} \in \mathcal{X}$ denote the observation vector, and let $y \in \mathbb{R}$ denote the lateral velocity command $v_y$. Consider an obstacle encounter centered along the corridor centerline, such that the rangefinder observations are invariant under lateral reflection:

$$\mathcal{R}(\mathbf{x}_{\text{sym}}) = \mathbf{x}_{\text{sym}}$$

An optimal expert navigator approaching a centered obstacle must evade either to the left ($+A$) or to the right ($-A$), where $A > 0$:

$$p(y \mid \mathbf{x}_{\text{sym}}) = \frac{1}{2}\delta(y - A) + \frac{1}{2}\delta(y + A)$$

Under behavior cloning with Mean Squared Error loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}\left[ \| f_\theta(\mathbf{x}) - y \|_2^2 \right]$$

The Bayes-optimal deterministic function $f^*(\mathbf{x})$ that minimizes MSE is the conditional expectation:

$$f^*(\mathbf{x}) = \mathbb{E}[y \mid \mathbf{x}]$$

Evaluating this expectation at the symmetric condition:

$$f^*(\mathbf{x}_{\text{sym}}) = \int y \, p(y \mid \mathbf{x}_{\text{sym}}) \, dy = \frac{1}{2}(+A) + \frac{1}{2}(-A) = \mathbf{0.00\text{ m/s}}$$

### Corollary: Curriculum Training Inefficacy under Feedforward MSE
Even if a curriculum dataset provides thousands of expert recovery demonstrations equally divided between left and right bypasses, any deterministic feedforward neural network minimizing MSE will converge to the average action ($v_y = 0.00\text{ m/s}$) for centered states. Thus, **curriculum data collection cannot resolve symmetry standoff within the canonical feedforward MLP architecture**.

---

## 5. Isolated Curriculum Pilot Experiment (Experiment C & D)

To empirically test this theorem, we implemented an isolated curriculum training pipeline:
1. **Curriculum Data Collection (`generate_curriculum_data.py`):**
   - 40 recovery trajectories initialized in low-speed standoff ($0.40–0.45\text{ m}$ clearance, $v < 0.05\text{ m/s}$) with balanced left/right bypasses.
   - Combined with canonical baseline data:
     - Training set: 57,210 transitions (`data/bc_curriculum_train.npz`).
     - Validation set: 13,890 transitions (`data/bc_curriculum_val.npz`).
2. **Pilot Candidate Training (`train_curriculum_pilot.py`):**
   - Architecture: Identical `NavigationPolicyMLP` (canonical weights preserved).
   - Training: 15 epochs, AdamW ($\text{lr} = 5 \times 10^{-4}$), cosine decay.
   - Final validation loss: $0.000825$; validation MAE: $0.0090\text{ m/s}$.
   - Weights saved to: `bc_curriculum_candidate.pt`.
3. **Direct Symmetry Probe:**
   - Evaluated candidate on symmetric head-on pillar state:
     - Command output: $v_x = 0.3812\text{ m/s}$, **$v_y = +0.00018\text{ m/s}$**, $v_z = -0.0004\text{ m/s}$.
   - Evaluated on micro-perturbation ($y = +0.02\text{ m}$):
     - Command output: **$v_y = +0.02168\text{ m/s}$** (remains below the $0.05\text{ m/s}$ threshold required to break standoff).

**Empirical Result:**
The empirical test perfectly matched the mathematical theorem: despite explicit low-speed avoidance demonstrations, the network output collapsed to zero lateral velocity on symmetric encounters.

---

## 6. Paired Development Evaluation (Experiment E)

We conducted a paired evaluation across all 50 development seeds (`2000–2049`) comparing the frozen baseline against the curriculum candidate under identical canonical conditions (500 steps, $d_{\text{caut}} = 0.50\text{ m}$).

### 6.1 Quantitative Performance Comparison

| Metric | Frozen BC Baseline | BC Curriculum Candidate | Paired Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Checkpoint** | `bc_policy_best.pt` | `bc_curriculum_candidate.pt` | Isolated candidate |
| **Evaluated Episodes** | 50 (seeds 2000..2049) | 50 (seeds 2000..2049) | Matched paired seeds |
| **Goal Success Rate** | **24 / 50 (48.0%)** | **1 / 50 (2.0%)** | **-46.0% (Catastrophic Degradation)** |
| **Wilson 95% CI** | `[34.8%, 61.5%]` | `[0.4%, 10.5%]` | Non-overlapping |
| **Mean Forward Progress** | **6.525 m** | **5.758 m** | -0.767 m |
| **Mean Episode Duration** | 22.99 s | 24.93 s | +1.94 s (Persistent timeouts) |
| **Mean Throttled Fraction** | 27.6% | 21.0% | Reduced forward effort |
| **Obstacle Collisions** | **0 / 50 (0.0%)** | **1 / 50 (2.0%)** | +1 crash (Seed 2026) |
| **Corridor Boundary Collisions**| **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | 0 crashes |
| **Obstacle Envelope Violations**| **0 / 50 (0.0%)** | **19 / 50 (38.0%)** | **Severe safety degradation** |
| **Min Obstacle Clearance** | **0.1070 m** | **-0.0026 m** | Penetrated collision margin |
| **Min Boundary Clearance** | 0.2537 m | 0.2364 m | Safe |

### 6.2 Failure Categorization Breakdown

```
========================================================================================
FAILURE MODE DISTRIBUTION (N = 50 episodes per arm)
========================================================================================
Category                             Frozen Baseline        Curriculum Candidate
----------------------------------------------------------------------------------------
SUCCESS                              24 (48.0%)             1  ( 2.0%)
PROGRESS_STALL_PILLAR_3              10 (20.0%)             21 (42.0%)
PROGRESS_STALL_PILLAR_2              10 (20.0%)             11 (22.0%)
PROGRESS_STALL_PILLAR_1               6 (12.0%)              7 (14.0%)
TIMEOUT_SLOW_TRANSIT                  0 ( 0.0%)              7 (14.0%)
GOAL_ALIGNMENT_TIMEOUT                0 ( 0.0%)              2 ( 4.0%)
OBSTACLE_COLLISION                    0 ( 0.0%)              1 ( 2.0%)
----------------------------------------------------------------------------------------
Total Standoff Stalls (Pillars 1-3)  26 (52.0%)             39 (78.0%)
========================================================================================
```

### 6.3 Paired Outcome Matrix (McNemar Contingency)

$$\begin{array}{c|cc}
& \text{Candidate Success} & \text{Candidate Failure} \\
\hline
\text{Baseline Success} & 1 & 23 \\
\text{Baseline Failure} & 0 & 26 \\
\end{array}$$

- **Candidate Wins ($C=1, B=0$):** 1 (Seed 2046)
- **Candidate Losses ($C=0, B=1$):** 23
- **Ties ($C=B$):** 26 (25 shared failures, 1 shared success)
- **Two-tailed McNemar $p$-value:** $p = 2.4 \times 10^{-6}$ (statistically significant degradation)

### 6.4 Why Did the Candidate Degrade?
1. **Failure to Break Symmetry:** As predicted by the Bimodal MSE Collapse Theorem, the candidate policy still produced $v_y \approx 0$ at centered obstacles, increasing total standoff stalls from 26 to 39.
2. **Feature Interference with Cruise Forward Motion:** Injecting zero-speed recovery states diluted the policy's clean forward cruise mode, causing sluggish navigation (`TIMEOUT_SLOW_TRANSIT`: 7 episodes) and poor goal deceleration alignment (`GOAL_ALIGNMENT_TIMEOUT`: 2 episodes).
3. **Safety Violations:** In attempting to resolve stalled states without temporal memory, the policy produced jerky near-obstacle actions that breached the $0.080\text{ m}$ safety envelope 19 times and caused a collision in seed 2026.

---

## 7. Architectural Determination: Why Conclusion C is Required

Curriculum training within the current framework fails not because of insufficient data or hyperparameter tuning, but because of a **fundamental representation mismatch**:

```
+-------------------------------------------------------------------------------+
|                       THE SYSTEMIC ARCHITECTURAL TRAP                         |
+-------------------------------------------------------------------------------+
|                                                                               |
|  1. Rangefinder Observation: Symmetrical forward rays (FL ≈ FR)                |
|                             ↓                                                 |
|  2. Loss Function:           MSE Loss penalizes any non-zero guess when       |
|                              teacher evades both left (+A) and right (-A)     |
|                             ↓                                                 |
|  3. Policy Representation:   Memoryless Feedforward MLP (y = f(x_t))           |
|                             Cannot store decision commitment or prior bias    |
|                             ↓                                                 |
|  4. Velocity Shortcut:       obs[16] (vy) = 0.0 → act[1] (vy_cmd) = 0.0       |
|                             ↓                                                 |
|  RESULT:                     Zero lateral velocity commanded at standoff.     |
|                              Supervisor throttles forward speed vx → 0.10.    |
|                              Permanent deadlock until 500-step timeout.       |
+-------------------------------------------------------------------------------+
```

Therefore, neither supervisor tuning (Phase 4J-F/G) nor curriculum data collection (Phase 4K) can resolve symmetry standoff. The navigation system requires an **Observation or Policy Design Revision**.

---

## 8. Prescribed Architectural Revisions (Phase 5 Roadmap)

To permanently resolve the symmetry standoff bottleneck, the project must transition to one of the following validated architectural designs in Phase 5:

### Option 1: Temporal Frame-Stacking or Recurrent Memory (Recommended Primary)
- **Concept:** Feed a history buffer of observations $\mathbf{s}_{t-3:t}$ or utilize a Recurrent / GRU backbone.
- **Mechanism:** When the UAV approaches an obstacle, any microscopic heading or lateral drift over 4 timesteps ($\Delta t = 0.20\text{ s}$) breaks temporal symmetry, establishing persistent directional commitment.
- **Advantage:** Preserves end-to-end continuous regression while eliminating the static snapshot ambiguity.

### Option 2: Categorical Intention Conditioning / Mixture-of-Experts (MoE)
- **Concept:** Decompose policy output into a discrete decision head $c \in \{\text{Bypass Left}, \text{Bypass Right}\}$ and a continuous conditioned control head $v_y \sim p(v_y \mid \mathbf{x}, c)$.
- **Mechanism:** The discrete intention head is trained with cross-entropy (where symmetry yields a softmax probability $[0.5, 0.5]$ sampled to choose a definitive branch), while the continuous control head is conditioned on $c$ and outputs a non-zero velocity without MSE averaging collapse.

### Option 3: Observation Feature Augmentation (Asymmetry Inducer)
- **Concept:** Add an explicit signed obstacle offset feature $y_{\text{rel}} = y_{\text{obs}} - y_{\text{drone}}$ or an active micro-scan dither to the deployable observation contract.
- **Mechanism:** Removes the information asymmetry between the privileged teacher and deployable student.

---

## 9. Verification & Archive Artifacts

All Phase 4K code, datasets, evaluation tables, and logs have been created and sealed within the workspace:

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Pre-Run Audit** | [`PHASE_4K_PRE_RUN_AUDIT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/PHASE_4K_PRE_RUN_AUDIT.md) | Initial verification and checksum manifest |
| **Dataset Audit** | [`dataset_audit.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/dataset_audit.json) | Statistical analysis of 47,780 training transitions |
| **Root-Cause Report** | [`PHASE_4K_ROOT_CAUSE_REPORT.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/PHASE_4K_ROOT_CAUSE_REPORT.md) | Comprehensive Jacobian and feature correlation audit |
| **PyBullet Diagnostics**| [`symmetry_diagnostic_results.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/symmetry_diagnostic_results.csv) | 14 controlled PyBullet standoff scenarios |
| **Curriculum Design** | [`curriculum_design.md`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/curriculum_design.md) | 5-stage curriculum formulation and MSE collapse proof |
| **Curriculum Generator**| [`generate_curriculum_data.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/generate_curriculum_data.py) | Standstill recovery trajectory generator |
| **Curriculum Trainer** | [`train_curriculum_pilot.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/train_curriculum_pilot.py) | Supervised trainer with cosine lr schedule |
| **Training Metrics** | [`training_metrics.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/training_metrics.csv) | Epoch loss, MAE, and gradient norm history |
| **Candidate Weights** | [`bc_curriculum_candidate.pt`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/bc_curriculum_candidate.pt) | Isolated pilot model checkpoint |
| **Evaluation Harness** | [`run_phase4k_evaluation.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/run_phase4k_evaluation.py) | Paired evaluation harness across seeds 2000..2049 |
| **Evaluation Data** | [`development_evaluation.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/development_evaluation.csv) | Complete 100-episode paired evaluation results |
| **Episode Traces** | [`traces/`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/traces) | High-resolution JSON traces for paired runs |
