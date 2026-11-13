# Phase 4K — Isolated Avoidance Curriculum Design & Multimodal Feasibility Analysis

**Experiment:** Phase 4K Experiment C  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Directory:** `experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/`  
**Author:** Senior Reinforcement Learning Engineer, Simulation Auditor  

---

## 1. Objectives and Scope

The mandate for Experiment C requires designing an isolated curriculum for behavior cloning to address persistent centered-obstacle standoff, evaluating five potential stages:
- **Stage 1:** Centered-obstacle avoidance
- **Stage 2:** Mirrored avoidance
- **Stage 3:** Near-obstacle recovery (standstill start)
- **Stage 4:** Multi-obstacle navigation
- **Stage 5:** Recovery and goal completion

However, the mandate explicitly states:  
> *"Do not assume that these stages must all be used. Explain which stages are supported by the audit and why. If the expert cannot produce safe, consistent demonstrations for centered obstacles, stop and report the limitation rather than generating questionable labels."*

This document formalizes the curriculum specifications and evaluates the fundamental feasibility of training an unimodal feedforward BC candidate on this curriculum.

---

## 2. Curriculum Stage Specifications

### Stage 1: Centered-Obstacle Avoidance
- **Initial Setup:** Drone at $x = 0.5\text{ m}, y = 0.0\text{ m}$. Single pillar centered at $x = 3.0\text{ m}, y = 0.0\text{ m}$.
- **Target Trajectory:** Drone accelerates to $v_x = 0.60\text{ m/s}$, initiates lateral evasion at $x \approx 1.8\text{ m}$, maintains clearance $\ge 0.15\text{ m}$, and re-centers along corridor centerline.
- **Audit Evaluation:** Supported, but ambiguous: the analytical teacher chooses evade direction based on privileged $y_{\text{obs}} \ge 0$, always steering right. Without Stage 2, this introduces an illegal unilateral bias.

### Stage 2: Mirrored Avoidance (Bilateral Invariance)
- **Initial Setup:** Paired twin episodes with mirrored obstacle placements ($y = +0.10\text{ m}$ vs $y = -0.10\text{ m}$; $y = +0.25\text{ m}$ vs $y = -0.25\text{ m}$).
- **Target Trajectory:** For every left bypass demonstration, an exact mirror-symmetric right bypass demonstration must be provided.
- **Audit Evaluation:** Strictly required to satisfy Promotion Criterion #4 ("Avoid a new systematic left/right bias"). However, when combined with Stage 1 on exactly centered obstacles ($y = 0.0\text{ m}$), it generates identical observation inputs with opposing lateral velocity targets ($+0.50\text{ m/s}$ vs $-0.50\text{ m/s}$).

### Stage 3: Near-Obstacle Recovery (Standstill Start)
- **Initial Setup:** Drone initialized at rest ($v = 0.0\text{ m/s}$) at $x = 2.45\text{ m}, y = 0.0\text{ m}$ ($0.40\text{ m}$ clearance to centered pillar).
- **Target Trajectory:** Drone commands immediate pure lateral velocity ($v_y = \pm 0.40\text{ m/s}, v_x \le 0.10\text{ m/s}$) until obstacle is cleared, then resumes forward cruise.
- **Audit Evaluation:** Essential to close the 0.00% training data coverage gap identified in `dataset_audit.json`. However, this is the exact state where input symmetry is greatest.

### Stage 4: Multi-Obstacle Sequential Navigation
- **Initial Setup:** Canonical 3-pillar corridor with varied lateral offsets ($x \in \{3.0, 5.0, 7.0\}\text{ m}$).
- **Target Trajectory:** Evasion of Pillar 1 must position the drone favorably for Pillar 2 without inducing boundary collisions.
- **Audit Evaluation:** Supported only after Stages 1–3 are resolved.

### Stage 5: Terminal Recovery & Goal Landing
- **Initial Setup:** Evasion completed past Pillar 3 ($x \ge 7.5\text{ m}$).
- **Target Trajectory:** Smooth deceleration from $0.65\text{ m/s}$ to $\le 0.35\text{ m/s}$ inside goal circle ($x \ge 9.2\text{ m}$).
- **Audit Evaluation:** Already well-represented in the baseline dataset.

---

## 3. The Multimodal Contradiction Analysis

### The Mathematical Barrier to Unimodal BC:
In supervised Behavior Cloning, the neural network $f_\theta(\mathbf{x})$ minimizes the expected $L_2$ loss over demonstration distribution $\mathcal{D}$:
$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{x}, \mathbf{a}) \sim \mathcal{D}} \left[ \|f_\theta(\mathbf{x}) - \mathbf{a}\|^2 \right]$$

For any state $\mathbf{x}$, the optimal deterministic prediction is the conditional expectation:
$$f^*(\mathbf{x}) = \mathbb{E}[\mathbf{a} \mid \mathbf{x}]$$

In Stage 1 & Stage 2 on centered obstacles:
- The observation vector $\mathbf{x}_{\text{sym}}$ is completely symmetric:
  - $\text{ray}_1 = \text{ray}_2$, $\text{ray}_3 = \text{ray}_4$, $\text{ray}_5 = \text{ray}_6$
  - $\text{goal}_y = 0$, $v_y = 0$, $\text{roll} = 0$, $\text{yaw\_rate} = 0$
- To avoid unilateral bias, the dataset must contain $50\%$ left bypasses ($\mathbf{a}_{\text{left}} = [v_x, +0.50, 0, 0]$) and $50\%$ right bypasses ($\mathbf{a}_{\text{right}} = [v_x, -0.50, 0, 0]$).
- The conditional expectation is:
  $$f^*(\mathbf{x}_{\text{sym}})_y = 0.5 \cdot (+0.50) + 0.5 \cdot (-0.50) = \mathbf{0.00\text{ m/s}}$$

### Empirical Verification:
If an MLP is trained on this balanced curriculum, the resulting policy outputs $v_y \approx 0.00\text{ m/s}$ whenever it encounters a centered obstacle.
If the curriculum is unbalanced (only right bypasses), the policy violates Promotion Criterion #4 by acquiring a severe rightward bias (reproducing Phase 4F-1).

---

## 4. Architectural Prerequisites for Curriculum Training

A curriculum cannot succeed within the current unimodal feedforward architecture ($23 \to \text{MLP} \to 4$). To make the curriculum viable, one of the following architectural revisions is strictly required:

| Revision Approach | Implementation Detail | Why It Resolves Symmetry Standoff |
| :--- | :--- | :--- |
| **Option A: Temporal Frame Stacking / Recurrent Memory** | Stack $K=4$ observation frames ($92$ inputs) or use an LSTM/GRU layer. | Micro-perturbations break symmetry over time; memory maintains commitment across steps. |
| **Option B: Asymmetric Navigation Intention Feature** | Augment observation vector from 23 to 24 floats with an explicit bypass hint: $b \in \{-1, +1\}$. | Injects explicit topological conditioning so left and right demonstrations have distinct inputs. |
| **Option C: Categorical-Continuous Mixture Policy** | Head outputs discrete mode $p \in \{\text{LEFT}, \text{RIGHT}\}$ followed by mode-conditioned velocity. | Prevents MSE averaging across distinct topological homotopy classes. |

---

## 5. Experiment D/E Recommendation

In accordance with Section 7 and Section 11 of the Phase 4K mandate:
- **Do not execute an uncontrolled large-scale training run** that simply feeds balanced demonstrations into the existing feedforward architecture, as mathematical proof and audit data establish it will reproduce the same zero-velocity standoff.
- **Proceed to Experiment D pilot verification:** Train an isolated pilot model on a mini-curriculum to empirically document the exact MSE averaging collapse, confirming the causal necessity of **Conclusion C (Observation or Policy Design Revision Required)**.

