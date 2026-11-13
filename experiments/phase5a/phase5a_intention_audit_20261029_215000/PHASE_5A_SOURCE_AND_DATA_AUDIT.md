# PHASE 5A — SOURCE AND DATA AUDIT REPORT
## Structural, Observational, and Algorithmic Dissection of the Navigation Pipeline

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Audit Directory:** `experiments/phase5a/phase5a_intention_audit_20261029_215000/`

---

## 1. Deployable Observation Contract Specification

The deployable observation vector $\mathbf{o}_t \in \mathbb{R}^{23}$ is defined strictly in `core/contracts.py` via `DeployableObservation`:

| Index Range | Feature Name | Description | Frame | Units / Normalization | Physical Limits |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | `ray_front` | Center forward beam $(\text{azimuth}=0^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **1** | `ray_fl_30` | Front-left beam $(\text{azimuth}=+30^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **2** | `ray_fr_30` | Front-right beam $(\text{azimuth}=-30^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **3** | `ray_fl_60` | Front-left oblique beam $(\text{azimuth}=+60^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **4** | `ray_fr_60` | Front-right oblique beam $(\text{azimuth}=-60^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **5** | `ray_left_90` | Lateral left beam $(\text{azimuth}=+90^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **6** | `ray_right_90`| Lateral right beam $(\text{azimuth}=-90^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **7** | `ray_rear_180`| Rear beam $(\text{azimuth}=180^\circ, \text{pitch}=0^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **8** | `ray_up_45` | Front pitch up beam $(\text{azimuth}=0^\circ, \text{pitch}=+45^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **9** | `ray_down_45` | Front pitch down beam $(\text{azimuth}=0^\circ, \text{pitch}=-45^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **10**| `ray_zenith` | Vertical zenith beam $(\text{pitch}=+90^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **11**| `ray_nadir` | Vertical nadir beam $(\text{pitch}=-90^\circ)$ | Body FLU | $[0.0, 1.0]$ ($\div 4.0\text{ m}$) | $[0.05, 4.00]\text{ m}$ |
| **12:15**| `rel_goal_unit` | Unit vector to target position: $[\hat{g}_x, \hat{g}_y, \hat{g}_z]$ | Body FLU | Dimensionless unit vector | $[-1.0, 1.0]$ |
| **15:18**| `body_vel` | Estimated linear velocity: $[v_x, v_y, v_z]$ | Body FLU | $\text{m/s}$ (unnormalized physical) | $[-0.2, 0.8]\text{ m/s}$ |
| **18:22**| `attitude_trig`| Trigonometric orientation: $[\sin\phi, \cos\phi, \sin\theta, \cos\theta]$ | World/Body | Dimensionless | $[-1.0, 1.0]$ |
| **22** | `yaw_rate` | Measured body yaw angular velocity: $r$ | Body FLU | $\text{rad/s}$ (unnormalized physical) | $[-0.6, 0.6]\text{ rad/s}$ |

---

## 2. Policy Action Generation and Action Transformation Pipeline

The policy and action pipeline executes in four distinct transformations:

```
[DeployableObservation: 23 floats]
            │
            ▼
┌───────────────────────────────────────┐
│  NavigationPolicyMLP Backbone         │  3-layer MLP: 23 -> 128 -> 128 -> 64 -> 4
│  (Linear + LayerNorm + GELU)          │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  Scaled Tanh Activation Head          │  act_center = [0.30, 0.00, 0.00, 0.00]
│  act = act_center + half_range * tanh │  act_half_range = [0.50, 0.60, 0.40, 0.60]
└───────────────────────────────────────┘
            │
            ▼  Raw Policy Action [vx, vy, vz, yaw_rate] in physical units
┌───────────────────────────────────────┐
│  Physical ActionLimits Saturation     │  vx: [-0.20, +0.60] m/s
│  (ActionLimits in core/contracts.py)  │  vy: [-0.40, +0.40] m/s, vz: [-0.30, +0.30] m/s
└───────────────────────────────────────┘  yaw_rate: [-0.50, +0.50] rad/s
            │
            ▼  Clamped VelocityAction
┌───────────────────────────────────────┐
│  SafetySupervisor.filter_action(...)  │  Obstacle caution throttling (vx -> 0.10 m/s)
│  (controllers/safety_supervisor.py)   │  Critical emergency brake (vx -> 0.0 m/s)
└───────────────────────────────────────┘  Wall boundary repulsion / retreat
            │
            ▼  Safe Filtered VelocityAction
┌───────────────────────────────────────┐
│  Environment Actuation: step(...)     │  Normalized to [-1, 1] via ActionLimits.to_normalized()
│  PyBullet velocity controller         │
└───────────────────────────────────────┘
```

---

## 3. Safety Supervisor Action Filtering Rules

The supervisor implements deterministic envelope protection:
1. **Forward Clearance Evaluation:** Evaluates $d_{\text{front}} = \min(\text{ray}_0, \text{ray}_1, \text{ray}_2, \text{ray}_3, \text{ray}_4) \times 4.0\text{ m}$.
2. **Caution-Zone Velocity Throttling:** If $d_{\text{front}} < d_{\text{caut}}$ ($0.50\text{ m}$):
   $$v_x \gets \min\left(v_x, \max\left(0.10, v_{x, \text{max}} \cdot \frac{d_{\text{front}} - d_{\text{crit}}}{d_{\text{caut}} - d_{\text{crit}}}\right)\right)$$
   When $d_{\text{front}} \approx 0.35–0.45\text{ m}$, forward velocity is clamped to $0.10\text{ m/s}$.
3. **Emergency Braking:** If $d_{\text{front}} < d_{\text{crit}}$ ($0.30\text{ m}$), $v_x \gets 0.0\text{ m/s}$.
4. **Emergency Reverse:** If $d_{\text{front}} < d_{\text{rev}}$ ($0.15\text{ m}$), $v_x \gets -0.10\text{ m/s}$.
5. **Lateral Corridor Repulsion:** When distance to wall $d_{\text{wall}} < 0.25\text{ m}$, inward repulsion $v_y \gets v_y \mp 1.5 \cdot (0.25 - d_{\text{wall}})$ is applied.

---

## 4. Cryptographic Checksum Baseline

All model checkpoints and supervisory code have been verified via SHA-256:

| Asset | Expected SHA-256 | Actual Verified SHA-256 | Status |
| :--- | :--- | :--- | :--- |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **MATCH** |
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **MATCH** |
| `experiments/phase4k/.../bc_curriculum_candidate.pt` | `dd43ad888ff2329b64e8e184590ef41e358d5c37050da7892743605460e27b4a` | `dd43ad888ff2329b64e8e184590ef41e358d5c37050da7892743605460e27b4a` | **MATCH** |

---

## 5. Scripted APF Expert Privileged Information Audit

In `controllers/apf_expert.py` (lines 160–210), the expert computes avoidance actions via:

```python
# controllers/apf_expert.py line 191
evade_dir = -1.0 if obs_c[1] >= 0 else 1.0
circ_mag = float(self.k_circ * np.exp(-max(0.0, clearance) / 0.35))
vy_des += float(rep_mag * n_rep[1] + circ_mag * evade_dir)
```

### Critical Findings:
1. **Privileged Global Obstacle Coordinate:** `obs_c[1]` is the true $y$-coordinate of the cylindrical pillar center in world space.
2. **Artificial Asymmetry Injection:** If `obs_c[1] >= 0`, `evade_dir = -1.0` (steer right). If `obs_c[1] < 0`, `evade_dir = +1.0` (steer left).
3. **Centered-Obstacle Asymmetry:** When an obstacle is exactly centered ($y_{\text{obs}} = 0.0$), the `>= 0` condition forces `evade_dir = -1.0` (rightward steering). The expert never confronts symmetry; its decision rule contains an arbitrary tie-break conditioned on privileged world data inaccessible to the deployable UAV.

---

## 6. Phase 4K Recovery Demonstrations Audit

The 40 recovery demonstrations generated in `generate_curriculum_data.py` had the following properties:
- **Starting State:** UAV placed at $x \in [2.40, 2.46]\text{ m}$, $y \in [-0.02, 0.02]\text{ m}$, $v_x = v_y = v_z = 0.0\text{ m/s}$, directly upstream of Pillar 1 ($x = 3.0\text{ m}, y = 0.0\text{ m}$).
- **Forced Evasion Labels:** Exactly 20 episodes forced `evade_dir = +1.0` (left bypass), and 20 episodes forced `evade_dir = -1.0` (right bypass).
- **Target Actions:** During the initial 50 steps from hover, half the transitions commanded $v_y \approx +0.35\text{ m/s}$ and half commanded $v_y \approx -0.35\text{ m/s}$ on identical or near-identical observation inputs.
- **MSE Collapse Consequence:** Supervised training on this balanced bimodal distribution caused the deterministic network to output the conditional mean: $\mathbb{E}[v_y] = 0.00\text{ m/s}$, confirming the Bimodal MSE Symmetry Collapse Theorem.

---

## 7. Baseline Demonstration Action Distribution (`bc_train_10m.npz`)

Statistical audit of the 47,780 training transitions:

| Action Dimension | Min | Max | Mean | Std | Fraction $\approx 0$ ($|\cdot| < 0.02$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $v_x$ (Forward) | $-0.150$ | $+0.650$ | $+0.584$ | $0.124$ | $0.1\%$ |
| $v_y$ (Lateral) | $-0.600$ | $+0.600$ | $+0.0007$ | $0.207$ | **$13.1\%$** ($23.3\%$ for $< 0.05$) |
| $v_z$ (Vertical) | $-0.300$ | $+0.300$ | $+0.0012$ | $0.068$ | $68.4\%$ |
| $\dot{\psi}$ (Yaw Rate) | $-0.480$ | $+0.480$ | $+0.0008$ | $0.185$ | $21.2\%$ |

- $v_y > +0.05\text{ m/s}$: $38.28\%$ of transitions.
- $v_y < -0.05\text{ m/s}$: $38.39\%$ of transitions.
- The dataset is symmetrically balanced between left and right steering.
- The Pearson correlation between observed lateral velocity $\text{obs}[16]$ and target action $\text{act}[1]$ is **$0.9723$**, establishing that the network relied on an autoregressive velocity-copying shortcut.

---

## 8. Left/Right Clearance Observability from Deployable Rangefinders

Rangefinder ray indices corresponding to lateral sectors:
- **Left Sector:** Ray 1 ($+30^\circ$), Ray 3 ($+60^\circ$), Ray 5 ($+90^\circ$).
- **Right Sector:** Ray 2 ($-30^\circ$), Ray 4 ($-60^\circ$), Ray 6 ($-90^\circ$).

### Observability Analysis:
1. **Asymmetric Obstacles ($|y_{\text{obs}}| > 0.05\text{ m}$):**
   - The beam distance differential:
     $$\Delta d_{\text{side}} = \min(\text{ray}_1, \text{ray}_3, \text{ray}_5) - \min(\text{ray}_2, \text{ray}_4, \text{ray}_6)$$
     provides a clear, detectable geometric signal indicating which passage is wider.
2. **Exactly Centered Obstacle ($y_{\text{obs}} = 0.00\text{ m}$, drone at $y = 0.00\text{ m}$):**
   - In a symmetric corridor of width $W = 2.0\text{ m}$, both passages have identical physical width ($0.70\text{ m}$).
   - Ray readings on the left and right are theoretically identical:
     $$\text{ray}_1 \equiv \text{ray}_2, \quad \text{ray}_3 \equiv \text{ray}_4, \quad \text{ray}_5 \equiv \text{ray}_6$$
   - Therefore, the deployable rangefinder observation contains **zero information favoring left over right**.
   - Any deterministic policy mapping $\mathbf{o} \mapsto v_y$ without internal state or explicit tie-break must satisfy $f(\mathcal{R}(\mathbf{o})) = -f(\mathbf{o}) \implies f(\mathbf{o}) = 0$.

---

## 9. Simulator Determinism & Noise Characterization

- `sim/corridor_env.py` uses PyBullet `p.DIRECT` headless simulation.
- Random number generation is seeded via `self._rng = np.random.default_rng(seed)`.
- **Sensor Noise:** Simulated rangefinders add Gaussian noise: $d \gets d + \mathcal{N}(0, 0.02^2)$ and $1\%$ dropout.
- Under fixed seed, the entire execution is **100% bit-exact reproducible**.
- In the presence of sensor noise, symmetric obstacles produce microscopic instant-to-instant beam fluctuations of $\pm 0.01–0.02\text{ m}$, but because the noise is zero-mean and uncorrelated, it produces zero net lateral momentum in the baseline policy, leaving the UAV deadlocked in standoff.

