# Phase 4J-D — Pre-Run Workspace and Configuration Audit

**Audit Date:** 2026-10-29  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Milestone:** Phase 4J-D Sealed Held-Out Generalization Evaluation  
**Subject:** Behavior-Cloning Baseline Policy (`models/checkpoints/bc_policy_best.pt`) with Isolated Supervisor Caution Distance $d_{\text{caut}} = 0.50\text{ m}$.

---

## 1. Repository State and Integrity Gate

### 1.1 Working Directory and Branch
- **Working Directory:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Active Branch:** `main`
- **Working Tree Policy:** Zero canonical code modifications. All prior experimental scripts, logs, and artifacts are strictly preserved.

### 1.2 Cryptographic File Hash Audit (SHA-256)
Every critical file participating in the evaluation pipeline has been independently hashed and verified prior to unsealing the held-out seed pool:

| File Path | SHA-256 Hash | Verification Status | Role in Phase 4J-D |
|:---|:---|:---|:---|
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **VERIFIED MATCH** | Frozen Candidate BC Model |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **VERIFIED MATCH** | Canonical Safety Supervisor (Preserved Unmodified) |
| `experiments/phase4j/isolated_supervisor_opt.py` | `d68fbc38b58e5f2e21e5ad81efc1bc9d053734fe9f4d98b993445784b6b2f83f` | **VERIFIED MATCH** | Isolated Experimental Supervisor Wrapper |
| `sim/corridor_env.py` | `4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66` | **VERIFIED MATCH** | PyBullet 10m Corridor Simulation Environment |
| `core/geometry.py` | `22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb` | **VERIFIED MATCH** | Corridor Geometry & Pillar Definitions |
| `core/contracts.py` | `1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab` | **VERIFIED MATCH** | Action Limits, VelocityAction & Sensor Contracts |
| `models/policy_net.py` | `fe5f8e81a09e4dd123d8cde011f4c7ac9d0511911d83fab1f550337a1f204ea9` | **VERIFIED MATCH** | NavigationPolicyMLP Neural Network Definition |
| `experiments/phase4j/phase4jc_results/FROZEN_CONFIGURATION_MANIFEST.json` | *Existing Phase 4J-C Manifest* | **CONFIRMED** | Baseline Reference Manifest |

---

## 2. Frozen Candidate Specification

### 2.1 Model Architecture & Checkpoint
- **Controller Identifier:** `Arm_B_BC_Baseline`
- **Model Checkpoint Path:** `models/checkpoints/bc_policy_best.pt`
- **Checkpoint SHA-256:** `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f`
- **Network Parameters:** 28,740 parameters
- **Architecture:** `NavigationPolicyMLP(obs_dim=23, act_dim=4)`
  - Input Layer: 23 floats $\to$ Linear(128) $\to$ LayerNorm $\to$ ReLU
  - Hidden Layer 1: Linear(128) $\to$ LayerNorm $\to$ ReLU
  - Hidden Layer 2: Linear(64) $\to$ LayerNorm $\to$ ReLU
  - Output Head: Linear(4) $\to$ tanh bounded to ActionLimits
- **Determinism:** Forward pass evaluated deterministically in PyTorch (`torch.no_grad()`, `model.eval()`).

### 2.2 Observation Space Contract (23 Dimensions)
- `[0:12]`: 12 horizontal rangefinder rays normalized to $[0.0, 1.0]$ over max range $4.0\text{ m}$.
- `[12:15]`: Relative goal unit direction in body FLU frame $[u_x, u_y, u_z]$ scaled by $\min(1.0, d_{\text{goal}} / 2.0)$.
- `[15:18]`: Body velocities $[v_x, v_y, v_z]\text{ m/s}$.
- `[18:22]`: Attitude angles $[\sin\phi, \cos\phi, \sin\theta, \cos\theta]$.
- `[22]`: Yaw rate $\dot{\psi}\text{ rad/s}$.

### 2.3 Action Space Limits
- $v_x \in [0.00, 0.60]\text{ m/s}$ (non-negative forward velocity)
- $v_y \in [-0.40, 0.40]\text{ m/s}$ (lateral velocity)
- $v_z \in [-0.30, 0.30]\text{ m/s}$ (vertical velocity)
- $\dot{\psi} \in [-1.00, 1.00]\text{ rad/s}$ (yaw rate)

---

## 3. Isolated Supervisor Configuration ($d_{\text{caut}} = 0.50\text{ m}$)

The evaluation uses `IsolatedOptimizedSupervisor` wrapping the safety logic:
- `obstacle_caution_dist_m`: **$0.50\text{ m}$** (Linear velocity throttling onset threshold)
- `critical_obstacle_dist_m`: **$0.30\text{ m}$** (Emergency forward brake threshold: $v_x = 0.0$) [STRICT INVARIANT]
- `emergency_reverse_dist_m`: **$0.15\text{ m}$** (Reverse retreat threshold: $v_x = -0.10\text{ m/s}$) [STRICT INVARIANT]
- `boundary_buffer_m`: **$0.25\text{ m}$** (Corridor wall lateral repulsion threshold) [STRICT INVARIANT]
- `min_safe_speed_mps`: **$0.10\text{ m/s}$** (Floor speed during caution throttling)
- `repulsion_gain`: **$1.5\text{ s}^{-1}$** [STRICT INVARIANT]
- `critical_lateral_dist_m`: **$0.32\text{ m}$** [STRICT INVARIANT]
- `caution_lateral_dist_m`: **$0.55\text{ m}$** [STRICT INVARIANT]
- **Oracle Geometric Monitor Exclusion:** As mandated, the isolated oracle geometric monitor is **NOT** included in this primary held-out evaluation. The candidate is evaluated under the exact frozen baseline supervisor configuration.

---

## 4. Corridor Environment and Simulation Horizons

- **Scenario:** 10-meter canonical corridor (`Corridor3D.get_10m_benchmark()`)
- **Dimensions:** Length $X \in [0.0, 10.0]\text{ m}$, Width $Y \in [-1.0, 1.0]\text{ m}$, Height $Z \in [0.0, 2.0]\text{ m}$.
- **Obstacles:** 3 cylindrical pillars at $X \in \{3.0, 5.0, 7.0\}\text{ m}$, radius $r = 0.20\text{ m}$, height $2.0\text{ m}$ (spanning full floor-to-ceiling height). Lateral offsets $Y \in [-0.35, 0.35]\text{ m}$ seeded per episode.
- **Drone Geometry:** Collision sphere with radius $r_{\text{drone}} = 0.15\text{ m}$, mass $1.2\text{ kg}$.
- **Physics Timestep:** $\Delta t = 0.05\text{ s}$ ($20\text{ Hz}$).
- **Episode Horizon:** Exactly **500 steps** ($25.0\text{ seconds}$).
- **Goal Criteria:** Drone center within $r_{\text{goal}} = 0.35\text{ m}$ of $(9.5, 0.0, 1.0)\text{ m}$ with terminal speed $v_{\text{term}} \le 0.35\text{ m/s}$.

---

## 5. Seed Set Integrity and Unsealing Authorization

### 5.1 Verification of Prior Seed Isolation
- **Phase 4J-A Development Set:** Seeds `1000..1009` (10 seeds)
- **Phase 4J-B / 4J-C Development Set:** Seeds `2000..2049` (50 paired seeds)
- **Reserved Held-Out Pool:** Seeds **`90000..90099`** (100 seeds)
- **Audit Result:** A comprehensive audit of all Phase 4J logs and source files confirms that seeds `90000..90099` have never been executed, inspected, or tuned against in Phase 4J. They are formally unsealed for single-pass evaluation now.

---

## 6. Pre-Run Gate Sign-off

| Checklist Item | Requirement | Actual Value / Status | Result |
|:---|:---|:---|:---|
| Checkpoint SHA-256 | `93c4d7d1d847...` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **PASS** |
| Canonical Supervisor SHA-256 | `9f84af22821f...` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **PASS** |
| Caution Distance | $0.50\text{ m}$ | $0.50\text{ m}$ | **PASS** |
| Critical Distance | $0.30\text{ m}$ | $0.30\text{ m}$ | **PASS** |
| Reverse Distance | $0.15\text{ m}$ | $0.15\text{ m}$ | **PASS** |
| Wall Buffer | $0.25\text{ m}$ | $0.25\text{ m}$ | **PASS** |
| Episode Horizon | 500 steps ($25.0\text{ s}$) | 500 steps ($25.0\text{ s}$) | **PASS** |
| Timestep | $0.05\text{ s}$ ($20\text{ Hz}$) | $0.05\text{ s}$ ($20\text{ Hz}$) | **PASS** |
| Held-Out Seed Range | `90000..90099` (100 seeds) | `90000..90099` (100 seeds) | **PASS** |
| Oracle Monitor Excluded | Yes (Baseline Only) | Confirmed Excluded | **PASS** |
| Zero Physical Flight | Mandatory | Zero hardware flight commands | **PASS** |

**Conclusion:** All pre-run gates pass. The evaluation is authorized to proceed.

