# Phase 4J-C — Mandatory Pre-Run Workspace and Configuration Audit

**Audit Date:** 2026-10-29  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Milestone:** Phase 4J-C Controlled BC Baseline Validation & Safety-Gate Audit  
**Target Subject:** Behavior-Cloning Baseline Policy (`models/checkpoints/bc_policy_best.pt`) under isolated supervisor caution distance $d_{\text{caut}} = 0.50\text{ m}$.

---

## 1. Git State and Repository Integrity

### 1.1 Working Directory and Branch
- **Working Directory:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Branch:** `main`
- **Pre-existing Working Tree State:** Unstaged modifications from prior phases are preserved (UI viewer files, legacy test files). No unstaged changes have been overwritten or discarded.

### 1.2 SHA-256 File Hash Manifest
Every file involved in the evaluation pipeline was hashed before execution:

| File Path | SHA-256 Hash | Status / Role |
|:---|:---|:---|
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **Target BC Baseline Checkpoint** (Verified) |
| `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | Canonical PPO Checkpoint (Preserved) |
| `experiments/phase4h/checkpoints/bc_phase4h_candidate.pt` | `1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b` | Phase 4H BC Candidate (Preserved, Not Evaluated as Subject) |
| `experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt` | `751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd` | Phase 4H PPO Candidate (Preserved) |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | Canonical Supervisor (Unmodified) |
| `experiments/phase4j/isolated_supervisor_opt.py` | `d68fbc38b58e5f2e21e5ad81efc1bc9d053734fe9f4d98b993445784b6b2f83f` | Isolated Experimental Supervisor Wrapper |
| `sim/corridor_env.py` | `4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66` | PyBullet Simulation Environment |
| `core/geometry.py` | `22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb` | Corridor & Obstacle Geometry |
| `core/contracts.py` | `1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab` | Observation & Action Definitions |
| `models/policy_net.py` | `fe5f8e81a09e4dd123d8cde011f4c7ac9d0511911d83fab1f550337a1f204ea9` | NavigationPolicyMLP Architecture |
| `experiments/phase4j/phase4jb_robustness_eval.py` | `88bdde7b5826d583d214d46cbdf14c0e94db8c266bbdc4a802d2f6baaee62aec` | Phase 4J-B Reference Evaluation Engine |

---

## 2. Checkpoint and Architecture Audit

### 2.1 Target Model Metadata (`bc_policy_best.pt`)
- **Training Epoch:** 25
- **Timestamp ISO:** `2026-10-24T06:02:16Z`
- **Validation Loss:** $0.000645$ (MAE: $0.00776$)
- **Network Parameters:** 28,740 parameters
- **Architecture:** `NavigationPolicyMLP(obs_dim=23, act_dim=4)`
  - Input Layer: 23 floats $\to$ Linear(128) $\to$ LayerNorm $\to$ ReLU
  - Hidden Layer 1: Linear(128) $\to$ LayerNorm $\to$ ReLU
  - Hidden Layer 2: Linear(64) $\to$ LayerNorm $\to$ ReLU
  - Output Head: Linear(4) $\to$ tanh bounded to action limits

### 2.2 Input Observation Space Contract (23 Dimensions)
- Dims `[0:12]`: Rangefinder distance rays (normalized to $[0.0, 1.0]$ over max range $4.0\text{ m}$).
- Dims `[12:15]`: Relative goal unit direction in body FLU frame: $[u_x, u_y, u_z]$ (scaled by $\min(1.0, d_{\text{goal}}/2.0)$).
- Dims `[15:18]`: Body velocities $[v_x, v_y, v_z]$ (m/s).
- Dims `[18:22]`: Trigonometric attitude $[\sin(\phi), \cos(\phi), \sin(\theta), \cos(\theta)]$.
- Dim `[22]`: Yaw rate $\dot{\psi}$ (rad/s).

### 2.3 Action Space Limits (`ActionLimits`)
- Forward velocity: $v_x \in [0.00, 0.60]\text{ m/s}$ (non-negative forward flight).
- Lateral velocity: $v_y \in [-0.40, 0.40]\text{ m/s}$.
- Vertical velocity: $v_z \in [-0.30, 0.30]\text{ m/s}$.
- Yaw rate: $\dot{\psi} \in [-1.00, 1.00]\text{ rad/s}$.

---

## 3. Simulator Environment and Geometric Specifications

### 3.1 Corridor Configuration (`Corridor3D.get_10m_benchmark()`)
- Length: $X \in [0.0, 10.0]\text{ m}$
- Full Width: $Y \in [-1.0, 1.0]\text{ m}$ (width $2.0\text{ m}$, centerline $Y = 0.0\text{ m}$)
- Height: $Z \in [0.0, 2.0]\text{ m}$ (floor $0.0\text{ m}$, ceiling $2.0\text{ m}$)
- Start Position: $(0.5, 0.0, 1.0)\text{ m}$ (with initial lateral perturbation $\Delta y \in [-0.1, 0.1]\text{ m}$)
- Destination Goal: $(9.5, 0.0, 1.0)\text{ m}$, acceptance radius $r_{\text{goal}} = 0.35\text{ m}$
- Drone Collision Geometry: Sphere with radius $r_{\text{drone}} = 0.15\text{ m}$, mass $1.2\text{ kg}$
- Obstacle Layout: 3 Cylindrical pillars at $X \in \{3.0, 5.0, 7.0\}\text{ m}$
  - Pillar radius: $r_{\text{pillar}} = 0.20\text{ m}$
  - Pillar height: $2.00\text{ m}$ (floor to ceiling)
  - Randomized lateral offsets: $Y_{\text{obs}} \in [-0.35, 0.35]\text{ m}$

### 3.2 Canonical Episode Horizon
- Physics timestep: $\Delta t = 0.05\text{ s}$ ($20\text{ Hz}$)
- Canonical Horizon: **500 steps** ($25.0\text{ seconds}$)
- Success criterion: Arrival within goal radius ($d_{\text{goal}} \le 0.35\text{ m}$) with terminal speed $v_{\text{term}} \le 0.35\text{ m/s}$.

---

## 4. Supervisor Configuration (`d_caut = 0.50 m`)

The isolated supervisor (`IsolatedOptimizedSupervisor`) configures the exact parameters:
- `obstacle_caution_dist_m`: **$0.50\text{ m}$** (onset of forward speed throttling)
- `critical_obstacle_dist_m`: **$0.30\text{ m}$** (emergency forward braking $v_x = 0.0$) [INVARIANT]
- `emergency_reverse_dist_m`: **$0.15\text{ m}$** (reverse retreat $v_x = -0.10$) [INVARIANT]
- `boundary_buffer_m`: **$0.25\text{ m}$** (corridor wall repulsion buffer) [INVARIANT]
- `repulsion_gain`: $1.5\text{ s}^{-1}$ [INVARIANT]
- `critical_lateral_dist_m`: $0.32\text{ m}$ [INVARIANT]
- `caution_lateral_dist_m`: $0.55\text{ m}$ [INVARIANT]

---

## 5. Seed Protection and Evaluation Set

- **Phase 4J-C Evaluation Seed Set:** Exactly **50 paired seeds: `2000..2049`**.
- **Reserved Held-Out Pool:** Seeds **`90000..90099`** are strictly protected and remain completely unexecuted, uninspected, and sealed.
- **Phase 4J-A Diagnostic Seeds:** Seeds `1000..1009` are preserved and separated.

