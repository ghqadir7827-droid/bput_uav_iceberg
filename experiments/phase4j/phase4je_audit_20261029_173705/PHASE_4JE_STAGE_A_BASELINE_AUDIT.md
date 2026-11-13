# Phase 4J-E — Stage A: Canonical Baseline and Experimental Contracts Audit

**Audit Date:** 2026-10-29  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Milestone:** Phase 4J-E Historical PPO Audit & Controlled BC vs PPO Comparison  
**Lead Auditor:** Independent Simulation-Validation & Experiment Auditor  

---

## 1. Repository State and Working Tree Audit

### 1.1 Workspace and Environment
- **Workspace Path:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Active Git Branch:** `main`
- **Python Interpreter:** Python 3.11.17 (`.venv/bin/python`)
- **Key Dependencies:**
  - `torch`: 2.14.1 (CPU inference)
  - `pybullet`: Oct 23 2026 build
  - `numpy`: 2.4.6
  - `pytest`: 9.1.1 (128 passing tests)

### 1.2 Git Status & Working Tree Modifications
Prior to beginning Phase 4J-E, the working tree contains uncommitted changes from earlier phases (UI viewer integration, diagnostic test suites, and hardware runner guards). These changes are preserved intact without modification:
- Modified files: `controllers/safety_supervisor.py`, `core/contracts.py`, `hardware/onnx_runner.py`, `models/actor_critic.py`, `models/train_ppo.py`, `scripts/run_pybullet_gui.py`, `scripts/run_quantitative_eval.py`, `scripts/smoke_test.py`, `tests/test_contracts.py`, `tests/test_safety_supervisor.py`, `viewer/eval_engine.py`, `viewer/server.py`, `viewer/static/*`.
- Untracked artifacts: `experiments/`, `reports/`, previous diagnostic test suites.
- Rule confirmation: Zero canonical files or checkpoints have been modified, overwritten, or deleted.

---

## 2. Experimental Contracts and Invariants

Every contract governing the simulation, observation, action, and evaluation spaces was audited directly against the Python source files:

| Dimension / Parameter | Canonical Definition | Source File & Location | Verified Value | Discrepancy / Notes |
|:---|:---|:---|:---|:---|
| **Corridor Length** | $X \in [0.0, 10.0]\text{ m}$ | `core/geometry.py:Corridor3D` | $10.0\text{ m}$ | None |
| **Corridor Width** | $Y \in [-1.0, 1.0]\text{ m}$ | `core/geometry.py:Corridor3D` | $2.0\text{ m}$ (centerline $Y=0.0$) | None |
| **Corridor Height** | $Z \in [0.0, 2.0]\text{ m}$ | `core/geometry.py:Corridor3D` | $2.0\text{ m}$ (floor 0, ceiling 2) | None |
| **Start Position** | $(0.5, 0.0, 1.0)\text{ m}$ | `core/geometry.py:Corridor3D` | $(0.5, 0.0, 1.0)\text{ m}$ | Small lateral perturbation $\Delta y \in [-0.1, 0.1]\text{ m}$ |
| **Goal Position** | $(9.5, 0.0, 1.0)\text{ m}$ | `core/geometry.py:Corridor3D` | $(9.5, 0.0, 1.0)\text{ m}$ | Total nominal forward transit: $9.0\text{ m}$ |
| **Goal Acceptance Radius** | $r_{\text{goal}} \le 0.35\text{ m}$ | `core/geometry.py:Corridor3D` | $0.35\text{ m}$ | None |
| **Terminal Speed Limit** | $v_{\text{term}} \le 0.35\text{ m/s}$ | Evaluation contracts | $0.35\text{ m/s}$ | Mandated for safe touchdown |
| **Drone Collision Geometry**| Sphere $r = 0.15\text{ m}$, mass $1.2\text{ kg}$ | `core/geometry.py`, `sim/corridor_env.py` | $r = 0.15\text{ m}$ | Center-to-surface radius |
| **Obstacle Geometry** | 3 Cylindrical pillars | `core/geometry.py` | $X \in \{3.0, 5.0, 7.0\}\text{ m}$ | Radius $r = 0.20\text{ m}$, height $2.0\text{ m}$, $Y \in [-0.35, 0.35]\text{ m}$ |
| **Simulation Timestep** | $\Delta t = 0.05\text{ s}$ ($20\text{ Hz}$) | `sim/corridor_env.py` | $0.05\text{ s}$ | Fixed-step physics |
| **Episode Horizon** | Exactly 500 steps ($25.0\text{ s}$) | `sim/corridor_env.py`, `core/geometry.py` | 500 steps | Discrepancy noted: Phase 4G used 200 steps (see Stage B) |
| **Collision Threshold** | $d \le 0.000\text{ m}$ | `sim/corridor_env.py`, `core/geometry.py` | $0.000\text{ m}$ | Physical contact |
| **Safety Envelope Threshold** | $d < 0.080\text{ m}$ ($8.0\text{ cm}$) | Evaluation specifications | $0.080\text{ m}$ | Standard near-miss violation margin |

---

## 3. Observation Space Contract (23 Dimensions)

The deployable observation vector defined in `core/contracts.py:DeployableObservation` consists of 23 float32 features:

1. **Dims `[0:12]` (12 floats) — Rangefinder LiDAR Rays:**
   - 8 Horizontal azimuth rays: $0^\circ$ (front), $\pm 30^\circ$, $\pm 60^\circ$, $\pm 90^\circ$ (sides), $180^\circ$ (rear).
   - 2 Vertical pitch rays: $+45^\circ$ (up-front), $-45^\circ$ (down-front).
   - 2 Orthogonal vertical rays: $+90^\circ$ (zenith/ceiling), $-90^\circ$ (nadir/floor).
   - Normalization: Normalized to $[0.0, 1.0]$ via $d / d_{\max}$ where $d_{\max} = 4.0\text{ m}$.
2. **Dims `[12:15]` (3 floats) — Relative Goal Unit Direction:**
   - Unit vector $[u_x, u_y, u_z]$ in body FLU frame pointing toward goal $(9.5, 0.0, 1.0)$.
   - Scaled by proximity attenuation $\min(1.0, d_{\text{goal}} / 2.0)$.
3. **Dims `[15:18]` (3 floats) — Body Velocities:**
   - Measured body FLU velocities $[v_x, v_y, v_z]$ in $\text{m/s}$.
4. **Dims `[18:22]` (4 floats) — Trigonometric Attitude:**
   - Attitude representation: $[\sin\phi, \cos\phi, \sin\theta, \cos\theta]$ (roll and pitch angles).
5. **Dim `[22]` (1 float) — Body Yaw Rate:**
   - Yaw angular rate $\dot{\psi}$ in $\text{rad/s}$.

---

## 4. Action Space Limits Contract

From `core/contracts.py:ActionLimits` and `sim/corridor_env.py`:
- **Forward Velocity ($v_x$):** $[-0.20, 0.60]\text{ m/s}$ (bounded in policy evaluation to non-negative $[0.00, 0.60]\text{ m/s}$ forward flight; negative speeds reserved for supervisor retreat).
- **Lateral Velocity ($v_y$):** $[-0.40, 0.40]\text{ m/s}$ ($+Y = \text{Left}$, $-Y = \text{Right}$).
- **Vertical Velocity ($v_z$):** $[-0.30, 0.30]\text{ m/s}$ ($+Z = \text{Up}$, $-Z = \text{Down}$).
- **Yaw Angular Rate ($\dot{\psi}$):** $[-0.50, 0.50]\text{ rad/s}$ ($+\text{CCW}$, $-\text{CW}$).

---

## 5. Checkpoint Inventory and SHA-256 Hashes

The candidate checkpoints selected for evaluation and comparison have been verified with cryptographic SHA-256 hashes:

| Checkpoint Identifier | File Path | SHA-256 Hash | Status / Role |
|:---|:---|:---|:---|
| **BC Baseline** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **Primary Evaluation Subject** |
| **PPO Baseline** | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **Primary Evaluation Subject** |
| **BC Phase 4H Candidate** | `experiments/phase4h/checkpoints/bc_phase4h_candidate.pt` | `1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b` | Secondary Diagnostic Subject |
| **PPO Phase 4H Candidate** | `experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt` | `751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd` | Secondary Diagnostic Subject |
| **Canonical Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | Canonical Source Reference |
| **Isolated Supervisor Opt**| `experiments/phase4j/isolated_supervisor_opt.py` | `d68fbc38b58e5f2e21e5ad81efc1bc9d053734fe9f4d98b993445784b6b2f83f` | Evaluation Wrapper ($d_{\text{caut}} = 0.50\text{ m}$) |
| **Simulation Environment**| `sim/corridor_env.py` | `4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66` | PyBullet Environment Source |
| **Geometry Definitions** | `core/geometry.py` | `22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb` | 10m Corridor Geometry Source |
| **Contracts Definition** | `core/contracts.py` | `1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab` | Observation & Action Limits |
| **Policy Net Architecture**| `models/policy_net.py` | `fe5f8e81a09e4dd123d8cde011f4c7ac9d0511911d83fab1f550337a1f204ea9` | NavigationPolicyMLP Source |
| **ActorCritic PPO Net** | `models/actor_critic.py` | `b7a8eec4828a849c9ddd36ad37d707baeb1bbecf0495b93c3a2b01a2eca30162` | ActorCriticPPO Source |

---

## 6. Audit Summary of Experimental Discrepancies Discovered

1. **Phase 4G Horizon Mismatch:** Historical Phase 4G training (`train_phase4g.py`) and evaluation hardcoded a 200-step ($10.0\text{ s}$) horizon, whereas the canonical horizon is 500 steps ($25.0\text{ s}$). At maximum forward velocity ($0.60\text{ m/s}$), reaching the $9.0\text{ m}$ distant goal within 200 steps was mathematically impossible.
2. **Missing `log_std` in PPO Baseline:** `models/checkpoints/ppo_policy_best.pt` does NOT contain the parameter `log_std`. It contains `model_state_dict` (actor) and `critic_state_dict` (critic). Thus, it does not represent a complete training resume state, but deterministic evaluation using the actor's mean output is fully and bit-exactly supported.
3. **Supervisor Caution Distance Evolution:** Milestone 4 evaluations were conducted with an early supervisor lacking caution-zone velocity throttling ($d_{\text{caut}}$ did not exist; only a hard brake at $0.30\text{ m}$ existed). Later phases (4B, 4C, 4H) introduced $d_{\text{caut}} = 0.80\text{ m}$, which triggered severe timeouts. Phase 4J optimized this to $d_{\text{caut}} = 0.50\text{ m}$.
4. **Seed Pool Isolation:** The completed held-out seed set `90000..90099` is strictly protected and remains unaccessed. All controlled comparisons in Phase 4J-E are conducted on the development seed set `2000..2049` (50 paired episodes).

**Conclusion:** All baseline contracts are verified. Stage A is complete.

