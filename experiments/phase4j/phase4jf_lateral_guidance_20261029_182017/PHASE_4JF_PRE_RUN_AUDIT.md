# Phase 4J-F — Pre-Run Workspace and Configuration Audit

**Audit Date:** 2026-10-29  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Milestone:** Phase 4J-F Isolated Proactive Lateral Guidance and Deadlock-Breaking Evaluation  
**Experiment Directory:** `experiments/phase4j/phase4jf_lateral_guidance_20261029_182017/`  
**Auditor:** Autonomous UAV Navigation & Simulation Validation Engineer  

---

## 1. Working Tree and Integrity Gate

### 1.1 Workspace Integrity
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Active Branch:** `main`
- **Working Tree State:** All existing uncommitted working-tree modifications from previous phases are preserved without deletion or reversion.
- **Experimental Isolation:** All source code, configs, tests, traces, and metrics for Phase 4J-F are strictly isolated in `experiments/phase4j/phase4jf_lateral_guidance_20261029_182017/`. Canonical files (`controllers/safety_supervisor.py`, `models/checkpoints/*`, `core/*`, `sim/*`) are unmodified.

### 1.2 Cryptographic File Hash Manifest (SHA-256)
Every critical file participating in the evaluation pipeline has been independently verified:

| File Path | SHA-256 Hash | Verification Status | Role in Phase 4J-F |
|:---|:---|:---|:---|
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **CONFIRMED** | Frozen Target Policy Checkpoint |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **UNMODIFIED** | Canonical Safety Supervisor |
| `experiments/phase4j/isolated_supervisor_opt.py` | `d68fbc38b58e5f2e21e5ad81efc1bc9d053734fe9f4d98b993445784b6b2f83f` | **CONFIRMED** | Baseline Supervisor Wrapper ($d_{\text{caut}} = 0.50\text{ m}$) |
| `sim/corridor_env.py` | `4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66` | **CONFIRMED** | 10m Corridor PyBullet Simulation |
| `core/geometry.py` | `22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb` | **CONFIRMED** | Corridor Geometry & Obstacle Specifications |
| `core/contracts.py` | `1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab` | **CONFIRMED** | Sensor Contracts & Action Bounds |
| `models/policy_net.py` | `fe5f8e81a09e4dd123d8cde011f4c7ac9d0511911d83fab1f550337a1f204ea9` | **CONFIRMED** | NavigationPolicyMLP Architecture |

---

## 2. Experimental Contracts and Invariants

Every contract governing the simulation, observation, action, and evaluation spaces was audited directly against the Python source files:

| Dimension / Parameter | Canonical Definition | Source File & Location | Verified Value | Discrepancy / Notes |
|:---|:---|:---|:---|:---|
| **Corridor Dimensions** | $10.0\text{ m} \times 2.0\text{ m} \times 2.0\text{ m}$ | `core/geometry.py:Corridor3D` | $X \in [0, 10], Y \in [-1, 1], Z \in [0, 2]$ | None |
| **Start Position** | $(0.5, 0.0, 1.0)\text{ m}$ | `core/geometry.py:Corridor3D` | $(0.5, 0.0, 1.0)\text{ m}$ | Small lateral perturbation $\Delta y \in [-0.1, 0.1]\text{ m}$ |
| **Goal Position** | $(9.5, 0.0, 1.0)\text{ m}$ | `core/geometry.py:Corridor3D` | $(9.5, 0.0, 1.0)\text{ m}$ | Total nominal forward transit: $9.0\text{ m}$ |
| **Goal Acceptance Radius** | $r_{\text{goal}} \le 0.35\text{ m}$ | `core/geometry.py:Corridor3D` | $0.35\text{ m}$ | None |
| **Terminal Speed Limit** | $v_{\text{term}} \le 0.35\text{ m/s}$ | Evaluation contracts | $0.35\text{ m/s}$ | Mandated for safe touchdown |
| **Drone Collision Geometry**| Sphere $r = 0.15\text{ m}$, mass $1.2\text{ kg}$ | `core/geometry.py`, `sim/corridor_env.py` | $r = 0.15\text{ m}$ | Center-to-surface radius |
| **Obstacle Geometry** | 3 Cylindrical pillars | `core/geometry.py` | $X \in \{3.0, 5.0, 7.0\}\text{ m}$ | Radius $r = 0.20\text{ m}$, height $2.0\text{ m}$, $Y \in [-0.35, 0.35]\text{ m}$ |
| **Simulation Timestep** | $\Delta t = 0.05\text{ s}$ ($20\text{ Hz}$) | `sim/corridor_env.py` | $0.05\text{ s}$ | Fixed-step physics |
| **Episode Horizon** | Exactly 500 steps ($25.0\text{ s}$) | `sim/corridor_env.py`, `core/geometry.py` | 500 steps | Invariant |
| **Collision Threshold** | $d \le 0.000\text{ m}$ | `sim/corridor_env.py`, `core/geometry.py` | $0.000\text{ m}$ | Physical contact |
| **Safety Envelope Threshold** | $d < 0.080\text{ m}$ ($8.0\text{ cm}$) | Evaluation specifications | $0.080\text{ m}$ | Standard near-miss violation margin |
| **Caution Distance ($d_{\text{caut}}$)** | $0.50\text{ m}$ | `isolated_supervisor_opt.py` | $0.50\text{ m}$ | Baseline reference |
| **Critical Braking Distance** | $0.30\text{ m}$ | Invariant safety barrier | $0.30\text{ m}$ | Hard forward brake ($v_x = 0.0$) |
| **Reverse Retreat Distance** | $0.15\text{ m}$ | Invariant safety barrier | $0.15\text{ m}$ | Reverse retreat ($v_x = -0.10\text{ m/s}$) |
| **Corridor Wall Buffer** | $0.25\text{ m}$ | Invariant safety barrier | $0.25\text{ m}$ | Lateral wall repulsion buffer |

---

## 3. Seed Set Integrity and Flight Safety Mandate

1. **Development Seed Set:** Exactly **50 development seeds (`2000..2049`)** are utilized for this paired experimental evaluation.
2. **Held-Out Seed Protection:** The reserved held-out seed pool **`90000..90099` remains strictly sealed, protected, and untouched**.
3. **Zero Physical Flight:** Absolutely zero physical flight commands or hardware communication to the RadioLink CrossFlight flight controller will occur during Phase 4J-F.

**Conclusion:** All pre-run checks pass. Phase 4J-F is authorized to proceed.

