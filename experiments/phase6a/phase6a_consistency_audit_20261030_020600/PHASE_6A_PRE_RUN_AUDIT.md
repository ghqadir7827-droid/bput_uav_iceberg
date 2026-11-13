# PHASE 6A PRE-RUN AUDIT & STARTING STATE VERIFICATION
**Autonomous UAV Navigation in GPS-Denied Corridors**  
**Run Identifier:** `phase6a_consistency_audit_20261030_020600`  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Audit Date:** October 30, 2026  

---

## 1. Workspace Isolation & Cryptographic Verification

All Phase 6A operations are confined to the canonical workspace `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav` and isolated inside the dedicated directory:  
`experiments/phase6a/phase6a_consistency_audit_20261030_020600/`.

Prior to introducing any edits, the working tree status was inspected via `git status`. All pre-existing untracked files and modifications were noted and preserved.

### 1.1 Cryptographic Integrity of Protected Assets

The SHA-256 hashes of the protected baselines were computed and verified against the canonical specification:

| Protected Asset | File Path | Expected SHA-256 Checksum | Measured SHA-256 Checksum | Gate Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Frozen BC Baseline** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **PASS (Exact Match)** |
| **Canonical Safety Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **PASS (Exact Match)** |
| **PPO Baseline Checkpoint** | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **PASS (Exact Match)** |
| **Sealed Held-Out Seeds** | Seeds `90000–90099` | Strictly Unread / Sealed | Sealed | **PROTECTED** |
| **Physical Hardware** | RadioLink CrossFlight / PX4 | Read-Only Lock | Inactive | **SAFE** |

### 1.2 Initial Hashes of Related Working Files

The following SHA-256 hashes represent the starting state of all candidate and evaluation files prior to Phase 6A:

```
9365aab7253c38793af5efa7cb7eb89b7e1db3bf60ea96e70df1671cdbbb4f50  models/checkpoints/policy_bc.onnx
38dd6575d9828b4616d740b296b067c7ee964fc7de1ff5e8f671c4ecb82a0ecf  models/checkpoints/policy_ppo.onnx
aab24029c02d25ed8008a2c300bad2065bd5d850adacded4eaccae6cfde1c053  controllers/guidance_controller.py
feff2d0cd627059f2ae258e6385160c22ebf88bcaa953a8df0dd4db6ca1cb72a  controllers/registry.py
6f0bb307825e741f4f93638c5ea7768ab66787d64c05bd430243f1a7a12934e5  hardware/onnx_runner.py
03959821018715f55eddf4433a73164d1687f60d927f11b72b5b53738f4ff97e  tests/test_guidance_controller.py
2763add9698bd1a9d37e6f3ac0356428c1055a5111de162abcd6c16a00fa2167  tests/test_onnx_runner.py
4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66  sim/corridor_env.py
22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb  core/geometry.py
1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab  core/contracts.py
```

---

## 2. Phase 6 Benchmark Verification & Mathematical Audit

The per-seed CSV (`experiments/phase6_recovery/phase6_recovery_20261030_003000/paired_development_results_phase6.csv`) and the summary JSON (`controller_comparison_summary.json`) were independently parsed and cross-checked using pure Python standard library routines.

### 2.1 Seed Matching and Simulation Invariants
- **Seed Consistency:** All 50 development seeds are strictly identical across all three controllers: integers `[2000, 2001, ..., 2049]`.
- **Horizon Invariant:** All 150 episodes were executed with the canonical 500-step limit ($25.0\text{ s}$ at $dt=0.05\text{ s}$).
- **Evaluation Criteria:** All controllers were evaluated under identical simulation conditions:
  - Destination goal radius: $0.35\text{ m}$
  - Maximum allowable landing speed: $0.35\text{ m/s}$
  - Obstacle critical threshold: $0.30\text{ m}$
  - Boundary buffer: $0.25\text{ m}$
  - Obstacle safety envelope: $0.080\text{ m}$
  - Wall safety envelope: $0.080\text{ m}$
- **Collision Classification:** Zero simulated physical collisions (defined as drone radius penetrating obstacle cylinder or corridor bounding plane, clearance $\le 0.0\text{ m}$) occurred in any of the 150 runs.

### 2.2 Independent Metric Recalculation

All metrics reported in the Phase 6 executive summary were re-derived directly from the 150 individual CSV rows:

| Controller | Trials | Calculated Success Rate | Summary JSON Rate | Min Obstacle Clearance | Min Wall Clearance | Mean Progress | Pillar 1 / 2 / 3 Passage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen BC Baseline** | 50 | **48.0% (24/50)** | 48.0% | 0.1070 m | 0.2537 m | 6.5255 m | 88.0% / 68.0% / 48.0% |
| **Phase 5B Stateless** | 50 | **70.0% (35/50)** | 70.0% | 0.0870 m | 0.1724 m | 7.9359 m | 96.0% / 94.0% / 78.0% |
| **Phase 6 Production** | 50 | **84.0% (42/50)** | 84.0% | 0.0957 m | 0.1942 m | 8.4752 m | 100.0% / 100.0% / 92.0% |

All figures match identically between raw episode logs and published summary metrics.

### 2.3 Exact Traceability of Minimum Clearance Episodes
- **Frozen BC Baseline:**
  - Minimum Obstacle Clearance: `0.1070 m` achieved on **Seed 2046** (Episode Outcome: `SUCCESS`).
  - Minimum Wall Clearance: `0.2537 m` achieved on **Seed 2000** (Episode Outcome: `SUCCESS`).
- **Phase 5B Stateless Guidance:**
  - Minimum Obstacle Clearance: `0.0870 m` achieved on **Seed 2042** (Episode Outcome: `PROGRESS_STALL_PILLAR_3`).
  - Minimum Wall Clearance: `0.1724 m` achieved on **Seed 2005** (Episode Outcome: `PROGRESS_STALL_PILLAR_3`).
- **Phase 6 Production Guidance:**
  - Minimum Obstacle Clearance: `0.0957 m` achieved on **Seed 2042** (Episode Outcome: `PROGRESS_STALL_PILLAR_3`).
  - Minimum Wall Clearance: `0.1942 m` achieved on **Seed 2042** (Episode Outcome: `PROGRESS_STALL_PILLAR_3`).

Both minimum clearances for Phase 6 safely exceed the mandatory $0.080\text{ m}$ threshold. Zero envelope violations occurred.

### 2.4 Paired Statistical Significance
The paired discordant table between Phase 6 Candidate and Frozen BC Baseline was recomputed:
- Candidate Wins ($C=1, B=0$): **23 seeds**
- Baseline Wins ($B=1, C=0$): **5 seeds**
- Ties ($C=1, B=1$): **19 seeds**
- Ties ($C=0, B=0$): **3 seeds**
- Total Discordant Pairs: $23 + 5 = 28$
- Exact Binomial Two-Sided McNemar Test: **$p = 0.000912234$**

---

## 3. Pre-Run Test Suite Baseline

The complete repository regression test suite was executed prior to initiating Phase 6A:
```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q
```
**Result:** **145 passed, 0 failed, 0 skipped** in $2.55\text{ s}$.

This establishes the clean, verified starting baseline for the Phase 6A campaign.

