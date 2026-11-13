# Phase 6B Change Ledger

**Phase:** Phase 6B — Targeted Pillar 3 Recovery Experiment  
**Timestamp:** 2026-10-30T02:58:00+05:30  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Experiment Root:** `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/`  

---

## 1. Summary of Changes

All algorithmic candidate development and diagnostic tooling was isolated entirely within `experiments/phase6b/` and `tests/`. 
In accordance with Section 7 (Configuration Consistency), explicit parameter configuration (`obstacle_caution_dist_m = 0.50`) was added to auxiliary evaluation and viewer tools without modifying the default implementation in `controllers/safety_supervisor.py`.

---

## 2. Protected Files (Strictly Unmodified)

| File Path | SHA-256 Checksum | Invariant Status |
| :--- | :--- | :--- |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **Zero Changes** |
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **Zero Changes** |
| `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **Zero Changes** |
| `core/guidance.py` | `c69b7ca069e1c43668718f940b0dcc3ea0f2437b99412d89c88d146b1032c5d9` | **Zero Changes** |
| `controllers/guidance_controller.py` | `2818fa497cd9aa1657114a895ce031edfc0687e1211cbead4003caffdd47dc21` | **Zero Changes** |
| `hardware/onnx_runner.py` | `d80e4d8f2cfbbd9e7184ddfc138ffeb2614d0c2bb4861c1413a583d417cc30cd` | **Zero Changes** |

---

## 3. Files Created in Phase 6B

### A. Isolated Candidate & Algorithm Implementations
1. `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/recovery_guidance.py`:
   - Pure-NumPy implementation of `PostEvasionRecoveryArbitrator` and `RecoveryConfig`.
   - Incorporates observable corridor width validation, post-evasion activation gating, hysteresis thresholds ($0.22\text{ m} / 0.08\text{ m}$), side clearance checks ($\ge 0.55\text{ m}$), and bounded lateral velocity setpoints.
2. `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/pillar3_recovery_controller.py`:
   - Standalone `IController` wrapper combining the frozen BC policy with `PostEvasionRecoveryArbitrator`.

### B. Diagnostic & Benchmark Tooling
3. `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/diagnose_phase6a_failures.py`:
   - Preflight reproduction harness for canonical development seeds 2000–2049 and per-step failure trace capture.
4. `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/analyze_pillar3_failures.py`:
   - Diagnostic analysis script reconstructing exact obstacle clearance geometries, control actions, and supervisor interventions.
5. `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/test_candidate_on_failures.py`:
   - Rapid execution harness for the 5 target failure seeds.
6. `experiments/phase6b/phase6b_pillar3_recovery_20261030_025800/benchmark_phase6b.py`:
   - Comprehensive paired evaluation harness across diagnostic scenarios and all 50 development seeds.

### C. Test Suites
7. `tests/test_pillar3_recovery.py`:
   - 12 comprehensive unit and integration tests verifying all Section 5 requirements.

---

## 4. Configuration Consistency Modifications (Section 7)

To ensure interactive testing, GUI visualization, and quantitative evaluation match production benchmark parameters without altering the default class definition in `controllers/safety_supervisor.py`:

1. `viewer/eval_engine.py`:
   - **Line 14:** Added `SafetySupervisorConfig` to import.
   - **Line 52–54:** Replaced default instantiation with:
     ```python
     self.supervisor_config = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
     self.safety_supervisor = SafetySupervisor(corridor=self.corridor, config=self.supervisor_config)
     ```
2. `scripts/run_quantitative_eval.py`:
   - **Line 25:** Added `SafetySupervisorConfig` to import.
   - **Line 63–65:** Replaced default instantiation with:
     ```python
     sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
     safety_sup = SafetySupervisor(corridor=env.corridor, config=sup_config)
     ```
3. `scripts/run_pybullet_gui.py`:
   - **Line 21:** Added `SafetySupervisorConfig` to import.
   - **Line 96–98:** Replaced default instantiation with:
     ```python
     sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
     safety_sup = SafetySupervisor(corridor=env.corridor, config=sup_config)
     ```

