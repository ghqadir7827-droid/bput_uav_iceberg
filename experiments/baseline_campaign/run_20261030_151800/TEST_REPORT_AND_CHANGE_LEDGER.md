# TEST REPORT AND CHANGE LEDGER
**Artifact ID:** `experiments/baseline_campaign/run_20261030_151800/TEST_REPORT_AND_CHANGE_LEDGER.md`  
**Execution Timestamp:** 2026-10-30T15:27:00  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Base Git Commit:** `f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f`  

---

## 1. Test Suite Execution Report

### 1.1 Baseline Campaign Harness Unit Tests
A dedicated test suite was implemented in [`tests/test_baseline_campaign.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/tests/test_baseline_campaign.py) to rigorously verify:
1. **Horizon Override:** Confirmed `env.max_steps = 1000` enables running beyond the default 500-step ceiling without premature truncation.
2. **Paired Obstacle Spawning:** Confirmed resetting the environment with the same seed generates identical obstacle coordinates and geometries across runs.
3. **Random Reproducibility:** Confirmed seeding `RandomBaselineController(seed=s)` produces deterministic, bit-for-bit identical action sequences.
4. **Seed Partition Disjointness:** Confirmed `seeds 10000..11199` are 100% disjoint from all existing test pools (`100–109`, `200–224`, `1000–1009`, `2000–2049`, and `90000–90099`).

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.17, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav
collected 4 items

tests/test_baseline_campaign.py::test_horizon_override_to_1000_steps PASSED [ 25%]
tests/test_baseline_campaign.py::test_paired_obstacle_spawning_parity PASSED [ 50%]
tests/test_baseline_campaign.py::test_random_baseline_reproducibility PASSED [ 75%]
tests/test_baseline_campaign.py::test_seed_split_integrity PASSED        [100%]

============================== 4 passed in 0.25s ===============================
```

### 1.2 Full Repository Test Suite Verification
The complete repository test suite was executed to ensure zero regressions were introduced:

```text
PYTHONPATH=. .venv/bin/pytest tests/ -q
........................................................................ [ 42%]
........................................................................ [ 85%]
.........................                                                [100%]
169 passed in 2.63s
```
**Verdict:** **169 / 169 tests PASSING**. Zero regressions across contracts, safety supervisor, models, or environment.

---

## 2. Integrity & Asset Cryptographic Hashes

The cryptographic hashes of all canonical models, safety supervisors, and environment files were verified before and after campaign execution:

| File Path | SHA-256 Checksum | Invariant Status |
| :--- | :--- | :--- |
| `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **PRESERVED (Unchanged)** |
| `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **PRESERVED (Unchanged)** |
| `models/checkpoints/ppo_policy_best.pt`| `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | **PRESERVED (Unchanged)** |
| `sim/corridor_env.py` | `4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66` | **PRESERVED (Unchanged)** |
| `core/contracts.py` | `1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab` | **PRESERVED (Unchanged)** |
| `core/geometry.py` | `22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb` | **PRESERVED (Unchanged)** |

---

## 3. Change Ledger

### 3.1 New Files Created
1. `tests/test_baseline_campaign.py`: Unit test verification harness for 1,000-step horizon, seed disjointness, and reproducibility.
2. `scripts/run_baseline_campaign.py`: Standalone evaluation script executing paired 1,200-seed benchmark, collecting full telemetry, and exporting all deliverables.
3. `experiments/baseline_campaign/run_20261030_151800/`: Complete experimental artifact suite.

### 3.2 Canonical Files Modified
- **NONE.** Zero canonical environment files, controller files, safety supervisor configurations, or checkpoints were altered.

