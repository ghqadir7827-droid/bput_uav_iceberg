# Baseline Evaluation: Seeds 1 to 1000 Inclusive (Reactive vs. Random)
**Directory:** `experiments/baseline_campaign/run_seeds_1_to_1000_20261030_153300/`  
**Execution Timestamp:** 2026-10-30T15:33:16  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Git Commit:** `f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f`  
**Status:** COMPLETE (2,000 / 2,000 episodes executed across seeds 1 to 1000 inclusive)

---

## 1. Scope & Objective

This benchmark evaluates the two fundamental baseline controllers across **seeds 1 through 1000 inclusive ($N=1,000$ seeds, 2,000 paired episodes total)** with a 1,000-step horizon override ($50.0$ seconds):

1. **`reactive_baseline` (`ReactiveBaselineController`):** Deterministic multi-beam rangefinder avoidance with dynamic speed scaling and goal tracking.
2. **`random_baseline` (`RandomBaselineController`):** Stochastic uniform velocity exploration ($v_x \in [0.0, 0.5]$ m/s, $v_y \in [-0.3, 0.3]$ m/s), seeded deterministically per episode.

Crucially, this seed span encompasses the legacy project history, including:
- Milestone 1 Validation seeds: `100–109`
- Milestone 4 Hard Negative crash seeds: `200–224` (specifically the 14 fatal crash seeds `[200, 203, 204, 206, 207, 210, 213, 215, 216, 217, 218, 220, 221, 224]`)
- Milestone 5 Diagnostic seed: `1000`

The canonical held-out evaluation pool (`90000–90099`) remains strictly disjoint and untouched.

---

## 2. Key Empirical Findings

| Metric Category | Reactive Baseline (`reactive_baseline`) | Random Exploration (`random_baseline`) | Comparative Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Evaluated Episodes** | 1,000 | 1,000 | 1,000 paired seeds |
| **Goal Completions (Success)** | **998 / 1,000 (99.8%)** | **97 / 1,000 (9.7%)** | **+90.1% (10.3x advantage)** |
| **Wilson 95% Confidence Interval** | **[99.27%, 99.95%]** | **[8.02%, 11.69%]** | Disjoint ($p < 10^{-15}$) |
| **Success at Step 500** | 618 / 1,000 (61.8%) | 0 / 1,000 (0.0%) | Random requires $>500$ steps |
| **Extended Horizon Gain (501–1000)** | **+380 completions (+38.0%)** | **+97 completions (+9.7%)** | Horizon extension crucial for slow cruising |
| **Obstacle Collisions** | **0 / 1,000 (0.0%)** | **0 / 1,000 (0.0%)** | Safety Supervisor 100% collision-free |
| **Wall Collisions** | **0 / 1,000 (0.0%)** | **0 / 1,000 (0.0%)** | Zero boundary breaches |
| **Safety Envelope Violations ($<0.08$m)** | **0 / 1,000 (0.0%)** | 1 / 1,000 (0.1%) | $d_{\text{min}} = 0.0873$m vs $0.0724$m |
| **Milestone 4 Hard Negatives (14 seeds)**| **14 / 14 (100.0%)** | **2 / 14 (14.3%)** | Reactive conquers crash seeds |
| **Both Succeed** | \multicolumn{3}{c|}{97 / 1,000 (9.7%)} |
| **Only Reactive Succeeds** | \multicolumn{3}{c|}{**901 / 1,000 (90.1%)**} |
| **Only Random Succeeds** | \multicolumn{3}{c|}{**0 / 1,000 (0.0%)** (Random never won where Reactive failed)} |
| **Both Fail** | \multicolumn{3}{c|}{2 / 1,000 (0.2%) (Seeds 99 and 686: dead-center pillars)} |
| **McNemar Chi-Square & p-value** | \multicolumn{3}{c|}{$\chi^2 = 899.00$, **$p < 10^{-15}$**} |

---

## 3. Deliverable Manifest

- `seed_manifest.json` — Exact 1,000 seed list (`1..1000`) and partition audit.
- `experiment_config.json` — Environment and supervisor configuration.
- `reactive_episode_results.csv` — Full metrics for all 1,000 reactive episodes.
- `random_episode_results.csv` — Full metrics for all 1,000 random episodes.
- `paired_seed_comparison.csv` — Paired differences across all 1,000 seeds.
- `aggregate_metrics.json` — Complete statistics and Wilson CIs.
- `failure_taxonomy.csv` — Failure classification counts and seed registers.
- `progress_500_vs_1000.csv` — Step 500 vs. step 1000 comparison.
- `representative_trajectories/` — Telemetry traces for characteristic successes and edge cases.
- `BASELINE_EVALUATION_SEEDS_1_TO_1000_REPORT.md` — Detailed technical audit report.

