# Baseline Evaluation Campaign: Reactive Obstacle Avoidance vs. Random Exploration
**Directory:** `experiments/baseline_campaign/run_20261030_151800/`  
**Execution Timestamp:** 2026-10-30T15:18:33  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Git Commit:** `f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f`  
**Status:** COMPLETE (2,400 / 2,400 episodes executed across 1,200 seeds)

---

## 1. Scope & Objective

This campaign conducts an exhaustive, paired empirical evaluation of two fundamental baseline controllers in the canonical GPS-denied UAV navigation environment:

1. **`reactive_baseline` (`ReactiveBaselineController`):**  
   Deterministic reactive obstacle avoidance utilizing multi-beam horizontal planar rangefinder rays, lateral clearance differential steering, goal tracking, and dynamic longitudinal velocity slowing.
2. **`random_baseline` (`RandomBaselineController`):**  
   Stochastic velocity exploration sampling continuous velocity commands uniformly ($v_x \in [0.0, 0.5]$ m/s, $v_y \in [-0.3, 0.3]$ m/s, $v_z \in [-0.2, 0.2]$ m/s, $\dot{\psi} \in [-0.3, 0.3]$ rad/s), seeded deterministically per episode with the environment seed.

Both controllers are evaluated on the **exact same 1,200 explicitly identified seeds (`10000–11199`)** with an evaluation-only horizon override of **1,000 steps ($50.0$ seconds)**.

---

## 2. Seed Split & Invariant Protections

To prevent data contamination and preserve existing evaluation contracts:
- **Evaluation Range:** `seeds 10000..11199` ($N=1,200$).
- **Verified Disjoint:**
  - Initial validation set (`100–109`): Strictly disjoint ($\min = 10000 > 109$).
  - Hard negative pool (`200–224`): Strictly disjoint.
  - Diagnostic scenarios (`1000–1009`): Strictly disjoint.
  - Development benchmark (`2000–2049`): Strictly disjoint.
  - Canonical held-out test set (`90000–90099`): Strictly disjoint and sealed ($\max = 11199 < 90000$).
- **Safety Supervisor Invariant:** Canonical deterministic dual-envelope safety supervisor (`controllers/safety_supervisor.py`) remained active for all 2,400 episodes with $d_{\text{caution}} = 0.50$m, $d_{\text{stop}} = 0.08$m, and active boundary repulsion.

---

## 3. Key Findings

| Metric | Reactive Baseline (`reactive_baseline`) | Random Exploration Baseline (`random_baseline`) | Delta / Paired Significance |
| :--- | :---: | :---: | :---: |
| **Evaluated Episodes** | 1,200 | 1,200 | 1,200 paired seeds |
| **Goal Completions (Success)** | **1,197 / 1,200 (99.75%)** | **116 / 1,200 (9.67%)** | $+90.08\%$ gain ($p < 10^{-15}$) |
| **Wilson 95% Confidence Interval** | **[99.27%, 99.91%]** | **[8.12%, 11.47%]** | Completely non-overlapping |
| **Success at Step 500** | 769 / 1,200 (64.08%) | 0 / 1,200 (0.00%) | Random never completed in 500 steps |
| **Extended Horizon Gain (501–1000)** | **+428 completions (+35.67%)** | **+116 completions (+9.67%)** | 1000-step horizon critical for slow, safe flight |
| **Obstacle Collisions** | **0 / 1,200 (0.00%)** | **0 / 1,200 (0.00%)** | Safety Supervisor prevented 100% crashes |
| **Wall Collisions** | **0 / 1,200 (0.00%)** | **0 / 1,200 (0.00%)** | Zero boundary breaches |
| **Safety Envelope Violations ($<0.08$m)** | 1 / 1,200 (0.08%) | 1 / 1,200 (0.08%) | Trace: grazed at 0.0777m and 0.0596m |
| **Mean Steps to Success** | 487.3 steps (24.4 s) | 840.0 steps (42.0 s) | Reactive is 1.7x faster |
| **Supervisor Interventions / Ep** | 141.8 steps (15.6% throttled) | 452.5 steps (35.6% throttled) | Random requires 3.2x more interventions |
| **Both Succeed** | \multicolumn{3}{c|}{116 / 1,200 (9.67%)} |
| **Only Reactive Succeeds** | \multicolumn{3}{c|}{1,081 / 1,200 (90.08%)} |
| **Only Random Succeeds** | \multicolumn{3}{c|}{**0 / 1,200 (0.00%)** (Random never won where Reactive failed)} |
| **Both Fail** | \multicolumn{3}{c|}{3 / 1,200 (0.25%) (Seeds 10009, 10131, 10667)} |

---

## 4. Deliverable Artifact Manifest

- `seed_manifest.json` — Exact 1,200 seed list and validation checks.
- `experiment_config.json` — Environment geometry, controller parameters, and supervisor config.
- `reactive_episode_results.csv` — Per-episode metrics for all 1,200 reactive runs.
- `random_episode_results.csv` — Per-episode metrics for all 1,200 random runs.
- `paired_seed_comparison.csv` — Paired differential analysis across all 1,200 seeds.
- `aggregate_metrics.json` — Full aggregate statistics with Wilson CIs, quantiles, and hypothesis tests.
- `failure_taxonomy.csv` — Mutual-exclusive outcome counts and sample seed registers.
- `progress_500_vs_1000.csv` — Extended-horizon progression for all episodes incomplete at step 500.
- `representative_trajectories/` — Step-by-step telemetry CSVs for characteristic successes, oscillations, timeouts, and recoveries.
- `BASELINE_CAMPAIGN_REPORT.md` — Comprehensive technical evaluation, root-cause forensics, and architectural recommendations.
- `TEST_REPORT_AND_CHANGE_LEDGER.md` — Harness test report and repository change ledger.

---

## 5. Independent Reproduction

To reproduce this benchmark identically:

```bash
cd /Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav
PYTHONPATH=. .venv/bin/python scripts/run_baseline_campaign.py \
    --seeds-start 10000 \
    --seeds-count 1200 \
    --max-steps 1000 \
    --output-dir experiments/baseline_campaign/run_reproduction
```

