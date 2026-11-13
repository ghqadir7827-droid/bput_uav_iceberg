# Phase 4F.1 Independent Forensic Audit of Evaluation Integrity

**Project:** Autonomous UAV Navigation in GPS-Denied 3D Corridors (Scaled 65 cm Physical Platform)  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Audit Target:** Phase 4F Implementation and Evaluation (`reports/PHASE4F_EVALUATION_REPORT.md`)  
**Git Branch:** `feature/v4-radial-adaptive-guidance`  
**Git Commit:** `f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f`  
**Active Checkpoint:** `models/checkpoints/bc_policy_scaled_v2.pt`  
**SHA-256 Checksum:** `d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1`  
**Date:** October 31, 2026  
**Auditor:** Independent Automated Verification Harness  
**Audit Verdict:** **Recommendation B — v5 remains a promising experimental candidate, but reporting/classification defects and state machine flaws must be corrected before promotion.**

---

## Executive Summary

This independent forensic audit was conducted to verify the internal consistency, physical validity, and reproducibility of the Phase 4F campaign. All 1,200 evaluation episodes (Stage B development and Stage C holdout) were independently re-simulated and re-analyzed alongside the raw data artifacts ([`data/phase4f_development_metrics.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/phase4f_development_metrics.json), [`data/phase4f_episode_outcomes.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/phase4f_episode_outcomes.json), [`data/phase4f_independent_confirmation_metrics.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/phase4f_independent_confirmation_metrics.json)).

### Core Findings
1. **Algorithmic Validation Confirmed:**
   - The 16-beam radial perception array successfully halves the maximum horizontal blind gap from $90.0^\circ$ to $45.0^\circ$ and strictly preserves the bit-exact 23-float observation contract for the frozen `v2` neural network policy.
   - In Seed 20334, the obstacle at $-123.8^\circ$ is intercepted by Rays 13 and 15, widening minimum clearance from $0.0892\text{ m}$ (v4) to **$0.1354\text{ m}$ (v5)** (+51.8% clearance improvement).
   - On truly impassable barriers ($w_{\text{gap}} = 0.60\text{ m} < 0.95\text{ m}$), the safe-arrest state machine achieves **25/25 (100.0%) safe rejections** in both Stage B and Stage C, eliminating the 23 lateral-scanning timeouts that plagued `v3` and `v4`.
   - On pristine holdout feasible routes (Stage C, $N=175$), `v5` achieves **86.9% clean success** (152/175), a statistically significant $+6.29\text{ pp}$ improvement over `v4` (80.6%, $p = 0.0347$) and $+14.29\text{ pp}$ over `v3` (72.6%, $p = 4.13 \times 10^{-5}$).

2. **Material Deficiencies, Reporting Contradictions & Flaws Identified:**
   - **Contradiction in Minimum Clearance (Audit A):** The Phase 4F report's summary table reported `Overall Min Clearance = 0.0786 m` for v5 in Stage B, contradicting the Level 1 text reporting a minimum clearance of $0.0055\text{ m}$ and a collision. Audit reveals that $0.0786\text{ m}$ was actually `successful_min_clearance_m` (the minimum clearance among successful goal arrivals). The actual overall minimum clearance across all 200 episodes is **$0.0055\text{ m}$**, caused by a wall-scrape boundary crash.
   - **Collision Episode Identification Error (Audit A):** The report cited Seed 20108 as the wall-scrape collision. Audit proves that Seed 20108 achieved clean success ($0.5664\text{ m}$ clearance); the collision actually occurred on **Seed 20104** (`CRASH_BOUNDARY`, $0.0055\text{ m}$ clearance).
   - **False-Positive Safe Rejection & Impossible Rate (Audit B):** The report claimed "26 safe rejections out of 25 strictly infeasible scenarios" ($104.0\%$). Seed 26444 is geometrically passable ($1.157\text{ m}$ bottleneck opening vs $0.65\text{ m}$ drone diameter), but was falsely rejected. Audit revealed that `min_traversable_width_m = 0.65 m` was applied to single-ray clearances rather than total corridor passage width.
   - **Forward Velocity Leak in Safe Arrest (Audit B & F):** In `SAFE_ARREST`, the controller suppressed lateral guidance ($v_y = 0$) but did not zero forward velocity ($v_x$). The drone drifted forward 3.19 meters through the corridor while in `SAFE_ARREST`, only stopping when the safety supervisor braked it near the next obstacle.
   - **Statistical Significance Distortion via Endpoint Pooling (Audit D):** The report claimed $p = 5.13 \times 10^{-6}$ for v5 vs v4. Audit proves that on feasible routes alone in Stage B, v5's clean success gain ($+6.29\text{ pp}$) is **not statistically significant** ($p = 0.0895$ at $\alpha = 0.05$). The reported extreme significance was driven entirely by resolving the 24 impassable barrier timeouts.
   - **Drift Statistic Error (Audit F):** The report claimed a mean hover drift of $7.2\text{ mm}$, which was merely the drift of Seed 20301. The true mean final drift across all 25 arrests is **$8.0\text{ mm}$** (peak hover drift: $17.0\text{ mm}$).

---

## 1. Evidence Identification and Provenance Audit

### 1.1 Integrity of Workspace and Checkpoint
- **Repository Working Tree:** Authoritative path `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`.
- **Git Branch:** `feature/v4-radial-adaptive-guidance` at commit `f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f`.
- **Frozen Neural Network Weights:** `models/checkpoints/bc_policy_scaled_v2.pt`
  - Bit-exact SHA-256: `d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1`
  - Audit status: **UNMODIFIED & IMMUTABLE**.
- **PyTest Invariant Suite:** All 229 unit tests pass in 4.52s.

### 1.2 Seed Space Provenance & Disjointness Analysis
The audit traced the provenance of all 400 evaluation seeds across the codebase:

| Seed Partition | Tier Breakdown | Seed Range | Count | Provenance & Historical Usage | Disjointness Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Stage B (Development)** | Level 1 (Basic) | 20100..20149 | 50 | Historical development and tuning set used in Phase 4E, 4E-B, and 4F. | Known development set; not held out. |
| | Level 2 (Standard) | 20000..20049 | 50 | | |
| | Level 3 (Challenging) | 20200..20249 | 50 | | |
| | Level 4 (Boundary) | 20300..20349 | 50 | | |
| **Stage C (Holdout)** | Level 1 (Basic) | 26100..26149 | 50 | Pre-registered in `data/independent_final_seed_manifest.json` (Phase 5). Executed for v2 vs v3 in `experiments/scaled_campaign/independent_final/`. | **Disjoint from training/tuning.** Zero overlap with demonstrations (`bc_train_scaled_v2.npz`, 406 seeds; `bc_val_scaled_v2.npz`, 90 seeds). Neither v4 nor v5 was ever tuned on these seeds. |
| | Level 2 (Standard) | 26200..26249 | 50 | | |
| | Level 3 (Challenging) | 26300..26349 | 50 | | |
| | Level 4 (Boundary) | 26400..26449 | 50 | | |

**Provenance Conclusion:** Stage C seeds are genuinely holdout for candidate `scaled_guided_bc_v5` (no parameters were fitted or tuned against them). However, they are not novel to the repository, having served as the Phase 5 benchmark set.

---

## 2. Audit A — Minimum-Clearance Contradiction

### 2.1 The Mathematical Contradiction
In `reports/PHASE4F_EVALUATION_REPORT.md`:
- Section 7.2 table reports `Overall Min Clearance = 0.0786 m` for v5.
- Section 7.3 Level 1 reports `min_clearance = 0.0055 m` and 1 wall-scrape collision.
- Section 7.3 Level 2 reports `min_clearance = 0.0871 m`.

An overall minimum over all episodes cannot mathematically exceed the minimum of any constituent sub-tier:
$$\min_{\text{all}} C \le \min_{\text{Level 1}} C$$
$$0.0786\text{ m} \le 0.0055\text{ m} \quad \mathbf{[FALSE]}$$

### 2.2 Independent Reconstruction from Raw Episode Data
Auditing [`data/phase4f_episode_outcomes.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/phase4f_episode_outcomes.json) and [`data/phase4f_development_metrics.json`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/phase4f_development_metrics.json):

```python
# From scripts/run_phase4f_evaluation.py lines 321-323:
overall_c = min(r.overall_min_clearance_m for r in c_recs) if c_recs else 0.0
succ_recs = [r for r in feas_recs if r.is_clean_success or r.is_envelope_breach_arrival]
succ_c = min(r.overall_min_clearance_m for r in succ_recs) if succ_recs else 0.0
```

- In `data/phase4f_development_metrics.json`:
  - `overall_min_clearance_m`: **`0.0055`**
  - `successful_min_clearance_m`: **`0.0786`**
- In `data/phase4f_episode_outcomes.json`:
  - Seed 20104: `outcome = "collision"`, `raw_termination = "CRASH_BOUNDARY"`, `min_obstacle_clearance_m = 0.6282`, `min_boundary_clearance_m = 0.0055`.
  - Seed 20248 (Level 3): `outcome = "clean_success"`, `min_obstacle_clearance_m = 0.0786`.

**Root Cause:** The table in `PHASE4F_EVALUATION_REPORT.md` suffered a transcription defect. The author copied `successful_min_clearance_m` ($0.0786\text{ m}$) into the table column labeled `Overall Min Clearance` instead of `overall_min_clearance_m` ($0.0055\text{ m}$). This masked the boundary crash in the top-level summary table.

### 2.3 Collision Seed Identification Error
The report narrative stated: `"1 collision (seed 20108 wall scrape)"`.
Direct audit of the episode records proves:
- **Seed 20108:** Achieved `clean_success` at step 682 with $0.5778\text{ m}$ obstacle clearance and $0.5664\text{ m}$ boundary clearance.
- **Seed 20104:** Suffered `CRASH_BOUNDARY` at step 452 with $0.0055\text{ m}$ boundary clearance.

The report misidentified the collision seed as 20108 due to a narrative typo.

---

## 3. Audit B — Safe Rejection, Seed 26444 & False-Positive Classification

### 3.1 The 26/25 Impossible Rate
In Stage C Level 4, there are 50 seeds: 25 even seeds (feasible) and 25 odd seeds (infeasible).
The Phase 4F report recorded 26 safe rejections out of 25 infeasible scenarios ($104.0\%$).
Independent inspection confirmed that the 26th safe rejection occurred on **Seed 26444**, which is an even, feasible seed.

### 3.2 Deep Forensic Audit of Seed 26444
1. **Geometry and Analytical Feasibility:**
   - Corridor: Width $2.80\text{ m}$ ($y \in [-1.40, +1.40]\text{ m}$), Height $2.40\text{ m}$.
   - Obstacle 1: Center $[7.000, -0.007, 1.200]\text{ m}$, Radius $0.250\text{ m}$.
   - Right channel boundary: Right wall at $y = +1.400\text{ m}$, Obstacle 1 right edge at $y = +0.243\text{ m}$.
   - Lateral passage opening:
     $$w_{\text{opening}} = 1.400 - 0.243 = 1.157\text{ m}$$
   - Airframe diameter: $w_{\text{uav}} = 0.650\text{ m}$.
   - Available maneuvering clearance:
     $$w_{\text{margin}} = 1.157 - 0.650 = 0.507\text{ m}$$
   - When centered in the opening ($y = 0.82\text{ m}$), obstacle clearance is $0.253\text{ m}$ and wall clearance is $0.253\text{ m}$, both exceeding the $0.150\text{ m}$ required safety envelope.
   - Ground-truth analytical feasibility: **FEASIBLE**.

2. **Why Controller v5 Rejected Seed 26444:**
   - In [`controllers/safe_arrest.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/controllers/safe_arrest.py) lines 135-136:
     ```python
     left_blocked = c_left < self.config.min_traversable_width_m  # 0.65m
     right_blocked = c_right < self.config.min_traversable_width_m  # 0.65m
     ```
   - Rangefinder clearances `c_left` and `c_right` are measured from the **center of the UAV** to the nearest surface.
   - When the UAV is flying down the center of the $1.157\text{ m}$ passage, distance from the UAV center to the right wall is $\approx 0.58\text{ m}$, and to the obstacle on the left is $\approx 0.58\text{ m}$.
   - Because $0.58\text{ m} < 0.65\text{ m}$, the state machine flagged `left_blocked = True` AND `right_blocked = True`!
   - Comparing single-ray body clearances to the full airframe diameter is a mathematical error.

3. **The Forward Velocity Leak:**
   - At step 373 ($x = 6.73\text{ m}$), the state machine accumulated 10 steps of false blockage and transitioned to `SAFE_ARREST`.
   - `suppress_lateral_guidance = True` forced $v_y = 0.0\text{ m/s}$.
   - **However, forward velocity $v_x$ was never zeroed!**
   - The neural network continued commanding $v_x \approx 0.15\text{ m/s}$.
   - Because $v_x > 0.05\text{ m/s}$, the hover dwell timer reset on every step.
   - The drone drifted forward through the passage past Obstacle 1, traveling **3.19 meters** while in the `SAFE_ARREST` state, until it reached $x = 9.92\text{ m}$ near Obstacle 2.
   - At $x = 9.92\text{ m}$, front proximity to Obstacle 2 caused the safety supervisor to brake the forward velocity below $0.05\text{ m/s}$.
   - Only then did the 40-step dwell timer complete, terminating the flight as `SAFE_ARREST` at step 847.

4. **Correct Semantic Classification:**
   Seed 26444 is a **FALSE-POSITIVE CONSERVATIVE REJECTION OF A FEASIBLE ROUTE**, not a correct safe rejection of an infeasible route.

---

## 4. Audit C — Complete Outcome Reconstruction

Every episode was independently classified into the mutually exclusive, exhaustive 8-outcome taxonomy. For all controllers and campaigns:
$$\sum \text{All Terminal Outcomes} = \text{Total Evaluated Episodes} = 200$$

### 4.1 Stage B Development Campaign ($N=200$ Episodes per Controller)

| Terminal Outcome Category | `scaled_guided_bc_v3` | `scaled_guided_bc_v4` | `scaled_guided_bc_v5` | Recomputed v5 vs Report |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Success (Feasible)** | 128 (73.1%) | 130 (74.3%) | **141 (80.6%)** | Confirmed (141/175) |
| **Envelope-Breaching Arrival** | 34 (19.4%) | 33 (18.9%) | **25 (14.3%)** | Confirmed (25/175) |
| **Collision** | 0 (0.0%) | 0 (0.0%) | **1 (0.5%)** | Confirmed (Seed 20104 wall crash) |
| **Timeout** | 13 (6.5%) | 12 (6.0%) | **8 (4.0%)** | Confirmed (8/200) |
| **Correct Safe Rejection (Infeasible)** | 1 (4.0%) | 1 (4.0%) | **25 (100.0%)** | Confirmed (25/25) |
| **False-Positive Feasible Rejection** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | Confirmed |
| **Unresolved Obstacle** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | Confirmed |
| **Unsafe or Unstable Arrest** | 24 (96.0% inf) | 24 (96.0% inf) | **0 (0.0%)** | Confirmed (v3/v4 timed out in scanning) |
| **Sum of All Outcomes** | **200 / 200** | **200 / 200** | **200 / 200** | **100% Exhaustive & Mutually Exclusive** |
| *True Overall Min Clearance (All Runs)* | $0.0001\text{ m}$ | $0.0003\text{ m}$ | **$0.0055\text{ m}$** | **Corrected from reported $0.0786\text{ m}$** |
| *Successful Min Clearance (Arrivals Only)* | $0.0001\text{ m}$ | $0.0003\text{ m}$ | **$0.0786\text{ m}$** | Confirmed (Level 3 Seed 20248) |

### 4.2 Stage C Independent Confirmation Campaign ($N=200$ Episodes per Controller)

| Terminal Outcome Category | `scaled_guided_bc_v3` | `scaled_guided_bc_v4` | `scaled_guided_bc_v5` | Recomputed v5 vs Report |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Success (Feasible)** | 127 (72.6%) | 141 (80.6%) | **152 (86.9%)** | Confirmed (152/175) |
| **Envelope-Breaching Arrival** | 38 (21.7%) | 26 (14.9%) | **15 (8.6%)** | Confirmed (15/175) |
| **Collision** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | Confirmed (Zero collisions) |
| **Timeout** | 10 (5.0%) | 8 (4.0%) | **7 (3.5%)** | Confirmed (7/200) |
| **Correct Safe Rejection (Infeasible)** | 1 (4.0%) | 0 (0.0%) | **25 (100.0%)** | **Corrected: exactly 25/25 (not 26/25)** |
| **False-Positive Feasible Rejection** | 0 (0.0%) | 0 (0.0%) | **1 (0.57%)** | **Isolated: Seed 26444** |
| **Unresolved Obstacle** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | Confirmed |
| **Unsafe or Unstable Arrest** | 24 (96.0% inf) | 25 (100.0% inf) | **0 (0.0%)** | Confirmed |
| **Sum of All Outcomes** | **200 / 200** | **200 / 200** | **200 / 200** | **100% Exhaustive & Mutually Exclusive** |
| *True Overall Min Clearance (All Runs)* | $0.0002\text{ m}$ | $0.0001\text{ m}$ | **$0.1049\text{ m}$** | Confirmed (Level 2 Seed 26207) |
| *Successful Min Clearance (Arrivals Only)* | $0.0002\text{ m}$ | $0.0001\text{ m}$ | **$0.1067\text{ m}$** | Confirmed (Level 3 Seed 26311) |

---

## 5. Audit D — Recomputed Paired Statistics

Paired statistics were recomputed using exact two-sided binomial tests (exact McNemar). Two separate analyses are required to prevent endpoint pooling artifacts:

### 5.1 Analysis 1: All Scenarios ($N=200$, Clean Success + Correct Safe Rejection)

| Controller Comparison | Both Valid | First Only | Second Only | Both Invalid | Paired Diff ($\Delta p$) | 95% Confidence Interval | Exact McNemar $p$-value | Asymptotic $\chi^2$ (with CC) | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage B: v3 vs v4** | 122 | 7 | 9 | 62 | $+1.00\text{ pp}$ | $[-2.92, +4.92]\text{ pp}$ | $p = 0.8036$ | $0.250$ ($0.063$) | Not Significant |
| **Stage B: v4 vs v5** | 119 | 12 | 47 | 22 | $+17.50\text{ pp}$ | $[+10.37, +24.63]\text{ pp}$ | **$p = 5.13 \times 10^{-6}$** | $20.76$ ($19.59$) | **Significant** |
| **Stage B: v3 vs v5** | 118 | 11 | 48 | 23 | $+18.50\text{ pp}$ | $[+11.42, +25.58]\text{ pp}$ | **$p = 1.24 \times 10^{-6}$** | $23.20$ ($21.97$) | **Significant** |
| **Stage C: v3 vs v4** | 121 | 7 | 20 | 52 | $+6.50\text{ pp}$ | $[+1.49, +11.51]\text{ pp}$ | **$p = 0.0192$** | $6.26$ ($5.33$) | **Significant** |
| **Stage C: v4 vs v5** | 135 | 6 | 42 | 17 | $+18.00\text{ pp}$ | $[+11.69, +24.31]\text{ pp}$ | **$p = 1.01 \times 10^{-7}$** | $27.00$ ($25.52$) | **Significant** |
| **Stage C: v3 vs v5** | 122 | 6 | 55 | 17 | $+24.50\text{ pp}$ | $[+17.64, +31.36]\text{ pp}$ | **$p = 5.38 \times 10^{-11}$** | $39.36$ ($37.77$) | **Significant** |

### 5.2 Analysis 2: Feasible Routes Only ($N=175$, Clean Success Strictly)

| Controller Comparison | Both Clean | First Only | Second Only | Neither Clean | Paired Diff ($\Delta p$) | 95% Confidence Interval | Exact McNemar $p$-value | Asymptotic $\chi^2$ (with CC) | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage B: v3 vs v4** | 121 | 7 | 9 | 38 | $+1.14\text{ pp}$ | $[-3.33, +5.62]\text{ pp}$ | $p = 0.8036$ | $0.250$ ($0.063$) | Not Significant |
| **Stage B: v4 vs v5** | 118 | 12 | 23 | 22 | $+6.29\text{ pp}$ | $[-0.27, +12.85]\text{ pp}$ | **$p = 0.0895$** | $3.46$ ($2.86$) | **NOT Significant at $\alpha = 0.05$** |
| **Stage B: v3 vs v5** | 117 | 11 | 24 | 23 | $+7.43\text{ pp}$ | $[+0.89, +13.96]\text{ pp}$ | **$p = 0.0410$** | $4.83$ ($4.11$) | **Significant** |
| **Stage C: v3 vs v4** | 121 | 6 | 20 | 28 | $+8.00\text{ pp}$ | $[+2.41, +13.59]\text{ pp}$ | **$p = 0.0094$** | $7.54$ ($6.50$) | **Significant** |
| **Stage C: v4 vs v5** | 135 | 6 | 17 | 17 | $+6.29\text{ pp}$ | $[+1.00, +11.58]\text{ pp}$ | **$p = 0.0347$** | $5.26$ ($4.35$) | **Significant** |
| **Stage C: v3 vs v5** | 121 | 6 | 31 | 17 | $+14.29\text{ pp}$ | $[+7.81, +20.76]\text{ pp}$ | **$p = 4.13 \times 10^{-5}$** | $16.89$ ($15.57$) | **Significant** |

### 5.3 Audit Insights on Statistical Significance
1. In Stage B, the claim of "overwhelming statistical superiority ($p = 5.13 \times 10^{-6}$)" is driven almost entirely by the safe-arrest state machine resolving the 24 infeasible barrier timeouts. On passable corridors, v5 was **not statistically significantly better than v4 ($p = 0.0895$)**.
2. In Stage C holdout, however, v5 achieved genuine, statistically significant navigation superiority on feasible corridors alone: $+6.29\text{ pp}$ over v4 ($p = 0.0347$) and $+14.29\text{ pp}$ over v3 ($p = 4.13 \times 10^{-5}$).
3. When analyzing paired binary outcomes, feasible navigation and impassable barrier arrest must never be pooled into a single composite without reporting the disaggregated components.

---

## 6. Audit E — Sensor Coverage and Seed 20334

### 6.1 16-Beam Radial Architecture Verification
- **Geometry:** 12 baseline beams plus 4 oblique rear-flank beams:
  - Ray 12: Azimuth $+120.0^\circ$, Elevation $0.0^\circ$ (Rear-left flank)
  - Ray 13: Azimuth $-120.0^\circ$, Elevation $0.0^\circ$ (Rear-right flank)
  - Ray 14: Azimuth $+135.0^\circ$, Elevation $0.0^\circ$ (Rear-left oblique)
  - Ray 15: Azimuth $-135.0^\circ$, Elevation $0.0^\circ$ (Rear-right oblique)
- **Circular Gap:** Maximum horizontal circular gap is strictly **$45.0^\circ$** ($180^\circ - 135^\circ$), halved from $90.0^\circ$ in the baseline.
- **Supervisor Integration:** Verified. [`controllers/safety_supervisor.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/controllers/safety_supervisor.py) queries [`compute_directional_clearances_m`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/core/contracts.py), which checks rays `(1, 3, 5, 12, 14)` for left clearance and `(2, 4, 6, 13, 15)` for right clearance.
- **Observation Projection Invariant:** Verified. [`DeployableObservation.to_policy_vector()`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/core/contracts.py) extracts `raw[:12]` and appends `raw[16:27]`, supplying exactly 23 floats to the frozen neural network with bit-exact consistency.

### 6.2 Seed 20334 Forensic Reproduction
Independent simulation of Seed 20334 across all three controllers:

| Controller | Observation Config | Obstacle Detection | Minimum Clearance | Envelope Steps | Total Steps | Final Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `scaled_guided_bc_v3` | 12 Beams | Undetected | $0.1154\text{ m}$ | 656 | 1,000 | `TIMEOUT` (Chatter oscillation) |
| `scaled_guided_bc_v4` | 12 Beams | Undetected | $0.0892\text{ m}$ | 20 | 672 | `ENVELOPE_BREACH_ARRIVAL` |
| `scaled_guided_bc_v5` | 16 Beams | **Intercepted (Rays 13/15)** | **$0.1354\text{ m}$** | 74 | 764 | `ENVELOPE_BREACH_ARRIVAL` |

**Audit Findings:**
- The obstacle is intercepted at relative bearing $-123.8^\circ$ by Ray 13 and Ray 15.
- The supervisor detects rightward obstacle constraint and reduces rightward velocity.
- The minimum clearance increases from $0.0892\text{ m}$ to **$0.1354\text{ m}$** (+51.8% widening).
- The Phase 4F report reported $0.1320\text{ m}$ due to a minor narrative typo; the true simulated value is $0.1354\text{ m}$.
- Both values strictly confirm that the safety envelope ($0.1500\text{ m}$) was still breached, correctly classifying the run as `ENVELOPE_BREACH_ARRIVAL`.
- **Coverage Limitations:** A 16-beam planar/elevated ray array cannot guarantee complete obstacle detection. Obstacles situated in vertical gaps ($> 45^\circ$ pitch) or thin structures between azimuth rays remain undetectable.

---

## 7. Audit F — Safe-Arrest Verification & Threshold Analysis

### 7.1 Infeasible Barrier Scenario Verification ($N=25$)
All 25 known infeasible barrier scenarios in Stage B (seeds 20301..20349, odd) and Stage C (seeds 26401..26449, odd) were independently simulated and traced:
- **Geometry:** Impassable bottleneck width $w_{\text{gap}} = 0.60\text{ m} < w_{\text{crit}} = 0.95\text{ m}$.
- **Verification Time:** Transition to `BLOCKED` occurs at mean $t = 3.65\text{ s}$; `VERIFIED_INFEASIBLE` confirmed after 10 evidence steps ($0.50\text{ s}$).
- **Kinematic Stability:**
  - Residual horizontal speed during hover: $\|\mathbf{v}_{xy}\| \le 0.015\text{ m/s}$ (configured limit: $0.050\text{ m/s}$).
  - Mean final position drift: **$8.0\text{ mm}$** ($0.0080\text{ m}$). (The reported $7.2\text{ mm}$ was the drift of Seed 20301 alone).
  - Maximum final position drift: **$12.6\text{ mm}$** ($0.0126\text{ m}$).
  - Peak drift during the entire 40-step dwell window: **$17.0\text{ mm}$** (allowable: $100.0\text{ mm}$).
  - Minimum surface clearance during hover: **$0.1138\text{ m}$** (safety floor: $0.050\text{ m}$).
- **Termination:** 25/25 episodes completed the 40-step ($2.0\text{ s}$) dwell and achieved clean `SAFE_REJECTION`.

### 7.2 Dynamic Validity of Dwell and Drift Thresholds
- The 40-step ($2.0\text{ s}$) dwell window is physically realistic for a 65 cm quadcopter. In PyBullet, body settling occurs within 15–20 steps ($0.75\text{--}1.0\text{ s}$); the additional 20 steps verify that ground effect or numerical draft does not induce translational instability.
- **Critical Flaw:** The state machine depends entirely on external obstacles halting forward velocity $v_x$. Without an obstacle in front of the vehicle, the state machine fails to brake the drone.

---

## 8. Master Discrepancy Register

| ID | Issue & Location | Original Claim | Raw Evidence | Recomputed Result | Status | Corrective Action |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **D1** | Min Clearance Contradiction (Sec 7.2) | v5 Stage B Overall Min Clearance = $0.0786\text{ m}$ | `overall_min_clearance_m` is $0.0055\text{ m}$ in JSON | True overall min clearance is $0.0055\text{ m}$; $0.0786\text{ m}$ is for successful arrivals | **CONTRADICTED** | Separate overall min clearance ($0.0055\text{ m}$) from arrival clearance ($0.0786\text{ m}$). |
| **D2** | Collision Seed ID (Sec 7.3) | Seed 20108 wall scrape | Seed 20108 was clean success ($0.5664\text{ m}$) | Collision occurred on Seed 20104 ($0.0055\text{ m}$) | **CONTRADICTED** | Correct collision seed to 20104. |
| **D3** | Level 4 Rejections (Sec 8.3) | 26 safe rejections out of 25 infeasible | 25 infeasible seeds in Level 4 | 25/25 correct safe rejections + 1 false-positive rejection (Seed 26444) | **CONTRADICTED** | Report false-positive rejections separately from correct rejections. |
| **D4** | Forward Velocity Leak in Safe Arrest | Vehicle anchors hover and halts | $v_x$ not zeroed; drone flew 3.19 m in SAFE_ARREST | Forward velocity leaks through until supervisor brakes vehicle | **CONTRADICTED** | Explicitly command $v_x = 0$ upon entering SAFE_ARREST. |
| **D5** | Safe Arrest Threshold Mismatch | Checks 65 cm drone passage | `c_left < 0.65m` checked on center-to-wall rays | Single-ray clearance cannot be compared to total drone diameter | **CONTRADICTED** | Check total passage opening ($c_{\text{left}} + c_{\text{right}} \ge 0.65\text{ m}$). |
| **D6** | Seed 20334 Clearance Value (Sec 6) | Clearance widened to $0.1320\text{ m}$ | JSON records $0.1354\text{ m}$ | True clearance is $0.1354\text{ m}$ (+51.8%) | **CONTRADICTED** | Correct clearance to $0.1354\text{ m}$. |
| **D7** | Statistical Significance Pooling (Sec 7.4) | $p = 5.13 \times 10^{-6}$ shows guidance superiority | Feasible clean success $p = 0.0895$ (n.s.) | Significance in Stage B driven by 24 barrier timeouts | **CONTRADICTED** | Report feasible navigation and barrier arrest significance separately. |
| **D8** | Hover Drift Statistic (Sec 5.3) | Mean drift is $7.2\text{ mm}$ | $7.2\text{ mm}$ is Seed 20301 only | True mean final drift is $8.0\text{ mm}$ (peak $17.0\text{ mm}$) | **CONTRADICTED** | Report true mean drift of $8.0\text{ mm}$. |
| **D9** | Holdout Seed Provenance (Sec 8.1) | Pristine holdout seeds | Pre-registered in Phase 5 manifest | Disjoint from training/tuning, but previously run in Phase 5 | **CONFIRMED (QUALIFIED)** | Document Phase 5 benchmark provenance. |

---

## 9. Final Recommendation & Action Plan

### Recommendation: **B. Candidate `scaled_guided_bc_v5` remains a promising experimental candidate, but reporting/classification defects and state machine flaws must be corrected before promotion.**

Promoting `scaled_guided_bc_v5` to the authoritative benchmark standard today is premature because:
1. The forward velocity leak in `SafeArrestStateMachine` represents an unverified control assumption that would cause flyaways if an infeasible barrier is declared in open air.
2. The single-ray threshold flaw causes false-positive aborts on passable $1.15\text{ m}$ corridors.
3. The reporting contradictions in the Phase 4F report (Overall Min Clearance $0.0786\text{ m}$ vs $0.0055\text{ m}$, Seed 20108 vs 20104, 26/25 safe rejections) compromise evaluation documentation integrity.

### Required Corrective Actions (Milestone Phase 4F.2):
1. **Patch `controllers/safe_arrest.py`:**
   - Command $v_x = 0.0\text{ m/s}$ in `VERIFIED_INFEASIBLE` and `SAFE_ARREST`.
   - Update traversability logic to evaluate combined opening width ($c_{\text{left}} + c_{\text{right}} \ge 0.65\text{ m} + 2 \cdot \text{margin}$) or compare single-side clearance to half-width plus margin ($0.475\text{ m}$).
2. **Standardize Evaluation Reporting Pipeline:**
   - Enforce distinct fields for `overall_min_clearance_all_runs_m` and `successful_arrivals_min_clearance_m`.
   - Disaggregate feasible-route navigation McNemar tests from barrier safe-rejection tests.
3. **Re-Run Automated Verification Suite:**
   - Execute `run_phase4f_evaluation.py` to confirm that Seed 26444 achieves `clean_success` and that Stage C safe rejections equal exactly 25/25 ($100.0\%$).

> [!CAUTION]
> **Simulation Boundary Disclaimer:** All evaluations were executed entirely in high-fidelity PyBullet simulation. Simulation results do not constitute proof of physical flight readiness or formal safety certification. Physical flight controllers remain disarmed and disconnected.

