# BASELINE EVALUATION REPORT: SEEDS 1 TO 1000 INCLUSIVE (REACTIVE VS. RANDOM)
**Artifact ID:** `experiments/baseline_campaign/run_seeds_1_to_1000_20261030_153300/BASELINE_EVALUATION_SEEDS_1_TO_1000_REPORT.md`  
**Execution Timestamp:** 2026-10-30T15:33:16 to 2026-10-30T15:38:15 (299.3 s execution)  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Lead Evaluator:** Principal Autonomous Robotics & Verification Engineer  
**Evaluation Scope:** Seeds 1 through 1000 inclusive ($N=1,000$ seeds, 2,000 paired episodes total)  

---

## 1. Executive Summary & Key Results

At the user's directive, an exhaustive paired evaluation was conducted across **seeds 1 through 1000 inclusive** for both the `reactive_baseline` and `random_baseline` controllers under the canonical 10-meter corridor benchmark with an evaluation-only horizon override of 1,000 steps ($50.0$ seconds).

```
+--------------------------------------------------------------------------------------------------------------------+
|                                      SEEDS 1 TO 1000 BENCHMARK SUMMARY (N = 1,000 SEEDS)                           |
+--------------------------------------------------------------------------------------------------------------------+
| Controller              | 500-Step Success | 1000-Step Success | 95% Wilson CI     | Collisions | Envelope Breaches |
|-------------------------+------------------+-------------------+-------------------+------------+-------------------|
| Reactive Baseline       | 618/1000 (61.8%) |  998/1000 (99.8%) | [99.27%, 99.95%]  | 0 (0.0%)   | 0 (0.0%)          |
| Random Exploration      |   0/1000 ( 0.0%) |   97/1000 ( 9.7%) | [ 8.02%, 11.69%]  | 0 (0.0%)   | 1 (0.1%)          |
|-------------------------+------------------+-------------------+-------------------+------------+-------------------|
| Paired Differential     | +618 (+61.8%)    |  +901 (+90.1%)    | McNemar p < 1e-15 | 0 diff     | -1 diff           |
+--------------------------------------------------------------------------------------------------------------------+
```

### Core Empirical Conclusions:
1. **Flawless Baseline Stability:** `reactive_baseline` completed **998 out of 1,000 episodes (99.8%)** [95% Wilson CI: 99.27% – 99.95%] with **0 collisions** and **0 safety envelope violations** ($d_{\text{min}} = 0.0873$m $> 0.08$m).
2. **Conquest of the 14 Milestone 4 Hard Negatives:** On the exact 14 fatal crash seeds from Milestone 4 (`[200, 203, 204, 206, 207, 210, 213, 215, 216, 217, 218, 220, 221, 224]`) where unshielded PPO crashed 100% of the time, the reactive baseline achieved **14 / 14 SUCCESS (100.0%)** with an average obstacle clearance of $0.151$m!
3. **Horizon Sensitivity Confirmed:** At 500 steps, `reactive_baseline` achieved 61.8% (618/1000). The extended horizon unlocked **+380 completions (+38.0%)**, confirming that slow, cautious cruising ($0.20$–$0.35$ m/s near obstacles) requires ~25–35 seconds (500–700 steps) to complete the 10m corridor with 3 sequential pillars.
4. **Random Exploration Bounds:** Random exploration achieved **0.0% at 500 steps** and **9.7% (97/1000) at 1,000 steps**. In **0 out of 1,000 seeds** did Random succeed where Reactive failed.

---

## 2. Forensic Analysis of the 14 Milestone 4 Hard-Negative Seeds

In Milestone 4, a major project crisis occurred when unshielded PPO crashed at high rates across seeds `200–224`. Exactly 14 seeds were identified as fatal crash configurations and designated as the "Hard Negative Pool".

The table below reports how `reactive_baseline` and `random_baseline` perform on these exact 14 seeds under the canonical Safety Supervisor:

| Hard Negative Seed | Reactive Outcome | Reactive Steps | Reactive Min Obs Clearance | Random Outcome | Random Steps | Random Min Obs Clearance | Obstacle Geometry / Challenge |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seed 200** | **SUCCESS** | 447 | 0.1587m | OSCILLATION | 1000 | 0.1197m | Centerline pillar at $x=3.0$ ($y \approx 0.02$) |
| **Seed 203** | **SUCCESS** | 393 | 0.1679m | OSCILLATION | 1000 | 0.1143m | Tight squeeze between Pillar 1 and 2 |
| **Seed 204** | **SUCCESS** | 567 | 0.1339m | **SUCCESS** | 851 | 0.1284m | Wide right-side opening |
| **Seed 206** | **SUCCESS** | 446 | 0.1576m | TIMEOUT_1000 | 1000 | 0.1233m | Staggered slalom |
| **Seed 207** | **SUCCESS** | 590 | 0.1266m | TIMEOUT_1000 | 1000 | 0.1245m | Near-wall Pillar 2 |
| **Seed 210** | **SUCCESS** | 518 | 0.1330m | OSCILLATION | 1000 | 0.1282m | Symmetric Pillar 2 entry |
| **Seed 213** | **SUCCESS** | 459 | 0.1451m | OSCILLATION | 1000 | 0.1164m | Acute wall pinch |
| **Seed 215** | **SUCCESS** | 373 | 0.1875m | OSCILLATION | 1000 | 0.1214m | Staggered offset |
| **Seed 216** | **SUCCESS** | 411 | 0.1515m | OSCILLATION | 1000 | 0.1247m | Tight central pinch |
| **Seed 217** | **SUCCESS** | 404 | 0.1759m | **SUCCESS** | 714 | 0.1512m | Wide lateral bypass |
| **Seed 218** | **SUCCESS** | 459 | 0.1550m | OSCILLATION | 1000 | 0.1361m | Sequential left-right slalom |
| **Seed 220** | **SUCCESS** | 382 | 0.1831m | OSCILLATION | 1000 | 0.1237m | Central Pillar 3 choke point |
| **Seed 221** | **SUCCESS** | 519 | 0.1216m | OSCILLATION | 1000 | 0.1238m | Narrow exit corridor |
| **Seed 224** | **SUCCESS** | 581 | 0.1299m | TIMEOUT_1000 | 1000 | 0.1067m | Right wall corner pinch |
| **TOTAL** | **14 / 14 (100.0%)** | **467.8** | **0.1519m (Min: 0.1216m)** | **2 / 14 (14.3%)** | **968.9** | **0.1245m** | **Reactive completes 100% of hard negatives** |

### Why Did Reactive Succeed Where Neural Networks Failed?
- **PPO collapsed** because oversampling these 14 seeds caused policy gradient explosion and mode collapse to unilateral right drift.
- **BC struggled (48%)** because symmetric ray returns caused zero lateral velocity commands ($v_y \approx 0$).
- **`reactive_baseline` succeeds (100%)** because:
  1. It actively decelerates ($v_x \to 0.20$ m/s) whenever rangefinder distance drops below $1.4$m.
  2. Its multi-beam lateral scoring (`d_fl + 0.6 * d_left`) breaks symmetry early ($1.5$m range), initiating avoidance long before reaching the supervisor's $0.50$m caution boundary.
  3. Its wall-centering buffers prevent it from pin-balling into corridor walls during sharp counter-turns.

---

## 3. Forensics of the 2 Reactive Baseline Failures

Out of 1,000 seeds, `reactive_baseline` failed on **only 2 seeds**:

### Failure 1: Seed 99 — `TIMEOUT_1000` (Symmetric Pillar Lock)
- **Obstacle Coordinates:** Pillar 1 ($y = +0.0042$), Pillar 2 ($y = +0.0456$), Pillar 3 ($y = +0.0083$).
- **Trace Analysis:** All three pillars are centered within $4$ cm of the corridor midline. The drone cleared Pillar 1 at step 160, but approached Pillar 2 at $x = 4.8$m with near-identical left and right ray distances.
- **Root Cause:** Symmetrical balance between obstacle repulsion and wall centering forces slowed forward speed to $v_x \approx 0.05$ m/s. The drone remained actively in flight but only reached $x = 5.987$m when step 1000 expired.

### Failure 2: Seed 686 — `OSCILLATION` (Dead-Center Mathematical Equilibrium)
- **Obstacle Coordinates:** Pillar 1 ($y = +0.0007$), Pillar 2 ($y = +0.0148$), Pillar 3 ($y = +0.0007$).
- **Trace Analysis:** Pillar 1 and Pillar 3 are positioned at $y = +0.0007$m ($0.7$ millimeters from the exact centerline!).
- **Root Cause:** Dead-center pillar alignment produced identical left and right sector clearances, causing the reactive steering rule to chatter between $+0.40$ m/s and $-0.40$ m/s. The vehicle oscillated at $x = 6.449$m until step 1000 expired.

---

## 4. Comprehensive Metrics Table (Seeds 1 to 1000)

| Metric | Reactive Baseline (`reactive_baseline`) | Random Exploration (`random_baseline`) | Paired Delta / Significance |
| :--- | :---: | :---: | :---: |
| **Total Episodes** | 1,000 | 1,000 | 1,000 paired seeds |
| **Goal Completions (Success)** | **998 / 1,000 (99.8%)** | **97 / 1,000 (9.7%)** | **+90.1% ($p < 10^{-15}$)** |
| **Wilson 95% Confidence Interval** | **[99.27%, 99.95%]** | **[8.02%, 11.69%]** | Completely non-overlapping |
| **Success at Step 500** | 618 / 1,000 (61.8%) | 0 / 1,000 (0.0%) | Random never completed in 500 steps |
| **Extended Horizon Gain** | **+380 (+38.0%)** | **+97 (+9.7%)** | Proves 500-step horizon was truncating safe runs |
| **Obstacle Collisions** | **0 / 1,000 (0.0%)** | **0 / 1,000 (0.0%)** | Zero crashes |
| **Wall Collisions** | **0 / 1,000 (0.0%)** | **0 / 1,000 (0.0%)** | Zero breaches |
| **Safety Envelope Violations** | **0 / 1,000 (0.0%)** | 1 / 1,000 (0.1%) | Overall min: $0.0873$m vs $0.0724$m |
| **Oscillations** | 1 / 1,000 (0.1%) | 498 / 1,000 (49.8%) | Random chatters against walls/pillars |
| **Timeouts at 1000** | 1 / 1,000 (0.1%) | 404 / 1,000 (40.4%) | Random wandering without reaching goal |
| **Mean Forward Progress** | **$8.961 \pm 0.126$m** | **$7.198 \pm 3.225$m** | Reactive: highly consistent |
| **Mean Steps to Success** | **$494.3 \pm 103.6$ steps** | **$863.5 \pm 75.0$ steps** | Reactive completes in ~24s vs 43s |
| **Pillar 1 Cleared** | 1,000 / 1,000 (100.0%) | 887 / 1,000 (88.7%) | Random drops 11.3% at Pillar 1 |
| **Pillar 2 Cleared** | 1,000 / 1,000 (100.0%) | 667 / 1,000 (66.7%) | Random drops 33.3% by Pillar 2 |
| **Pillar 3 Cleared** | 998 / 1,000 (99.8%) | 476 / 1,000 (47.6%) | Random drops 52.4% by Pillar 3 |
| **Both Succeed** | \multicolumn{3}{c|}{97 / 1,000 (9.7%)} |
| **Only Reactive Succeeds** | \multicolumn{3}{c|}{**901 / 1,000 (90.1%)**} |
| **Only Random Succeeds** | \multicolumn{3}{c|}{**0 / 1,000 (0.0%)** (Random never won where Reactive failed)} |
| **Both Fail** | \multicolumn{3}{c|}{2 / 1,000 (0.2%) (Seeds 99 and 686)} |
| **McNemar Chi-Square** | \multicolumn{3}{c|}{$\chi^2 = 899.00$, **$p < 10^{-15}$**} |

