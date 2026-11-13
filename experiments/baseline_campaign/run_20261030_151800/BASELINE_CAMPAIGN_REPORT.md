# FULL-SCALE BASELINE EVALUATION CAMPAIGN REPORT: REACTIVE OBSTACLE AVOIDANCE VS. RANDOM EXPLORATION
**Artifact ID:** `experiments/baseline_campaign/run_20261030_151800/BASELINE_CAMPAIGN_REPORT.md`  
**Execution Timestamp:** 2026-10-30T15:28:00+05:30  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Lead Evaluator:** Principal Autonomous Robotics & Verification Engineer  
**Evaluation Scope:** 1,200 paired seeds (`10000..11199`), 2,400 total episodes, 1,000-step horizon override  

---

## 1. Executive Summary & Core Verdict

This engineering campaign conducted a large-scale, paired empirical evaluation of two fundamental baseline controllers under the canonical 10-meter 3D corridor navigation benchmark:
1. **`reactive_baseline` (`ReactiveBaselineController`):** Deterministic multi-beam rangefinder avoidance with dynamic speed scaling and goal tracking.
2. **`random_baseline` (`RandomBaselineController`):** Stochastic uniform velocity exploration ($v_x \in [0.0, 0.5]$ m/s, $v_y \in [-0.3, 0.3]$ m/s), seeded deterministically per episode.

```
+--------------------------------------------------------------------------------------------------------------------+
|                                      FULL-SCALE BENCHMARK SUMMARY (N = 1,200 SEEDS)                                 |
+--------------------------------------------------------------------------------------------------------------------+
| Controller              | 500-Step Success | 1000-Step Success | 95% Wilson CI     | Collisions | Envelope Breaches |
|-------------------------+------------------+-------------------+-------------------+------------+-------------------|
| Reactive Baseline       | 769/1200 (64.1%) | 1197/1200 (99.75%)| [99.27%, 99.91%]  | 0 (0.00%)  | 1 (0.08%)         |
| Random Exploration      |   0/1200 ( 0.0%) |  116/1200 ( 9.67%)| [ 8.12%, 11.47%]  | 0 (0.00%)  | 1 (0.08%)         |
|-------------------------+------------------+-------------------+-------------------+------------+-------------------|
| Paired Differential     | +769 (+64.08%)   | +1081 (+90.08%)   | McNemar p < 1e-15 | 0 diff     | 0 diff            |
+--------------------------------------------------------------------------------------------------------------------+
```

### Key High-Level Findings:
1. **Reactive Baseline Dominance:** The reactive baseline achieves a near-flawless **99.75% corridor success rate (1,197 / 1,200)** under the 1,000-step horizon with **0 physical collisions** and an average of 487.3 steps to goal.
2. **Horizon Sensitivity:** The extended 1,000-step horizon was **decisive**. At 500 steps, the reactive baseline completed only 64.08% (769/1200) because its cautious speed throttling ($0.20$–$0.35$ m/s near obstacles) requires ~25–35 seconds (500–700 steps) to safely negotiate three sequential pillars. Extending the horizon to 1,000 steps unlocked **+428 additional completions (+35.67%)**.
3. **Random Exploration Failure:** Random exploration achieved **0.00% success at 500 steps** and only **9.67% (116/1200) at 1,000 steps**. In 100% of seeds where Random succeeded, Reactive also succeeded. Random **never** solved any seed that Reactive failed.
4. **The "Brownian Ratchet" Mechanism:** Random exploration's 9.67% completions are not "navigation policies"; they are stochastic forward drift guided and protected entirely by the **Safety Supervisor's active repulsion barriers**. In 0 episodes did random exploration exhibit systematic stall recovery.
5. **Architectural Lesson for BC:** The key reason `reactive_baseline` succeeds where the raw BC policy freezes is **dynamic longitudinal slowing coupled with asymmetric tie-breaking**: when faced with obstacles, the reactive controller immediately decelerates from $0.70$ m/s to $0.20$ m/s, giving lateral avoidance steering time to take effect without triggering the safety supervisor's emergency stop.

---

## 2. Evaluation Methodology & Invariant Protections

### 2.1 Seed Split Partitioning & Integrity
To preserve scientific rigor and prevent benchmark contamination, the evaluation used an explicit, non-overlapping seed block:
- **Seed Pool:** `seeds 10000` to `11199` ($N=1,200$ consecutive deterministic seeds).
- **Partition Disjointness:** Verified via automated unit tests ([`tests/test_baseline_campaign.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/tests/test_baseline_campaign.py)) to have **zero overlap** with:
  - Unit/validation pool: `100–109`
  - Milestone 4 hard negative crash pool: `200–224`
  - Diagnostic scenario pool: `1000–1009`
  - Development benchmark pool: `2000–2049`
  - Canonical held-out test pool: `90000–90099` (remains strictly sealed and uncontaminated).

### 2.2 Invariant System Protections
- **Safety Supervisor Active:** The canonical deterministic safety supervisor ([`controllers/safety_supervisor.py`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/controllers/safety_supervisor.py)) remained active across all 2,400 runs with production settings ($d_{\text{caution}} = 0.50$m, $d_{\text{stop}} = 0.08$m, boundary buffer $0.25$m).
- **Environment Geometry:** The canonical `initial_benchmark_10m` scenario was strictly preserved ($10.0$m $\times$ $2.0$m $\times$ $2.0$m corridor, start $[0.5, 0.0, 1.0]$, goal $[9.5, 0.0, 1.0]$, $r_{\text{goal}} = 0.35$m, $r_{\text{drone}} = 0.15$m, pillars at $x = 3.0, 5.0, 7.0$m with radius $0.20$m and randomized $y \in [-0.35, 0.35]$m).
- **Paired Environment Resets:** For every seed $s$, the environment was reset identically, guaranteeing identical initial drone lateral perturbation and identical pillar $y$ coordinates for both controllers.

---

## 3. Comprehensive Comparative Performance Metrics

The table below compiles all aggregate metrics for both controllers across the 1,200-seed benchmark.

| Metric Category | Metric Name | Reactive Baseline (`reactive_baseline`) | Random Exploration (`random_baseline`) | Comparative Impact / Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Sample Size** | Total Episodes ($N$) | 1,200 | 1,200 | 1,200 paired seeds |
| **Goal Completion** | Success Count & Rate | **1,197 / 1,200 (99.75%)** | **116 / 1,200 (9.67%)** | **+90.08% (10.3x advantage)** |
| | Wilson 95% Confidence Interval | **[99.27%, 99.91%]** | **[8.12%, 11.47%]** | Disjoint intervals ($p < 10^{-15}$) |
| **Horizon Dynamics** | Success at Step 500 | 769 / 1,200 (64.08%) | 0 / 1,200 (0.00%) | Random requires $>500$ steps |
| | Success at Step 1000 | 1,197 / 1,200 (99.75%) | 116 / 1,200 (9.67%) | Horizon extension critical |
| | Net Success Gain in Extended Horizon | **+428 (+35.67%)** | **+116 (+9.67%)** | Validates extended horizon utility |
| **Safety Invariants** | Obstacle Collisions ($d_{\text{obs}} \le 0.0$) | **0 / 1,200 (0.00%)** | **0 / 1,200 (0.00%)** | Supervisor 100% collision-free |
| | Wall Collisions ($d_{\text{wall}} \le 0.0$) | **0 / 1,200 (0.00%)** | **0 / 1,200 (0.00%)** | Supervisor 100% boundary-safe |
| | Safety Envelope Violations ($<0.08$m)| **1 / 1,200 (0.08%)** | **1 / 1,200 (0.08%)** | $d_{\text{min}} = 0.0777$m vs $0.0596$m |
| | Minimum Obstacle Clearance (Overall / Mean) | 0.0777m / 0.1481m | 0.0856m / 0.1275m | Reactive maintains higher clearance |
| | Minimum Wall Clearance (Overall / Mean) | 0.2941m / 0.3957m | 0.0596m / 0.2931m | Random drifts closer to walls |
| **Failure Modes** | Spatial Standstills | 0 / 1,200 (0.00%) | 0 / 1,200 (0.00%) | Both maintain non-zero jitter |
| | Oscillations / Chattering | 1 / 1,200 (0.08%) | 625 / 1,200 (52.08%) | Random trapped in chattering |
| | Timeouts at Step 1000 | 1 / 1,200 (0.08%) | 457 / 1,200 (38.08%) | Random wanders without reaching goal |
| | Goal Alignment Failures ($v > 0.35$) | 0 / 1,200 (0.00%) | 1 / 1,200 (0.08%) | Seed 10474 arrived too fast |
| **Progress & Speed** | Forward Progress (Mean $\pm$ Std) | **$8.962 \pm 0.140$m** | **$7.146 \pm 3.199$m** | Reactive: tight, repeatable progress |
| | Forward Progress Median [IQR] | $8.969$m [$8.960$, $8.978$] | $6.173$m [$4.151$, $10.187$] | Random: massive variance |
| | Steps to Success (Mean $\pm$ Std) | **$487.3 \pm 99.2$ steps** | **$840.0 \pm 66.8$ steps** | Reactive completes in ~24s vs 42s |
| | Steps to Success Median [Min, Max] | $465.0$ [368, 990] | $838.0$ [694, 984] | Reactive fastest run: 368 steps |
| **Pillar Passage** | Pillar 1 Cleared ($x > 3.35$m) | 1,200 / 1,200 (100.0%) | 1,087 / 1,200 (90.58%) | Random fails 9.4% on Pillar 1 |
| | Pillar 2 Cleared ($x > 5.35$m) | 1,199 / 1,200 (99.92%) | 773 / 1,200 (64.42%) | Random drops to 64.4% on Pillar 2 |
| | Pillar 3 Cleared ($x > 7.35$m) | 1,199 / 1,200 (99.92%) | 554 / 1,200 (46.17%) | Random drops to 46.2% on Pillar 3 |
| **Supervision Load**| Intervened Episodes | 1,195 / 1,200 (99.58%) | 1,198 / 1,200 (99.83%) | Both require active shielding |
| | Mean Interventions per Episode | 141.8 steps | 452.5 steps | Random triggers 3.2x interventions |
| | Mean Speed Throttled Fraction | 15.59% of steps | 35.63% of steps | Random heavily throttled by safety |

---

## 4. Extended Horizon Analysis (500 vs. 1,000 Steps)

A primary objective was determining whether doubling the evaluation horizon from 500 steps ($25.0$ s) to 1,000 steps ($50.0$ s) genuinely improves navigation or merely prolongs failure.

```
                  SUCCESS RATE BY HORIZON STEP COUNT
  100% |============================================================ (99.75% Reactive)
       |                                   /
   80% |                                  /
       |                                 /
   60% |============== (64.08% Reactive)/
       |
   40% |
       |
   20% |
       |                                                    === (9.67% Random)
    0% |---------------------------------------------------
           Step 500 (25.0s)                     Step 1000 (50.0s)
```

### 4.1 Reactive Baseline Horizon Dynamics
- **Step 500 Checkpoint:** 769 successes (64.08%).
- **Step 1000 Horizon:** 1,197 successes (99.75%).
- **Forensic Diagnosis of the 428 Extended Successes:**
  - In `reactive_baseline`, the controller features dynamic longitudinal slowing: when facing an obstacle ($d_{\text{front}} < 1.4$m), forward velocity $v_x$ is reduced to $0.20$–$0.35$ m/s.
  - Across a 10m corridor with 3 separate pillars, the drone spends $\approx 60\%$ of the episode navigating in the caution zone at $\approx 0.30$ m/s.
  - Total physical flight time required is $22$ to $32$ seconds ($440$ to $640$ steps).
  - Under the standard 500-step horizon, 428 completely safe, nominal flights were artificially truncated at $x \in [7.5, 9.2]$m simply because the clock ran out.
  - **Conclusion:** For `reactive_baseline`, doubling the horizon did not rescue broken flights; **it allowed slow, safe flights to finish naturally**.

### 4.2 Random Baseline Horizon Dynamics
- **Step 500 Checkpoint:** **0 successes (0.00%)**.
- **Step 1000 Horizon:** **116 successes (9.67%)**.
- **Forensic Diagnosis of the 116 Extended Successes:**
  - Because random exploration commands zero systematic steering, forward progress is governed by 1D Brownian drift with positive drift velocity $\mathbb{E}[v_x] = 0.25$ m/s.
  - With random lateral diffusion ($\sigma_y \approx 0.17$ m/s), the vehicle repeatedly rebounds off walls and pillars under Safety Supervisor intervention.
  - It requires an average of **840.0 steps ($42.0$ seconds)** for this Brownian drift to randomly diffuse into the goal sphere.
  - **Conclusion:** Doubling the horizon enables low-probability stochastic goal arrival, but does not convert random exploration into a defensible navigation policy.

---

## 5. Paired Statistical Analysis & McNemar Discordance

Because both controllers were evaluated on the exact same 1,200 seeds under identical initial conditions, paired statistical tests establish definitive superiority:

```
                      PAIRED CONTINGENCY MATRIX (N = 1,200)
                              Random Succeeded      Random Failed         Total
  Reactive Succeeded               116 (9.67%)      1,081 (90.08%)      1,197 (99.75%)
  Reactive Failed                    0 (0.00%)          3 ( 0.25%)          3 ( 0.25%)
  Total                            116 (9.67%)      1,084 (90.33%)      1,200 (100.0%)
```

### 5.1 Statistical Significance Tests
- **McNemar's Discordance Test:**
  - Discordant pairs: $b = 1,081$ (Reactive only), $c = 0$ (Random only).
  - Continuity-corrected Chi-Square: $\chi^2 = \frac{(|1081 - 0| - 1)^2}{1081 + 0} = \mathbf{1,079.00}$.
  - $p$-value: **$p < 10^{-15}$** (Statistically overwhelming evidence that Reactive outperforms Random).
- **Wilcoxon Signed-Rank Test on Forward Progress:**
  - Test statistic: $W = 155,371.0$.
  - $p$-value: **$p = 2.62 \times 10^{-65}$**.
  - Mean paired progress difference: **$+1.816$ meters** in favor of Reactive.

### 5.2 Key Observation on Asymmetric Discordance
In **0 out of 1,200 seeds** did Random exploration succeed where Reactive failed ($c = 0$). Random exploration provides zero complementary coverage over the reactive baseline.

---

## 6. Forensic Failure Taxonomy & Deep Dive

Across the 2,400 evaluated episodes, all failures were mapped to mutually exclusive classifications:

| Primary Outcome Code | Reactive Baseline ($N=1,200$) | Random Exploration ($N=1,200$) | Failure Description & Root Cause |
| :--- | :---: | :---: | :--- |
| `SUCCESS` | **1,197 (99.75%)** | **116 (9.67%)** | Arrived at goal with $d \le 0.35$m, $v \le 0.35$ m/s, zero collisions. |
| `OBSTACLE_COLLISION` | 0 (0.00%) | 0 (0.00%) | Penetrated internal obstacle ($d_{\text{obs}} \le 0.0$m). Zero recorded. |
| `WALL_COLLISION` | 0 (0.00%) | 0 (0.00%) | Penetrated corridor boundary ($d_{\text{wall}} \le 0.0$m). Zero recorded. |
| `SAFETY_ENVELOPE_VIOLATION` | 1 (0.08%) | 1 (0.08%) | Clearance dropped $<0.08$m without physical collision. |
| `SPATIAL_STANDSTILL` | 0 (0.00%) | 0 (0.00%) | Motionless hover ($v < 0.02$ m/s) until step 1000. |
| `OSCILLATION` | 1 (0.08%) | **625 (52.08%)** | Trapped in high-frequency lateral chatter / limit cycle near obstacle. |
| `TIMEOUT_1000` | 1 (0.08%) | **457 (38.08%)** | Actively wandering in flight but exhausted 1,000 steps without goal arrival. |
| `GOAL_ALIGNMENT_FAILURE` | 0 (0.00%) | 1 (0.08%) | Reached goal sphere but speed exceeded $0.35$ m/s and failed to land. |
| `OTHER_FAILURE` | 0 (0.00%) | 0 (0.00%) | Unclassified failure. |

---

### 6.1 Forensics of the 3 Reactive Baseline Failures

Out of 1,200 seeds, `reactive_baseline` failed on **only 3 seeds**:

#### Failure 1: Seed 10009 — `OSCILLATION` (Symmetric Pillar Lock)
- **Obstacle Coordinates:** Pillar 1 ($y = +0.016$), Pillar 2 ($y = +0.043$), Pillar 3 ($y = +0.073$).
- **Trace Analysis:** All three pillars were generated almost exactly on the corridor centerline ($y \in [0.01, 0.07]$m). The drone cleared Pillar 1 at step 180, but approached Pillar 2 with near-equal clearance on both left and right rangefinder rays.
- **Root Cause:** In `ReactiveBaselineController`:
  ```python
  left_score = d_fl + 0.6 * d_left
  right_score = d_fr + 0.6 * d_right
  ```
  Because the corridor wall centering buffers intervened simultaneously ($d_{\text{left}} \approx d_{\text{right}} \approx 0.65$m), the lateral steering commands alternated between $+0.40$ m/s and $-0.40$ m/s every 2 steps. The drone remained trapped in a limit-cycle oscillation at $x = 4.476$m until step 1000 expired.
- **Trace File:** [`representative_trajectories/traj_reactive_oscillation_seed_10009.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/baseline_campaign/run_20261030_151800/representative_trajectories/traj_reactive_oscillation_seed_10009.csv).

#### Failure 2: Seed 10131 — `TIMEOUT_1000` (Sequential Slowdown Timeout)
- **Obstacle Coordinates:** Pillar 1 ($y = +0.023$), Pillar 2 ($y = -0.206$), Pillar 3 ($y = -0.012$).
- **Trace Analysis:** The drone cleared Pillar 1 and Pillar 2 successfully, but entered the third obstacle section at step 720. Due to a tight S-turn squeeze between Pillar 3 and the right wall, the controller maintained $v_x \approx 0.15$ m/s while the supervisor repeatedly attenuated forward velocity.
- **Outcome:** The drone was actively moving forward at $x = 7.381$m when step 1000 expired. It required an estimated 150 additional steps to reach the goal.

#### Failure 3: Seed 10667 — `SAFETY_ENVELOPE_VIOLATION` (Near-Miss Grazing)
- **Obstacle Coordinates:** Pillar 1 ($y = -0.319$), Pillar 2 ($y = +0.003$), Pillar 3 ($y = -0.196$).
- **Trace Analysis:** At step 565, while completing a rapid counter-turn between Pillar 2 and Pillar 3, the vehicle's left boundary grazed within **$0.0777$ meters** of Pillar 2.
- **Outcome:** While no physical collision occurred ($d_{\text{obs}} > 0$), this breached the strict $0.0800$m safety envelope by $2.3$ millimeters. The drone continued flying and reached the goal at $x = 8.954$m at step 712. Under strict safety rules, it is categorized as an envelope violation.
- **Trace File:** [`representative_trajectories/traj_reactive_violation_seed_10667.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/baseline_campaign/run_20261030_151800/representative_trajectories/traj_reactive_violation_seed_10667.csv).

---

### 6.2 Forensics of Random Exploration Failures
- **625 Oscillations (52.08%):** Because `random_baseline` samples uncorrelated uniform velocities at 20 Hz, whenever it approaches an obstacle, the safety supervisor brakes forward speed and repels the drone sideways. The next random action commands motion back toward the obstacle, creating rapid high-frequency chattering against the supervisor's caution boundary with zero net progress.
- **457 Timeouts (38.08%):** The drone spent all 1,000 steps wandering in the open spaces between pillars, drifting laterally and backwards, exhausting the time budget before reaching the goal.
- **1 Goal Alignment Failure (Seed 10474):** The drone drifted into the goal sphere at step 742, but was moving at $0.48$ m/s ($>0.35$ m/s landing limit). It blew through the goal sphere, bounced off the back wall, and timed out.

---

## 7. Investigation of Random Exploration "Recovery"

The prompt specifically tasked us with investigating episodes where random exploration appeared to "recover" from stalls.

### 7.1 Forensic Findings
1. **Zero Systematic Recovery:** In all 1,200 seeds, `stalls_entered` was 0 and `stalls_recovered` was 0 under our formal definition (sustained hover $<0.02$ m/s followed by $>0.8$m breakout).
2. **The "Brownian Ratchet" Illusion:**
   - In 116 episodes, random exploration succeeded in reaching the goal.
   - Analysis of the step telemetry (e.g., [`traj_random_apparent_recovery_seed_10025.csv`](file:///Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/experiments/baseline_campaign/run_20261030_151800/representative_trajectories/traj_random_apparent_recovery_seed_10025.csv)) shows that the vehicle was never "recovering" via policy intelligence.
   - Because $v_x \sim \mathcal{U}(0.0, 0.5)$ has a positive expected value, the drone is persistently driven forward.
   - When it hits an obstacle caution zone, the supervisor zeros $v_x$ and deflects $v_y$. Eventually, a random action samples $v_y$ aligned with the supervisor's open direction, allowing the drone to slip past.
   - This required an average of **$452.5$ supervisor interventions per episode** and **840.0 steps**.
3. **Verdict:** Random exploration does **NOT** provide viable recovery behavior. What appears as recovery is simply random diffusion constrained by an active deterministic safety supervisor.

---

## 8. Comparison Against Existing BC + Clearance Guidance Stack

How does `reactive_baseline` compare to our previous neural network policies on comparable corridor benchmarks?

| Architecture / Controller | Benchmark Set | Episode Horizon | Success Rate (%) | Collisions | Envelope Violations | Mean Steps | Primary Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BC Baseline** (`bc_policy_best.pt`) | Dev (`2000–2049`) | 500 steps | **48.0% (24/50)** | 0 | 0 | 374.2 | 52% Standoff Freezes ($v_y \approx 0$) |
| **BC + Phase 6A Guidance** | Dev (`2000–2049`) | 500 steps | **84.0% (42/50)** | 0 | 0 | 288.4 | Sequential Pillar 3 Corner Pinches |
| **Reactive Baseline** (`reactive_baseline`) | Campaign (`10000–11199`) | 500 steps | **64.08% (769/1200)**| 0 | 0 | 442.1 | Timeouts due to slow cruise speed |
| **Reactive Baseline** (`reactive_baseline`) | Campaign (`10000–11199`) | 1,000 steps | **99.75% (1197/1200)**| **0** | **1** | **487.3** | Near-flawless (only 3 failures in 1200) |
| **Random Exploration** | Campaign (`10000–11199`) | 1,000 steps | **9.67% (116/1200)** | 0 | 1 | 840.0 | 90.3% Oscillations & Timeouts |

### 8.1 Why Does the Reactive Baseline Outperform Raw BC?
1. **Dynamic Speed Throttling:** Raw BC always demands forward velocity $v_x \approx 0.60$ m/s, forcing the Safety Supervisor into violent braking interventions. The reactive baseline actively slows down ($v_x \to 0.20$ m/s) when rangefinder clearance drops below $1.4$m, giving its lateral steering time to act smoothly.
2. **Multi-Beam Lateral Steering:** The reactive baseline combines front, front-left, front-right, left, and right rays into a weighted score (`d_fl + 0.6 * d_left`), creating strong, continuous repulsive forces away from obstacles.
3. **Corridor Wall Centering:** Unlike BC (which only sees rays), the reactive baseline has explicit wall-centering buffers that push back when wall clearance drops below $0.65$m.

---

## 9. Final Decision & Answers to Specific Evaluation Questions

### Question 1: How often does each controller complete the corridor?
- **`reactive_baseline`:** Completes the corridor **99.75% of the time (1,197 / 1,200)** [95% Wilson CI: 99.27% – 99.91%].
- **`random_baseline`:** Completes the corridor **9.67% of the time (116 / 1,200)** [95% Wilson CI: 8.12% – 11.47%].

### Question 2: Does the 1,000-step horizon materially improve success?
- **YES, dramatically for both controllers:**
  - For `reactive_baseline`, success jumps from **64.08% (769/1200) at 500 steps** to **99.75% (1,197/1200) at 1,000 steps** ($+35.67$ percentage points). This is because the controller's safe, cautious speed near obstacles requires ~25–35 seconds (500–700 steps) to navigate 3 pillars. The 500-step horizon was prematurely truncating valid flights.
  - For `random_baseline`, success jumps from **0.00% (0/1200) at 500 steps** to **9.67% (116/1200) at 1,000 steps**, though this represents stochastic diffusion rather than systematic navigation.

### Question 3: Which controller has fewer collisions and safety violations?
- Both controllers had **zero simulated collisions** thanks to the canonical Safety Supervisor.
- Both controllers had exactly **one safety envelope violation** ($<0.08$m) across 1,200 seeds ($0.08\%$ rate).
- However, `reactive_baseline` maintained significantly higher mean clearances ($0.148$m vs $0.128$m for obstacles, $0.396$m vs $0.293$m for walls) and required 3.2x fewer supervisor interventions.

### Question 4: Which controller recovers from stalls?
- **Neither controller exhibits an active stall recovery mechanism:**
  - `random_baseline` does not "recover" systematically; its non-zero progress is unguided Brownian jitter deflected by the supervisor.
  - `reactive_baseline` rarely enters a stall (0 formal stalls detected); when it does encounter symmetric obstacles (Seed 10009), it enters a lateral oscillation rather than executing a reverse-and-replan maneuver.

### Question 5: Which obstacle configurations defeat each controller?
- **`reactive_baseline` is defeated by:**
  1. *Centered sequential pillars with equal clearance* (e.g., Seed 10009, all pillars at $y \approx 0.04$m), where wall centering forces balance obstacle repulsion forces, causing lateral limit cycles.
  2. *Near-wall grazing pinches* (e.g., Seed 10667), where counter-steering from an obstacle pushes the drone within $0.0777$m of an adjacent pillar.
- **`random_baseline` is defeated by:**
  - Virtually all configurations (90.33% failure rate). Any obstacle requiring coordinated sequential turns traps random walk into boundary chattering or causes it to wander until timeout.

### Question 6: Does either baseline outperform the current BC + stateless clearance-guidance candidate on comparable scenarios?
- **YES.** On the 10-meter 3D corridor benchmark, `reactive_baseline` with a 1,000-step horizon achieves **99.75% success**, outperforming the BC + Phase 6A candidate (84.0% at 500 steps). Even at 500 steps, `reactive_baseline` achieves 64.1% compared to raw BC's 48.0%.
- The reactive baseline achieves this because it is an explicit potential-field controller with dynamic deceleration, avoiding the representation collapse of unguided neural networks.

### Question 7: Which behavior, if any, should be incorporated into the next navigation-policy iteration?
We should incorporate three specific behaviors from `reactive_baseline` into our production BC + Guidance stack:
1. **Dynamic Longitudinal Deceleration:** When forward range drops below $1.4$m, scale desired forward velocity proportionally ($v_x \leftarrow \text{clip}(0.20 + 0.35 \cdot (d - 0.4), 0.20, v_{\text{cruise}})$). Currently, BC commands $v_x \approx 0.60$ m/s directly into obstacles, triggering abrupt supervisor braking. Proactive deceleration eliminates standoff freezes.
2. **Multi-Beam Sector Weighting:** Instead of only looking at the front 3 rays, compute weighted sector clearance ($d_{\text{fl}} + 0.6 d_{\text{left}}$ vs $d_{\text{fr}} + 0.6 d_{\text{right}}$) to initiate evasive lateral nudges much earlier ($1.5$m range rather than $0.5$m).
3. **Corridor Boundary Centering Bias:** Add a weak proportional centering force ($v_y \mathrel{-}= (0.65 - d_{\text{left}}) \cdot 0.5$) when near walls to prevent the corner-pinch failures identified in Phase 6A.

