# Forensic Failure Taxonomy: 1,250-Episode Scaled UAV Benchmark
**Dataset:** `experiments/scaled_campaign/bc_comparison/bc_comparison_episodes_all.csv`  **Controllers:** `reactive_baseline`, `original_raw_bc`, `original_guided_bc`, `scaled_raw_bc`, `scaled_guided_bc`  **Corridor Tiers:** Level 1 (Basic), Level 2 (Standard), Level 3 (Challenging), Level 4 (Feasibility)
---
## 1. Global Performance & Failure Breakdown by Level
| Tier | Controller | Feasible Episodes | Successes (%) | Timeouts (%) | Collisions (Obs/Wall) | Envelope Violations (<0.15m) | Min Clearance Across ALL (m) | Mean Steps (All) | Mean Interventions |
|---|---|---|---|---|---|---|---|---|---|
| Level 1 (Basic) | `original_guided_bc` | 50 | 31 (62.0%) | 19 (38.0%) | 0/0 | 22 | 0.0386m | 804.2 | 212.0 |
| Level 1 (Basic) | `original_raw_bc` | 50 | 16 (32.0%) | 34 (68.0%) | 0/0 | 38 | 0.0372m | 906.6 | 475.3 |
| Level 1 (Basic) | `reactive_baseline` | 50 | 38 (76.0%) | 12 (24.0%) | 0/0 | 29 | 0.0587m | 686.7 | 345.0 |
| Level 1 (Basic) | `scaled_guided_bc` | 50 | 24 (48.0%) | 26 (52.0%) | 0/0 | 27 | 0.0893m | 820.7 | 449.9 |
| Level 1 (Basic) | `scaled_raw_bc` | 50 | 12 (24.0%) | 38 (76.0%) | 0/0 | 39 | 0.0930m | 915.5 | 605.3 |
| Level 2 (Standard) | `original_guided_bc` | 100 | 42 (42.0%) | 54 (54.0%) | 0/4 | 87 | 0.0002m | 894.5 | 498.5 |
| Level 2 (Standard) | `original_raw_bc` | 100 | 36 (36.0%) | 64 (64.0%) | 0/0 | 90 | 0.0002m | 917.7 | 560.9 |
| Level 2 (Standard) | `reactive_baseline` | 100 | 90 (90.0%) | 10 (10.0%) | 0/0 | 74 | 0.0621m | 613.4 | 225.6 |
| Level 2 (Standard) | `scaled_guided_bc` | 100 | 68 (68.0%) | 32 (32.0%) | 0/0 | 46 | 0.0720m | 702.0 | 387.7 |
| Level 2 (Standard) | `scaled_raw_bc` | 100 | 29 (29.0%) | 71 (71.0%) | 0/0 | 90 | 0.0172m | 900.5 | 671.3 |
| Level 3 (Challenging) | `original_guided_bc` | 50 | 29 (58.0%) | 21 (42.0%) | 0/0 | 39 | 0.0151m | 824.9 | 437.9 |
| Level 3 (Challenging) | `original_raw_bc` | 50 | 7 (14.0%) | 43 (86.0%) | 0/0 | 48 | 0.0002m | 980.7 | 753.6 |
| Level 3 (Challenging) | `reactive_baseline` | 50 | 43 (86.0%) | 7 (14.0%) | 0/0 | 44 | 0.0472m | 626.0 | 247.1 |
| Level 3 (Challenging) | `scaled_guided_bc` | 50 | 21 (42.0%) | 29 (58.0%) | 0/0 | 34 | 0.0683m | 818.7 | 588.8 |
| Level 3 (Challenging) | `scaled_raw_bc` | 50 | 4 (8.0%) | 45 (90.0%) | 1/0 | 46 | 0.0000m | 967.8 | 853.5 |
| Level 4 (Boundary & Feasibility) | `original_guided_bc` | 25 | 14 (56.0%) | 11 (44.0%) | 0/0 | 25 | 0.0002m | 866.4 | 541.5 |
| Level 4 (Boundary & Feasibility) | `original_raw_bc` | 25 | 4 (16.0%) | 21 (84.0%) | 0/0 | 25 | 0.0025m | 985.9 | 797.0 |
| Level 4 (Boundary & Feasibility) | `reactive_baseline` | 25 | 10 (40.0%) | 15 (60.0%) | 0/0 | 24 | 0.0413m | 817.8 | 558.7 |
| Level 4 (Boundary & Feasibility) | `scaled_guided_bc` | 25 | 4 (16.0%) | 21 (84.0%) | 0/0 | 24 | 0.0760m | 940.6 | 810.5 |
| Level 4 (Boundary & Feasibility) | `scaled_raw_bc` | 25 | 2 (8.0%) | 21 (84.0%) | 2/0 | 25 | 0.0012m | 913.6 | 821.9 |

---
## 2. In-Depth Failure Taxonomy: Level 2 Target (`scaled_guided_bc`)
In Level 2 Standard ($3.8 \times 2.5 \times 14.0$ m, $N=100$), `scaled_guided_bc` achieved **68 successes and 32 failures**.
- **Total Collision Count:** **0** (0 obstacle collisions, 0 wall collisions).  - **Total Timeout Count:** **32** (100% of all failures are timeouts at step 1,000).  - **Mean Safety Interventions in Failed Episodes:** **801.2 / 1,000 steps (80.1% duration)**.  - **Mean Cruising Speed in Failed Episodes:** **0.201 m/s** (vs 0.535 m/s in successful episodes).
### Stage-Wise Breakdown of Failures
1. **Stage 3 — Obstacle 3 Choke Deadlock ($X \approx 9.5$–$10.5$ m):** **22 / 32 episodes (68.8% of all failures)**     - *Mechanism:* Drone evades Obstacle 1 and 2 to the right ($Y \approx -0.7$ m), then centers. Approaching Obstacle 3 at $(10.5, -0.57)$, the obstacle is at azimuth $-37^\circ$. Rays 0, 1, 2 register clear space ($c_{\text{front}} > 3.0$ m), causing Guidance to output `NONE`. BC predicts forward flight into the obstacle. Ray 4 hits the obstacle at $<0.5$ m, triggering supervisor `obstacle_brake`. The drone freezes at $X \approx 10.0$–$10.4$ m until step 1,000 timeout.     - *Representative Seeds:* `[20000, 20012, 20015, 20024, 20026, 20027, 20031, 20034]`
2. **Stage 1 — Obstacle 1 Approach Stall ($X \approx 2.5$–$3.5$ m):** **6 / 32 episodes (18.8% of all failures)**     - *Mechanism:* Early deceleration before Obstacle 1 without sufficient lateral commitment. The drone slows to $<0.10$ m/s, enters supervisor caution/stopping mode, and does not build lateral momentum to clear the pillar.     - *Representative Seeds:* `[20016, 20022, 20033, 20056, 20058, 20091]`
3. **Stage 2 — Inter-Obstacle Transition ($X \approx 6.0$–$7.5$ m):** **3 / 32 episodes (9.4% of all failures)**     - *Mechanism:* Sluggish recentering after evading Obstacle 1 leaves the drone off-center when meeting Obstacle 2, resulting in an emergency brake lock.     - *Representative Seeds:* `[20009, 20045, 20068]`
4. **Stage 4 — Terminal Goal Acquisition ($X \ge 11.0$ m):** **1 / 32 episodes (3.1% of all failures)**     - *Mechanism:* Drone reaches $X = 12.2$ m but goal-seeking release or slight lateral drift leaves it hovering just outside the $0.50$ m acceptance sphere until step 1,000.     - *Representative Seeds:* `[20073]`
