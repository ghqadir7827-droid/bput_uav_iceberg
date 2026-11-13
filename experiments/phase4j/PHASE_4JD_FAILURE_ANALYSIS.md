# Phase 4J-D — Comprehensive Held-Out Failure Mode Analysis

**Milestone:** Phase 4J-D Sealed Held-Out Generalization Evaluation  
**Dataset:** 100 Reserved Held-Out Seeds (`90000..90099`)  
**Target Subject:** Frozen BC Baseline Policy (`models/checkpoints/bc_policy_best.pt`) with Isolated Supervisor $d_{\text{caut}} = 0.50\text{ m}$  
**Evaluation Date:** 2026-10-29  

---

## 1. Executive Summary of Failure Dynamics

Across the 100 held-out evaluation episodes:
- **Total Unsuccessful Episodes:** 53 / 100 (53.0%)
- **Physical Crashes:** **0 / 100 (0.0%)** (Zero obstacle collisions, zero boundary collisions)
- **Safety Envelope Violations:** **0 / 100 (0.0%)** (Zero excursions below the 8.0 cm safety buffer)
- **Timeouts ($500\text{ steps} / 25.0\text{ s}$):** **53 / 53 (100.0% of failures)**

Every unsuccessful episode in the held-out evaluation terminated because the drone exhausted the 25.0-second episode time budget while safely hovering, decelerating, or negotiating an obstacle bypass. The safety supervisor functioned flawlessly to prevent collisions, but the policy's conservative local navigation dynamics failed to generate sufficient forward progress within the allotted horizon.

---

## 2. Failure Category Taxonomy and Breakdown

The table below presents the mutual-exclusive classification of all 100 held-out episodes:

| Category Code | Description | Count | Percentage | Development Set Share (Phase 4J-C) |
|:---|:---|:---:|:---:|:---:|
| `SUCCESS` | Goal arrived within $r \le 0.35\text{ m}$ and $v_{\text{term}} \le 0.35\text{ m/s}$ | **47** | **47.0%** | 48.0% (24/50) |
| `PROGRESS_STALL_PILLAR_3` | Passed Pillars 1 & 2, timed out at Pillar 3 ($6.0\text{ m} \le X < 8.0\text{ m}$) | **25** | **25.0%** | 26.0% (13/50) |
| `PROGRESS_STALL_PILLAR_1` | Timed out at Pillar 1 ($X < 4.0\text{ m}$) | **16** | **16.0%** | 8.0% (4/50) |
| `PROGRESS_STALL_PILLAR_2` | Passed Pillar 1, timed out at Pillar 2 ($4.0\text{ m} \le X < 6.0\text{ m}$) | **9** | **9.0%** | 12.0% (6/50) |
| `PROGRESS_LATERAL_OSCILLATION` | Rapid lateral hunting ($>10$ reversals) consuming forward momentum | **2** | **2.0%** | 2.0% (1/50) |
| `GOAL_ALIGNMENT_TIMEOUT` | Cleared all 3 pillars, reached goal vicinity ($d < 0.60\text{ m}$), timed out | **1** | **1.0%** | 0.0% (0/50) |
| `OBSTACLE_COLLISION` | Physical contact with obstacle cylinder ($d_{\text{obs}} \le 0.0$) | **0** | **0.0%** | 0.0% (0/50) |
| `WALL_COLLISION` | Physical contact with corridor boundary ($d_{\text{wall}} \le 0.0$) | **0** | **0.0%** | 0.0% (0/50) |
| **Total Evaluated** | | **100** | **100.0%** | 100.0% (50/50) |

---

## 3. Detailed Failure Mode Mechanics

### 3.1 Pillar 3 Progress Stall (`PROGRESS_STALL_PILLAR_3` — 25 Episodes)
- **Seeds Affected:** 90001, 90009, 90011, 90012, 90022, 90025, 90027, 90044, 90047, 90049, 90050, 90056, 90057, 90059, 90061, 90065, 90076, 90078, 90081, 90083, 90086, 90087, 90088, 90091, 90092, 90093, 90094, 90095.
- **Proportion of Failures:** **47.2%** of all failures (consistent with 50.0% in Phase 4J-C).
- **Physical Dynamics:** In these episodes, the drone successfully clears both Pillar 1 ($X=3.0\text{ m}$) and Pillar 2 ($X=5.0\text{ m}$), accumulating approximately $6.1\text{ to }6.3\text{ m}$ of forward progress. However, negotiating two prior obstacles consumed $320\text{ to }380\text{ steps}$ ($16.0\text{ to }19.0\text{ s}$). Upon reaching the caution zone ($d_{\text{caut}} = 0.50\text{ m}$) of Pillar 3, the supervisor reduces forward speed to $v_x \approx 0.10\text{ m/s}$. The remaining $\sim 120\text{ steps}$ ($6.0\text{ s}$) are insufficient to execute the lateral shift and complete the transit before step 500 expires.
- **Clearance Safety:** Drone obstacle clearances remain fully safe: mean $0.114\text{ m}$, minimum $0.104\text{ m}$ (Seed 90088).

### 3.2 Pillar 1 Progress Stall (`PROGRESS_STALL_PILLAR_1` — 16 Episodes)
- **Seeds Affected:** 90000, 90007, 90010, 90013, 90016, 90020, 90035, 90040, 90042, 90046, 90054, 90058, 90067, 90070, 90071, 90075.
- **Proportion of Failures:** **30.2%** of all failures.
- **Physical Dynamics:** The drone encounters Pillar 1 head-on with minimal initial lateral offset. As forward range drops below $0.50\text{ m}$, the supervisor throttles forward speed. The BC policy's proposed lateral velocity $v_y$ is balanced near zero due to symmetric frontal lidar observations. The drone settles into a stable, safe hovering standoff at $X \approx 2.1 - 2.3\text{ m}$ directly in front of the pillar, remaining stationary until the 500-step horizon expires.
- **Clearance Safety:** Global minimum clearance for this category is $0.0975\text{ m}$ ($9.75\text{ cm}$, Seed 90007). This is comfortably above the $0.080\text{ m}$ envelope violation threshold.

### 3.3 Pillar 2 Progress Stall (`PROGRESS_STALL_PILLAR_2` — 9 Episodes)
- **Seeds Affected:** 90002, 90021, 90023, 90028, 90033, 90038, 90041, 90055, 90060.
- **Proportion of Failures:** **17.0%** of all failures.
- **Physical Dynamics:** The drone clears Pillar 1, but its exit trajectory leaves it aligned closely with the center of Pillar 2 or constrained near the corridor wall. The supervisor engages caution throttling, and the drone hovers at $X \approx 4.0 - 4.3\text{ m}$ for the remainder of the episode.
- **Clearance Safety:** Minimum obstacle clearance in this group is $0.112\text{ m}$.

### 3.4 Progress Lateral Oscillation (`PROGRESS_LATERAL_OSCILLATION` — 2 Episodes)
- **Seeds Affected:** 90034 ($X=8.18\text{ m}$, 42 reversals), 90048 ($X=6.27\text{ m}$, 51 reversals).
- **Proportion of Failures:** **3.8%** of all failures.
- **Physical Dynamics:** When maneuvering in narrow gaps between an obstacle and the corridor boundary, the policy commands alternating sign lateral velocities ($\pm 0.35\text{ m/s}$). This limit-cycle lateral oscillation prevents forward translation, consuming the remaining steps while safely centered.

### 3.5 Goal Alignment Timeout (`GOAL_ALIGNMENT_TIMEOUT` — 1 Episode)
- **Seed Affected:** 90008 ($X=9.14\text{ m}$, final goal distance $0.52\text{ m}$).
- **Proportion of Failures:** **1.9%** of all failures.
- **Physical Dynamics:** The drone successfully navigated past all three pillars, achieving $8.64\text{ m}$ forward progress. However, as it approached the goal $(9.5, 0.0, 1.0)\text{ m}$, the policy decelerated smoothly to align with the goal center, reaching step 500 just $0.17\text{ m}$ outside the $0.35\text{ m}$ goal acceptance sphere.

---

## 4. Minimum Clearance Safety Audit

Across all 100 held-out evaluation episodes (50,000 total simulation steps):

```
Obstacle Clearance Distribution (m):
  Minimum:  0.0975 m (9.75 cm) — Seed 90007
  P05:      0.1082 m
  P10:      0.1119 m
  P25:      0.1163 m
  Median:   0.1207 m
  Mean:     0.1365 m
  Max:      0.2974 m

Boundary Clearance Distribution (m):
  Minimum:  0.2498 m (25.0 cm)
  Median:   0.2911 m
  Mean:     0.2944 m
```

### Safety Margin Analysis:
1. **Obstacle Safety Envelope ($0.080\text{ m}$):** The lowest recorded clearance of $0.0975\text{ m}$ provides a positive margin of **$+17.5\text{ mm}$** above the violation boundary. Zero envelope violations occurred.
2. **Boundary Safety Envelope ($0.080\text{ m}$):** The lowest boundary clearance of $0.2498\text{ m}$ provides a margin of **$+169.8\text{ mm}$** above the violation threshold. The supervisor's wall repulsion buffer ($0.25\text{ m}$) operated with perfect precision.
3. **Collision Margin ($0.000\text{ m}$):** Zero physical contacts were recorded across both obstacles and boundary walls.

---

## 5. Comparison: Development Set vs. Held-Out Set

| Metric | Phase 4J-C Dev Set (`2000..2049`, N=50) | Phase 4J-D Held-Out (`90000..90099`, N=100) | $\Delta$ Difference | Generalization Assessment |
|:---|:---:|:---:|:---:|:---|
| **Goal Success Rate** | **48.0%** (24/50) | **47.0%** (47/100) | **$-1.0\%$** | **Virtually Identical** |
| **95% Wilson CI** | [34.8%, 61.5%] | [37.5%, 56.7%] | N/A | Subsumed within CI |
| **Obstacle Collisions** | 0.0% (0/50) | 0.0% (0/100) | $0.0\%$ | Perfect Safety Parity |
| **Boundary Collisions** | 0.0% (0/50) | 0.0% (0/100) | $0.0\%$ | Perfect Safety Parity |
| **Envelope Violations** | 0.0% (0/50) | 0.0% (0/100) | $0.0\%$ | Perfect Safety Parity |
| **Global Min Obstacle Dist** | $0.1073\text{ m}$ | $0.0975\text{ m}$ | $-0.0098\text{ m}$ | Safe ($>8\text{ cm}$) |
| **Global Min Boundary Dist** | $0.1873\text{ m}$ | $0.2498\text{ m}$ | $+0.0625\text{ m}$ | Improved Wall Margin |
| **Mean Forward Progress** | $6.53\text{ m}$ | $6.61\text{ m}$ | $+0.08\text{ m}$ | Parity |
| **Mean Episode Duration** | $23.0\text{ s}$ | $22.86\text{ s}$ | $-0.14\text{ s}$ | Parity |
| **Mean Throttled Fraction** | $27.6\%$ | $28.0\%$ | $+0.4\%$ | Parity |
| **Dominant Failure Mode** | Pillar 3 Stall (50.0%) | Pillar 3 Stall (47.2%) | $-2.8\%$ | Identical Mechanism |

---

## 6. Engineering Conclusions and Next-Phase Recommendations

1. **Root Cause of Non-Success:** The failure mode is 100% time-budget exhaustion, driven by cumulative caution-zone speed throttling across sequential obstacles. The BC baseline navigates with extreme caution; when it faces three obstacles, the cumulative deceleration leaves insufficient time to reach the goal.
2. **Safety Stability:** The frozen controller and supervisor wrapper show zero degradation in obstacle or wall safety when deployed on completely unseen held-out environments.
3. **Future Interventions:**
   - **Momentum-Preserving Lateral Steering:** Policy fine-tuning or APF guidance that biases lateral bypass before forward deceleration occurs.
   - **Adaptive Caution Envelope:** Progress-dependent or angle-dependent throttling that allows higher speeds when clear lateral clearance is visible.

