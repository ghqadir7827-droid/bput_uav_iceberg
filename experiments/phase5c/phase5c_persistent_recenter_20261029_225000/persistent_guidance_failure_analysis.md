# PHASE 5C — PERSISTENT GUIDANCE FAILURE ANALYSIS
## Comprehensive Forensic Investigation of Controller C Degradation

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Isolation Directory:** `experiments/phase5c/phase5c_persistent_recenter_20261029_225000/`

---

## 1. Executive Summary & Core Diagnostic Finding

In the paired development benchmark across all 50 development seeds (`2000–2049`), Controller C (Persistent Clearance Guidance with `SETTLING` Re-centering) suffered a **catastrophic performance collapse**:

| Controller Configuration | Goal Successes | Success Rate (%) | Wilson 95% CI | Obstacle Violations (<0.08m) | Min Obstacle Clearance | Mean Throttled Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Controller A (Frozen Baseline)** | 24 / 50 | 48.0% | `[34.8%, 61.5%]` | **0** | **0.1070 m** | 27.57% |
| **Controller B (Stateless Guidance)** | **35 / 50** | **70.0%** | `[56.2%, 80.9%]` | **0** | **0.0870 m** | **20.18%** |
| **Controller C (Persistent Guidance)**| **17 / 50** | **34.0%** | `[22.4%, 47.8%]` | **1 Violation (Seed 2001)** | **0.0223 m (FAILED)** | **37.15% (Choked)** |

### Paired Comparison Matrix:
- **Controller B vs Controller A:** **+11 net wins** (18 wins, 7 losses, $p = 0.0433$) — **Statistically significant improvement**.
- **Controller C vs Controller B:** **-18 net losses** (4 wins, **22 losses**, $p = 0.0005$) — **Severe, statistically significant degradation**.
- **Controller C vs Controller A:** **-7 net losses** (7 wins, 14 losses, $p = 0.1892$) — **Performs worse than the unguided baseline**.
- **Safety Envelope Ineligibility:** In Seed 2001, Controller C recorded a minimum obstacle clearance of **$0.0223\text{ m}$**, violating the mandatory $0.080\text{ m}$ safety envelope. Controller C is strictly disqualified from promotion.

---

## 2. Quantitative State Occupancy Analysis

Auditing `state_transition_metrics.csv` across all 50 episodes reveals why Controller C degraded:

```
========================================================================================
CONTROLLER C STATE OCCUPANCY & TRANSITION METRICS (N = 50 episodes, 500 steps max)
========================================================================================
State Name                 Mean Steps / Episode   Percentage   Design Intent vs Reality
----------------------------------------------------------------------------------------
NOMINAL (Cruise)           98.8 steps             19.8%        Expected > 50%; actual < 20%
AVOID_LEFT (Latched)      210.2 steps             42.0%        Trapped by exit condition
AVOID_RIGHT (Latched)     140.0 steps             28.0%        Trapped by exit condition
----------------------------------------------------------------------------------------
TOTAL AVOIDANCE LATCHED   350.2 steps             70.0%        SEVERE OVER-STEERING TRAP
----------------------------------------------------------------------------------------
SETTLING (Re-centering)    36.0 steps              7.2%        Truncated by preemption
Mean Settling Preemptions   5.30 per episode       —           Aborted before centering
========================================================================================
```

---

## 3. The Four Specific Failure Mechanisms of Controller C

### Mechanism 1: Latch Lock-in in Multi-Obstacle Corridors
The exit condition for `AVOID_LEFT` and `AVOID_RIGHT` was specified as:
$$\text{Exit Condition:} \quad \left(C_{\text{front}} > 1.25\text{ m} \quad \text{AND} \quad (x_{\text{drone}} - x_{\text{trigger}}) \ge 0.65\text{ m}\right) \quad \text{OR} \quad \text{step\_count} \ge 50$$
- In the canonical corridor, internal pillars are spaced approximately $2.0–2.5\text{ m}$ apart ($X \approx 3.0, 5.0, 7.0\text{ m}$).
- When the UAV clears Pillar 1 at $X = 3.3\text{ m}$, the rangefinder beam to Pillar 2 (located at $X = 5.0\text{ m}$) measures a distance of $5.0 - 3.3 = 1.7\text{ m}$.
- As the UAV advances to $X = 3.7\text{ m}$, $C_{\text{front}}$ towards Pillar 2 drops below $1.25\text{ m}$ **before** the UAV can register an open forward corridor!
- Consequently, the condition $C_{\text{front}} > 1.25\text{ m}$ is **almost never met** in a sequential multi-obstacle corridor.
- The state machine falls back to its 50-step timeout ($2.5\text{ seconds}$). During these 50 steps, the controller continues to blindly command maximum lateral velocity ($v_y = \pm 0.25\text{ m/s}$), pinning the drone against the corridor wall.

### Mechanism 2: Massive Throttling & Coupled Boundary Deadlock
- Because the UAV was locked in avoidance for **$70.0\%$ of every episode**, it flew continuously along the corridor walls ($|Y| \approx 0.60–0.75\text{ m}$).
- Inward wall distance frequently fell below the supervisor's active boundary buffer ($0.25\text{ m}$).
- Whenever a downstream obstacle appeared ahead, the supervisor clamped forward velocity to $0.10\text{ m/s}$ while applying inward boundary repulsion, resulting in constant deadlock.
- Mean throttled fraction jumped from **$20.18\%$ in Controller B to $37.15\%$ in Controller C** (+17.0% increase in throttled flight time).
- Mid-corridor progress collapsed: Pillar 2 clearance fell from **$94.0\%$ in Controller B down to $76.0\%$ in Controller C**.

### Mechanism 3: The Seed 2001 Near-Collision Envelope Violation ($0.0223\text{ m}$)
In Seed 2001, Controller C recorded an obstacle clearance of **$0.0223\text{ m}$** (penetrating the $0.080\text{ m}$ safety envelope):
- **Step 270 ($X = 4.90\text{ m}, Y = -0.63\text{ m}$):** UAV had cleared Pillar 1 and was approaching Pillar 2 at $X = 5.0\text{ m}, Y = -0.25\text{ m}$.
- **Step 275–282 ($X = 5.00–5.15\text{ m}$):** The UAV was trapped in `AVOID_LEFT` setpoint ($v_y = +0.25\text{ m/s}$) while executing a high-speed diagonal traverse across the face of Pillar 2.
- At Step 281 ($X = 5.145\text{ m}, Y = -0.55\text{ m}$), clearance to the cylindrical pillar surface dropped to **$0.0223\text{ m}$** ($2.2\text{ cm}$), triggering emergency supervisor interventions before clearing.
- This proves that state latching without continuous clearance feedback can force the UAV into dangerous diagonal interception vectors.

### Mechanism 4: Re-Centering Preemption Churn
- The `SETTLING` state averaged only **$36.0\text{ steps}$ ($1.8\text{ seconds}$)** per episode across an average of **5.3 preemptions**.
- Re-centering was aborted almost as soon as it began because the next obstacle's forward rays triggered $C_{\text{front}} < 0.85\text{ m}$.
- Rather than smoothly returning the UAV to the centerline, `SETTLING` introduced chattering transitions that disrupted the forward velocity profile.

---

## 4. Full Breakdown of the 22 Discordant Episodes (B Wins vs C Losses)

In 22 out of 50 seeds, Controller B (Stateless) achieved full goal completion while Controller C (Persistent) failed:

| Seed | Controller B Outcome | Controller C Outcome | Controller C Progress | Failure Stage | C Standoff Duration | C Throttled Fraction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2002** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.074\text{ m}$ | Pillar 2 | $5.25\text{ s}$ | $39.4\%$ |
| **2003** | `SUCCESS` ($8.66\text{ m}$) | `TIMEOUT_SLOW_TRANSIT` | $8.334\text{ m}$ | Terminal | $0.45\text{ s}$ | $34.0\%$ |
| **2007** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.132\text{ m}$ | Pillar 3 | $3.85\text{ s}$ | $38.6\%$ |
| **2008** | `SUCCESS` ($8.66\text{ m}$) | `TIMEOUT_SLOW_TRANSIT` | $7.914\text{ m}$ | Pillar 3 | $0.50\text{ s}$ | $35.2\%$ |
| **2009** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.093\text{ m}$ | Pillar 3 | $3.80\text{ s}$ | $44.4\%$ |
| **2011** | `SUCCESS` ($8.65\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.165\text{ m}$ | Pillar 2 | $8.05\text{ s}$ | $58.6\%$ |
| **2012** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.077\text{ m}$ | Pillar 2 | $6.45\text{ s}$ | $53.0\%$ |
| **2013** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $7.185\text{ m}$ | Pillar 3 | $4.35\text{ s}$ | $45.8\%$ |
| **2014** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.191\text{ m}$ | Pillar 2 | $5.60\text{ s}$ | $46.8\%$ |
| **2016** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.204\text{ m}$ | Pillar 3 | $3.65\text{ s}$ | $37.2\%$ |
| **2019** | `SUCCESS` ($8.66\text{ m}$) | `TIMEOUT_SLOW_TRANSIT` | $8.335\text{ m}$ | Terminal | $1.10\text{ s}$ | $27.4\%$ |
| **2023** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.341\text{ m}$ | Pillar 2 | $8.55\text{ s}$ | $74.2\%$ |
| **2024** | `SUCCESS` ($8.66\text{ m}$) | `TIMEOUT_SLOW_TRANSIT` | $8.411\text{ m}$ | Terminal | $3.90\text{ s}$ | $37.4\%$ |
| **2026** | `SUCCESS` ($8.65\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.201\text{ m}$ | Pillar 3 | $3.20\text{ s}$ | $39.6\%$ |
| **2028** | `SUCCESS` ($8.66\text{ m}$) | `TIMEOUT_SLOW_TRANSIT` | $8.330\text{ m}$ | Terminal | $1.35\text{ s}$ | $41.0\%$ |
| **2031** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.149\text{ m}$ | Pillar 3 | $2.35\text{ s}$ | $42.6\%$ |
| **2034** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.303\text{ m}$ | Pillar 3 | $3.95\text{ s}$ | $45.0\%$ |
| **2039** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.079\text{ m}$ | Pillar 2 | $6.10\text{ s}$ | $47.4\%$ |
| **2043** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.327\text{ m}$ | Pillar 3 | $3.70\text{ s}$ | $39.8\%$ |
| **2045** | `SUCCESS` ($8.65\text{ m}$) | `TIMEOUT_SLOW_TRANSIT` | $8.062\text{ m}$ | Terminal | $1.10\text{ s}$ | $25.8\%$ |
| **2047** | `SUCCESS` ($8.65\text{ m}$) | `PROGRESS_STALL_PILLAR_3` | $6.262\text{ m}$ | Pillar 3 | $3.50\text{ s}$ | $39.6\%$ |
| **2048** | `SUCCESS` ($8.66\text{ m}$) | `PROGRESS_STALL_PILLAR_2` | $4.113\text{ m}$ | Pillar 2 | $5.60\text{ s}$ | $50.6\%$ |

---

## 5. Why Stateless Guidance (Controller B) Remains Definitively Preferred

Stateless Clearance Guidance (Controller B) outclasses both the baseline and persistent guidance because:
1. **Continuous Real-Time Responsiveness:** Operates at $20\text{ Hz}$ based strictly on instantaneous geometry. The moment an obstacle shifts from front to side, guidance naturally relinquishes control, enabling the baseline's nominal cruise and goal deceleration behaviors to operate unhindered.
2. **Absence of Memory Traps:** Never gets locked into an obsolete directional commitment. If the UAV encounters alternating obstacles, it adapts instantly to the new opening without waiting for arbitrary timeout clocks.
3. **Flawless Safety Profile:** Across 50 diverse seeds, Controller B recorded zero collisions, zero envelope violations, and a safe minimum obstacle clearance of $0.0870\text{ m}$.
4. **Definitive Benchmark Superiority:** Achieved a **$70.0\%$ success rate** (+22.0% over baseline, +36.0% over persistent guidance), with $p = 0.0433$ significance.

Therefore, **Conclusion C: STATELESS GUIDANCE REMAINS PREFERRED** is the sole defensible scientific determination.

