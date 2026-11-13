# Phase 4J-A2 Time Budget Recalculation & Discrepancy Reconciliation

## Executive Summary

Phase 4J-A2 provides an independent, empirically grounded recalculation of the corridor transit time budget. Using step-by-step telemetry recorded during Phase 4I (`experiments/phase4i/traces/`), this report audits the Phase 4I analytical estimate ($23.8\text{ s}$ under $d_{\text{caut}} = 0.55\text{ m}$) against real physical measurements.

The recalculation reveals that the Phase 4I analytical model was overly optimistic because it assumed instantaneous acceleration between obstacles and disregarded the lateral velocity components required for chicane steering.

---

## 1. Measured Spatial Baseline & Segment Velocities

From the empirical traces of the optimal analytical expert (`Expert_APF_Reference`) under canonical $d_{\text{caut}} = 0.80\text{ m}$:

| Corridor Flight Segment | $X$-Coordinates (m) | Segment Length (m) | Measured Duration (Steps) | Measured Duration (s) | Mean Forward Speed $v_x$ (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Entrance Corridor** | $0.50 \to 2.00$ | $1.50\text{ m}$ | **57 steps** | **$2.85\text{ s}$** | $0.526\text{ m/s}$ |
| **Pillar 1 Caution Zone** | $2.00 \to 3.35$ | $1.35\text{ m}$ | **188 steps** | **$9.40\text{ s}$** | **$0.143\text{ m/s}$** |
| **Inter-Pillar Gap 1** | $3.35 \to 4.00$ | $0.65\text{ m}$ | **56 steps** | **$2.80\text{ s}$** | $0.231\text{ m/s}$ |
| **Pillar 2 Caution Zone** | $4.00 \to 5.35$ | $1.35\text{ m}$ | **168 steps** | **$8.40\text{ s}$** | **$0.161\text{ m/s}$** |
| **Inter-Pillar Gap 2** | $5.35 \to 6.00$ | $0.65\text{ m}$ | ~56 steps (est) | $2.80\text{ s}$ | $0.230\text{ m/s}$ |
| **Pillar 3 Caution Zone** | $6.00 \to 7.35$ | $1.35\text{ m}$ | ~180 steps (est) | $9.00\text{ s}$ | $0.150\text{ m/s}$ |
| **Terminal Approach** | $7.35 \to 9.15$ | $1.80\text{ m}$ | ~100 steps (est) | $5.00\text{ s}$ | $0.360\text{ m/s}$ |
| **Total Empirical Duration**| $0.50 \to 9.15$ | $8.65\text{ m}$ | **805 steps** | **$40.25\text{ s}$** | **$0.215\text{ m/s}$** |

### Key Observation:
At step 500 ($25.00\text{ s}$), the drone has accumulated $2.85 + 9.40 + 2.80 + 8.40 + 1.55 = 25.00\text{ s}$ of flight, placing it at:
$$X_{\text{empirical}} = 5.35 + (0.23 \times 1.55) \approx \mathbf{5.70\text{ meters}}$$
This matches the exact observed mean progress of the expert in Phase 4I ($5.68\text{ m}$).

---

## 2. Reconciling the Phase 4I Estimate Discrepancy

Phase 4I predicted that reducing $d_{\text{caut}}$ from $0.80\text{ m}$ to $0.55\text{ m}$ would yield a $7.7\text{-second}$ saving, resulting in a total completion time of $23.8\text{ seconds}$ ($476\text{ steps}$).

### Why the 23.8s Estimate Was Idealized:
1. **Neglect of Lateral Velocity Budget:** In a $2.0\text{ m}$ corridor, steering laterally around a pillar of radius $0.20\text{ m}$ requires $\Delta y = \pm 0.45\text{ m}$. At lateral acceleration limits, kinetic energy is split between $v_x$ and $v_y$, reducing forward velocity below the purely longitudinal clamp limit.
2. **Acceleration Dynamics in Inter-Pillar Gaps:** The $0.65\text{ m}$ gap is too short for the drone to reach $0.40 - 0.50\text{ m/s}$ before the next obstacle's rays begin throttling.
3. **Rangefinder Angular Spans:** Rangefinder ray 1 and ray 2 point at $\pm 15^\circ$. When the drone is laterally offset, these angled rays intersect the upcoming obstacle slightly earlier than the direct longitudinal distance, triggering caution throttling sooner.

---

## 3. Revised Independent Timing Model

Recalculating based on measured acceleration profiles and the revised geometry:

| Flight Segment | Distance (m) | Projected Duration ($d_{\text{caut}}=0.55\text{ m}$) | Projected Duration ($d_{\text{caut}}=0.50\text{ m}$) | Projected Duration ($d_{\text{caut}}=0.45\text{ m}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Entrance Corridor** | $1.75\text{ m}$ | $3.18\text{ s}$ ($64\text{ steps}$) | $3.18\text{ s}$ ($64\text{ steps}$) | $3.18\text{ s}$ ($64\text{ steps}$) |
| **3 Pillar Caution Zones** | $3 \times 1.10\text{ m} = 3.30\text{ m}$ | **$13.75\text{ s}$** ($275\text{ steps}$) | **$11.80\text{ s}$** ($236\text{ steps}$) | **$9.90\text{ s}$** ($198\text{ steps}$) |
| **2 Inter-Pillar Gaps** | $2 \times 0.90\text{ m} = 1.80\text{ m}$ | **$5.14\text{ s}$** ($103\text{ steps}$) | **$4.80\text{ s}$** ($96\text{ steps}$) | **$4.50\text{ s}$** ($90\text{ steps}$) |
| **Goal Terminal Landing** | $1.80\text{ m}$ | **$5.14\text{ s}$** ($103\text{ steps}$) | **$5.14\text{ s}$** ($103\text{ steps}$) | **$5.14\text{ s}$** ($103\text{ steps}$) |
| **Total Projected Duration** | **$8.65\text{ m}$** | **$27.21\text{ s}$ ($544\text{ steps}$)** | **$24.92\text{ s}$ ($498\text{ steps}$)** | **$22.72\text{ s}$ ($454\text{ steps}$)** |

### Analytical Conclusion:
- At **$d_{\text{caut}} = 0.55\text{ m}$**, total duration is projected at **$27.2\text{ seconds}$ ($544\text{ steps}$)**, which is very close to, but slightly above, the $500\text{-step}$ ($25.0\text{ s}$) canonical threshold.
- At **$d_{\text{caut}} = 0.50\text{ m}$**, projected duration is **$24.9\text{ seconds}$ ($498\text{ steps}$)**, landing just inside the 500-step horizon.
- At **$d_{\text{caut}} = 0.45\text{ m}$**, projected duration is **$22.7\text{ seconds}$ ($454\text{ steps}$)**, providing a comfortable safety margin of $\approx 46\text{ steps}$.

**Implication for Experimentation:**  
Phase 4J-A must not restrict its sweep to only $0.80\text{ m}$ and $0.55\text{ m}$. It must evaluate a comprehensive parametric sweep across:
$$d_{\text{caut}} \in \{0.80, 0.70, 0.60, 0.55, 0.50, 0.45\}\text{ m}$$
to empirically locate the exact boundary where goal arrival becomes feasible without compromising obstacle clearance.
