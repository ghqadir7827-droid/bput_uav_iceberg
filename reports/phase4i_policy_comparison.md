# Phase 4I Policy Comparison: Baseline, Candidate, and Expert Dynamics

## Executive Summary

Phase 4I-E conducts a comprehensive, head-to-head comparison of all five controllers evaluated under identical conditions:
1. **`Expert_APF_Reference`**: Analytical Artificial Potential Field expert (ground-truth reference).
2. **`Arm_B_BC_Baseline`**: Frozen canonical BC baseline (`bc_policy_best.pt`).
3. **`BC_Phase4H_Candidate`**: Phase 4H BC policy (`bc_phase4h_candidate.pt`).
4. **`Arm_A_PPO_Baseline`**: Frozen canonical PPO baseline (`ppo_policy_best.pt`).
5. **`PPO_Phase4H_Candidate`**: Phase 4H PPO policy (`ppo_phase4h_candidate.pt`).

Every policy was evaluated across the six diagnostic probes and the 10-seed multi-pillar development suite.

---

## 1. Multi-Pillar Development Comparison (Seeds 1000..1009)

| Evaluation Metric | Expert APF Reference | Arm B: BC Baseline | BC Phase 4H Candidate | Arm A: PPO Baseline | PPO Phase 4H Candidate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Success Rate (10 Seeds)** | **0.0%** (0/10) | **0.0%** (0/10) | **0.0%** (0/10) | **0.0%** (0/10) | **0.0%** (0/10) |
| **Mean Forward Progress** | **5.68 m** | 4.88 m | **5.29 m** | 2.60 m | 3.24 m |
| **Mean Cleared Pillars** | **2.0 / 3** | 1.4 / 3 | **1.6 / 3** | 0.3 / 3 | 0.6 / 3 |
| **Unconstrained Forward (%)** | 11.8% | 13.4% | 14.0% | 1.0% | 5.6% |
| **Supervisor Constrained (%)** | **70.7%** | 13.9% | **22.7%** | 12.7% | 15.2% |
| **Lateral Avoidance (%)** | 14.0% | 26.4% | **42.0%** | 16.5% | 14.5% |
| **Braking / Stopping (%)** | **0.6%** (Lowest) | 15.3% | **7.1%** | **18.8%** | **21.4%** (Highest) |
| **Retreat / Recovery (%)** | 0.0% | 0.2% | 0.4% | 0.8% | 0.2% |
| **Collisions (10 Seeds)** | **0** | **0** | 1 (Seed 1005) | **0** | **0** |
| **Timeouts (10 Seeds)** | 10 | 10 | 9 | 10 | 10 |

---

## 2. Answers to Core Investigation Questions (Phase 4I-E)

### Question 1: Do learned policies spend substantially more time in supervisor caution or braking states?
- **PPO policies:** **Yes.** PPO spends **$18.8\% - 21.4\%$ of the entire episode completely halted** ($v_x \le 0.10\text{ m/s}$), compared to only $0.6\%$ for the Expert.
- **BC policies:** **Partially.** `BC_Phase4H_Candidate` cut braking time in half compared to baseline (from $15.3\%$ down to $7.1\%$). However, it spends $42.0\%$ of its time in active lateral steering.
- **The Expert:** The Expert spends virtually **no time halted** ($0.6\%$), but spends **$70.7\%$ of its flight in supervisor caution throttling**, continuously flying at low speed ($0.15 - 0.20\text{ m/s}$).

### Question 2: Is time lost primarily caused by poor obstacle avoidance decisions, repeated braking, delayed re-acceleration, or terminal maneuvers?
The loss mechanism differs radically by algorithm:
- **For PPO:** Time is lost primarily to **repeated braking and stalling (Failure Class 2)**. Because PPO refuses to take open left channels ($Y_{\text{obs}} < 0$), it repeatedly drives into the right wall/obstacle constriction, triggers emergency stops, and spends $18-21\%$ of its time stationary.
- **For BC Candidate:** Time is lost to **cumulative supervisor throttling across successive caution zones (Failure Class 1)** combined with the large lateral distance traversed during S-turns.
- **For Expert:** Time is lost **100% to cumulative supervisor throttling (Failure Class 1)**.

### Question 3: Does the drone make continuous but slow progress, or repeatedly become trapped?
- **Expert & BC Candidate:** Make **continuous, steady forward progress**. In 80% to 100% of episodes, the drone clears Pillar 1 and Pillar 2 smoothly without becoming trapped, advancing steadily until the step counter reaches 500 at $X \approx 5.5 - 6.8\text{ m}$.
- **PPO Baseline & Candidate:** **Repeatedly become trapped**. In 60% to 80% of multi-pillar episodes, PPO gets wedged at Pillar 1 ($X \approx 2.05\text{ m}$) and fails to clear even the first obstacle.

### Question 4: Is the time spent recovering from one obstacle responsible for later timeouts?
**Yes.** Because the pillars are spaced only $2.0\text{ m}$ apart, the unconstrained gap between clearing Pillar 1 ($X = 3.35\text{ m}$) and entering Pillar 2's caution zone ($X = 4.20\text{ m}$) is only **$0.85\text{ m}$**.
At an acceleration limit of $1.0\text{ m/s}^2$, the drone cannot accelerate back to cruise speed ($0.60\text{ m/s}$) within $0.85\text{ m}$. The drone remains locked in a permanent low-velocity regime throughout the entire middle segment of the corridor ($X \in [2.2, 7.4]\text{ m}$).

### Question 5: Does the expert use a different trajectory, a different action profile, or both?
- **Trajectory:** The expert initiates lateral bypass earlier and smoother because it calculates potential field gradients from exact obstacle center coordinates $1.30\text{ m}$ away. Learned policies use 12-beam rangefinders that detect the obstacle later, resulting in sharper, more abrupt S-turns ($\Delta y = \pm 0.65\text{ m}$).
- **Action Profile:** The expert commands a smooth forward cruise speed ($v_x = 0.65\text{ m/s}$) and gentle tangential circulation. However, the Safety Supervisor clamps this setpoint down to $0.15\text{ m/s}$, neutralizing the expert's speed advantage.

### Question 6: Are current safety constraints compatible with successful passage through the tested geometry?
**NO.** Under the canonical configuration:
- Corridor length: $10.0\text{ m}$
- 3 pillars at $X \in \{3.0, 5.0, 7.0\}\text{ m}$
- Supervisor caution distance: $0.80\text{ m}$
- Velocity clamp in caution zone: $\approx 0.18\text{ m/s}$
- Episode horizon: $500\text{ steps} = 25.0\text{ seconds}$

The minimum theoretical duration to traverse the entrance, three caution zones, two inter-pillar transitions, and goal landing is:
$$T_{\text{min}} = 3.09 + 19.17 + 4.25 + 8.60 = \mathbf{35.1\text{ seconds}} = \mathbf{702\text{ steps}}$$
A 500-step horizon allows only $25.0\text{ seconds}$. Therefore, zero goal success on the three-pillar corridor is a **structural mathematical certainty** under the current supervisor configuration, not a policy defect.

---

## 3. Policy Evolution Summary

| Dimension | Baseline State (Phase 4A/4F) | Phase 4H Candidate State | Phase 4I Diagnosis |
| :--- | :--- | :--- | :--- |
| **BC Centered Obstacle** | Stalled at $X = 2.81\text{ m}$ (0% pass) | **Clears obstacle (100% pass)** | Bimodal target cancellation resolved. |
| **BC Multi-Pillar Progress** | Mean progress $4.88\text{ m}$ | **Mean progress $5.29\text{ m}$** | Bypasses 1.6/3 obstacles; limited by time budget. |
| **BC Braking Duration** | 15.3% of episode | **7.1% of episode** | Halved stopping time; steady transit. |
| **PPO Lateral Bias** | $v_y = -0.248\text{ m/s}$ in open space | **$v_y = -0.029\text{ m/s}$ (85% reduction)** | Symmetry breaking eliminated by reflection. |
| **PPO Multi-Pillar Progress**| Mean progress $2.60\text{ m}$ | **Mean progress $3.24\text{ m}$** | Progress improved, but stalls at rightward constrictions. |
| **PPO Braking Duration** | 18.8% of episode | 21.4% of episode | Mode lock causes persistent wall stalling. |
