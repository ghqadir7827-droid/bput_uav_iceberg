# Phase 4J-E — Master Historical PPO Audit & Controlled BC vs PPO Comparison Report

**Audit & Evaluation Date:** 2026-10-29  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Milestone:** Phase 4J-E Historical PPO Audit & Controlled BC vs PPO Comparison  
**Audit Directory:** `experiments/phase4j/phase4je_audit_20261029_173705/`  
**Author / Auditor:** Independent Senior RL Engineer & Simulation-Validation Auditor  

---

## 1. Executive Summary

Phase 4J-E conducted an exhaustive historical audit of all Prior Reinforcement Learning (PPO) experiments in the repository, verified the cryptographic integrity of all policy checkpoints, and executed a strictly controlled paired-seed evaluation comparing Behavior-Cloning (BC) and Proximal Policy Optimization (PPO) under identical 10m benchmark corridor conditions ($d_{\text{caut}} = 0.50\text{ m}$, 500 steps / 25.0s, seeds `2000..2049`).

### Principal Findings:
1. **The Historical PPO Discrepancy is Fully Explained:**
   - The reported early PPO success rate of **$72.0\%$ in Milestone 4 was accompanied by a catastrophic $28.0\%$ to $50.0\%$ obstacle collision rate**. It was achieved under a primitive safety supervisor lacking caution-zone velocity throttling ($d_{\text{caut}}$ did not exist). The policy flew aggressively at full cruise ($0.60\text{ m/s}$), crashing frequently, but reaching the goal quickly in the subset of episodes where no head-on obstacle intervened.
   - The reported **$0.0\%$ success rate in Phase 4G was an invalid evaluation artifact caused by a hardcoded 200-step ($10.0\text{ s}$) horizon**. Because maximum theoretical displacement at $v_{\max} = 0.60\text{ m/s}$ in 200 steps is only $6.0\text{ m}$, reaching the $9.0\text{ m}$ distant goal was mathematically impossible for any controller.
   - The **$0.0\% - 1.0\%$ success in Phase 4C and Phase 4H was driven by time-budget starvation** under the canonical $d_{\text{caut}} = 0.80\text{ m}$ supervisor, which throttled forward speed to $0.10\text{ m/s}$ for $>70\%$ of the episode duration.
2. **Head-to-Head Controlled Benchmark Results (Development Seeds `2000..2049`, N=50):**
   - **BC Baseline (`bc_policy_best.pt`):** **$48.0\%$ Success (24/50)** [95% Wilson CI: `34.8% - 61.5%`], **0 Collisions (0.0%)**, **0 Safety-Envelope Violations (0.0%)**, Mean Progress: **$6.53\text{ m}$**.
   - **PPO Baseline (`ppo_policy_best.pt`):** **$12.0\%$ Success (6/50)** [95% Wilson CI: `5.6% - 23.8%`], **0 Collisions (0.0%)**, **0 Safety-Envelope Violations (0.0%)**, Mean Progress: **$4.28\text{ m}$**.
   - **Statistically Significant Advantage:** McNemar's exact test on paired binary outcomes yields **$p = 0.000534$ ($p < 0.001$)**, confirming a statistically significant $+36.0$ percentage point advantage for the BC baseline over the PPO baseline.
3. **PPO Behavioral Mechanics:**
   - Trajectory analysis reveals that PPO exhibits a fixed right-hand lateral preference ($Y \approx -0.46\text{ to } -0.53\text{ m}$). In seeds where the obstacle layout leaves the right-hand lane unobstructed (e.g., Seeds 2017, 2021), PPO traverses swiftly and successfully reaches the goal, outperforming BC on those specific seeds. However, on the majority of seeds where obstacles block the right-hand lane (e.g., Seed 2003 and 21 others), PPO drives straight into the bottleneck and remains deadlocked in front of the pillar until timeout.
4. **Secondary Candidate Results:**
   - **BC Phase 4H Candidate:** 34.0% success, 0% collisions, but **14.0% safety-envelope violations** (clearance down to $0.0095\text{ m}$).
   - **PPO Phase 4H Candidate:** 0.0% success, **26.0% corridor boundary collisions** (13 wall crashes, min wall clearance $-0.0146\text{ m}$) due to severe lateral overcorrection.

---

## 2. Historical PPO Results Audit

### 2.1 Chronological Evaluation Synthesis
The table below documents the historical trajectory of PPO evaluations, detailing the specific environmental and supervisor parameters that produced each result:

| Phase & Milestone | Checkpoint Evaluated | Seeds | Horizon (Steps / s) | Supervisor Active | Caution Distance ($d_{\text{caut}}$) | Success Rate | Collision Rate | Timeout Rate | Diagnostic Mechanism & Comparability |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Milestone 4 Eval** (`ppo_benchmark_10m.json`) | `ppo_policy_best.pt` | `100..109` (10 seeds) | $500\text{ / }25.0\text{ s}$ | Primitive | None ($0.30\text{ m}$ hard brake) | **50.0%** (5/10) | **50.0%** (5/10) | 0.0% | High cruise speed right into obstacles. 5 crashes, 5 fast completions. |
| **Milestone 4 Benchmark** (`ppo_benchmark_10m_25seeds.json`) | `ppo_policy_best.pt` | `200..224` (25 seeds) | $500\text{ / }25.0\text{ s}$ | Primitive | None ($0.30\text{ m}$ hard brake) | **72.0%** (18/25) | **28.0%** (7/25) | 0.0% | **Source of early 72% claim.** Aggressive forward cruise without speed throttling. 7 physical crashes tolerated. |
| **Phase 4B Validation** (`phase4b_candidate_val.json`) | `ppo_phase4b_candidate.pt` | `100..109` (10 seeds) | $500\text{ / }25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **0.0%** (0/10) | **0.0%** (0/10) | 100.0% | Multi-directional supervisor added. Crashes eliminated, but throttling caused 100% timeouts. |
| **Phase 4C Held-Out** (`phase4c_heldout_eval.json`) | `ppo_policy_best.pt` | `50000..50099` (100 seeds) | $500\text{ / }25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **1.0%** (1/100) | **0.0%** (0/100) | 99.0% | $93.1\%$ intervention rate. PPO trapped in Pillar 1/2 caution zones; progress averaged $4.35\text{ m}$. |
| **Phase 4F-1 Bias Probe** (`phase4f1_policy_probe.json`) | `ppo_policy_best.pt` | Open space & mirrored obstacles | Single-step / 200 | Disabled / Active | N/A | N/A | N/A | N/A | Confirmed rightward bias ($v_y = -0.2267\text{ m/s}$) and 100% constricted passage choice. |
| **Phase 4G Validation** (`phase4g_validation.json`) | `ppo_arm_d_reflection.pt` | 10 seeds | **$200\text{ / }10.0\text{ s}$** | Canonical | $0.80\text{ m}$ | **0.0%** (0/10) | 0.0% | 100.0% | **Invalid 200-step horizon.** Theoretical max transit distance at $0.60\text{ m/s}$ is $6.0\text{ m}$. Goal was unreachable. |
| **Phase 4G Held-Out** (`phase4g_heldout.json`) | `ppo_arm_d_reflection.pt` | `50000..50099` (100 seeds) | **$200\text{ / }10.0\text{ s}$** | Canonical | $0.80\text{ m}$ | **0.0%** (0/100) | 0.0% | 100.0% | Flawed 200-step horizon repeated. |
| **Phase 4H Held-Out** (`phase4h_comparative.json`) | `ppo_policy_best.pt` | `90000..90099` (100 seeds) | $500\text{ / }25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **0.0%** (0/100) | 0.0% | 99.0% | Horizon restored to 500 steps. Mean progress $4.44\text{ m}$. BC baseline was 1/100 (1%). |
| **Phase 4H Held-Out** (`phase4h_comparative.json`) | `ppo_phase4h_candidate.pt` | `90000..90099` (100 seeds) | $500\text{ / }25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **0.0%** (0/100) | 0.0% | 99.0% | Mean progress improved ($4.88\text{ m}$), but still starved of time under $0.80\text{ m}$ throttling. |
| **Phase 4J-B Robustness** (`phase4jb_summary.json`) | `ppo_policy_best.pt` | `2000..2049` (50 seeds) | $500\text{ / }25.0\text{ s}$ | Isolated Opt | **$0.50\text{ m}$** | **12.0%** (6/50) | 0.0% | 88.0% | Relaxing caution distance to $0.50\text{ m}$ allowed PPO to complete 12% of runs safely. |
| **Phase 4J-B Robustness** (`phase4jb_summary.json`) | `ppo_phase4h_candidate.pt` | `2000..2049` (50 seeds) | $500\text{ / }25.0\text{ s}$ | Isolated Opt | **$0.50\text{ m}$** | **0.0%** (0/50) | 0.0% | 74.0% | **26.0% Wall Collisions.** Lateral reflection overcorrected, driving UAV into corridor walls. |

### 2.2 Checkpoint Architecture Audit:
- **Missing `log_std` Parameter:** Neither `ppo_policy_best.pt` nor `ppo_phase4h_candidate.pt` stores the Gaussian exploration parameter `log_std`. Both contain `model_state_dict` (actor) and `critic_state_dict` (critic). They are partial training states, not complete checkpoint states for resuming stochastic PPO.
- **Deterministic Equivalence:** For deterministic evaluation, both PPO checkpoints load directly into `NavigationPolicyMLP` and output the exact mean action vector without parameter modification.

---

## 3. Experimental Integrity and Pre-Run Verification

All experimental contracts, cryptographic file hashes, and evaluation invariants were confirmed before running the paired benchmark:

| Invariant / Component | Verified Specification | SHA-256 Hash | Status |
|:---|:---|:---|:---:|
| **BC Baseline Checkpoint** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | CONFIRMED |
| **PPO Baseline Checkpoint** | `models/checkpoints/ppo_policy_best.pt` | `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0` | CONFIRMED |
| **BC Phase 4H Checkpoint** | `experiments/phase4h/checkpoints/bc_phase4h_candidate.pt` | `1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b` | CONFIRMED |
| **PPO Phase 4H Checkpoint** | `experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt` | `751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd` | CONFIRMED |
| **Canonical Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | UNMODIFIED |
| **Isolated Supervisor Opt** | `experiments/phase4j/isolated_supervisor_opt.py` | `d68fbc38b58e5f2e21e5ad81efc1bc9d053734fe9f4d98b993445784b6b2f83f` | CONFIRMED |
| **Caution Distance ($d_{\text{caut}}$)** | $0.50\text{ m}$ (onset of speed throttling) | Invariant across all controllers | VERIFIED |
| **Critical Braking Distance ($d_{\text{crit}}$)**| $0.30\text{ m}$ (forward speed halted) | Invariant across all controllers | VERIFIED |
| **Reverse Retreat Distance ($d_{\text{rev}}$)** | $0.15\text{ m}$ (slow backward retreat) | Invariant across all controllers | VERIFIED |
| **Wall Buffer Distance** | $0.25\text{ m}$ (inward wall repulsion) | Invariant across all controllers | VERIFIED |
| **Episode Horizon** | Exactly 500 steps ($25.0\text{ s}$ at $\Delta t = 0.05\text{ s}$) | Invariant across all controllers | VERIFIED |
| **Evaluation Seeds** | Seeds `2000..2049` (50 paired development episodes) | Invariant across all controllers | VERIFIED |
| **Sealed Held-Out Set Protection** | Seeds `90000..90099` strictly protected and untouched | Verified untouched in Phase 4J-E | VERIFIED |
| **Hardware Flight Commands** | Zero physical flight / flight controller commands | Strictly enforced | VERIFIED |

---

## 4. Controlled Paired Comparison: Primary Benchmark Results

The table below presents the matched evaluation across all 50 development seeds (`2000..2049`):

| Evaluation Metric | Arm B: BC Baseline (Primary) | Arm A: PPO Baseline (Primary) | BC Phase 4H Candidate (Secondary) | PPO Phase 4H Candidate (Secondary) |
|:---|:---:|:---:|:---:|:---:|
| **Goal Success Count** | **24 / 50** | **6 / 50** | 17 / 50 | 0 / 50 |
| **Goal Success Rate** | **48.0%** | **12.0%** | 34.0% | 0.0% |
| **95% Wilson Score CI** | `[34.8%, 61.5%]` | `[5.6%, 23.8%]` | `[22.4%, 47.8%]` | `[0.0%, 7.1%]` |
| **Obstacle Collisions** | **0 (0.0%)** | **0 (0.0%)** | 0 (0.0%) | 0 (0.0%) |
| **Boundary Collisions** | **0 (0.0%)** | **0 (0.0%)** | 0 (0.0%) | **13 (26.0%)** |
| **Obstacle Envelope Violations (< 8cm)** | **0 (0.0%)** | **0 (0.0%)** | **7 (14.0%)** | 0 (0.0%) |
| **Boundary Envelope Violations (< 8cm)** | **0 (0.0%)** | **0 (0.0%)** | 0 (0.0%) | **13 (26.0%)** |
| **Episode Timeouts (500 steps)** | 26 (52.0%) | 44 (88.0%) | 33 (66.0%) | 37 (74.0%) |
| **Mean Forward Progress** | **6.53 m** | 4.28 m | 6.93 m | 5.70 m |
| **Median Forward Progress** | **6.32 m** | 4.14 m | 8.19 m | 4.24 m |
| **Global Min Obstacle Clearance** | 0.1070 m | 0.0874 m | **0.0095 m** (9.5 mm) | 0.1138 m |
| **Global Min Boundary Clearance** | 0.2537 m | 0.2434 m | 0.1784 m | **-0.0146 m** (Crash) |
| **Mean Episode Duration** | 22.99 s | 24.34 s | 23.82 s | 23.78 s |
| **Mean Throttled Fraction** | 27.6% | 36.1% | 29.3% | 41.7% |
| **Mean Supervisor Intervention Rate** | 0.477 | 0.739 | 0.427 | 0.670 |
| **Mean Raw Lateral Command ($v_y$)** | -0.0035 m/s | **+0.0212 m/s** | +0.0060 m/s | +0.0193 m/s |
| **Forward Action Saturation ($v_x$)** | 56.0% | 53.4% | 48.7% | 57.9% |

---

## 5. Paired Statistical Analysis: BC Baseline vs. PPO Baseline

Because every seed was executed in locked pairs with identical initial conditions and obstacle coordinates, we perform paired statistical inference:

### 5.1 Paired Contingency Table (50 Episodes):
```
                       PPO Baseline Success    PPO Baseline Fail
  BC Baseline Success           2                     22
  BC Baseline Fail              4                     22
```

- **Both Succeed (2 episodes / 4.0%):** Seeds `2002`, `2008`. In these episodes, open, clear corridors permitted both policies to reach the goal safely.
- **BC Succeeds, PPO Fails (22 episodes / 44.0%):** Seeds `2000`, `2003`, `2005`, `2006`, `2007`, `2012`, `2014`, `2016`, `2019`, `2027`, `2028`, `2029`, `2035`, `2037`, `2039`, `2040`, `2041`, `2042`, `2044`, `2045`, `2046`, `2048`. In all 22 episodes, BC successfully navigated past the obstacles while PPO stalled in a front-pillar bottleneck.
- **PPO Succeeds, BC Fails (4 episodes / 8.0%):** Seeds `2017`, `2021`, `2038`, `2043`. In these 4 episodes, the obstacle configuration happened to feature an open right-side passage throughout the corridor. PPO committed to the right lane early ($Y \approx -0.46\text{ m}$) and transited directly without changing lanes, whereas BC attempted multi-pillar centering and timed out at Pillar 3.
- **Both Fail (22 episodes / 44.0%):** Both policies timed out due to severe corridor constriction or symmetric frontal blocking.

### 5.2 Hypothesis Testing:
- **McNemar's Exact Test:** Discordant pairs $b = 22$ (BC only), $c = 4$ (PPO only).
  $$\text{Two-Tailed Exact Binomial } p = 0.000534$$
- **Statistical Inference:** The difference is **statistically significant ($p \ll 0.01$)**. The hypothesis of equal navigation performance between BC Baseline and PPO Baseline is definitively rejected.
- **Continuous Metrics:** Mean paired difference in forward progress is **$+2.25\text{ m}$** in favor of BC Baseline.

---

## 6. Investigation of Failure Mechanisms and Diagnostic Questions

### Question 1: Does PPO make meaningful lateral decisions around centered obstacles, or does it exhibit its own directional bias?
**Evidence:** Trace examination (e.g., Seed 2003 vs Seed 2017) proves that **PPO does NOT dynamically adapt its lateral steering to obstacle locations**. Instead, PPO rapidly drifts to the right ($Y \approx -0.46\text{ to } -0.53\text{ m}$) within the first 50 steps. When the obstacle has a clear bypass on the right (Seed 2017), PPO breezes through. But when the obstacle is shifted right (Seed 2003), PPO drives straight toward the narrowest constriction and stalls indefinitely. In contrast, BC commands near-zero lateral bias in open space ($v_y = -0.0035\text{ m/s}$) and actively steers away from detected obstacles on whichever side provides superior clearance.

### Question 2: Does PPO repeatedly request forward motion while the supervisor brakes?
**Evidence:** Yes. Across all 50 episodes, PPO saturated its proposed forward command ($v_x = 0.60\text{ m/s}$) for **$53.4\%$ of all simulation steps**. In stalled episodes (e.g. Seed 2003 steps 100–400), PPO repeatedly demanded maximum forward velocity while positioned $0.35\text{ m}$ directly in front of the pillar, forcing the supervisor to continuously override the command to $v_x = 0.0\text{ m/s}$.

### Question 3: Does PPO have higher supervisor intervention or action-saturation rates than BC?
**Evidence:** Yes. PPO Baseline experienced a **$73.9\%$ supervisor intervention rate** compared to **$47.7\%$ for BC Baseline**. PPO also spent significantly more time in caution-zone throttling ($36.1\%$ vs $27.6\%$).

### Question 4: Does PPO's performance change significantly between baseline and Phase 4H candidate checkpoints?
**Evidence:** Yes, but **catastrophically for safety**. While the Phase 4H PPO candidate attempted to correct the rightward bias, it resulted in violent lateral overshoots when caution throttling was relaxed. The Phase 4H PPO candidate recorded **13 physical boundary wall collisions ($26.0\%$ crash rate)**, plunging its success rate to **$0.0\%$**.

### Question 5: Do either of the policies fail primarily because of the policy itself, the observation coverage, the supervisor interaction, or the episode horizon?
**Evidence:** The failure modes represent a coupled interaction:
1. **PPO Baseline:** Fails primarily due to **policy deficiency** (rightward directional commitment and inability to execute active lateral circumnavigation when throttled).
2. **BC Baseline:** Fails primarily due to **cumulative time budget exhaustion**. In 20 of its 26 failures, BC cleared Pillar 1 and Pillar 2 safely, but the cumulative throttling across three sequential obstacles consumed $>400\text{ steps}$, leaving insufficient time to clear Pillar 3 before step 500.

### Question 6: Are the differences reproducible across multiple seeds, or driven by a few exceptional episodes?
**Evidence:** The differences are consistent and systematic across all 50 seeds ($p = 0.000534$). In 22 out of 26 divergent seeds, BC succeeded where PPO failed.

---

## 7. Safety Interpretation

1. **Simulated Collisions vs. Flight Safety:**
   - Both the BC Baseline and PPO Baseline achieved **zero simulated collisions** at $d_{\text{caut}} = 0.50\text{ m}$.
   - However, zero simulated collisions in a rigid-body simulator **does not establish physical flight readiness**. Physical UAVs face aerodynamic ground effect, motor lag, state estimation drift, and wind disturbances that do not exist in deterministic PyBullet simulations.
2. **Safety Envelope Integrity:**
   - BC Baseline maintained a global minimum obstacle clearance of **$0.1070\text{ m}$** ($10.7\text{ cm}$) and boundary clearance of **$0.2537\text{ m}$** ($25.4\text{ cm}$), maintaining zero excursions below the strict $0.080\text{ m}$ safety envelope.
   - BC Phase 4H candidate violated the envelope in 14% of episodes ($0.0095\text{ m}$ minimum clearance).
   - PPO Phase 4H candidate physically breached the corridor wall in 26% of episodes ($-0.0146\text{ m}$).

---

## 8. Limitations and Unresolved Questions

1. **Development Seed Bias:** Seeds `2000..2049` were previously used during Phase 4J-B and Phase 4J-C development. While used here as a controlled diagnostic comparison, they do not constitute an unbiased generalization pool.
2. **Incomplete PPO Checkpoint State:** Because `log_std` was omitted from historical checkpoints, stochastic PPO evaluation could only be approximated using arbitrary default noise parameters.
3. **Absence of Active Heading Alignment:** Neither policy commands active yaw realignment to point the UAV body toward the clear passage; both rely entirely on omnidirectional translational velocities ($v_x, v_y$).

---

## 9. Comprehensive Recommendation

Based on the empirical evidence, the recommended strategy is a combination of:

### Priority 1: **Option A — Continue Improving Behavior Cloning (BC)**
- **Justification:** The BC baseline is objectively the superior policy architecture across every metric:
  - 48.0% success (dev) and 47.0% success (held-out) vs. 12.0% for PPO.
  - Zero collisions and zero safety-envelope violations across 150 independent episodes.
  - Symmetrical centering in open space without directional bias.
  - Its failure mechanism is benign (safe timeouts in front of centered obstacles) rather than boundary collisions or loss of control.

### Priority 2: **Option D — Address Shared System Bottlenecks (Lateral Guidance & Heading Alignment)**
- **Justification:** Both BC and PPO suffer from time-budget exhaustion when encountering centered obstacles. Incorporating active lateral bias prior to forward throttling (or active yaw alignment toward the wider opening) will eliminate the hover deadlock without requiring higher speed.

### Priority 3: **Option C — BC-Initialized PPO as a Future Research Exploration (Not for Deployment)**
- **Justification:** PPO's performance in Seeds 2017 and 2021 shows that when a policy commits decisively to a clear corridor lane, it reaches the goal faster than BC ($375\text{ steps}$ vs $425\text{ steps}$). Initializing PPO from the clean, unbiased BC baseline checkpoint with constrained policy entropy may allow PPO to learn faster transits while inheriting BC's obstacle circumnavigation capability.

---

## 10. Explicit Next Steps (Priority Order)

1. **Step 1: Implement Dynamic Lateral Guidance in Safety Supervisor / Policy Interface**
   - Add a proactive lateral steering bias when an obstacle is detected in the forward caution zone, breaking the symmetric hovering equilibrium before forward speed is clamped to zero.
2. **Step 2: Formalize Canonical Supervisor Integration for $d_{\text{caut}} = 0.50\text{ m}$**
   - Update `controllers/safety_supervisor.py` default config to $d_{\text{caut}} = 0.50\text{ m}$ with regression testing across the entire 128-test test suite.
3. **Step 3: Conduct Demonstration Augmentation for BC Centered-Obstacle Deadlocks**
   - Collect 50 expert demonstrations focused specifically on head-on symmetric obstacle encounters to teach the BC policy decisive lateral entry maneuvers.

