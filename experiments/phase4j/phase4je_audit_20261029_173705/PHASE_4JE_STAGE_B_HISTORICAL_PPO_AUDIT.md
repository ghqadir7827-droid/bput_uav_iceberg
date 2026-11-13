# Phase 4J-E — Stage B: Historical PPO Results Audit

**Audit Date:** 2026-10-29  
**Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Milestone:** Phase 4J-E Historical PPO Audit & Controlled BC vs PPO Comparison  
**Lead Auditor:** Independent Simulation-Validation & Experiment Auditor  

---

## 1. Executive Summary: The PPO Performance Trajectory

A central paradox in the project's development history is that **early PPO evaluations reported high success rates (50% to 72%)**, whereas **subsequent benchmark evaluations reported near-zero success (0% to 1%)**.

This audit resolves that paradox by examining the underlying source code, checkpoint metadata, and historical evaluation logs across Milestones 4 through Phase 4J:

1. **The Milestone 4 Success Artifact (October 24, 2026):**
   - The reported $72.0\%$ success in `ppo_benchmark_10m_25seeds.json` and $70.0\%$ in `ppo_policy_best.pt` occurred under a **primitive supervisor lacking caution-zone velocity throttling**.
   - Under that configuration, PPO was allowed to fly forward at maximum speed ($0.60\text{ m/s}$) until within $0.30\text{ m}$ of obstacles.
   - **Crucially, PPO suffered a $28.0\%$ to $50.0\%$ obstacle crash rate in Milestone 4.** When it did not collide, it traversed the corridor in only $18.0\text{ s}$ ($360\text{ steps}$), reaching the goal before the 500-step timeout could trigger.
2. **The Phase 4B/4C Safety Lockdown Bottleneck (October 26–27, 2026):**
   - To eliminate high-speed obstacle collisions, a multi-directional safety supervisor with caution-zone velocity throttling ($d_{\text{caut}} = 0.80\text{ m}$) was introduced.
   - In Phase 4C (100 held-out seeds `50000..50099`), collisions dropped to **0.0%**, but **timeouts exploded to 99.0%** (1.0% success, mean progress $4.35\text{ m}$).
   - The combination of PPO's systematic rightward lateral bias ($v_y = -0.23\text{ m/s}$) and prolonged throttling to $0.10\text{ m/s}$ trapped the UAV in safe hovering deadlocks in front of obstacles.
3. **The Phase 4G 200-Step Horizon Flaw (October 29, 2026):**
   - In Phase 4G, attempts were made to train reflection-augmented PPO to eliminate the lateral bias.
   - However, the training and evaluation scripts (`train_phase4g.py`, `scripts/run_phase4g_validation.py`, `scripts/run_phase4g_heldout.py`) hardcoded a **200-step (10.0 s) horizon** instead of the canonical 500-step (25.0 s) horizon.
   - At max speed ($0.60\text{ m/s}$), maximum theoretical displacement in 200 steps is $6.0\text{ m}$. Reaching the $9.0\text{ m}$ distant goal was **mathematically impossible**, guaranteeing a 0.0% reported success rate regardless of policy quality.
4. **Phase 4H & Phase 4J Caution-Distance Recovery:**
   - In Phase 4H (restored to 500 steps, $d_{\text{caut}} = 0.80\text{ m}$), PPO baseline achieved 0/100 (99% timeouts, 0% collisions), and Phase 4H PPO candidate achieved 0/100 (99% timeouts, 0% collisions).
   - In Phase 4J-B, reducing $d_{\text{caut}}$ to $0.50\text{ m}$ improved PPO baseline success from 0% to **12.0% (6/50)** with 0% collisions, while the BC baseline surged from 1% to **48.0% (24/50)** with 0% collisions. However, the Phase 4H PPO candidate suffered **26.0% wall collisions** at $d_{\text{caut}} = 0.50\text{ m}$.

---

## 2. Checkpoint Architecture and Compatibility Audit

Four primary checkpoints were audited at the tensor level:

| Property | `bc_policy_best.pt` | `ppo_policy_best.pt` | `bc_phase4h_candidate.pt` | `ppo_phase4h_candidate.pt` |
|:---|:---|:---|:---|:---|
| **Location** | `models/checkpoints/` | `models/checkpoints/` | `experiments/phase4h/checkpoints/` | `experiments/phase4h/checkpoints/` |
| **SHA-256 Hash** | `93c4d7d1d847...` | `33f6d8b705ed...` | `1fdb2286170b...` | `751431cda829...` |
| **Policy Network** | `NavigationPolicyMLP` | `NavigationPolicyMLP` | `NavigationPolicyMLP` | `NavigationPolicyMLP` |
| **Parameter Count** | 28,740 parameters | 28,740 parameters | 28,740 parameters | 28,740 parameters |
| **Actor Weights Present** | Yes (`model_state_dict`, 16 keys) | Yes (`model_state_dict`, 16 keys) | Yes (`model_state_dict`, 16 keys) | Yes (`model_state_dict`, 16 keys) |
| **Critic Weights Present**| No | Yes (`critic_state_dict`, 14 keys) | No | Yes (`critic_state_dict`, 14 keys) |
| **`log_std` Parameter Saved** | N/A | **NO (Missing)** | N/A | **NO (Missing)** |
| **Training Resume State** | Partial (Optimizer state saved) | Incomplete (No optimizer, no `log_std`) | Partial (Optimizer state saved) | Incomplete (No optimizer, no `log_std`) |
| **Inference Determinism** | Exact deterministic mean | Exact deterministic mean | Exact deterministic mean | Exact deterministic mean |
| **Stochastic Inference** | Not supported | Requires fallback `init_log_std = -1.2` | Not supported | Requires fallback `init_log_std = -1.2` |
| **Observation Contract** | 23 Deployable features | 23 Deployable features | 23 Deployable features | 23 Deployable features |
| **Action Contract** | 4 Continuous actions | 4 Continuous actions | 4 Continuous actions | 4 Continuous actions |

### Critical Finding on Checkpoint Integrity:
- Both `ppo_policy_best.pt` and `ppo_phase4h_candidate.pt` save the actor's weights as `model_state_dict` using the exact `NavigationPolicyMLP` architecture.
- Neither checkpoint contains `log_std`. Consequently, neither checkpoint can resume stochastic PPO training without re-initializing exploration noise.
- **For deterministic evaluation, both PPO checkpoints are completely compatible with `NavigationPolicyMLP`.** The deterministic mean action pass is identical between `ActorCriticPPO.get_action_and_value(..., deterministic=True)` and `NavigationPolicyMLP.forward(..., bounded=True)`.

---

## 3. Chronological Audit of Historical PPO Evaluations

The table below synthesizes all historical PPO evaluations across the codebase, identifying the exact experimental conditions and explaining the observed results:

| Milestone / Experiment | Checkpoint Evaluated | Seeds Used | Horizon (Steps / s) | Supervisor Active | Caution Distance ($d_{\text{caut}}$) | Success Rate | Collision Rate | Timeout Rate | Root Cause / Comparability |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Milestone 4 Validation** (`ppo_benchmark_10m.json`) | `ppo_policy_best.pt` | `100..109` (10 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Primitive | **None** ($0.30\text{ m}$ hard brake only) | **50.0%** (5/10) | **50.0%** (5/10) | 0.0% | **High speed, high crashes.** No caution throttling. Traversed corridor in $17.9\text{ s}$ when avoiding collision. |
| **Milestone 4 Benchmark** (`ppo_benchmark_10m_25seeds.json`) | `ppo_policy_best.pt` | `200..224` (25 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Primitive | **None** ($0.30\text{ m}$ hard brake only) | **72.0%** (18/25) | **28.0%** (7/25) | 0.0% | **Primary source of "promising" 72% claim.** High crash rate tolerated; fast transit when clear. |
| **Phase 4B Validation** (`phase4b_candidate_validation_metrics.json`) | `ppo_phase4b_candidate.pt` | `100..109` (10 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **0.0%** (0/10) | **0.0%** (0/10) | 100.0% | Multi-directional caution throttling introduced. Collisions eliminated; timeouts exploded. |
| **Phase 4C Held-Out** (`phase4c_heldout_eval_ppo_baseline.json`) | `ppo_policy_best.pt` | `50000..50099` (100 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **1.0%** (1/100) | **0.0%** (0/100) | 99.0% | $93.1\%$ supervisor intervention rate. Mean progress $4.35\text{ m}$. PPO trapped in Pillar 1 & 2 caution zones. |
| **Phase 4F-1 Bias Probe** (`phase4f1_policy_probe_metrics.json`) | `ppo_policy_best.pt` vs `bc_policy_best.pt` | Open space & mirrored obstacles | Single-step & 200 steps | Disabled / Active | N/A | N/A | N/A | N/A | **Rightward bias confirmed:** PPO commanded $v_y = -0.2267\text{ m/s}$ in open space. Chose narrow right-side gap in 100% of tests. BC remained centered ($v_y = -0.0005\text{ m/s}$). |
| **Phase 4G Validation** (`phase4g_validation_metrics.json`) | `ppo_arm_d_reflection.pt` | 10 seeds | **$200\text{ steps} / 10.0\text{ s}$** | Canonical | $0.80\text{ m}$ | **0.0%** (0/10) | 0.0% | 100.0% | **Flawed 200-step horizon.** Goal displacement is $9.0\text{ m}$. Max travel distance at $0.60\text{ m/s}$ is $6.0\text{ m}$. 100% timeout guaranteed. |
| **Phase 4G Held-Out** (`phase4g_heldout_metrics.json`) | `ppo_arm_d_reflection.pt` | `50000..50099` (100 seeds) | **$200\text{ steps} / 10.0\text{ s}$** | Canonical | $0.80\text{ m}$ | **0.0%** (0/100) | 0.0% | 100.0% | Same 200-step horizon flaw. |
| **Phase 4H Held-Out** (`phase4h_comparative_metrics.json`) | `ppo_policy_best.pt` | `90000..90099` (100 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **0.0%** (0/100) | 0.0% | 99.0% | Restored to 500 steps. Mean progress $4.44\text{ m}$. BC baseline was 1/100 (1%). |
| **Phase 4H Held-Out** (`phase4h_comparative_metrics.json`) | `ppo_phase4h_candidate.pt` | `90000..90099` (100 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Canonical | $0.80\text{ m}$ | **0.0%** (0/100) | 0.0% | 99.0% | Mean progress $4.88\text{ m}$. Bias reduced, but still timed out under $0.80\text{ m}$ caution zone. |
| **Phase 4J-B Robustness** (`phase4jb_summary.json`) | `ppo_policy_best.pt` | `2000..2049` (50 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Isolated Opt | **$0.50\text{ m}$** | **12.0%** (6/50) | 0.0% | 88.0% | Under reduced throttling ($0.50\text{ m}$), PPO baseline success increased to 12.0% (vs BC 48.0%). Zero collisions. |
| **Phase 4J-B Robustness** (`phase4jb_summary.json`) | `ppo_phase4h_candidate.pt` | `2000..2049` (50 seeds) | $500\text{ steps} / 25.0\text{ s}$ | Isolated Opt | **$0.50\text{ m}$** | **0.0%** (0/50) | 0.0% | 74.0% | **26.0% Wall Collisions!** Phase 4H candidate overcompensated laterally, striking corridor boundaries when throttling was reduced. |

---

## 4. Synthesis: Verified Facts vs. Plausible Explanations vs. Unresolved Questions

### 4.1 Verified Historical Facts:
1. **The 72% success claim in Milestone 4 came with a 28% collision rate:** It was produced before the caution-zone velocity throttling was implemented. The policy did not possess superior obstacle circumvention; it flew aggressively forward at full cruise until hard-braking at $0.30\text{ m}$.
2. **The 0% success in Phase 4G was an invalid evaluation artifact:** `train_phase4g.py` and evaluation scripts enforced a 200-step (10.0 s) cutoff, making goal arrival physically impossible.
3. **The 0%–1% success in Phase 4C and Phase 4H was driven by time-budget starvation:** The canonical supervisor's $0.80\text{ m}$ caution zone throttled both BC and PPO forward velocity for $>70\%$ of the episode duration, exceeding the 25.0 s horizon.
4. **PPO Baseline has a confirmed rightward lateral bias ($v_y \approx -0.23\text{ m/s}$):** Confirmed independently in Phase 4F-0, Phase 4F-1, and Phase 4G audits.
5. **Phase 4H PPO Candidate suffers catastrophic boundary collisions when throttling is relaxed:** Recorded 26.0% wall collisions at $d_{\text{caut}} = 0.50\text{ m}$ and 34.0% wall collisions at $d_{\text{caut}} = 0.45\text{ m}$.

### 4.2 Plausible Explanations:
- PPO's reinforcement learning objective (which penalized distance to goal and time) rewarded high forward velocity over conservative lateral clearances. When constrained by the supervisor's caution zone, the policy could not adapt its forward intent, leading to continuous saturation of forward commands ($v_x = 0.60\text{ m/s}$) while the supervisor clamped speed to $0.10\text{ m/s}$.
- The Behavior Cloning baseline, having been trained on expert demonstrations with smooth lateral bypasses and centered flight, maintains better equilibrium in the corridor, allowing it to capitalize on the $0.50\text{ m}$ caution distance far more effectively than PPO.

### 4.3 Unresolved Questions Addressed in Stage D:
- Under identical evaluation conditions at $d_{\text{caut}} = 0.50\text{ m}$ on development seeds `2000..2049`:
  - Exactly how does PPO Baseline perform head-to-head against BC Baseline?
  - Does PPO Baseline ever succeed on seeds where BC Baseline fails?
  - What are the precise action-level differences when negotiating centered pillars?

**Conclusion:** The historical audit is complete. Direct comparisons between Milestone 4, Phase 4G, and Phase 4J cannot be made without accounting for horizon and supervisor differences. Stage C & D will establish the first rigorous, matched-condition evaluation.

