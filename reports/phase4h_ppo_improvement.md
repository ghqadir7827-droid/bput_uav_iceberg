# Phase 4H PPO Navigation Policy Improvement Report

## Executive Summary

Phase 4H-D developed and evaluated an improved Proximal Policy Optimization (PPO) candidate (`experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt`) targeting the confirmed failure modes identified in Phase 4F and Phase 4G:
1. **Unilateral Mode Lock (Rightward Bias):** RL optimization in symmetric corridors broke symmetry, causing the original PPO baseline (`ppo_policy_best.pt`) to develop persistent rightward lateral drift ($v_y \approx -0.25\text{ m/s}$), rendering right-shifted obstacles impassable.
2. **Horizon Discrepancy & Lack of Braking:** Phase 4G trained under an artificial 200-step horizon ($10\text{ s}$), forcing the drone into aggressive forward rushing ($v_x \ge 0.50\text{ m/s}$) with insufficient braking, causing terminal speed violations ($v_{\text{term}} > 0.35\text{ m/s}$) and high-speed supervisor emergency stops.

The Phase 4H PPO candidate applies a single, rigorously justified principal intervention:
- **Canonical 500-Step Episode Horizon:** Training rollouts and episode bounds use the canonical $500\text{ steps}$ ($25.0\text{ s}$ at $dt=0.05\text{ s}$), matching the physical flight duration required to traverse the $10\text{ m}$ corridor safely.
- **Bilateral Reflection Augmentation:** 50% alternating mirrored rollouts enforce bilateral spatial symmetry across the corridor centerline, eliminating unilateral lateral mode lock.
- **Terminal Arrival Braking Shaping:** A terminal reward structure rewarding arrival speed $\le 0.35\text{ m/s}$ ($+25.0$ return) while penalizing overspeeding ($+10.0$ return).

---

## 1. PPO Training Configuration & Architecture

### 1.1 Policy & Value Architecture
- **Model:** Actor-Critic MLP (`ActorCriticPPO`)
  - **Shared Observation:** 23-float `DeployableObservation`
  - **Actor:** 3 hidden layers $(128, 128, 64)$ with LayerNorm, GELU, and bounded output head
  - **Critic:** 3 hidden layers $(128, 128, 64)$ with LayerNorm, GELU, and scalar value head
  - **Exploration:** Gaussian action distribution with state-independent log-std initialized to $-1.20$.
- **Warmstart Source:** Loaded actor and critic weights from `experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt`. Note: As mandated by safeguards, this is an actor/critic weight initialization, not an exact optimizer state resume.
- **Optimizer:** AdamW ($\text{lr} = 1\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-4}$)
- **Rollout Length:** 1,024 steps per iteration
- **Batch Size:** 64, with 5 PPO update epochs per iteration
- **PPO Clipping:** $\epsilon = 0.20$
- **Discount & GAE:** $\gamma = 0.99$, $\lambda = 0.95$

### 1.2 Strict Seed Protocol & Safeguards
- Training seeds were dynamically sampled from non-reserved ranges $[3000, 9999]$ and verified against leakage.
- **Exclusion Pools:**
  - `VALIDATION_SEEDS = [100..109]`
  - `PHASE4C_HELD_OUT = [50000..50099]`
  - `PHASE4G_HELD_OUT = [80000..80099]`
  - `PHASE4H_HELD_OUT = [90000..90099]` (strictly reserved for Phase 4H-E held-out benchmark)
- **Zero Flight Controller Interaction:** Verified purely in PyBullet simulation.
- **Immutable Canonical Artifacts:** `models/checkpoints/ppo_policy_best.pt` remains unmodified.

---

## 2. Reward Formulation

The dense shaped reward function across rollouts is formulated as:
$$r_t = r_{\text{progress}} + r_{\text{obs}} + r_{\text{wall}} + r_{\text{smooth}} + r_{\text{term}}$$
Where:
- **Progress:** $r_{\text{progress}} = 4.0 \cdot (d_{\text{goal}}(t-1) - d_{\text{goal}}(t))$
- **Obstacle Buffer Penalty:** $r_{\text{obs}} = -2.5 \cdot (0.40 - d_{\text{obs}})^2$ if $d_{\text{obs}} < 0.40\text{ m}$ else $0.0$
- **Wall Buffer Penalty:** $r_{\text{wall}} = -2.5 \cdot (0.35 - d_{\text{wall}})^2$ if $d_{\text{wall}} < 0.35\text{ m}$ else $0.0$
- **Action Smoothness:** $r_{\text{smooth}} = -0.015 \cdot \|a_t - a_{t-1}\|^2$
- **Terminal Outcome & Braking:**
  $$r_{\text{term}} = \begin{cases} +25.0 & \text{if reached goal with } v_{\text{term}} \le 0.35\text{ m/s} \\ +10.0 & \text{if reached goal with } v_{\text{term}} > 0.35\text{ m/s} \\ -20.0 & \text{if collision (obstacle or boundary)} \end{cases}$$

---

## 3. Candidate Checkpoint & Provenance
- **Candidate Checkpoint:** `experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt`
- **Training Script:** `experiments/phase4h/train_ppo_phase4h.py`
- **Training Log:** `experiments/phase4h/logs/ppo_training_log.json`
- **Verification:** Evaluated against baseline under the unified comparative protocol in Phase 4H-E.
