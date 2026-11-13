# Phase 4J-C — Comprehensive Failure Mode Analysis

**Evaluation Configuration:** `Arm_B_BC_Baseline` (`models/checkpoints/bc_policy_best.pt`), $d_{\text{caut}} = 0.50\text{ m}$, 50 Paired Seeds (`2000..2049`)  
**Total Episodes:** 50  
**Successful Episodes:** 24 (48.0%)  
**Unsuccessful Episodes:** 26 (52.0%)  
**Collisions:** 0 (0.0%)  
**Safety Envelope Violations (<8cm):** 0 (0.0%)

---

## 1. Executive Summary & Failure Categorization

In the controlled reproduction across 50 paired development seeds, **100% of unsuccessful episodes were timeouts** caused by local progress stalls in front of one of the three corridor pillars.

There were **zero high-speed collisions, zero boundary crashes, and zero safety envelope violations** ($d_{\text{obs}} \ge 0.107\text{ m}$ across all 25,000 executed steps). The safety supervisor successfully arrested forward velocity and prevented physical contact in every single scenario. However, because the BC baseline policy lacked assertive lateral steering authority in specific obstacle geometries, it entered limit cycles of forward-surge $\to$ emergency brake $\to$ retreat $\to$ forward-surge, consuming the remaining episode time until the canonical 500-step ($25.0\text{ s}$) timeout.

---

## 2. Failure Mode Taxonomy Table

| Failure Category | Episodes Affected | % of Failed (26) | % of Total (50) | Representative Seeds | Primary Evidence / Telemetry Signatures | Diagnosis Status |
|:---|:---:|:---:|:---:|:---|:---|:---:|
| **Progress Stall at Pillar 1 ($X \approx 2.7\text{–}3.0\text{ m}$)** | 6 | 23.1% | 12.0% | `2004, 2015, 2018, 2024, 2032, 2039` | Final $X \in [2.72, 2.85]\text{ m}$; 52 to 128 lateral reversals; throttled $35\text{–}43\%$; min clearance $0.113\text{–}0.126\text{ m}$. | **CONFIRMED** |
| **Progress Stall at Pillar 2 ($X \approx 4.6\text{–}5.0\text{ m}$)** | 10 | 38.5% | 20.0% | `2009, 2010, 2020, 2025, 2026, 2030, 2031, 2033, 2034, 2047` | Cleared Pillar 1 successfully; final $X \in [4.60, 4.82]\text{ m}$; 20 to 125 lateral reversals; throttled $32\text{–}40\%$; min clearance $0.110\text{–}0.126\text{ m}$. | **CONFIRMED** |
| **Progress Stall at Pillar 3 ($X \approx 6.6\text{–}7.0\text{ m}$)** | 10 | 38.5% | 20.0% | `2001, 2011, 2013, 2017, 2021, 2022, 2023, 2036, 2045, 2048` | Cleared Pillars 1 & 2; final $X \in [6.58, 6.85]\text{ m}$; 16 to 69 lateral reversals; throttled $26\text{–}37\%$; min clearance $0.108\text{–}0.123\text{ m}$. | **CONFIRMED** |
| **Obstacle Collision** | 0 | 0.0% | 0.0% | None | Zero instances of $d_{\text{obs}} \le 0.000\text{ m}$. | **CONFIRMED (Zero)** |
| **Boundary Collision** | 0 | 0.0% | 0.0% | None | Zero instances of $d_b \le 0.000\text{ m}$. | **CONFIRMED (Zero)** |
| **Goal Overshoot / Terminal Velocity Failure** | 0 | 0.0% | 0.0% | None | In all 24 episodes entering the goal sphere, terminal velocity was $\le 0.25\text{ m/s}$ (well below $0.35\text{ m/s}$ limit). | **CONFIRMED (Zero)** |
| **Safety-Monitoring Failure** | 0 | 0.0% | 0.0% | None | No undetected intrusions below the $0.080\text{ m}$ envelope. | **CONFIRMED (Zero)** |

---

## 3. Detailed Trajectory Case Studies

### 3.1 Case Study A: Pillar 1 Deadlock (Seed 2004)
- **Obstacle Layout:**
  - Pillar 0 (Pillar 1): Center $(3.000, 0.231, 1.000)\text{ m}$
  - Pillar 1 (Pillar 2): Center $(5.000, -0.187, 1.000)\text{ m}$
  - Pillar 2 (Pillar 3): Center $(7.000, 0.054, 1.000)\text{ m}$
- **Trajectory Progression:**
  - Steps 0–75: Drone accelerates smoothly along corridor centerline from $X = 0.50\text{ m}$ to $X = 2.45\text{ m}$, reaching $v_x = 0.58\text{ m/s}$.
  - Step 80: Forward sensor detects Pillar 0 at distance $0.30\text{ m}$. The supervisor triggers `Front obstacle detected -> Forward brake`, clamping $v_x = 0.0\text{ m/s}$.
  - Steps 85–120: The BC policy continues commanding maximum forward drive ($v_x^{\text{raw}} = 0.68\text{ m/s}$) with near-zero lateral steering ($v_y^{\text{raw}} \in [-0.04, +0.02]\text{ m/s}$).
  - Step 124: With forward progress blocked, the drone triggers stalled recovery reverse retreat ($v_x = -0.10\text{ m/s}, v_y = +0.15\text{ m/s}$), backing the vehicle up to $X = 2.61\text{ m}$.
  - Steps 128–500: As soon as the retreat clears the caution envelope, the policy surges forward again directly into the pillar ($v_x^{\text{raw}} = 0.72\text{ m/s}, v_y^{\text{raw}} \approx 0.0\text{ m/s}$), immediately re-triggering the brake. The drone oscillates between $X = 2.61\text{ m}$ and $X = 2.75\text{ m}$ with 123 lateral reversals until step 500 expires.
- **Root Cause:** Policy fails to commit to a lateral bypass trajectory when approaching an obstacle with a small lateral offset ($Y_{\text{obs}} = +0.23\text{ m}$).

### 3.2 Case Study B: Successful Multi-Pillar Traversal (Seed 2008)
- **Obstacle Layout:**
  - Pillar 0: Center $(3.000, -0.342, 1.000)\text{ m}$
  - Pillar 1: Center $(5.000, 0.315, 1.000)\text{ m}$
  - Pillar 2: Center $(7.000, -0.288, 1.000)\text{ m}$
- **Trajectory Progression:**
  - Because the pillars have large alternating offsets ($\Delta Y \approx \pm 0.30\text{–}0.35\text{ m}$), the wide corridor gap ($> 1.15\text{ m}$) is immediately salient in the rangefinder observations.
  - The policy commands sustained lateral velocities ($v_y \approx +0.35\text{ m/s}$ past Pillar 0, $-0.38\text{ m/s}$ past Pillar 1, $+0.32\text{ m/s}$ past Pillar 2).
  - Clearances: Pillar 0 min clearance $= 0.284\text{ m}$; Pillar 1 min clearance $= 0.266\text{ m}$; Pillar 2 min clearance $= 0.301\text{ m}$.
  - Arrival: Reaches the goal at step 341 ($17.05\text{ s}$), decelerating cleanly to $v_{\text{term}} = 0.182\text{ m/s}$.

---

## 4. Synthesis of Primary and Secondary Causes

1. **Primary Failure Driver:**
   - **Bimodal Policy Steering Behavior:** When an obstacle is offset by $> 0.28\text{ m}$, the policy steers around it with ease. When an obstacle is placed closer to centerline ($|Y_{\text{obs}}| \le 0.20\text{ m}$), the imitation learning policy suffers perceptual indecision (attracted equally to left and right bypass paths), resulting in near-zero lateral velocity commands ($v_y \approx 0.0\text{ m/s}$) that cause the vehicle to plow directly toward the obstacle.
2. **Secondary Mitigating Driver:**
   - **Supervisor Invariant Protection:** The supervisor's critical brake ($0.30\text{ m}$) and emergency reverse ($0.15\text{ m}$) operate with 100% reliability, preserving at least $10.7\text{ cm}$ of clearance at all times. The supervisor successfully trades mission progress for vehicle survival.

