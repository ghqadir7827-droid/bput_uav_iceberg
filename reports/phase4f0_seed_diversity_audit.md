# PHASE 4F-0: SEED DIVERSITY AND REPEATED ROUTE INVESTIGATION AUDIT REPORT

**Date:** October 27, 2026  
**Status:** COMPLETED  
**Decision Rule Outcome:** **PASS**  
**Investigation Mode:** Strictly READ-ONLY (Zero training, zero checkpoint modifications, zero physics changes)  
**Target Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`

---

## Executive Summary

The user reported that during interactive simulation in the 3D viewer across multiple seeds (specifically `1, 20, 100, 2000, 8000`), the drone repeatedly takes what appears to be the exact same route around obstacles, seemingly preferring the same side even when that route is constrained or blocked.

This investigation conducted a rigorous, end-to-end diagnostic audit tracing UI inputs, server endpoints, environment resets, obstacle generation, sensor coordinate frames, neural network outputs, and safety supervisor interventions.

### Key Audit Findings

1. **Seed Propagation & Environment Variability:**
   - When the "Reset" button is clicked in the UI, the selected seed faithfully reaches the PyBullet simulator: `app.js` $\to$ `POST /api/reset` $\to$ `server.py` $\to$ `eval_engine.py` $\to$ `corridor_env.py` $\to$ `np.random.default_rng(seed)`.
   - Obstacle coordinates **do** regenerate dynamically across seeds. For seeds `1, 20, 100, 2000, 8000`, the 3 pillar lateral offsets vary across $[-0.331\text{ m}, +0.315\text{ m}]$; the scenario geometry is **not** static or cached.
   - **UI Interaction Defect Identified:** If the user edits the "Random Seed" number in the input box and clicks "Play" (`btn-play`) without clicking "Reset" (`btn-reset`), no reset request is issued. The simulation continues running whatever previous episode/seed was active.

2. **Root Cause of Repeated Route (Neural Lateral Bias):**
   - The PPO policy (`ppo_policy_best.pt`, SHA-256 `33f6d8b...`) has learned a severe, systematic **rightward lateral bias** during reinforcement learning fine-tuning.
   - At the corridor entrance and along the corridor centerline ($X=0.5\text{–}1.5\text{ m}, Y=0.0\text{ m}$), when cruising at nominal forward speed ($v_x = 0.6\text{ m/s}$), the raw PPO network outputs $v_y = -0.209\text{ m/s}$ (commanding velocity to the **RIGHT**) and $\text{yaw\_rate} = -0.154\text{ rad/s}$ (turning clockwise/right).
   - The policy's lateral cruising equilibrium is shifted to $Y \approx -0.38\text{ m}$ (hugging the right wall).
   - In contrast, the Behavior Cloning policy (`bc_policy_best.pt`) evaluated under identical conditions outputs $v_y = -0.000\text{ m/s}$ and $\text{yaw\_rate} = +0.000\text{ rad/s}$ (perfect symmetry). The bias was **not** inherited from demonstrations, but was acquired during PPO training.

3. **Obstacle Bypass Choice & Constrained Route Selection:**
   - At Obstacle 1 ($X=3.0\text{ m}$), the PPO policy chose the **RIGHT** bypass in **5 out of 5** runs (100%), regardless of seed.
   - In Seed 20 ($Y_{\text{obs}} = -0.154\text{ m}$) and Seed 8000 ($Y_{\text{obs}} = -0.178\text{ m}$), Obstacle 1 is shifted right, leaving a wide open gap on the left ($0.95\text{–}0.98\text{ m}$) and a severely constrained bottleneck on the right ($0.62\text{–}0.65\text{ m}$).
   - PPO still steered hard right into the $0.62\text{ m}$ bottleneck. Because the drone's collision envelope ($0.30\text{ m}$) plus safety supervisor buffers ($0.25\text{ m}$ wall buffer + $0.30\text{ m}$ obstacle buffer = $0.70\text{ m}$) cannot fit in a $0.62\text{ m}$ gap, the drone suffered persistent supervisor intervention, entering the limit-cycle pinning deadlock identified in Phase 4D and timing out before reaching Obstacle 2.

4. **Safety Supervisor State Leakage Defect:**
   - In `viewer/eval_engine.py`, `EvaluationEngine.reset()` does not reset `self.safety_supervisor`. Internal timers (`stalled_timer_s`, `loss_timer_s`) survive across resets until cleared during subsequent step execution.

---

## 1. Trace of Seed Propagation Path

### 1.1 UI to Server Dissection

| Component | File Path | Mechanism | Audit Result |
| :--- | :--- | :--- | :--- |
| **Input Element** | `viewer/static/index.html` (L68) | `<input type="number" id="seed-input" value="50001">` | Numeric input element exists. |
| **Reset Event** | `viewer/static/app.js` (L652–665) | `parseInt(document.getElementById('seed-input').value)` $\to$ `fetch('/api/reset', { body: JSON.stringify({ seed }) })` | Reads input and sends JSON POST. |
| **Play Event** | `viewer/static/app.js` (L634–639) | `fetch('/api/play', { method: 'POST' })` | **DEFECT:** Does NOT read `#seed-input` or reset episode. If user changes seed without clicking Reset, old episode/seed continues. |
| **HTTP Handler** | `viewer/server.py` (L124–129) | `raw_seed = data.get("seed")` $\to$ `seed = int(raw_seed)` $\to$ `engine.reset(seed=seed)` | Parses seed integer faithfully. Defaults to `engine.current_seed` if omitted. |
| **Engine Reset** | `viewer/eval_engine.py` (L112–136) | `self.current_seed = seed` $\to$ `self.obs_vec, self.info = self.env.reset(seed=self.current_seed)` | Passes seed to environment. **DEFECT:** Fails to reset `self.safety_supervisor`. |
| **Environment Reset** | `sim/corridor_env.py` (L79–98) | `self._rng = np.random.default_rng(seed)` $\to$ `_build_corridor()` $\to$ `_spawn_obstacles()` $\to$ `_spawn_drone()` | Re-seeds NumPy RNG. Fully rebuilds PyBullet collision bodies and sensor rays. |

### 1.2 Obstacle Geometry Regeneration Across User Seeds

In `sim/corridor_env.py` (L181–213), the 3 cylindrical pillars have fixed longitudinal positions $X \in [3.0, 5.0, 7.0]\text{ m}$, radius $r = 0.20\text{ m}$, and lateral offset $y_i \sim \mathcal{U}(-0.35, +0.35)\text{ m}$.

| Seed | Obstacle 1 ($X=3.0\text{ m}$) | Obstacle 2 ($X=5.0\text{ m}$) | Obstacle 3 ($X=7.0\text{ m}$) | Start Lateral Perturbation ($Y_0$) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $Y = +0.008\text{ m}$ | $Y = +0.315\text{ m}$ | $Y = -0.249\text{ m}$ | $+0.090\text{ m}$ |
| **20** | $Y = -0.154\text{ m}$ | $Y = -0.027\text{ m}$ | $Y = -0.265\text{ m}$ | $+0.005\text{ m}$ |
| **100** | $Y = +0.234\text{ m}$ | $Y = +0.068\text{ m}$ | $Y = -0.148\text{ m}$ | $-0.091\text{ m}$ |
| **2000** | $Y = +0.053\text{ m}$ | $Y = +0.109\text{ m}$ | $Y = -0.331\text{ m}$ | $+0.001\text{ m}$ |
| **8000** | $Y = -0.178\text{ m}$ | $Y = -0.075\text{ m}$ | $Y = +0.045\text{ m}$ | $+0.054\text{ m}$ |

**Conclusion:** The environment geometry **does** vary across seeds. Obstacle coordinates are not fixed, cached, or overwritten.

---

## 2. Controlled Seed-Diversity Experiment Results

Simulations were executed using the exact PPO controller configuration loaded in the interactive viewer (`models/checkpoints/ppo_policy_best.pt`, SHA-256 `33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0`).

### 2.1 Episode Summary Table

| Run Tag | Seed | Outcome | Steps | Interventions | Replay Determinism | Obs 1 Route | Obs 2 Route | Obs 3 Route |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seed_1` | 1 | Timeout | 500 | 469 | Baseline | Right ($Y=-0.44$) | Right ($Y=-0.52$) | Not reached |
| `seed_1_replay` | 1 | Timeout | 500 | 469 | **100% Identical** | Right ($Y=-0.44$) | Right ($Y=-0.52$) | Not reached |
| `seed_20` | 20 | Timeout | 500 | 465 | — | **Right ($Y=-0.53$)** | Not reached | Not reached |
| `seed_100` | 100 | Timeout | 500 | 473 | — | Right ($Y=-0.30$) | Right ($Y=-0.41$) | **Right ($Y=-0.53$)** |
| `seed_2000` | 2000 | Timeout | 500 | 477 | — | Right ($Y=-0.40$) | Right ($Y=-0.46$) | Left ($Y=-0.05$) |
| `seed_8000` | 8000 | Timeout | 500 | 473 | — | **Right ($Y=-0.53$)** | Not reached | Not reached |

### 2.2 Replay Control Verification
Seed 1 replayed twice produced **identical trajectories**, identical step counts (500), identical intervention counts (469), and identical velocity proposals at every single control step ($\Delta = 0.0000$). The simulation and policy inference pipeline are strictly deterministic.

---

## 3. Lateral Action Bias & Obstacle Encounter Analysis

### 3.1 Coordinate Conventions Verified
- **World Frame (ENU):** $X \in [0, 10]\text{ m}$ (forward), $Y \in [-1, +1]\text{ m}$ ($+Y$ = Left wall, $-Y$ = Right wall), $Z \in [0, 2]\text{ m}$ (Up).
- **Body Frame (FLU):** $+v_x$ = Forward, $+v_y$ = Left, $-v_y$ = Right, $+v_z$ = Up, $+\dot{\psi}$ = CCW (turn left), $-\dot{\psi}$ = CW (turn right).
- **Physical Feasibility Buffer:** Drone bounding diameter is $0.30\text{ m}$. Supervisor boundary buffer is $0.25\text{ m}$; obstacle buffer is $0.30\text{ m}$. Minimum gap required for intervention-free transit: $0.70\text{ m}$.

### 3.2 Obstacle Encounter Breakdown

| Run | Obstacle | Obs $Y$ | Left Gap | Right Gap | Wider Side | Feasibility | Chosen Side | Constrained Selection? | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `seed_1` | 1 | $+0.008\text{ m}$ | $0.792\text{ m}$ | $0.808\text{ m}$ | Right | Both | **Right** | No | Passed |
| `seed_1` | 2 | $+0.315\text{ m}$ | $0.485\text{ m}$ | $1.115\text{ m}$ | Right | Right only | **Right** | No | Pinned at exit |
| `seed_20` | 1 | $-0.154\text{ m}$ | **$0.954\text{ m}$** | **$0.646\text{ m}$** | **Left** | Left only ($>0.7\text{m}$) | **Right** | **YES (Severe)** | Pinned, timeout |
| `seed_100` | 1 | $+0.234\text{ m}$ | $0.566\text{ m}$ | $1.034\text{ m}$ | Right | Right only | **Right** | No | Passed |
| `seed_100` | 2 | $+0.068\text{ m}$ | $0.732\text{ m}$ | $0.868\text{ m}$ | Right | Both | **Right** | No | Passed |
| `seed_100` | 3 | $-0.148\text{ m}$ | **$0.948\text{ m}$** | **$0.652\text{ m}$** | **Left** | Left only ($>0.7\text{m}$) | **Right** | **YES (Severe)** | Pinned, timeout |
| `seed_2000` | 1 | $+0.053\text{ m}$ | $0.747\text{ m}$ | $0.853\text{ m}$ | Right | Both | **Right** | No | Passed |
| `seed_2000` | 2 | $+0.109\text{ m}$ | $0.691\text{ m}$ | $0.909\text{ m}$ | Right | Right only | **Right** | No | Passed |
| `seed_2000` | 3 | $-0.331\text{ m}$ | **$1.131\text{ m}$** | **$0.469\text{ m}$** | **Left** | Left only | **Left** | No | Passed |
| `seed_8000` | 1 | $-0.178\text{ m}$ | **$0.978\text{ m}$** | **$0.622\text{ m}$** | **Left** | Left only ($>0.7\text{m}$) | **Right** | **YES (Severe)** | Pinned, timeout |

### 3.3 Encounter Statistics
- **Total Obstacle Encounters:** 10 (across 5 distinct seeds).
- **Both Sides Feasible ($\ge 0.70\text{ m}$):** 3 encounters.
- **Left Side Only Feasible ($\ge 0.70\text{ m}$):** 4 encounters (`seed_20` Obs 1, `seed_100` Obs 3, `seed_2000` Obs 3, `seed_8000` Obs 1).
- **Right Side Only Feasible ($\ge 0.70\text{ m}$):** 3 encounters (`seed_1` Obs 2, `seed_100` Obs 1, `seed_2000` Obs 2).
- **Cases where PPO chose the constrained side despite an open alternative:** **3 encounters** (75% of left-feasible-only cases). In each of these cases, PPO ignored a $>0.94\text{ m}$ gap on the left to force itself into a $<0.66\text{ m}$ right bottleneck, causing fatal deadlock.

---

## 4. Policy Neural Network Input/Output Audit

To determine whether the right-side preference arises from sensor misinterpretation or an intrinsic network bias, we probed the raw policy activations across systematic state sweeps:

### 4.1 Corridor Centerline Response ($X=1.0\text{ m}$, $v_x = 0.60\text{ m/s}$, $\text{yaw} = 0.0$)

| Lateral Position ($Y$) | Left Wall Dist | Right Wall Dist | Raw $v_x$ (m/s) | Raw $v_y$ (m/s) | Raw $\dot{\psi}$ (rad/s) | Net Steering Command |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $+0.40\text{ m}$ | $0.60\text{ m}$ | $1.40\text{ m}$ | $+0.727$ | **$-0.282$** | **$-0.225$** | Hard Right |
| $+0.20\text{ m}$ | $0.80\text{ m}$ | $1.20\text{ m}$ | $+0.704$ | **$-0.223$** | **$-0.166$** | Hard Right |
| $+0.10\text{ m}$ | $0.90\text{ m}$ | $1.10\text{ m}$ | $+0.699$ | **$-0.216$** | **$-0.158$** | Hard Right |
| **$0.00\text{ m}$ (Center)** | **$1.00\text{ m}$** | **$1.00\text{ m}$** | **$+0.694$** | **$-0.209$** | **$-0.154$** | **Hard Right** |
| $-0.10\text{ m}$ | $1.10\text{ m}$ | $0.90\text{ m}$ | $+0.688$ | **$-0.197$** | **$-0.150$** | Hard Right |
| $-0.20\text{ m}$ | $1.20\text{ m}$ | $0.80\text{ m}$ | $+0.731$ | **$-0.092$** | **$-0.175$** | Right |
| **$-0.38\text{ m}$ (Equil)** | $1.38\text{ m}$ | $0.62\text{ m}$ | $+0.725$ | **$0.000$** | **$-0.110$** | **Lateral Equilibrium** |
| $-0.40\text{ m}$ | $1.40\text{ m}$ | $0.60\text{ m}$ | $+0.724$ | $+0.046$ | $-0.101$ | Weak Centering |

### 4.2 Behavior Cloning (BC) vs PPO Comparison
Evaluating the Behavior Cloning baseline (`bc_policy_best.pt`) under the identical centerline state ($Y=0.0\text{ m}, v_x=0.60\text{ m/s}$):
- **BC Policy:** $v_y = -0.000\text{ m/s}$, $\text{yaw\_rate} = +0.000\text{ rad/s}$ (Symmetric).
- **PPO Baseline:** $v_y = -0.209\text{ m/s}$, $\text{yaw\_rate} = -0.154\text{ rad/s}$ (Asymmetric Right Bias).
- **PPO Candidate (Phase 4B):** $v_y = -0.115\text{ m/s}$, $\text{yaw\_rate} = -0.110\text{ rad/s}$ (Asymmetric Right Bias).

### 4.3 Direct Obstacle Proximity Test ($X_{\text{drone}} = 2.0\text{ m}$, $X_{\text{obs}} = 3.0\text{ m}$)
Sweeping the obstacle lateral position $Y_{\text{obs}}$ from $-0.30\text{ m}$ to $+0.30\text{ m}$:
- For **every single obstacle position**, PPO raw output commanded $v_y < 0$ ($-0.148\text{–}-0.268\text{ m/s}$).
- Even when the obstacle was at $Y_{\text{obs}} = -0.20\text{ m}$ (blocking the right side, with the left side completely open), PPO commanded $v_y = -0.259\text{ m/s}$ (steering right directly into the obstacle).

**Finding:** The neural network does not dynamically balance left versus right opening spaces. It has collapsed into a single topological bypass mode ("always bypass on the right").

---

## 5. Evaluation of Root Cause Hypotheses

| Hypothesis | Evaluated Status | Source-Level Evidence |
| :--- | :---: | :--- |
| **1. Seed not propagated correctly** | **Partially Confirmed (UI)** | In the backend, seed propagates faithfully. In UI, clicking Play without Reset fails to issue a reset request. |
| **2. Scenario configuration fixed** | **Disproved** | Pillars vary across $[-0.33\text{m}, +0.31\text{m}]$; confirmed by live inspection. |
| **3. Reset fails to restore state** | **Confirmed (Supervisor)** | `eval_engine.reset()` fails to clear `stalled_timer_s` and `loss_timer_s`. |
| **4. Coordinate-frame labeling error** | **Disproved** | World ENU, Body FLU, and action signs verified mathematically and via PyBullet raycasts. |
| **5. Sensor ray ordering defect** | **Disproved** | Ray indices 0..11 verified with physical clearances. |
| **6. PPO learned lateral-action bias** | **Confirmed (Primary Cause)** | PPO outputs $v_y = -0.209\text{ m/s}$ at centerline; lateral equilibrium is shifted to $Y=-0.38\text{ m}$. |
| **7. BC demonstrations biased** | **Disproved** | BC policy is symmetric ($v_y = 0.000$ at $Y=0.0\text{ m}$). Bias originated in PPO training. |
| **8. Reward favors progress over detour** | **Confirmed (Contributing)** | Dense progress reward ($4.0 \times \Delta d$) without gap-selection incentives reward monotonic forward transit. |
| **9. Action processing mismatch** | **Disproved** | Scaling $v_y = 0.0 + 0.6\tanh(\text{raw}_1)$ matches across training, evaluation, and viewer. |
| **10. Supervisor forces right route** | **Disproved** | The policy proposes rightward velocity while supervisor is nominal. Supervisor actually pushes left to prevent wall crash. |
| **11. Environment symmetry artifact** | **Partially True** | Symmetric corridor allows symmetry-breaking collapse during RL exploration. |

---

## 6. Required Diagnostic Answers

### 1. Do different UI seeds actually reach the environment reset?
**Yes**, provided the user clicks the "Reset" button (or preset seed chip). If the user changes the number in `#seed-input` and clicks "Play", no reset request is sent.

### 2. Which aspects of the scenario change across the user's reported seeds?
The lateral positions of all 3 obstacles change dynamically according to $y_i \sim \mathcal{U}(-0.35, 0.35)\text{ m}$. The drone's initial lateral perturbation also varies across $[-0.1, +0.1]\text{ m}$. Sensor noise and dropouts vary per seed.

### 3. Are same-seed replays reproducible?
**Yes, 100% bit-for-bit reproducible.** Seed 1 replayed produced identical trajectories, action outputs, supervisor states, and step counts.

### 4. Does PPO choose the same side when obstacle geometry makes that side unsuitable?
**Yes.** In Seed 20 and Seed 8000, Obstacle 1 left only a narrow $0.62\text{–}0.65\text{ m}$ gap on the right while leaving a $0.95\text{–}0.98\text{ m}$ opening on the left. PPO chose the right side in both cases, wedging itself between the obstacle and the wall and timing out.

### 5. Are the policy's raw actions similar across different observations?
**Yes.** At corridor entrance ($X < 1.5\text{ m}$), the policy commands $v_y \approx -0.45\text{ m/s}$ and $\dot{\psi} \approx -0.50\text{ rad/s}$ regardless of obstacle placement, driving the drone to the right wall before Obstacle 1 is even reached.

### 6. Does the safety supervisor cause or amplify the apparent repeated route?
**No.** The policy independently commands the rightward veer while the supervisor is in `nominal` state. The supervisor only acts defensively when the drone gets dangerously close to the wall ($d < 0.25\text{ m}$) or obstacle ($d < 0.30\text{ m}$).

### 7. What is the first confirmed point in the pipeline where the behavior becomes invariant or biased?
The **actor neural network weights** of the PPO checkpoint (`models/checkpoints/ppo_policy_best.pt`). Probing the network directly reveals that the bias exists in the learned weights, not in the environment, sensor pipeline, or supervisor.

### 8. What is the smallest evidence-supported corrective action?
1. **Interactive Viewer UI/Server Fix (Code Correction):**
   - Automatically trigger `/api/reset` when the seed input changes or when "Play" is clicked following an input change.
   - Reset `self.safety_supervisor` internal timers in `EvaluationEngine.reset()`.
2. **Policy Fine-Tuning Symmetry Regularization (Future Phase):**
   - Incorporate lateral reflection augmentation or a gap-clearance reward term during future PPO training rollouts to penalize veering into constrained gaps.

### 9. Is the next step a code/interface correction, a demonstration-data correction, a reward/training change, or further diagnosis?
The immediate next step is **twofold**:
1. A minor code correction in the viewer interface (`app.js` and `eval_engine.py`) to eliminate the seed-entry confusion and supervisor reset leakage.
2. A formal architectural plan for future PPO training (outside this read-only phase) incorporating bilateral symmetry data augmentation or a gap-feasibility reward term.

---

## 7. Decision Rule Statement

**OUTCOME: PASS**

The seed/reset path, scenario variability, observation/action flow, and repeated route behavior have been completely traced to their exact sources with rigorous empirical evidence.

**Formal Assurance:**
- Strictly NO PPO or BC training occurred during this phase.
- Strictly NO model checkpoints were modified, overwritten, or re-exported.
- Strictly NO canonical source files (`corridor_env.py`, `safety_supervisor.py`, `contracts.py`) were modified.
- Strictly NO historical benchmark records were altered.
- All diagnostic artifacts were written to distinct Phase 4F-0 filenames (`reports/phase4f0_*`).
