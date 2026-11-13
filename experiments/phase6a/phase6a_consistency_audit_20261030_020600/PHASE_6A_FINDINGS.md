# PHASE 6A TECHNICAL FINDINGS: CONSISTENCY AUDIT & ROOT CAUSE ANALYSIS
**Autonomous UAV Navigation in GPS-Denied Corridors**  
**Run Identifier:** `phase6a_consistency_audit_20261030_020600`  
**Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Audit Date:** October 30, 2026  

---

## 1. Finding F-1: Safety Supervisor Configuration Default Audit

### 1.1 Source-Code Inspection
In `controllers/safety_supervisor.py` lines 34–47:
```python
@dataclass
class SafetySupervisorConfig:
    boundary_buffer_m: float = 0.25
    critical_obstacle_dist_m: float = 0.30
    emergency_reverse_dist_m: float = 0.15
    repulsion_gain: float = 1.5
    max_stale_duration_s: float = 0.40
    max_loss_duration_s: float = 1.50
    deceleration_limit_mps2: float = 0.8
    obstacle_caution_dist_m: float = 0.80  # <-- CLASS DEFAULT
    critical_lateral_dist_m: float = 0.32
    caution_lateral_dist_m: float = 0.55
    stalled_duration_s: float = 0.35
```

### 1.2 Call-Site Inventory & Classification
Every instantiation of `SafetySupervisor` across the workspace was cataloged:

| Calling Module | Line | Instantiation Form | Effective $d_{\text{caut}}$ | Operational Impact |
| :--- | :---: | :--- | :---: | :--- |
| `hardware/onnx_runner.py` | 60–65 | `SafetySupervisorConfig(obstacle_caution_dist_m=0.50)` | **0.50 m** | **Explicit & Correct** |
| `experiments/phase6_recovery/...` | 361 | `IsolatedOptimizedSupervisor(caution_distance_m=0.50)` | **0.50 m** | **Explicit & Correct** |
| `experiments/phase4j/isolated_supervisor_opt.py` | 51–56 | `SafetySupervisorConfig(obstacle_caution_dist_m=...)` | **Explicit** | **Protected & Isolated** |
| `viewer/eval_engine.py` | 52, 91 | `SafetySupervisor(corridor=self.corridor)` | **0.80 m** | **DEFECT:** Web GUI viewer silently starves time budget |
| `scripts/run_quantitative_eval.py` | 63 | `SafetySupervisor(corridor=env.corridor)` | **0.80 m** | **DEFECT:** CLI benchmark tool silently starves time budget |
| `scripts/run_pybullet_gui.py` | 96 | `SafetySupervisor(corridor=env.corridor)` | **0.80 m** | **DEFECT:** Interactive GUI simulation silently uses 0.80 m |
| `scripts/smoke_test.py` | 44 | `SafetySupervisor(corridor=env.corridor)` | **0.80 m** | Uses class default (smoke test only) |
| `tests/test_safety_supervisor.py` | Various | Various (many pass explicit configs, some default) | **Mixed** | Unit tests evaluate specific boundaries |

### 1.3 Architectural Assessment
The safety supervisor file (`controllers/safety_supervisor.py`) must remain cryptographically frozen in this phase. However, the audit reveals a significant discrepancy:
- The Phase 6 recovery benchmark and ONNX runner were properly evaluated with $d_{\text{caut}} = 0.50\text{ m}$.
- However, the user-facing tools (`viewer/eval_engine.py`, `scripts/run_quantitative_eval.py`, `scripts/run_pybullet_gui.py`) rely on the bare class constructor, which silently enforces $0.80\text{ m}$. This causes any user evaluating the system through the Web GUI or CLI scripts to experience the historical 1% timeout starvation bug.
- **Resolution Strategy:** Callers in the production and evaluation harnesses must explicitly pass `SafetySupervisorConfig(obstacle_caution_dist_m=0.50)` without altering the protected class definition.

---

## 2. Finding F-2: Guidance Implementation Parity & Divergence Audit

### 2.1 Side-by-Side Code Comparison
A line-by-line inspection comparing `controllers/guidance_controller.py` with `hardware/onnx_runner.py` revealed three distinct logic divergences:

#### Divergence A: Asymmetric Deadband Tie-Breaking
- **In `controllers/guidance_controller.py` (lines 166–177):**
  ```python
  if wall_r > wall_l + 0.10:
      selected_dir = "TIE_BREAK_RIGHT"
      vy_cmd = min(vy_cmd, -0.25)
  elif wall_l > wall_r + 0.10:
      selected_dir = "TIE_BREAK_LEFT"
      vy_cmd = max(vy_cmd, 0.25)
  else:
      selected_dir = f"TIE_BREAK_{self.default_tie_breaker}"
      if self.default_tie_breaker == "RIGHT":
          vy_cmd = min(vy_cmd, -0.25)
      else:
          vy_cmd = max(vy_cmd, 0.25)
  ```
- **In `hardware/onnx_runner.py` (lines 132–137):**
  ```python
  if wall_r > wall_l + 0.10:
      vy_cmd = min(vy_cmd, -0.25)
      self.last_guidance_dir = "TIE_BREAK_RIGHT"
  else:
      vy_cmd = max(vy_cmd, 0.25)
      self.last_guidance_dir = "TIE_BREAK_LEFT"
  ```
  *Impact:* The edge runner omitted the symmetric `elif wall_l > wall_r + 0.10` check. It unconditionally defaults to leftward tie-breaking unless the right wall has a $0.10\text{ m}$ advantage.

#### Divergence B: Directional Hysteresis Substring Matching
- **In `controllers/guidance_controller.py` (lines 155–156):**
  Uses exact identity list membership:  
  `self.last_dir in ["LEFT", "TIE_BREAK_LEFT"]`
- **In `hardware/onnx_runner.py` (lines 122–123):**
  Uses substring matching:  
  `"LEFT" in self.last_guidance_dir`  
  *Impact:* If `last_guidance_dir` is `"WALL_AVOID_LEFT"`, the edge runner treats it as a leftward turn and shifts the hysteresis thresholds $l_{th}$ and $r_{th}$ by $h = 0.04\text{ m}$. In the simulator controller, `WALL_AVOID_LEFT` does not trigger this hysteresis shift.

#### Divergence C: Reset Lifecycle
- `guidance_controller.py` provides `reset()` to clear `self.last_dir = "NONE"`.
- `onnx_runner.py` lacked a reset interface for guidance state across distinct mission episodes.

### 2.2 Architectural Parity Resolution
To eliminate code duplication and guarantee bit-for-bit equivalence between edge hardware and simulation, the pure guidance arbitration logic should be extracted into a shared, zero-dependency module:  
`core/guidance.py` (`StatelessClearanceGuidanceArbitrator`).  
Because it uses only standard NumPy and dataclasses, it introduces zero PyTorch dependencies into the edge runner while ensuring 100% behavioral parity.

---

## 3. Finding F-3: Production ONNX Test Coverage Audit

### 3.1 Existing Test Suite Analysis
Inspection of `tests/test_onnx_runner.py` lines 11–95 revealed that every unit test explicitly instantiates the runner with:
```python
runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_ppo.onnx")
```
- **Finding:** The unit test suite does not execute a single line of code against the production artifact `models/checkpoints/policy_bc.onnx`.
- **Consequence:** If `policy_bc.onnx` were deleted, had mismatched layer dimensions, or output non-finite values, the repository test suite would report 100% success.
- **Remediation Plan:**
  1. Add comprehensive tests verifying default runner instantiation (which resolves to `policy_bc.onnx`).
  2. Validate input shape `(1, 23)` and output shape `(1, 4)`.
  3. Validate finite output bounds against `ActionLimits`.
  4. Test guidance activation, terminal release, and safety supervisor end-to-end integration with the BC model.
  5. Retain PPO tests under a legacy regression section.

---

## 4. Finding F-4: Wide-Left Regression Forensic Analysis

### 4.1 Diagnostic Case Geometry
- Obstacle: Cylinder centered at $(x=3.0\text{ m}, y=-0.25\text{ m})$, radius $r=0.15\text{ m}$.
- Corridor bounds: $y \in [-1.0\text{ m}, +1.0\text{ m}]$ (width $2.0\text{ m}$).
- Left Opening: From obstacle left edge ($y=-0.10\text{ m}$) to left wall ($y=+1.0\text{ m}$) = **$1.10\text{ m}$ of open passage**.
- Right Opening: From obstacle right edge ($y=-0.40\text{ m}$) to right wall ($y=-1.0\text{ m}$) = **$0.60\text{ m}$ of restricted passage**.

### 4.2 Forensic Trace Timeline
Examining the recorded episode `repro_Scenario_C_Wide_Left_Phase6_Production_Guidance.json`:
1. **Steps 0–45:** The drone cruises forward from $(x=1.0, y=0.0)$. The frozen BC baseline commands a slight rightward drift ($v_y \approx -0.01$ to $-0.03\text{ m/s}$), bringing the UAV to $y = -0.34\text{ m}$ by step 49.
2. **Step 49 ($C_{\text{front}} = 0.88\text{ m} \to 0.71\text{ m}$):** Guidance activates. Because the UAV is at $y = -0.34\text{ m}$, the obstacle center ($y = -0.25\text{ m}$) is situated to its front-left.
3. **Steps 49–65:** The controller evaluates sector clearances. For several steps, $\Delta = C_{\text{left}} - C_{\text{right}}$ is within the deadband, commanding `TIE_BREAK_LEFT`.
4. **Step 66 ($x=2.35\text{ m}, y=-0.34\text{ m}$):** The front-left rays strike the obstacle cylinder ($C_{\text{left}} = 0.359\text{ m}$), while the right ray looks into open space ($C_{\text{right}} = 0.765\text{ m}$). $\Delta = C_{\text{left}} - C_{\text{right}} = -0.406\text{ m} < -0.08\text{ m}$.
5. **The Trap Closes:** The controller selects `RIGHT` ($v_y = -0.25\text{ m/s}$). It drives directly into the narrow $0.60\text{ m}$ gap on the right.
6. **Step 79 ($x=2.58\text{ m}, y=-0.50\text{ m}$):** Right wall clearance drops to $0.347\text{ m} < 0.35\text{ m}$. The wall-pinch guardrail fires `WALL_AVOID_LEFT` ($v_y = +0.25\text{ m/s}$).
7. **Spatial Oscillation & Deadlock:** The UAV is now caught between the obstacle face (requiring rightward evasion) and the corridor wall (requiring leftward evasion). The safety supervisor detects obstacle proximity $<0.30\text{ m}$, throttles forward speed to $0.10\text{ m/s}$, trips the stalled duration timer ($0.35\text{ s}$), and commands reverse retreat. The episode times out at step 500 at $x=2.69\text{ m}$ ($1.69\text{ m}$ progress).

### 4.3 Why Phase 5B Succeeded on This Case
Phase 5B lacked wall-awareness arbitration. When it navigated toward the obstacle, it tie-broke `LEFT` earlier, entering the $1.10\text{ m}$ left opening where wall clearance was abundant ($>0.50\text{ m}$).

---

## 5. Terminal-Distance Contract Audit

### 5.1 Observation Vector Construction in Simulator
In `sim/corridor_env.py` lines 307–322:
```python
world_rel_goal = self.corridor.goal_position - pos
dist_goal = np.linalg.norm(world_rel_goal)
scale = float(min(1.0, dist_goal / 2.0))
if dist_goal > 1e-4:
    unit_world_goal = (world_rel_goal / dist_goal) * scale
else:
    unit_world_goal = np.zeros(3, dtype=np.float32)

cos_y = np.cos(yaw)
sin_y = np.sin(yaw)
body_goal_x = cos_y * unit_world_goal[0] + sin_y * unit_world_goal[1]
body_goal_y = -sin_y * unit_world_goal[0] + cos_y * unit_world_goal[1]
body_goal_z = unit_world_goal[2]
body_goal = np.array([body_goal_x, body_goal_y, body_goal_z], dtype=np.float32)
```

### 5.2 Mathematical Analysis of the Terminal Release Invariant
1. `dist_goal` is the 3D Euclidean distance in physical meters between the drone position $(x, y, z)$ and the goal position $(x_g, y_g, z_g)$.
2. When $\text{dist\_goal} \ge 2.0\text{ m}$, $\text{scale} = 1.0$. The vector `unit_world_goal` is a unit vector ($\|\mathbf{u}\|_2 = 1.0$).
3. When $\text{dist\_goal} < 2.0\text{ m}$, $\text{scale} = \text{dist\_goal} / 2.0$. The vector becomes:
   $$\mathbf{u} = \frac{\mathbf{g}_{\text{world}}}{\text{dist\_goal}} \cdot \frac{\text{dist\_goal}}{2.0} = \frac{\mathbf{g}_{\text{world}}}{2.0}$$
   Its 3D Euclidean norm is:
   $$\|\mathbf{u}\|_2 = \frac{\text{dist\_goal}}{2.0}$$
4. The rotation from world coordinates to body FLU is an orthogonal planar rotation around the $Z$-axis, which is norm-preserving:
   $$\|\mathbf{g}_{\text{body}}\|_2 = \|\mathbf{u}\|_2 = \frac{\text{dist\_goal}}{2.0}$$
5. In `controllers/guidance_controller.py` lines 132–134:
   ```python
   goal_vec = obs_vec[12:15]
   goal_norm = float(np.linalg.norm(goal_vec))
   in_terminal_zone = (goal_norm < 0.50)
   ```
   Substituting the norm equality:
   $$\text{goal\_norm} < 0.50 \iff \frac{\text{dist\_goal}}{2.0} < 0.50 \iff \text{dist\_goal} < 1.00\text{ m}$$

### 5.3 Conclusion on Terminal Contract
The condition `goal_norm < 0.50` is **strictly equivalent to 3D distance to goal $< 1.00\text{ m}$**. There is no ambiguity. The units and scaling are fully consistent across `sim/corridor_env.py`, `core/contracts.py`, `controllers/guidance_controller.py`, and `hardware/onnx_runner.py`. The threshold should remain at $0.50$ (normalized) representing $1.0\text{ m}$ physical distance.

