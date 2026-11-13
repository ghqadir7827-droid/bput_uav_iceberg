# PHASE 5B — IMPLEMENTATION AND SIGN CONVENTION AUDIT REPORT
## Rigorous Technical Audit of Alternative A Clearance-Based Guidance

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Isolation Directory:** `experiments/phase5b/phase5b_clearance_audit_20261029_220000/`

---

## 1. Resolution of the Directional-Bias Discrepancy

Earlier project reports contained seemingly conflicting descriptions of policy bias (referring in some contexts to a "rightward bias" and in others to a "leftward bias"). A source-level audit of the coordinate conventions, policy heads, and diagnostic logs resolves this completely:

### Coordinate Frame & Sign Conventions
1. **World Frame (PyBullet):**
   - $+X$: Longitudinal forward corridor axis ($X \in [0.0, 10.0]\text{ m}$).
   - $+Y$: Transverse lateral axis ($Y \in [-1.0, +1.0]\text{ m}$).
     - **Left Wall:** $Y = +1.0\text{ m}$.
     - **Right Wall:** $Y = -1.0\text{ m}$.
     - **Corridor Centerline:** $Y = 0.0\text{ m}$.
   - $+Z$: Vertical altitude axis ($Z \in [0.0, 2.0]\text{ m}$).
2. **Body Frame (FLU):**
   - $+X_b$: Forward.
   - $+Y_b$: Left (aligned with $+Y$ in world frame when $\text{yaw} = 0$).
   - $+Z_b$: Up (aligned with $+Z$ in world frame).
3. **Action Vector Conventions (`core/contracts.py`):**
   - $v_x$: Forward speed setpoint ($[-0.20, +0.60]\text{ m/s}$).
   - $v_y$: Lateral velocity setpoint ($[-0.40, +0.40]\text{ m/s}$).
     - **$v_y > 0$ strictly commands motion to the LEFT ($+Y$).**
     - **$v_y < 0$ strictly commands motion to the RIGHT ($-Y$).**
   - $v_z$: Vertical velocity setpoint ($[-0.30, +0.30]\text{ m/s}$).
   - $\dot{\psi}$ (`yaw_rate`): Yaw angular velocity setpoint ($[-0.50, +0.50]\text{ rad/s}$, $+CCW, -CW$).
4. **Rangefinder Azimuth Signs (`core/contracts.py`):**
   - Ray 0: $\text{azimuth} = 0.0^\circ$ (Front center).
   - Ray 1: $\text{azimuth} = +30.0^\circ$ (Front Left).
   - Ray 2: $\text{azimuth} = -30.0^\circ$ (Front Right).
   - Ray 3: $\text{azimuth} = +60.0^\circ$ (Side Front Left).
   - Ray 4: $\text{azimuth} = -60.0^\circ$ (Side Front Right).
   - Ray 5: $\text{azimuth} = +90.0^\circ$ (Lateral Left).
   - Ray 6: $\text{azimuth} = -90.0^\circ$ (Lateral Right).

### The Root of the Historical Discrepancy
- In Phase 4F-1 (`reports/phase4f1_mirrored_route_comparison.csv`), the **PPO baseline** exhibited a strong **RIGHTWARD bias** ($v_y < 0$, chosen side = "right" in $90\%$ of evaluations).
- In contrast, the **frozen BC baseline (`bc_policy_best.pt`)** exhibits a net **LEFTWARD bias**:
  - In an open corridor at $Y = 0.0\text{ m}$, the BC policy commands $v_y = +0.0366\text{ m/s}$ (steering left).
  - When an obstacle is shifted to the right ($Y_{\text{obs}} = -0.10\text{ m}$), the wide opening is on the left; the BC baseline's inherent leftward drift steers it cleanly around the obstacle ($8.15\text{ m}$ SUCCESS).
  - When an obstacle is shifted to the left ($Y_{\text{obs}} = +0.10\text{ m}$), the wide opening is on the right; the BC baseline cannot overcome its internal bias, drifts into the obstacle caution envelope, and freezes at $X \approx 1.76\text{ m}$ (`STANDOFF_TIMEOUT`).
- **Conclusion:** There is no code defect in the coordinate frames. The discrepancy was purely terminological in historical text comparing PPO (right-biased) against BC (left-biased).

---

## 2. Technical Audit of Alternative A Implementation

Inspection of `AltAClearanceController` in `experiments/phase5a/.../run_phase5a_diagnostics.py` establishes the exact operational logic:

### 1. Rangefinder Sector Assignments
- **Front Sector:** Indices $\{0, 1, 2\}$ ($\text{azimuth} \in [-30^\circ, +30^\circ]$).
- **Left Sector:** Indices $\{1, 3, 5\}$ ($\text{azimuth} \in \{+30^\circ, +60^\circ, +90^\circ\}$).
- **Right Sector:** Indices $\{2, 4, 6\}$ ($\text{azimuth} \in \{-30^\circ, -60^\circ, -90^\circ\}$).

### 2. Clearance Metric Calculation
- The raw deployable observation vector contains normalized distance rays $r_i \in [0.0, 1.0]$.
- The rule de-normalizes rays by the maximum sensor range:
  $$d_i = r_i \times 4.0\text{ m}$$
- Sector metrics are computed via minimum pooling:
  $$C_{\text{front}} = \min(d_0, d_1, d_2)$$
  $$C_{\text{left}} = \min(d_1, d_3, d_5)$$
  $$C_{\text{right}} = \min(d_2, d_4, d_6)$$
  $$\Delta = C_{\text{left}} - C_{\text{right}}$$

### 3. Physical Meaning of Metrics
- In `sim/corridor_env.py`, each ray returns:
  $$d_{\text{ray}} = d_{\text{offset}} + \alpha \cdot (R_{\text{max}} - d_{\text{offset}}) + \epsilon_{\text{noise}}$$
  where $d_{\text{offset}} = 0.18\text{ m}$ (drone collision radius + buffer).
- Therefore, $d_i$ represents the line-of-sight distance along the beam ray from the drone sensor center to the nearest obstacle or boundary surface. It is **not** true 3D Euclidean surface-to-surface clearance, but a directional ray clearance.

### 4. Activation Threshold
- Activation condition: $C_{\text{front}} < 0.85\text{ m}$.
- When $C_{\text{front}} \ge 0.85\text{ m}$, the rule is dormant:
  $$v_{y, \text{cmd}} = v_{y, \text{BC}}$$
  This ensures that in open flight, the policy retains nominal goal-seeking forward cruise.

### 5. Side Selection & Asymmetry Handling
When $C_{\text{front}} < 0.85\text{ m}$:
- **Left Side Wider ($\Delta > +0.08\text{ m}$):**
  $$v_{y, \text{cmd}} = \max(v_{y, \text{BC}}, +0.25\text{ m/s})$$
  Enforces a positive (leftward) velocity setpoint towards the wider passage.
- **Right Side Wider ($\Delta < -0.08\text{ m}$):**
  $$v_{y, \text{cmd}} = \min(v_{y, \text{BC}}, -0.25\text{ m/s})$$
  Enforces a negative (rightward) velocity setpoint towards the wider passage.

### 6. Symmetry Tie-Breaking
When $|\Delta| \le 0.08\text{ m}$ (symmetric encounter):
- The rule activates its deterministic tie-breaker:
  $$\text{selected\_dir} = \text{TIE\_BREAK\_LEFT}$$
  $$v_{y, \text{cmd}} = \max(v_{y, \text{BC}}, +0.25\text{ m/s})$$
- This injects the non-zero lateral velocity impulse that the feedforward MLP policy mathematically collapses to zero.

### 7. Step-to-Step Persistence (Statelessness)
- **Alternative A is completely stateless.**
- It maintains no memory, latch, or hysteresis.
- At every simulation step ($\Delta t = 0.05\text{ s}$), it recalculates $C_{\text{front}}$, $C_{\text{left}}$, and $C_{\text{right}}$.
- If sensor noise ($\sigma = 0.02\text{ m}$) or rapid obstacle passage causes $\Delta$ to fluctuate near the $\pm 0.08\text{ m}$ threshold, the rule can chatter or drop the command prematurely.

### 8. Interaction with Policy & Safety Supervisor
- The guidance rule modifies only the commanded lateral velocity $v_y$.
- $v_x$, $v_z$, and $\dot{\psi}$ are passed unchanged from the BC policy.
- The action is clamped to physical `ActionLimits` ($v_y \in [-0.40, +0.40]\text{ m/s}$).
- The resulting action is fed into `SafetySupervisor.filter_action(...)`. If the UAV approaches within $0.25\text{ m}$ of a corridor wall, the supervisor's boundary repulsion safely overrides or limits $v_y$.

### 9. Multi-Obstacle Vulnerability Identified
- Because Alternative A is stateless and purely local, clearing an obstacle by steering left brings the UAV close to the left wall ($Y \approx 0.45–0.55\text{ m}$).
- If the subsequent obstacle is placed on the left side of the corridor, the UAV enters the encounter with unfavorable positioning, reducing the available reaction time and increasing the likelihood of boundary throttling or timeout.

