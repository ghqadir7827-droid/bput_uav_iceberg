# Phase 4J-B-A — Source-Level Distance and Safety Audit

## 1. Executive Summary

This report establishes the source-verified mathematical definitions, sensor implementations, supervisor filtering mechanics, stopping dynamics, and collision semantics governing the GPS-denied UAV navigation simulation.

Every statement in this document has been independently verified against the source code of:
- `controllers/safety_supervisor.py`
- `sim/corridor_env.py`
- `core/geometry.py`
- `core/contracts.py`
- `experiments/phase4j/isolated_supervisor_opt.py`
- `experiments/phase4j/caution_distance_sweep.py`

---

## 2. Obstacle-Distance Convention and Sensor Geometry

### 2.1 Definition of `eff_front`
In `controllers/safety_supervisor.py` (lines 127–128):
```python
dir_clear = compute_directional_clearances_m(sensor_input)
eff_front = float(min(dir_clear.front_m, min_front_range))
```
`eff_front` is the minimum distance returned across the forward-facing rangefinder sensor array (specifically rays at index 0, 1, and 2 in `compute_directional_clearances_m`, and rays 0, 1, 2, 3, 4 in `compute_forward_clearance_m`).

### 2.2 Reference Point and Measurement Origin
In `sim/corridor_env.py` (lines 257–284), multi-beam rangefinder rays are cast using PyBullet `rayTestBatch`:
- The ray starts at an offset outside the drone body:
  $$\mathbf{r}_{\text{start}} = \mathbf{p}_{\text{drone}} + d_{\text{offset}} \cdot \hat{\mathbf{u}}$$
  where $d_{\text{offset}} = r_{\text{drone}} + 0.03\text{ m} = 0.15\text{ m} + 0.03\text{ m} = 0.18\text{ m}$, and $\hat{\mathbf{u}}$ is the unit ray direction.
- The ray extends to maximum range:
  $$\mathbf{r}_{\text{end}} = \mathbf{p}_{\text{drone}} + d_{\text{max}} \cdot \hat{\mathbf{u}} \quad (d_{\text{max}} = 4.0\text{ m})$$
- When a ray strikes an obstacle surface at fraction $f_{\text{hit}} \in [0, 1]$, the distance from the ray origin is $f_{\text{hit}} \cdot (d_{\text{max}} - d_{\text{offset}})$.
- The simulator converts this into the returned distance (line 283):
  $$\text{dist} = d_{\text{offset}} + f_{\text{hit}} \cdot (d_{\text{max}} - d_{\text{offset}})$$
- **Mathematical Implication:** The reported sensor distance `dist` is measured **from the drone's center of mass $\mathbf{p}_{\text{drone}}$ to the obstacle's outer collision surface**.

### 2.3 Accounting for Obstacle Geometry (Pillar Radius)
- Obstacles are spawned in PyBullet as cylinders with collision radius $r_{\text{pillar}} = 0.20\text{ m}$ (and height $2.0\text{ m}$).
- The ray test detects the physical outer cylinder surface.
- The pillar's radius is therefore automatically excluded from the ray distance: the sensor reads the distance to the **outer boundary of the pillar**, not its center.
- Analytically, for a drone at $(x_d, y_d)$ and pillar at $(x_p, y_p)$:
  $$\text{dist}_{\text{sensor}} = \sqrt{(x_d - x_p)^2 + (y_d - y_p)^2} - r_{\text{pillar}}$$

### 2.4 Physical Air Gap vs. Sensor Distance
The drone's physical collision body is a sphere of radius $r_{\text{drone}} = 0.15\text{ m}$.
Therefore, the true physical air gap (clearance between drone outer skin and obstacle outer skin) is:
$$\text{Air Gap} = \text{eff\_front} - r_{\text{drone}} = \text{eff\_front} - 0.15\text{ m}$$
Center-to-center Euclidean distance between drone and pillar is:
$$\text{Center-to-Center Distance} = \text{eff\_front} + r_{\text{pillar}} = \text{eff\_front} + 0.20\text{ m}$$

### 2.5 Verification of the Phase 4J-A Statement ($0.80\text{ m} \iff 0.65\text{ m}$ Air Gap)
- **Status:** **VERIFIED FROM SOURCE**.
- At configured caution threshold $d_{\text{caut}} = 0.80\text{ m}$, throttling activates when $\text{eff\_front} < 0.80\text{ m}$.
- The physical air gap at this activation boundary is:
  $$\text{Air Gap}_{\text{caut}} = 0.80\text{ m} - 0.15\text{ m} = 0.65\text{ m}$$
- The center-to-center distance at activation is:
  $$\text{Dist}_{\text{center}} = 0.80\text{ m} + 0.20\text{ m} = 1.00\text{ m}$$

### 2.6 Activation Conditions for Supervisor Responses
| Action Mode | Trigger Condition in Code | Center-to-Surface Distance (`eff_front`) | Physical Air Gap ($\text{eff\_front} - 0.15\text{ m}$) | Center-to-Center Distance ($\text{eff\_front} + 0.20\text{ m}$) |
|:---|:---|:---:|:---:|:---:|
| **Caution Throttling** | `eff_front < caution_dist` | $< d_{\text{caut}}$ (e.g. $0.80\text{ m}$) | $< 0.65\text{ m}$ (at $0.80\text{ m}$) | $< 1.00\text{ m}$ (at $0.80\text{ m}$) |
| **Emergency Braking** | `eff_front < critical_obstacle_dist_m` | $< 0.30\text{ m}$ | $< 0.15\text{ m}$ | $< 0.50\text{ m}$ |
| **Emergency Reverse** | `eff_front < emergency_reverse_dist_m` | $< 0.15\text{ m}$ | $< 0.00\text{ m}$ (Contact) | $< 0.35\text{ m}$ |
| **Stalled Recovery Retreat** | `vx <= 0.05` and `eff_front < 0.38 m` for $\ge 0.35\text{ s}$ | $< 0.38\text{ m}$ | $< 0.23\text{ m}$ | $< 0.58\text{ m}$ |

*Key Nuance on Emergency Reverse ($0.15\text{ m}$):* Since physical contact occurs when the air gap reaches $0.00\text{ m}$ ($\text{eff\_front} = 0.15\text{ m}$), a true physical measurement of $\text{eff\_front} < 0.15\text{ m}$ represents an obstacle collision (`d_obs <= 0.0`). The emergency reverse branch ($v_x = -0.10\text{ m/s}$) can only be triggered in practice if sensor noise causes $\text{eff\_front}$ to dip below $0.15\text{ m}$ while the physical air gap remains slightly positive ($\approx 1\text{–}10\text{ mm}$), or during stalled recovery retreat.

---

## 3. Actual Velocity-Filter Behaviour

Following the complete control flow of `filter_action` in `controllers/safety_supervisor.py` (lines 66–326):

1. **Input Clamping:**
   Raw action is bounded by physical limits: $v_x \in [0.0, 0.60]\text{ m/s}$, $v_y \in [-0.40, 0.40]\text{ m/s}$, $v_z \in [-0.30, 0.30]\text{ m/s}$, $\dot{\psi} \in [-1.0, 1.0]\text{ rad/s}$.
2. **Forward Proportional Throttling:**
   When $0.30\text{ m} \le \text{eff\_front} < d_{\text{caut}}$:
   $$\text{frac} = \frac{\text{eff\_front} - 0.30}{d_{\text{caut}} - 0.30}, \quad v_{x,\text{max}}^{\text{caut}} = \max\left(0.10, \text{frac} \cdot 0.60\right)$$
   $$\text{if } v_x > v_{x,\text{max}}^{\text{caut}}: \quad v_x = v_{x,\text{max}}^{\text{caut}}$$
   - **Enforces a minimum safe speed:** The throttle clamp never forces $v_x$ below $0.10\text{ m/s}$ in the caution zone.
   - **Forward velocity is capped**, but not artificially increased if proposed $v_x$ is already slower than the cap.
3. **No Dynamic Acceleration Limits in Obstacle Filtering:**
   - During normal flight, caution throttling, and emergency braking, **no acceleration or deceleration rate limits ($\text{m/s}^2$) are enforced on the action setpoint**.
   - Rate limiting is ONLY active during persistent localization loss (`curr_vx - decel * dt`).
4. **Multi-Directional Lateral and Vertical Arbitration:**
   - Lateral clearance is defined symmetrically against walls and obstacles:
     $$C_{\text{left}} = \min(d_{\text{left\_wall}}, d_{\text{left\_obs}}), \quad C_{\text{right}} = \min(d_{\text{right\_wall}}, d_{\text{right\_obs}})$$
   - If $C_{\text{side}} < d_{\text{crit\_lat}} = 0.32\text{ m}$, motion toward that side is completely prohibited ($v_y = 0.0$).
   - If both sides are constrained ($C_{\text{left}} < 0.32$ and $C_{\text{right}} < 0.32$), lateral motion is zeroed ($v_y = 0.0$) and forward velocity is clamped to $0.05\text{ m/s}$ if $\text{eff\_front} < d_{\text{caut}}$.
   - Active boundary repulsion is applied when within $\text{buf} = 0.25\text{ m}$ of walls:
     $$v_y \mathrel{-}= \text{repulsion\_gain} \cdot \text{penetration} \cdot \text{headroom}$$
5. **Statefulness and Hysteresis:**
   - Forward caution throttling and emergency braking are **completely memoryless / stateless**. They evaluate pure instantaneous threshold inequalities at every step without hysteresis.
   - Stalled recovery is **stateful**: it accumulates `self.stalled_timer_s` when $v_x \le 0.05$ and $\text{eff\_front} < 0.38\text{ m}$. It triggers reverse retreat when timer reaches $0.35\text{ s}$, and resets the timer only when $\text{eff\_front} \ge 0.45\text{ m}$ or $v_x > 0.10\text{ m/s}$.

---

## 4. Emergency Stopping Calculation: Simulation vs. Physical Reality

### 4.1 Stopping Distance in Simulation
- In `sim/corridor_env.py` (lines 376–383), velocity commands are applied via:
  `p.resetBaseVelocity(self.drone_id, linearVelocity=[world_vx, world_vy, world_vz])`
  followed by `p.stepSimulation()`.
- When $\text{eff\_front} < 0.30\text{ m}$, the supervisor sets $v_x = 0.0$.
- In the simulation, this zero velocity takes effect on the rigid body at the next physics step ($\Delta t = 0.05\text{ s}$).
- During this single transition step, if the vehicle was entering at throttled minimum speed $v_x = 0.10\text{ m/s}$, the distance traversed before velocity is zeroed is:
  $$\Delta x_{\text{sim}} = v_x \cdot \Delta t = 0.10\text{ m/s} \times 0.05\text{ s} = 0.005\text{ m} = 5.0\text{ mm}$$
- **Verification:** The claim in Phase 4J-A that stopping from $0.10\text{ m/s}$ in simulation takes less than $0.01\text{ m}$ is **verified mathematically and empirically** ($5.0\text{ mm} < 10.0\text{ mm}$).

### 4.2 Physical Reality on a Real UAV (RadioLink CrossFlight)
- A physical quadrotor cannot execute instantaneous velocity resets. Velocity changes require tilting the airframe to generate horizontal thrust components.
- For a physical drone with mass $m = 1.2\text{ kg}$ and maximum tilt deceleration $a_{\text{max}} \approx 0.8\text{ m/s}^2$ (as specified in `SafetySupervisorConfig.deceleration_limit_mps2`):
  - Stopping distance from $v_0 = 0.10\text{ m/s}$:
    $$s_{\text{brake}} = \frac{v_0^2}{2 a} = \frac{0.01}{2 \times 0.8} = 0.00625\text{ m} = 6.25\text{ mm}$$
    Adding communication latency ($\tau_{\text{lat}} \approx 0.05\text{ s}$): $s_{\text{lat}} = v_0 \cdot \tau_{\text{lat}} = 0.005\text{ m} = 5.0\text{ mm}$.
    Total physical stopping distance from $0.10\text{ m/s}$ is $\approx 11.25\text{ mm} \approx 0.011\text{ m}$.
  - Stopping distance from unthrottled maximum speed $v_0 = 0.60\text{ m/s}$:
    $$s_{\text{brake}} = \frac{0.60^2}{2 \times 0.8} = \frac{0.36}{1.6} = 0.225\text{ m} = 22.5\text{ cm}$$
    Adding communication latency: $s_{\text{lat}} = 0.60 \times 0.05 = 0.030\text{ m} = 3.0\text{ cm}$.
    Total physical stopping distance from $0.60\text{ m/s}$ is $\mathbf{25.5\text{ cm}}$!
- **Architectural Consequence:** The physical necessity of the caution zone is to provide the required distance runway ($s_{\text{decel}}$) to shed kinetic energy:
  - If $d_{\text{caut}} = 0.50\text{ m}$, the distance between caution onset ($0.50\text{ m}$) and emergency braking ($0.30\text{ m}$) is **$0.20\text{ m}$ ($20\text{ cm}$)**.
  - Required deceleration to reach $0.10\text{ m/s}$ across $20\text{ cm}$ from $0.60\text{ m/s}$ is:
    $$a = \frac{v_0^2 - v_f^2}{2 \Delta s} = \frac{0.36 - 0.01}{2 \times 0.20} = \frac{0.35}{0.40} = 0.875\text{ m/s}^2$$
    which closely matches the physical capability of the flight controller ($0.80\text{–}1.0\text{ m/s}^2$).
  - If $d_{\text{caut}}$ were reduced to $0.40\text{ m}$ or $0.35\text{ m}$, the deceleration runway would shrink to $10\text{ cm}$ or $5\text{ cm}$, requiring decelerations of $1.75\text{–}3.5\text{ m/s}^2$, which exceeds physical pitch limits and guarantees high-speed barrier penetration.

---

## 5. Boundary and Collision Semantics

### 5.1 Obstacle Collision Detection
- In `core/geometry.py` (lines 31–48 and 173–179), obstacle clearance is evaluated analytically:
  ```python
  xy_dist = np.linalg.norm(p[:2] - c[:2]) - self.radius
  # for cylinder:
  obs_clear = min_dist - self.drone_radius
  ```
- In `check_termination` (lines 208–210):
  ```python
  d_obs = self.obstacle_clearance(pos, obstacles)
  if d_obs <= 0.0:
      return TerminationReason.CRASH_OBSTACLE, f"Obstacle collision (clearance={d_obs:.3f}m)"
  ```
- **Collision is detected when the analytical air gap is $\le 0.000\text{ m}$.**
- PyBullet physics collision contacts are NOT used for termination checks; exact analytical Euclidean formulas are used.

### 5.2 Boundary Collision Detection
- In `core/geometry.py` (lines 144–155 and 203–205):
  $$\text{boundary\_distance} = \min(y_{\text{max}} - y, y - y_{\text{min}}, z_{\text{ceiling}} - z, z - z_{\text{floor}}) - r_{\text{drone}}$$
- If $\text{boundary\_distance} \le 0.0$, `check_termination` triggers `TerminationReason.CRASH_BOUNDARY`.

### 5.3 Accounting for Drone Radius in Clearances
- Both `obstacle_clearance` and `corridor_boundary_distance` **explicitly subtract the drone radius $r_{\text{drone}} = 0.15\text{ m}$**.
- Therefore:
  - $\text{Clearance} > 0$: Physical gap exists.
  - $\text{Clearance} = 0$: Skin-to-skin contact.
  - $\text{Clearance} < 0$: Physical interpenetration.
- The project requirement mandates maintaining a clearance $\ge 0.08\text{ m}$. Any clearance in the range $[0.000, 0.080)\text{ m}$ is an envelope violation without being a collision.

### 5.4 Order of Evaluation and Metric Tracking Bug in Evaluation Harness
- In `CorridorSimEnv.step()`:
  1. `p.stepSimulation()` moves the vehicle.
  2. `_get_observations()` computes new privileged clearances at the new position.
  3. `check_termination()` evaluates whether $d_{\text{obs}} \le 0$ or $d_b \le 0$.
  4. Returns `terminated = True` and `info["obstacle_clearance"]`.
- In the Phase 4J-A evaluation runner (`experiments/phase4j/caution_distance_sweep.py`), `min_obs_clear` was updated at line 119 using `priv` from the *start* of the step loop. When `env.step()` caused a collision termination, the while loop exited before line 119 could update `min_obs_clear` with the terminal state's negative clearance!
- Consequently, reported `min_obs_clearance_m` for crashing episodes reflected the clearance at step $k-1$ (positive) rather than the crash clearance ($\le 0$).
- **Correction for Phase 4J-B:** The evaluation layer must explicitly update `min_obs_clear` and `min_wall_clear` using the final `info["privileged_state"]` after `env.step()`.

