"""scratch/test_unified_supervisor.py - Diagnostic test for Unified SafetySupervisor."""

import numpy as np
from sim.corridor_env import CorridorSimEnv
from controllers.registry import ControllerRegistry
from core.contracts import VelocityAction, compute_forward_clearance_m
from controllers.safety_supervisor import SafetySupervisor, SafetyState, SafetySupervisorConfig

class UnifiedSupervisor(SafetySupervisor):
    def filter_action(self, proposed_action, current_position, current_velocity, min_front_range, dt, is_localization_healthy=True, telemetry_age_s=0.0):
        if not is_localization_healthy or telemetry_age_s > self.config.max_stale_duration_s:
            self.loss_timer_s += dt
            if self.loss_timer_s > self.config.max_loss_duration_s:
                safe_act = VelocityAction(vx=0.0, vy=0.0, vz=-0.15, yaw_rate=0.0)
                return safe_act, SafetyState.EMERGENCY_LAND, "Emergency landing"
            curr_vx = current_velocity[0]
            curr_vy = current_velocity[1]
            decel = self.config.deceleration_limit_mps2 * dt
            safe_vx = np.sign(curr_vx) * max(0.0, abs(curr_vx) - decel)
            safe_vy = np.sign(curr_vy) * max(0.0, abs(curr_vy) - decel)
            return VelocityAction(vx=float(safe_vx), vy=float(safe_vy), vz=0.0, yaw_rate=0.0), SafetyState.LOCALIZATION_FAILSAFE, "Failsafe"
        self.loss_timer_s = 0.0

        vx = float(np.clip(proposed_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps))
        vy = float(np.clip(proposed_action.vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(proposed_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))
        yaw_rate = float(np.clip(proposed_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps))

        obs_intervened = False
        obs_reason = ""
        caution_dist = 0.80

        # Obstacle check
        if min_front_range < self.config.critical_obstacle_dist_m:
            obs_intervened = True
            if min_front_range < self.config.emergency_reverse_dist_m:
                vx = -0.10
                obs_reason = f"Emergency obstacle proximity ({min_front_range:.2f}m) -> Reversing"
            else:
                vx = 0.0
                obs_reason = f"Front obstacle detected ({min_front_range:.2f}m < {self.config.critical_obstacle_dist_m:.2f}m) -> Forward brake"
        elif min_front_range < caution_dist:
            obs_intervened = True
            frac = (min_front_range - self.config.critical_obstacle_dist_m) / (caution_dist - self.config.critical_obstacle_dist_m)
            max_caut_vx = max(0.10, frac * self.limits.max_fwd_vel_mps)
            if vx > max_caut_vx:
                vx = max_caut_vx
                obs_reason = f"Obstacle caution zone ({min_front_range:.2f}m) -> Throttled"
            else:
                obs_intervened = False

        # Boundary check (never bypassed!)
        d_left, d_right, d_ceil, d_floor = self.corridor.boundary_clearances(current_position)
        buf = self.config.boundary_buffer_m
        b_intervened = False
        b_details = []

        if d_left < buf:
            b_intervened = True
            vy = min(vy, 0.0) - self.config.repulsion_gain * (buf - d_left)
            b_details.append(f"left={d_left:.2f}m")
        if d_right < buf:
            b_intervened = True
            vy = max(vy, 0.0) + self.config.repulsion_gain * (buf - d_right)
            b_details.append(f"right={d_right:.2f}m")
        if d_ceil < buf:
            b_intervened = True
            vz = min(vz, 0.0) - self.config.repulsion_gain * (buf - d_ceil)
            b_details.append(f"ceil={d_ceil:.2f}m")
        if d_floor < buf:
            b_intervened = True
            vz = max(vz, 0.0) + self.config.repulsion_gain * (buf - d_floor)
            b_details.append(f"floor={d_floor:.2f}m")

        min_b = min(d_left, d_right, d_ceil, d_floor)
        if b_intervened:
            if min_b <= 0.0:
                vx = 0.0
            elif min_front_range < caution_dist:
                b_ratio = max(0.0, min_b / buf)
                vx = min(vx, max(0.05, b_ratio * self.limits.max_fwd_vel_mps * 0.5))

        vy = float(np.clip(vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))
        safe_action = VelocityAction(vx=float(vx), vy=vy, vz=vz, yaw_rate=float(yaw_rate))
        self.last_safe_action = safe_action

        b_str = ", ".join(b_details)
        if obs_intervened and b_intervened:
            return safe_action, SafetyState.OBSTACLE_BRAKE, f"Combined: {obs_reason} | {b_str}"
        elif obs_intervened:
            return safe_action, SafetyState.OBSTACLE_BRAKE, obs_reason
        elif b_intervened:
            return safe_action, SafetyState.BOUNDARY_INTERVENTION, f"Boundary active ({b_str})"
        return safe_action, SafetyState.NOMINAL, "All safety envelopes satisfied"

if __name__ == "__main__":
    registry = ControllerRegistry()
    controller = registry.get_controller("ppo_fine_tuned")
    env = CorridorSimEnv(scenario="10m", dt=0.05, seed=50001)
    obs, info = env.reset(seed=50001)
    sup = UnifiedSupervisor(env.corridor)

    for step in range(1, 500):
        proposed_act = controller.predict(obs)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()
        safe_act, s_state, s_reason = sup.filter_action(
            proposed_action=proposed_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=front_dist,
            dt=env.dt
        )
        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, info = env.step(norm_applied)
        b_dist = info["boundary_distance"]
        obs_clr = info["obstacle_clearance"]
        if step % 25 == 0 or term or trunc:
            print(f"Step {step:3d}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) front_dist={front_dist:.2f} "
                  f"applied=(vx={safe_act.vx:.2f}, vy={safe_act.vy:.2f}, vz={safe_act.vz:.2f}) "
                  f"state={s_state.value} b_dist={b_dist:.3f} obs_clr={obs_clr:.3f}")
        if term or trunc:
            print("Termination:", info.get("termination_reason"), info.get("termination_detail"))
            break
