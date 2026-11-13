"""diagnose_anomaly.py - Investigate 3mm Clearance Anomaly in Seed 1007."""

import numpy as np
import torch
from sim.corridor_env import CorridorSimEnv
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m, compute_directional_clearances_m

device = torch.device("cpu")
model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
ckpt = torch.load("experiments/phase4h/checkpoints/bc_phase4h_candidate.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=0.50)

seed = 1007
obs, info = env.reset(seed=seed)
supervisor.reset_state()

print("Obstacles spawned:")
for i, o in enumerate(env.obstacles):
    print(f"  Obs {i}: center={o.center}, radius={o.radius}, height={o.height}")

steps = 0
done = False
min_obs_val = 100.0
min_step = -1

history = []

while not done and steps < 500:
    priv = info["privileged_state"]
    curr_pos = env._get_drone_position()
    curr_vel = priv.global_velocity
    obs_clear = float(priv.obstacle_clearance)
    
    if obs_clear < min_obs_val:
        min_obs_val = obs_clear
        min_step = steps

    front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

    with torch.no_grad():
        t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
    raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))

    clamped_act = VelocityAction(
        vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
        vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
        vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
        yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
    )

    safe_act, s_state, s_reason = supervisor.filter_action(
        proposed_action=clamped_act,
        current_position=priv.global_position,
        current_velocity=priv.global_velocity,
        min_front_range=front_dist,
        dt=env.dt,
        is_localization_healthy=True,
        telemetry_age_s=0.0,
        observation=obs,
    )

    # Directional clearances
    dir_clear = compute_directional_clearances_m(obs)

    history.append({
        "step": steps,
        "pos": curr_pos.tolist(),
        "vel": curr_vel.tolist(),
        "obs_clear": obs_clear,
        "front_dist": front_dist,
        "dir_clear": {
            "front": dir_clear.front_m,
            "left": dir_clear.left_m,
            "right": dir_clear.right_m,
            "rear": dir_clear.rear_m,
        },
        "raw_act": (raw_act.vx, raw_act.vy, raw_act.vz, raw_act.yaw_rate),
        "safe_act": (safe_act.vx, safe_act.vy, safe_act.vz, safe_act.yaw_rate),
        "state": s_state.value,
        "reason": s_reason,
    })

    norm_applied = safe_act.to_normalized(env.action_limits)
    obs, rew, term, trunc, info = env.step(norm_applied)
    steps += 1
    done = term or trunc

term_r = info.get("termination_reason", "unknown")
print(f"Episode finished: steps={steps}, term={term_r}, min_obs_clearance={min_obs_val:.5f} at step {min_step}")

print("\nTrace around min step:")
start_idx = max(0, min_step - 8)
end_idx = min(len(history), min_step + 9)
for h in history[start_idx:end_idx]:
    p = h["pos"]
    dc = h["dir_clear"]
    print(f"Step {h['step']:3d}: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) obs_clear={h['obs_clear']:.5f} front={dc['front']:.2f} left={dc['left']:.2f} right={dc['right']:.2f} rear={dc['rear']:.2f} safe_vx={h['safe_act'][0]:.2f} safe_vy={h['safe_act'][1]:.2f} state={h['state']} reason={h['reason'][:50]}")

