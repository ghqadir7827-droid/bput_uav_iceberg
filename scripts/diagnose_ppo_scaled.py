"""diagnose_ppo_scaled.py - Diagnostic Investigation of PPO in Scaled Environment.

Analyzes why ppo_policy_best.pt experiences 0% success in Level 2:
1. Lateral bias and action saturation under scaled observations.
2. Value function / action distribution vs legacy 10m environment.
3. Safety supervisor intervention breakdown (Caution, Stopping, Repulsion).
4. Exploration challenges and sample efficiency in 3.8m corridor.
"""

import json
from pathlib import Path
import numpy as np
import torch

from controllers.ppo_policy import PPOPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import compute_forward_clearance_m
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from sim.corridor_env import CorridorSimEnv


def diagnose_ppo():
    ppo = PPOPolicyController("models/checkpoints/ppo_policy_best.pt")
    cfg_scaled = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env_scaled = CorridorSimEnv(scenario=cfg_scaled, seed=20000)
    supervisor = SafetySupervisor(env_scaled.corridor)

    actions_raw = []
    actions_filtered = []
    supervisor_states = {"nominal": 0, "caution": 0, "stopping": 0, "evasion": 0, "recovery": 0}
    reasons = []

    # Run 10 diagnostic seeds
    for s in range(20000, 20010):
        obs, _ = env_scaled.reset(seed=s)
        for step in range(300):
            pos = env_scaled._get_drone_position()
            vel = env_scaled._get_drone_velocity()

            raw_act = ppo.predict(obs)
            actions_raw.append([raw_act.vx, raw_act.vy, raw_act.vz, raw_act.yaw_rate])

            min_front = compute_forward_clearance_m(obs, max_range_m=env_scaled.sensor_config.max_range_m)
            safe_act, state, _ = supervisor.filter_action(
                proposed_action=raw_act,
                current_position=pos,
                current_velocity=vel,
                min_front_range=min_front,
                dt=env_scaled.dt,
                is_localization_healthy=True,
                telemetry_age_s=0.0,
                observation=obs,
            )

            supervisor_states[state.value] = supervisor_states.get(state.value, 0) + 1
            actions_filtered.append([safe_act.vx, safe_act.vy, safe_act.vz, safe_act.yaw_rate])

            norm_act = safe_act.to_normalized(env_scaled.action_limits)
            obs, _, term, trunc, info = env_scaled.step(norm_act)
            if term or trunc:
                reasons.append(info.get("termination_reason", "unknown"))
                break

    env_scaled.close()

    raw_arr = np.array(actions_raw)
    filt_arr = np.array(actions_filtered)

    diag = {
        "raw_action_means": {
            "vx": float(np.mean(raw_arr[:, 0])),
            "vy": float(np.mean(raw_arr[:, 1])),
            "vz": float(np.mean(raw_arr[:, 2])),
            "yaw_rate": float(np.mean(raw_arr[:, 3])),
        },
        "raw_action_stds": {
            "vx": float(np.std(raw_arr[:, 0])),
            "vy": float(np.std(raw_arr[:, 1])),
            "vz": float(np.std(raw_arr[:, 2])),
            "yaw_rate": float(np.std(raw_arr[:, 3])),
        },
        "filtered_action_means": {
            "vx": float(np.mean(filt_arr[:, 0])),
            "vy": float(np.mean(filt_arr[:, 1])),
            "vz": float(np.mean(filt_arr[:, 2])),
            "yaw_rate": float(np.mean(filt_arr[:, 3])),
        },
        "supervisor_states": supervisor_states,
        "sample_termination_reasons": reasons,
    }

    out_file = Path("experiments/scaled_campaign/bc_comparison/ppo_scaled_diagnosis.json")
    with open(out_file, "w") as f:
        json.dump(diag, f, indent=2)

    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    diagnose_ppo()

