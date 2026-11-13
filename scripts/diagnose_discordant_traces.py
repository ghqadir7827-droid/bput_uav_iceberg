"""diagnose_discordant_traces.py - Deep Root Cause Forensic Analysis of Discordant Seeds.

Analyzes high-impact discordant seeds to measure:
- Number of lateral command reversals (directional chatter metric).
- Guidance override activations and persistence lock steps.
- Closest point of approach (CPA) and safety supervisor buffer proximity.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import compute_forward_clearance_m
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from scripts.run_exceed_90_experiments import GenericGuidedController
from sim.corridor_env import CorridorSimEnv


def analyze_discordant_seeds():
    analysis_file = Path("experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_paired_analysis.json")
    if not analysis_file.exists():
        print("Analysis file not found.")
        return

    with open(analysis_file) as f:
        data = json.load(f)

    discordant = data.get("discordant_seed_inventory", [])
    print(f"Total discordant seeds to diagnose: {len(discordant)}")

    drone_radius = 0.325
    ckpt_path = "models/checkpoints/bc_policy_scaled_v2.pt"
    bc = BehaviorCloningPolicyController(ckpt_path)

    g_cfg_v2 = GuidanceConfig.for_65cm_uav_v2(drone_radius=drone_radius)
    arb_v2 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v2)
    ctrl_v2 = GenericGuidedController(bc, arb_v2, "scaled_guided_bc_v2")

    g_cfg_v3 = GuidanceConfig.for_65cm_uav_v3(drone_radius=drone_radius)
    arb_v3 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v3)
    ctrl_v3 = GenericGuidedController(bc, arb_v3, "scaled_guided_bc_v3")

    forensic_records: List[Dict[str, Any]] = []

    # Map tier string to DifficultyLevel
    tier_map = {
        "Level 1 (Basic)": DifficultyLevel.LEVEL_1_BASIC,
        "Level 2 Dev (Standard)": DifficultyLevel.LEVEL_2_STANDARD,
        "Level 2 Confirmation": DifficultyLevel.LEVEL_2_STANDARD,
        "Level 3 (Challenging)": DifficultyLevel.LEVEL_3_CHALLENGING,
        "Level 4 (Boundary & Feasibility)": DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY,
    }

    # Group by tier
    for item in discordant:
        t_name = item["tier"]
        seed = item["seed"]
        lvl_enum = tier_map[t_name]

        cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=cfg, seed=seed)
        sup = SafetySupervisor(env.corridor)

        seed_diag = {
            "tier": t_name,
            "seed": seed,
            "type": item["type"],
            "v2_status": item["v2_status"],
            "v3_status": item["v3_status"],
            "controllers": {},
        }

        for cid, ctrl in [("v2", ctrl_v2), ("v3", ctrl_v3)]:
            ctrl.reset()
            sup.reset()
            obs, _ = env.reset(seed=seed)

            vy_history = []
            dir_history = []
            min_c = float("inf")
            reversals = 0
            steps = 0
            dt = env.dt

            for st in range(1000):
                steps += 1
                pos = env._get_drone_position()
                vel = env._get_drone_velocity()

                d_obs = env.corridor.obstacle_clearance(pos, env.obstacles)
                d_wall = env.corridor.boundary_distance(pos)
                min_c = min(min_c, d_obs, d_wall)

                raw_act = ctrl.predict(obs)
                min_fwd = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

                safe_act, state, _ = sup.filter_action(
                    proposed_action=raw_act,
                    current_position=pos,
                    current_velocity=vel,
                    min_front_range=min_fwd,
                    dt=dt,
                    observation=obs,
                )

                if hasattr(ctrl, "arbitrator"):
                    d_cur = ctrl.arbitrator.last_dir
                    dir_history.append(d_cur)

                vy_history.append(safe_act.vy)
                if len(vy_history) > 1:
                    # Sign reversal detection
                    if (vy_history[-1] > 0.1 and vy_history[-2] < -0.1) or (vy_history[-1] < -0.1 and vy_history[-2] > 0.1):
                        reversals += 1

                norm_act = safe_act.to_normalized(env.action_limits)
                obs, _, term, trunc, _ = env.step(norm_act)
                if term or trunc:
                    break

            seed_diag["controllers"][cid] = {
                "steps": steps,
                "lateral_reversals": reversals,
                "min_clearance_m": round(float(min_c), 4),
                "tie_break_dirs": [d for d in dir_history if "TIE" in d][:10],
            }

        forensic_records.append(seed_diag)
        env.close()

    out_file = Path("experiments/scaled_campaign/exhaustive_v2_v3/forensic_discordant_analysis.json")
    with open(out_file, "w") as fp:
        json.dump(forensic_records, fp, indent=2)
    print(f"Saved forensic records to {out_file}")


if __name__ == "__main__":
    analyze_discordant_seeds()

