"""diagnose_level4_failures.py - Detailed Forensic Diagnosis of Level 4 Regressions and Failures.

Analyzes every Level 4 episode in Exhaustive (20300..20349) and Untouched (20350..20399) where:
- v3 fails while v2 succeeds (regressions).
- v3 feasible failures (timeouts).
- Infeasible barrier safe rejection behavior.
- Full 65cm footprint clearance geometry.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from sim.corridor_env import CorridorSimEnv
from core.physical_dimensions import ScaledEnvironmentFactory, DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from scripts.run_exceed_90_experiments import GenericGuidedController


def analyze_level4():
    print("=" * 80)
    print("PHASE 3: LEVEL 4 FORENSIC DIAGNOSIS")
    print("=" * 80)

    # 1. Load episode records
    p_ex = Path("experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv")
    p_un = Path("experiments/scaled_campaign/untouched_confirmation/untouched_episodes.csv")

    with open(p_ex) as f:
        ex_rows = list(csv.DictReader(f))
    with open(p_un) as f:
        un_rows = list(csv.DictReader(f))

    l4_ex = [r for r in ex_rows if "Level 4" in r["tier"]]
    l4_un = [r for r in un_rows if "Level 4" in r["tier"]]

    # Pair by seed
    def get_paired_l4(rows):
        paired = {}
        for r in rows:
            s = int(r["seed"])
            cid = r["controller_id"]
            if s not in paired:
                paired[s] = {}
            paired[s][cid] = r
        return paired

    paired_ex = get_paired_l4(l4_ex)
    paired_un = get_paired_l4(l4_un)

    target_seeds = []
    # Exhaustive regressions (v2 succ, v3 fail)
    for s, p in paired_ex.items():
        v2_ok = p["scaled_guided_bc_v2"]["status"] in ["SUCCESS", "SAFE_REJECT"]
        v3_ok = p["scaled_guided_bc_v3"]["status"] in ["SUCCESS", "SAFE_REJECT"]
        if v2_ok and not v3_ok:
            target_seeds.append((s, "Exhaustive", p["scaled_guided_bc_v2"]["is_feasible"] == "True"))

    # Untouched regressions
    for s, p in paired_un.items():
        v2_ok = p["scaled_guided_bc_v2"]["status"] in ["SUCCESS", "SAFE_REJECT"]
        v3_ok = p["scaled_guided_bc_v3"]["status"] in ["SUCCESS", "SAFE_REJECT"]
        if v2_ok and not v3_ok:
            target_seeds.append((s, "Untouched", p["scaled_guided_bc_v2"]["is_feasible"] == "True"))

    # Also any feasible v3 failures
    for s, p in paired_ex.items():
        if p["scaled_guided_bc_v3"]["is_feasible"] == "True" and p["scaled_guided_bc_v3"]["status"] != "SUCCESS":
            if (s, "Exhaustive", True) not in target_seeds:
                target_seeds.append((s, "Exhaustive", True))

    for s, p in paired_un.items():
        if p["scaled_guided_bc_v3"]["is_feasible"] == "True" and p["scaled_guided_bc_v3"]["status"] != "SUCCESS":
            if (s, "Untouched", True) not in target_seeds:
                target_seeds.append((s, "Untouched", True))

    print(f"Total Level 4 Target Seeds to Trace: {len(target_seeds)}")
    for s, src, feas in sorted(target_seeds):
        print(f"  Seed {s} ({src}) - Feasible: {feas}")

    # Trace each seed
    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY)
    checker = RouteFeasibilityChecker(PhysicalDroneModel(tip_to_tip_diagonal_m=0.65, collision_radius_m=0.325))
    bc = BehaviorCloningPolicyController("models/checkpoints/bc_policy_scaled_v2.pt")

    forensics = []

    for seed, src, feas in sorted(target_seeds):
        env = CorridorSimEnv(scenario=cfg, seed=seed)
        feas_res = checker.evaluate_feasibility(env.corridor, env.obstacles)

        record = {
            "seed": seed,
            "source": src,
            "is_feasible": feas,
            "bottleneck_gap_m": round(feas_res.min_bottleneck_gap_m, 4),
            "feasibility_status": str(feas_res.status),
            "corridor_width_m": cfg.width,
            "corridor_walls_y": [-cfg.width / 2.0, cfg.width / 2.0],
            "obstacles": [{"x": float(o.center[0]), "y": float(o.center[1]), "r": float(o.radius)} for o in env.obstacles],
            "controllers": {},
        }

        # Calculate theoretical corridor bottlenecks for 0.65m drone (radius 0.325m)
        bottlenecks = []
        wall_y = cfg.width / 2.0  # 1.40m
        for o in env.obstacles:
            ox, oy, r = float(o.center[0]), float(o.center[1]), float(o.radius)
            # Left gap: between obstacle left edge (oy + r) and left wall (+wall_y)
            gap_left = wall_y - (oy + r)
            # Right gap: between obstacle right edge (oy - r) and right wall (-wall_y)
            gap_right = (oy - r) - (-wall_y)
            # Available surface-to-surface clearance for 0.65m drone (radius 0.325m)
            # If drone is centered in gap, clearance to each boundary is: (gap - 0.65) / 2
            clearance_left = (gap_left - 0.65) / 2.0
            clearance_right = (gap_right - 0.65) / 2.0
            bottlenecks.append({
                "obstacle_x": ox,
                "obstacle_y": oy,
                "gap_left_m": round(gap_left, 3),
                "clearance_left_m": round(clearance_left, 3),
                "gap_right_m": round(gap_right, 3),
                "clearance_right_m": round(clearance_right, 3),
            })
        record["bottleneck_geometry"] = bottlenecks

        for cid, g_fn in [("v2", GuidanceConfig.for_65cm_uav_v2), ("v3", GuidanceConfig.for_65cm_uav_v3)]:
            obs, _ = env.reset(seed=seed)
            arb = StatelessClearanceGuidanceArbitrator(config=g_fn())
            ctrl = GenericGuidedController(bc, arb, cid)
            sup = SafetySupervisor(env.corridor)

            traj = []
            min_c = float("inf")
            min_wall = float("inf")
            min_obs = float("inf")
            env_steps = 0
            interv_steps = 0
            steps = 0
            stalled_steps = 0
            reversals = 0
            vy_history = []

            for st in range(1000):
                steps += 1
                pos = env._get_drone_position()
                vel = env._get_drone_velocity()

                d_obs = env.corridor.obstacle_clearance(pos, env.obstacles)
                d_wall = env.corridor.boundary_distance(pos)
                min_c = min(min_c, d_obs, d_wall)
                min_obs = min(min_obs, d_obs)
                min_wall = min(min_wall, d_wall)

                if d_obs < 0.15 or d_wall < 0.15:
                    env_steps += 1

                raw_act = ctrl.predict(obs)
                min_fwd = float(min(obs[0:5])) * 4.0

                safe_act, state, reason = sup.filter_action(raw_act, pos, vel, min_fwd, env.dt, observation=obs)
                if state.value != "nominal":
                    interv_steps += 1

                vy_history.append(safe_act.vy)
                if len(vy_history) > 1:
                    if (vy_history[-1] > 0.1 and vy_history[-2] < -0.1) or (vy_history[-1] < -0.1 and vy_history[-2] > 0.1):
                        reversals += 1

                if abs(vel[0]) < 0.05 and abs(vel[1]) < 0.05 and pos[0] > 2.0:
                    stalled_steps += 1

                if st % 50 == 0 or st == 999:
                    traj.append({
                        "step": st,
                        "x": round(float(pos[0]), 3),
                        "y": round(float(pos[1]), 3),
                        "vx": round(float(vel[0]), 3),
                        "vy": round(float(vel[1]), 3),
                        "d_obs": round(float(d_obs), 3),
                        "d_wall": round(float(d_wall), 3),
                        "arb_dir": arb.last_dir,
                        "sup_state": state.value,
                    })

                norm_act = safe_act.to_normalized(env.action_limits)
                obs, _, term, trunc, info = env.step(norm_act)
                if term or trunc:
                    term_reason = info.get("termination_reason", "timeout" if trunc else "unknown")
                    final_pos = env._get_drone_position()
                    break

            record["controllers"][cid] = {
                "final_status": term_reason,
                "total_steps": steps,
                "final_x": round(float(final_pos[0]), 3),
                "final_y": round(float(final_pos[1]), 3),
                "min_clearance_m": round(float(min_c), 4),
                "min_obs_clearance_m": round(float(min_obs), 4),
                "min_wall_clearance_m": round(float(min_wall), 4),
                "envelope_violation_steps": env_steps,
                "intervention_steps": interv_steps,
                "stalled_steps": stalled_steps,
                "lateral_reversals": reversals,
                "sampled_trajectory": traj,
            }

        env.close()
        forensics.append(record)

    out_file = Path("experiments/scaled_campaign/level4_failure_forensics.json")
    with open(out_file, "w") as fp:
        json.dump(forensics, fp, indent=2)
    print(f"\nSaved Level 4 forensics to {out_file}")


if __name__ == "__main__":
    analyze_level4()
