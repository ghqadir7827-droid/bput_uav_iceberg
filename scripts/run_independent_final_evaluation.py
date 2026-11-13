"""run_independent_final_evaluation.py - Independent Final Holdout Evaluation.

Executes both v2 and v3 across the pre-registered disjoint holdout suite:
- Level 1: 26100..26149 (N=50)
- Level 2: 26200..26249 (N=50)
- Level 3: 26300..26349 (N=50)
- Level 4: 26400..26449 (N=50: 25 feasible, 25 infeasible)

Outputs:
- experiments/scaled_campaign/independent_final/independent_final_episodes.csv
- experiments/scaled_campaign/independent_final/independent_final_summary.json
"""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from scripts.run_exceed_90_experiments import GenericGuidedController
from scripts.run_exhaustive_v2_vs_v3_evaluation import ExhaustiveEpisodeRecord, run_single_episode, summarize_episodes
from sim.corridor_env import CorridorSimEnv


def run_independent_evaluation():
    print("=" * 80)
    print("PHASE 5: INDEPENDENT FINAL HOLDOUT EVALUATION (N=200 SEEDS)")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/independent_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    drone_radius = 0.325
    checker = RouteFeasibilityChecker(PhysicalDroneModel(tip_to_tip_diagonal_m=0.65, collision_radius_m=0.325))

    ckpt_path = "models/checkpoints/bc_policy_scaled_v2.pt"
    bc = BehaviorCloningPolicyController(ckpt_path)

    # Controller 1: Frozen v2 Baseline
    g_cfg_v2 = GuidanceConfig.for_65cm_uav_v2(drone_radius=drone_radius)
    arb_v2 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v2)
    ctrl_v2 = GenericGuidedController(bc, arb_v2, "scaled_guided_bc_v2")

    # Controller 2: Candidate v3 with Directional Persistence
    g_cfg_v3 = GuidanceConfig.for_65cm_uav_v3(drone_radius=drone_radius)
    arb_v3 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v3)
    ctrl_v3 = GenericGuidedController(bc, arb_v3, "scaled_guided_bc_v3")

    controllers = [
        ("scaled_guided_bc_v2", ctrl_v2),
        ("scaled_guided_bc_v3", ctrl_v3),
    ]

    tier_suites = [
        ("Level 1 Independent (Basic)", 1, DifficultyLevel.LEVEL_1_BASIC, list(range(26100, 26150))),
        ("Level 2 Independent (Standard)", 2, DifficultyLevel.LEVEL_2_STANDARD, list(range(26200, 26250))),
        ("Level 3 Independent (Challenging)", 3, DifficultyLevel.LEVEL_3_CHALLENGING, list(range(26300, 26350))),
        ("Level 4 Independent (Boundary & Feasibility)", 4, DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(26400, 26450))),
    ]

    all_records: List[ExhaustiveEpisodeRecord] = []
    summary_dict: Dict[str, Any] = {}
    t_start = time.time()

    for tier_name, lvl_num, lvl_enum, seeds in tier_suites:
        print(f"\nEvaluating {tier_name} [N={len(seeds)} Seeds, Range: {seeds[0]}..{seeds[-1]}]")
        cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=cfg, seed=seeds[0])
        sup = SafetySupervisor(env.corridor)

        for cid, ctrl in controllers:
            t0 = time.time()
            tier_records: List[ExhaustiveEpisodeRecord] = []
            for s in seeds:
                rec = run_single_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=sup,
                    checker=checker,
                    tier_name=tier_name,
                    level_num=lvl_num,
                    seed=s,
                    controller_id=cid,
                )
                tier_records.append(rec)
                all_records.append(rec)

            elapsed = time.time() - t0
            summary_dict[f"{tier_name}_{cid}"] = summarize_episodes(tier_records)
            sm = summary_dict[f"{tier_name}_{cid}"]
            if sm["infeasible_total"] > 0:
                print(
                    f"  {cid:<22} | Feas: {sm['feasible_success']:>2}/{sm['feasible_total']} ({sm['feasible_success_pct']}%) "
                    f"| Safe Rej: {sm['infeasible_safe_rejections']:>2}/{sm['infeasible_total']} "
                    f"| Cols: {sm['total_collision_episodes']} | Envs: {sm['envelope_violation_episodes']} | {elapsed:.1f}s"
                )
            else:
                print(
                    f"  {cid:<22} | Succ: {sm['feasible_success']:>2}/{sm['feasible_total']} ({sm['feasible_success_pct']}%) "
                    f"[95% CI: {sm['feasible_wilson_ci_95']}] | Cols: {sm['total_collision_episodes']} | Envs: {sm['envelope_violation_episodes']} | {elapsed:.1f}s"
                )

        env.close()

    total_time = time.time() - t_start
    print(f"\nAll 400 independent holdout episodes completed in {total_time:.1f}s.")

    # Save CSV
    csv_file = out_dir / "independent_final_episodes.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))
    print(f"Saved independent episode records to {csv_file}")

    # Compute Overall Summaries
    for cid in ["scaled_guided_bc_v2", "scaled_guided_bc_v3"]:
        c_recs = [r for r in all_records if r.controller_id == cid]
        summary_dict[f"OVERALL_{cid}"] = summarize_episodes(c_recs)

    # Save Summary JSON
    json_file = out_dir / "independent_final_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved independent summary to {json_file}")


if __name__ == "__main__":
    run_independent_evaluation()

