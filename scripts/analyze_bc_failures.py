"""analyze_bc_failures.py - Detailed Forensic Failure Taxonomy from bc_comparison_episodes_all.csv.

Extracts:
1. Status breakdown for scaled_guided_bc and scaled_raw_bc across Level 1, 2, 3, 4.
2. For Level 2 scaled_guided_bc (the 32 failing episodes):
   - Terminal distance to goal, terminal position, total steps.
   - Stage of failure: approach (X < 3), obstacle 1 (3 <= X < 6), obstacle 2 (6 <= X < 9),
     obstacle 3 (9 <= X < 12), goal acquisition (X >= 12).
   - Min clearances, safety interventions, speeds.
3. Comparative failure modes with reactive baseline and original guided BC.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def analyze():
    csv_path = Path("experiments/scaled_campaign/bc_comparison/bc_comparison_episodes_all.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} episode records.")

    # Group by level and controller
    by_lvl_ctrl = defaultdict(list)
    for r in records:
        key = (int(r["level"]), r["level_name"], r["controller_id"])
        by_lvl_ctrl[key].append(r)

    # 1. Level 2 scaled_guided_bc breakdown
    l2_guided = by_lvl_ctrl[(2, "Level 2 (Standard)", "scaled_guided_bc")]
    print("\n" + "=" * 80)
    print(f"LEVEL 2: scaled_guided_bc Analysis (N={len(l2_guided)})")
    print("=" * 80)

    successes = [r for r in l2_guided if r["success"] == "True"]
    failures = [r for r in l2_guided if r["success"] != "True"]
    print(f"Successes: {len(successes)}/100 ({len(successes)}%)")
    print(f"Failures:  {len(failures)}/100 ({len(failures)}%)")

    fail_statuses = defaultdict(list)
    for r in failures:
        fail_statuses[r["status"]].append(r)

    for st, eps in fail_statuses.items():
        print(f"\nStatus: {st} (Count: {len(eps)})")
        term_dists = [float(r["terminal_dist_to_goal_m"]) for r in eps]
        steps = [int(r["total_steps"]) for r in eps]
        interventions = [int(r["safety_interventions"]) for r in eps]
        speeds = [float(r["mean_speed_mps"]) for r in eps]
        min_clears = [float(r["overall_min_clearance_m"]) for r in eps]

        print(f"  Mean terminal distance to goal: {np.mean(term_dists):.2f}m (min={np.min(term_dists):.2f}m, max={np.max(term_dists):.2f}m)")
        print(f"  Mean steps: {np.mean(steps):.1f}")
        print(f"  Mean safety interventions: {np.mean(interventions):.1f}")
        print(f"  Mean speed: {np.mean(speeds):.3f} m/s")
        print(f"  Min clearance: {np.min(min_clears):.4f}m")

        # Sample seeds
        seeds = [int(r["seed"]) for r in eps]
        print(f"  Seeds ({len(seeds)}): {seeds[:15]}...")

    # Classify failure stage by terminal distance to goal
    # Goal is at X=13.2, Y=0.0
    # X=0.8 start. Obstacles at ~3.5, 7.0, 10.5
    print("\n--- Failure Stage Classification (by terminal distance to goal) ---")
    stalls_near_start = []      # dist > 9.0 (X < 4.2, Obstacle 1)
    stalls_mid_obstacle2 = []   # 5.5 < dist <= 9.0 (4.2 <= X < 7.7, Obstacle 2)
    stalls_mid_obstacle3 = []   # 2.5 < dist <= 5.5 (7.7 <= X < 10.7, Obstacle 3)
    terminal_goal_miss = []     # dist <= 2.5 (X >= 10.7, Near Goal)

    for r in failures:
        d = float(r["terminal_dist_to_goal_m"])
        s = int(r["seed"])
        interv = int(r["safety_interventions"])
        min_c = float(r["overall_min_clearance_m"])
        info = {"seed": s, "term_dist": d, "interventions": interv, "min_clear": min_c, "speed": float(r["mean_speed_mps"])}
        if d > 9.0:
            stalls_near_start.append(info)
        elif d > 5.5:
            stalls_mid_obstacle2.append(info)
        elif d > 2.0:
            stalls_mid_obstacle3.append(info)
        else:
            terminal_goal_miss.append(info)

    print(f"Stage 1 - Stuck before/at Obstacle 1 (X < 4.5m, dist > 9.0m): {len(stalls_near_start)} episodes")
    for x in stalls_near_start[:5]:
        print(f"   seed={x['seed']}: term_dist={x['term_dist']:.2f}m, interv={x['interventions']}, min_clear={x['min_clear']:.3f}m, speed={x['speed']:.3f}m/s")

    print(f"Stage 2 - Stuck between Obstacle 1 & 2 (4.5m <= X < 8.0m, 5.5 < dist <= 9.0m): {len(stalls_mid_obstacle2)} episodes")
    for x in stalls_mid_obstacle2[:5]:
        print(f"   seed={x['seed']}: term_dist={x['term_dist']:.2f}m, interv={x['interventions']}, min_clear={x['min_clear']:.3f}m, speed={x['speed']:.3f}m/s")

    print(f"Stage 3 - Stuck between Obstacle 2 & 3 (8.0m <= X < 11.0m, 2.0 < dist <= 5.5m): {len(stalls_mid_obstacle3)} episodes")
    for x in stalls_mid_obstacle3[:5]:
        print(f"   seed={x['seed']}: term_dist={x['term_dist']:.2f}m, interv={x['interventions']}, min_clear={x['min_clear']:.3f}m, speed={x['speed']:.3f}m/s")

    print(f"Stage 4 - Terminal Goal Miss / Stalled near Goal (X >= 11.0m, dist <= 2.0m): {len(terminal_goal_miss)} episodes")
    for x in terminal_goal_miss:
        print(f"   seed={x['seed']}: term_dist={x['term_dist']:.2f}m, interv={x['interventions']}, min_clear={x['min_clear']:.3f}m, speed={x['speed']:.3f}m/s")

    # Output detailed report
    summary_report = {
        "level_2_failures_total": len(failures),
        "stage_breakdown": {
            "obstacle_1_choke": len(stalls_near_start),
            "obstacle_2_choke": len(stalls_mid_obstacle2),
            "obstacle_3_choke": len(stalls_mid_obstacle3),
            "terminal_goal_miss": len(terminal_goal_miss),
        },
        "sample_stuck_seeds": {
            "obstacle_1": [x["seed"] for x in stalls_near_start],
            "obstacle_2": [x["seed"] for x in stalls_mid_obstacle2],
            "obstacle_3": [x["seed"] for x in stalls_mid_obstacle3],
            "terminal_goal": [x["seed"] for x in terminal_goal_miss],
        }
    }
    out_path = Path("experiments/scaled_campaign/bc_comparison/failure_analysis_l2.json")
    with open(out_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    analyze()

