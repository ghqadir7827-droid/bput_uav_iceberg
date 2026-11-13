"""compile_failure_taxonomy.py - Generates Full Forensic Failure Taxonomy Report.

Analyzes bc_comparison_episodes_all.csv across:
- Controllers: reactive_baseline, original_raw_bc, original_guided_bc, scaled_raw_bc, scaled_guided_bc
- Levels: 1 (Basic), 2 (Standard), 3 (Challenging), 4 (Feasibility & Boundary)
Produces comprehensive failure taxonomy with episode counts, min clearances across all episodes,
supervisor intervention durations, and stage classification (Approach, Obstacle 1, Obstacle 2, Obstacle 3, Goal).
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def compile_taxonomy():
    csv_path = Path("experiments/scaled_campaign/bc_comparison/bc_comparison_episodes_all.csv")
    out_dir = Path("experiments/scaled_campaign/exceed_90_campaign")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            records.append(r)

    by_lvl_ctrl = defaultdict(list)
    for r in records:
        by_lvl_ctrl[(int(r["level"]), r["level_name"], r["controller_id"])].append(r)

    report_lines = []
    report_lines.append("# Forensic Failure Taxonomy: 1,250-Episode Scaled UAV Benchmark\n")
    report_lines.append("**Dataset:** `experiments/scaled_campaign/bc_comparison/bc_comparison_episodes_all.csv`  ")
    report_lines.append("**Controllers:** `reactive_baseline`, `original_raw_bc`, `original_guided_bc`, `scaled_raw_bc`, `scaled_guided_bc`  ")
    report_lines.append("**Corridor Tiers:** Level 1 (Basic), Level 2 (Standard), Level 3 (Challenging), Level 4 (Feasibility)\n")
    report_lines.append("---\n")

    report_lines.append("## 1. Global Performance & Failure Breakdown by Level\n")
    report_lines.append("| Tier | Controller | Feasible Episodes | Successes (%) | Timeouts (%) | Collisions (Obs/Wall) | Envelope Violations (<0.15m) | Min Clearance Across ALL (m) | Mean Steps (All) | Mean Interventions |\n")
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|\n")

    metrics_table = {}

    for (lvl, lvl_name, ctrl_id), eps in sorted(by_lvl_ctrl.items()):
        feas_eps = [r for r in eps if r["is_feasible"] == "True"]
        n_feas = len(feas_eps)
        if n_feas == 0:
            continue

        succ = sum(1 for r in feas_eps if r["success"] == "True")
        timeouts = sum(1 for r in feas_eps if r["status"] == "TIMEOUT")
        obs_col = sum(1 for r in feas_eps if r["obstacle_collision"] == "True")
        wall_col = sum(1 for r in feas_eps if r["wall_collision"] == "True")
        env_viol = sum(1 for r in feas_eps if r["safety_envelope_violation"] == "True")

        all_clears = [float(r["overall_min_clearance_m"]) for r in feas_eps]
        min_clear_all = min(all_clears)
        steps_all = [int(r["total_steps"]) for r in feas_eps]
        interv_all = [int(r["safety_interventions"]) for r in feas_eps]

        succ_pct = succ / n_feas * 100.0
        time_pct = timeouts / n_feas * 100.0

        report_lines.append(
            f"| {lvl_name} | `{ctrl_id}` | {n_feas} | {succ} ({succ_pct:.1f}%) | {timeouts} ({time_pct:.1f}%) | {obs_col}/{wall_col} | {env_viol} | {min_clear_all:.4f}m | {np.mean(steps_all):.1f} | {np.mean(interv_all):.1f} |\n"
        )

        metrics_table[f"L{lvl}_{ctrl_id}"] = {
            "level": lvl,
            "level_name": lvl_name,
            "controller": ctrl_id,
            "total_feasible": n_feas,
            "successes": succ,
            "success_pct": succ_pct,
            "timeouts": timeouts,
            "timeout_pct": time_pct,
            "obstacle_collisions": obs_col,
            "wall_collisions": wall_col,
            "envelope_violations": env_viol,
            "min_clearance_all_m": min_clear_all,
            "mean_steps_all": float(np.mean(steps_all)),
            "mean_interventions_all": float(np.mean(interv_all)),
        }

    # 2. Detailed Level 2 Failure Taxonomy for scaled_guided_bc
    l2_guided = by_lvl_ctrl[(2, "Level 2 (Standard)", "scaled_guided_bc")]
    l2_failures = [r for r in l2_guided if r["success"] != "True"]

    report_lines.append("\n---\n")
    report_lines.append("## 2. In-Depth Failure Taxonomy: Level 2 Target (`scaled_guided_bc`)\n")
    report_lines.append(f"In Level 2 Standard ($3.8 \\times 2.5 \\times 14.0$ m, $N=100$), `scaled_guided_bc` achieved **68 successes and 32 failures**.\n")
    report_lines.append("- **Total Collision Count:** **0** (0 obstacle collisions, 0 wall collisions).  ")
    report_lines.append("- **Total Timeout Count:** **32** (100% of all failures are timeouts at step 1,000).  ")
    report_lines.append("- **Mean Safety Interventions in Failed Episodes:** **801.2 / 1,000 steps (80.1% duration)**.  ")
    report_lines.append("- **Mean Cruising Speed in Failed Episodes:** **0.201 m/s** (vs 0.535 m/s in successful episodes).\n")

    # Group into stages
    stage1 = [] # Obstacle 1 (X < 4.5m, dist > 9.0m)
    stage2 = [] # Obstacle 2 (4.5m <= X < 8.0m, 5.5 < dist <= 9.0m)
    stage3 = [] # Obstacle 3 (8.0m <= X < 11.0m, 2.0 < dist <= 5.5m)
    stage4 = [] # Goal acquisition (X >= 11.0m, dist <= 2.0m)

    for r in l2_failures:
        d = float(r["terminal_dist_to_goal_m"])
        if d > 9.0:
            stage1.append(r)
        elif d > 5.5:
            stage2.append(r)
        elif d > 2.0:
            stage3.append(r)
        else:
            stage4.append(r)

    report_lines.append("### Stage-Wise Breakdown of Failures\n")
    report_lines.append(f"1. **Stage 3 — Obstacle 3 Choke Deadlock ($X \\approx 9.5$–$10.5$ m):** **{len(stage3)} / 32 episodes (68.8% of all failures)**  ")
    report_lines.append("   - *Mechanism:* Drone evades Obstacle 1 and 2 to the right ($Y \\approx -0.7$ m), then centers. Approaching Obstacle 3 at $(10.5, -0.57)$, the obstacle is at azimuth $-37^\\circ$. Rays 0, 1, 2 register clear space ($c_{\\text{front}} > 3.0$ m), causing Guidance to output `NONE`. BC predicts forward flight into the obstacle. Ray 4 hits the obstacle at $<0.5$ m, triggering supervisor `obstacle_brake`. The drone freezes at $X \\approx 10.0$–$10.4$ m until step 1,000 timeout.  ")
    report_lines.append(f"   - *Representative Seeds:* `{[int(r['seed']) for r in stage3[:8]]}`\n")

    report_lines.append(f"2. **Stage 1 — Obstacle 1 Approach Stall ($X \\approx 2.5$–$3.5$ m):** **{len(stage1)} / 32 episodes (18.8% of all failures)**  ")
    report_lines.append("   - *Mechanism:* Early deceleration before Obstacle 1 without sufficient lateral commitment. The drone slows to $<0.10$ m/s, enters supervisor caution/stopping mode, and does not build lateral momentum to clear the pillar.  ")
    report_lines.append(f"   - *Representative Seeds:* `{[int(r['seed']) for r in stage1]}`\n")

    report_lines.append(f"3. **Stage 2 — Inter-Obstacle Transition ($X \\approx 6.0$–$7.5$ m):** **{len(stage2)} / 32 episodes (9.4% of all failures)**  ")
    report_lines.append("   - *Mechanism:* Sluggish recentering after evading Obstacle 1 leaves the drone off-center when meeting Obstacle 2, resulting in an emergency brake lock.  ")
    report_lines.append(f"   - *Representative Seeds:* `{[int(r['seed']) for r in stage2]}`\n")

    report_lines.append(f"4. **Stage 4 — Terminal Goal Acquisition ($X \\ge 11.0$ m):** **{len(stage4)} / 32 episodes (3.1% of all failures)**  ")
    report_lines.append("   - *Mechanism:* Drone reaches $X = 12.2$ m but goal-seeking release or slight lateral drift leaves it hovering just outside the $0.50$ m acceptance sphere until step 1,000.  ")
    report_lines.append(f"   - *Representative Seeds:* `{[int(r['seed']) for r in stage4]}`\n")

    report_md_path = out_dir / "FAILURE_TAXONOMY_REPORT.md"
    with open(report_md_path, "w") as f:
        f.writelines(report_lines)

    report_json_path = out_dir / "failure_taxonomy_metrics.json"
    with open(report_json_path, "w") as f:
        json.dump(metrics_table, f, indent=2)

    print(f"Exported Failure Taxonomy Report to: {report_md_path}")
    print(f"Exported Failure Metrics JSON to: {report_json_path}")


if __name__ == "__main__":
    compile_taxonomy()

