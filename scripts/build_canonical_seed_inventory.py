"""build_canonical_seed_inventory.py - Comprehensive Seed Inventory Builder.

Catalogs:
1. Canonical Scaled Evaluation Suite (250 seeds across Levels 1-4)
2. Scaled Demonstration Datasets (v1 & v2 train/val seeds)
3. Legacy Unscaled Benchmark and Demonstration Seeds (10m corridor / 30cm drone)

Outputs:
- data/seed_inventory.json
- data/seed_inventory.csv
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

from core.physical_dimensions import (
    DifficultyLevel,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from sim.corridor_env import CorridorSimEnv


def build_inventory():
    print("Building Canonical Seed Inventory...")
    checker = RouteFeasibilityChecker(PhysicalDroneModel())

    records: List[Dict[str, Any]] = []

    # 1. Scaled Evaluation Suite (250 seeds)
    suites = [
        ("Level 1 (Basic)", 1, DifficultyLevel.LEVEL_1_BASIC, range(20100, 20150), "scaled_evaluation_level1_dev"),
        ("Level 2 (Standard Dev)", 2, DifficultyLevel.LEVEL_2_STANDARD, range(20000, 20050), "scaled_evaluation_level2_dev"),
        ("Level 2 (Standard Confirmation)", 2, DifficultyLevel.LEVEL_2_STANDARD, range(20050, 20100), "scaled_evaluation_level2_confirmation"),
        ("Level 3 (Challenging)", 3, DifficultyLevel.LEVEL_3_CHALLENGING, range(20200, 20250), "scaled_evaluation_level3_dev"),
        ("Level 4 (Feasibility & Boundary)", 4, DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, range(20300, 20350), "scaled_evaluation_level4_dev"),
    ]

    for tier_name, lvl_num, lvl_enum, seed_range, provenance in suites:
        cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=cfg, seed=list(seed_range)[0])
        for s in seed_range:
            env.reset(seed=s)
            feas_res = checker.evaluate_feasibility(env.corridor, env.obstacles)

            feas_cat = "FEASIBLE" if feas_res.is_feasible else "INFEASIBLE_BLOCKED"
            if lvl_num == 4 and (s % 2 == 1):
                assert not feas_res.is_feasible, f"Odd seed {s} in Level 4 should be infeasible!"
            
            records.append({
                "seed": s,
                "domain": "scaled_65cm",
                "tier": tier_name,
                "level": lvl_num,
                "corridor_length_m": round(cfg.x_end - cfg.x_start, 2),
                "corridor_width_m": cfg.width,
                "corridor_height_m": cfg.height,
                "obstacle_count": len(env.obstacles),
                "obstacle_radius_m": cfg.pillar_radius,
                "target_velocity_mps": 0.50,
                "min_required_clearance_m": 0.15,
                "feasibility_category": feas_cat,
                "bottleneck_gap_m": round(feas_res.min_bottleneck_gap_m, 4),
                "provenance": provenance,
                "source_manifest": "ScaledEnvironmentFactory",
                "is_active_eval": True
            })
        env.close()

    print(f"Recorded {len(records)} scaled evaluation seeds.")

    # 2. Scaled Demonstrations (v1 and v2)
    # v2 train: 40003..40405 (250 seeds accepted)
    with open("data/demonstrations/bc_train_scaled_v2_audit.json") as fp:
        audit_tr_v2 = json.load(fp)
        for ep in audit_tr_v2["accepted_episodes"]:
            records.append({
                "seed": ep["seed"],
                "domain": "scaled_65cm",
                "tier": "Level 2 (Standard)",
                "level": 2,
                "corridor_length_m": 14.0,
                "corridor_width_m": 3.8,
                "corridor_height_m": 2.5,
                "obstacle_count": 3,
                "obstacle_radius_m": 0.25,
                "target_velocity_mps": 0.50,
                "min_required_clearance_m": 0.12,
                "feasibility_category": "FEASIBLE",
                "bottleneck_gap_m": -1.0,
                "provenance": "bc_train_scaled_v2_accepted",
                "source_manifest": "data/demonstrations/bc_train_scaled_v2_audit.json",
                "is_active_eval": False
            })

    # v2 val: 40500..40589 (60 seeds accepted)
    with open("data/demonstrations/bc_val_scaled_v2_audit.json") as fp:
        audit_va_v2 = json.load(fp)
        for ep in audit_va_v2["accepted_episodes"]:
            records.append({
                "seed": ep["seed"],
                "domain": "scaled_65cm",
                "tier": "Level 2 (Standard)",
                "level": 2,
                "corridor_length_m": 14.0,
                "corridor_width_m": 3.8,
                "corridor_height_m": 2.5,
                "obstacle_count": 3,
                "obstacle_radius_m": 0.25,
                "target_velocity_mps": 0.50,
                "min_required_clearance_m": 0.12,
                "feasibility_category": "FEASIBLE",
                "bottleneck_gap_m": -1.0,
                "provenance": "bc_val_scaled_v2_accepted",
                "source_manifest": "data/demonstrations/bc_val_scaled_v2_audit.json",
                "is_active_eval": False
            })

    # v1 train: 30000..30300 (200 seeds accepted)
    with open("data/demonstrations/bc_train_scaled_audit.json") as fp:
        audit_tr_v1 = json.load(fp)
        for ep in audit_tr_v1["accepted_episodes"]:
            records.append({
                "seed": ep["seed"],
                "domain": "scaled_65cm",
                "tier": "Level 2 (Standard)",
                "level": 2,
                "corridor_length_m": 14.0,
                "corridor_width_m": 3.8,
                "corridor_height_m": 2.5,
                "obstacle_count": 3,
                "obstacle_radius_m": 0.25,
                "target_velocity_mps": 0.50,
                "min_required_clearance_m": 0.12,
                "feasibility_category": "FEASIBLE",
                "bottleneck_gap_m": -1.0,
                "provenance": "bc_train_scaled_v1_accepted",
                "source_manifest": "data/demonstrations/bc_train_scaled_audit.json",
                "is_active_eval": False
            })

    # v1 val: 30201..30273 (50 seeds accepted)
    with open("data/demonstrations/bc_val_scaled_audit.json") as fp:
        audit_va_v1 = json.load(fp)
        for ep in audit_va_v1["accepted_episodes"]:
            records.append({
                "seed": ep["seed"],
                "domain": "scaled_65cm",
                "tier": "Level 2 (Standard)",
                "level": 2,
                "corridor_length_m": 14.0,
                "corridor_width_m": 3.8,
                "corridor_height_m": 2.5,
                "obstacle_count": 3,
                "obstacle_radius_m": 0.25,
                "target_velocity_mps": 0.50,
                "min_required_clearance_m": 0.12,
                "feasibility_category": "FEASIBLE",
                "bottleneck_gap_m": -1.0,
                "provenance": "bc_val_scaled_v1_accepted",
                "source_manifest": "data/demonstrations/bc_val_scaled_audit.json",
                "is_active_eval": False
            })

    # 3. Legacy Unscaled Ranges
    legacy_ranges = [
        (range(1, 1001), "legacy_10m_reactive_qualification", "10m_corridor_unscaled"),
        (range(10000, 11200), "legacy_10m_reactive_extended_eval", "10m_corridor_unscaled"),
        (range(12000, 12100), "legacy_10m_guidance_eval", "10m_corridor_unscaled"),
        (range(2000, 2050), "legacy_10m_val_and_eval", "10m_corridor_unscaled"),
        (range(50000, 50100), "legacy_10m_phase4e_b_eval", "10m_corridor_unscaled"),
        (range(80000, 80100), "legacy_10m_phase4g_heldout", "10m_corridor_unscaled"),
        (range(90000, 90100), "legacy_10m_phase4_protected_baseline", "10m_corridor_unscaled"),
    ]

    for s_range, prov, dom in legacy_ranges:
        for s in s_range:
            records.append({
                "seed": s,
                "domain": dom,
                "tier": "Legacy (10m Corridor / 30cm UAV)",
                "level": 0,
                "corridor_length_m": 10.0,
                "corridor_width_m": 2.0,
                "corridor_height_m": 2.0,
                "obstacle_count": 3,
                "obstacle_radius_m": 0.15,
                "target_velocity_mps": 0.50,
                "min_required_clearance_m": 0.10,
                "feasibility_category": "FEASIBLE",
                "bottleneck_gap_m": -1.0,
                "provenance": prov,
                "source_manifest": "legacy_benchmarks",
                "is_active_eval": False
            })

    print(f"Total inventory records: {len(records)}")

    # Write CSV
    csv_path = Path("data/seed_inventory.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Saved {csv_path}")

    # Write JSON
    json_path = Path("data/seed_inventory.json")
    summary = {
        "total_records": len(records),
        "scaled_evaluation_suite_count": sum(1 for r in records if r["is_active_eval"]),
        "scaled_demonstration_count": sum(1 for r in records if "bc_" in r["provenance"]),
        "legacy_unscaled_count": sum(1 for r in records if "legacy" in r["provenance"]),
        "inventory": records
    }
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved {json_path}")

if __name__ == "__main__":
    build_inventory()
