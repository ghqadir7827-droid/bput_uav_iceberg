"""prepare_independent_evaluation.py - Pre-registers Independent Final Holdout Seeds.

Generates data/independent_final_seed_manifest.json before any evaluation takes place:
- Level 1 (Basic): Seeds 26100..26149 (N=50)
- Level 2 (Standard): Seeds 26200..26249 (N=50)
- Level 3 (Challenging): Seeds 26300..26349 (N=50)
- Level 4 (Boundary & Feasibility): Seeds 26400..26449 (N=50: 25 even=feasible, 25 odd=infeasible)

Total: 200 seeds (400 paired episodes).
"""

import json
from pathlib import Path


def generate_manifest():
    manifest_path = Path("data/independent_final_seed_manifest.json")
    manifest = {
        "title": "Phase 5 Independent Final Evaluation Seed Manifest",
        "description": "Pristine, disjoint holdout seed set generated and frozen before execution.",
        "timestamp_iso": "2026-10-31T01:34:00Z",
        "total_seeds": 200,
        "partitions": {
            "Level 1 (Basic)": {
                "level": 1,
                "seed_range": [26100, 26149],
                "count": 50,
                "feasibility": "All Feasible",
            },
            "Level 2 (Standard)": {
                "level": 2,
                "seed_range": [26200, 26249],
                "count": 50,
                "feasibility": "All Feasible",
            },
            "Level 3 (Challenging)": {
                "level": 3,
                "seed_range": [26300, 26349],
                "count": 50,
                "feasibility": "All Feasible",
            },
            "Level 4 (Boundary & Feasibility)": {
                "level": 4,
                "seed_range": [26400, 26449],
                "count": 50,
                "feasibility": "25 Feasible (Even), 25 Infeasible (Odd)",
            },
        },
    }
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Pre-registered manifest written to {manifest_path}")


if __name__ == "__main__":
    generate_manifest()
