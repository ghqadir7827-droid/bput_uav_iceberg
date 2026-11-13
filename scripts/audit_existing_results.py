"""audit_existing_results.py - Full First-Principles Statistical & Integrity Audit.

Audits:
1. experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv
2. experiments/scaled_campaign/untouched_confirmation/untouched_episodes.csv
3. All claims, denominators, percentages, Wilson CIs, McNemar tests, and equivalence claims.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Wilson score interval without continuity correction."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95%
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact binomial test for paired discordant counts."""
    total = n10 + n01
    if total == 0:
        return 1.0
    k = min(n10, n01)
    prob = sum(math.comb(total, i) * (0.5 ** total) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def audit_datasets():
    print("=" * 80)
    print("PHASE 1: FIRST-PRINCIPLES DATASET AUDIT")
    print("=" * 80)

    p_ex = Path("experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv")
    p_un = Path("experiments/scaled_campaign/untouched_confirmation/untouched_episodes.csv")

    with open(p_ex) as f:
        ex_rows = list(csv.DictReader(f))
    with open(p_un) as f:
        un_rows = list(csv.DictReader(f))

    print(f"Loaded {len(ex_rows)} rows from exhaustive_episodes.csv")
    print(f"Loaded {len(un_rows)} rows from untouched_episodes.csv")

    for name, rows in [("Exhaustive", ex_rows), ("Untouched", un_rows)]:
        print(f"\n--- Checking integrity of {name} records ---")
        # Check seed uniqueness per controller
        for cid in ["scaled_guided_bc_v2", "scaled_guided_bc_v3"]:
            c_rows = [r for r in rows if r["controller_id"] == cid]
            seeds = [int(r["seed"]) for r in c_rows]
            print(f"  {cid}: {len(c_rows)} episodes, {len(set(seeds))} unique seeds, min={min(seeds)}, max={max(seeds)}")
            assert len(c_rows) == len(set(seeds)), f"Duplicate seeds detected in {name} for {cid}!"

        # Check collisions
        cols = [r for r in rows if r["collision_episode"] == "True"]
        print(f"  Total collisions in {name}: {len(cols)}")

        # Check status breakdown
        status_counts = {}
        for r in rows:
            k = (r["tier"], r["controller_id"], r["status"], r["is_feasible"])
            status_counts[k] = status_counts.get(k, 0) + 1

        print("  Status breakdown by tier & controller:")
        for (tier, cid, st, feas), cnt in sorted(status_counts.items()):
            print(f"    {tier:<35} | {cid:<22} | feas={feas:<5} | status={st:<12} : {cnt}")


if __name__ == "__main__":
    audit_datasets()
