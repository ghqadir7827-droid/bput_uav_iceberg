"""analyze_exhaustive_sweep_statistics.py - Paired Statistics & Discordant Seed Forensics.

Reads:
- experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv
Outputs:
- experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_paired_analysis.json
- Printed paired McNemar contingency tables, clearance percentiles, and detailed root-cause breakdowns.
"""

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np


def mcnemar_exact(n10: int, n01: int) -> float:
    """Exact two-tailed binomial test for paired McNemar contingency table.
    
    n10: count where v2 succeeds and v3 fails.
    n01: count where v3 succeeds and v2 fails.
    """
    total = n10 + n01
    if total == 0:
        return 1.0
    k = min(n10, n01)
    # two-tailed cumulative binomial probability with p=0.5
    prob = sum(math.comb(total, i) * (0.5 ** total) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95%
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def analyze_sweep(csv_path: str = "experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv"):
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: {path} not found.")
        return

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group rows by (tier, seed)
    paired_by_seed: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        tier = r["tier"]
        seed = int(r["seed"])
        cid = r["controller_id"]
        key = (tier, seed)
        if key not in paired_by_seed:
            paired_by_seed[key] = {}
        paired_by_seed[key][cid] = r

    # Analysis by Tier and Overall
    tiers = list(dict.fromkeys(r["tier"] for r in rows))

    tier_analysis: Dict[str, Any] = {}
    all_discordant: List[Dict[str, Any]] = []

    for t in tiers:
        t_pairs = [v for k, v in paired_by_seed.items() if k[0] == t]
        n_pairs = len(t_pairs)

        # Count outcomes
        n_both_succ = 0
        n_v2_only = 0
        n_v3_only = 0
        n_both_fail = 0

        discordant_in_tier = []

        for pair in t_pairs:
            r2 = pair.get("scaled_guided_bc_v2")
            r3 = pair.get("scaled_guided_bc_v3")
            if not r2 or not r3:
                continue

            s2 = (r2["status"] in ["SUCCESS", "SAFE_REJECT"])
            s3 = (r3["status"] in ["SUCCESS", "SAFE_REJECT"])
            seed = int(r2["seed"])

            if s2 and s3:
                n_both_succ += 1
            elif s2 and not s3:
                n_v2_only += 1
                diff_rec = {
                    "tier": t,
                    "seed": seed,
                    "type": "REGRESSION_V2_WON",
                    "v2_status": r2["status"],
                    "v3_status": r3["status"],
                    "v2_min_clearance_m": float(r2["overall_min_clearance_m"]),
                    "v3_min_clearance_m": float(r3["overall_min_clearance_m"]),
                    "v2_steps": int(r2["total_steps"]),
                    "v3_steps": int(r3["total_steps"]),
                    "v2_interventions": int(r2["safety_interventions"]),
                    "v3_interventions": int(r3["safety_interventions"]),
                    "bottleneck_gap_m": float(r2["bottleneck_gap_m"]),
                }
                discordant_in_tier.append(diff_rec)
                all_discordant.append(diff_rec)
            elif not s2 and s3:
                n_v3_only += 1
                diff_rec = {
                    "tier": t,
                    "seed": seed,
                    "type": "RECOVERY_V3_WON",
                    "v2_status": r2["status"],
                    "v3_status": r3["status"],
                    "v2_min_clearance_m": float(r2["overall_min_clearance_m"]),
                    "v3_min_clearance_m": float(r3["overall_min_clearance_m"]),
                    "v2_steps": int(r2["total_steps"]),
                    "v3_steps": int(r3["total_steps"]),
                    "v2_interventions": int(r2["safety_interventions"]),
                    "v3_interventions": int(r3["safety_interventions"]),
                    "bottleneck_gap_m": float(r2["bottleneck_gap_m"]),
                }
                discordant_in_tier.append(diff_rec)
                all_discordant.append(diff_rec)
            else:
                n_both_fail += 1

        p_mcnemar = mcnemar_exact(n_v2_only, n_v3_only)
        v2_total_succ = n_both_succ + n_v2_only
        v3_total_succ = n_both_succ + n_v3_only

        ci_v2 = compute_wilson_ci(v2_total_succ, n_pairs)
        ci_v3 = compute_wilson_ci(v3_total_succ, n_pairs)

        tier_analysis[t] = {
            "n_pairs": n_pairs,
            "both_succeed": n_both_succ,
            "v2_only_succeed": n_v2_only,
            "v3_only_succeed": n_v3_only,
            "both_fail": n_both_fail,
            "v2_success_count": v2_total_succ,
            "v2_success_pct": round(v2_total_succ / n_pairs * 100.0, 2),
            "v2_wilson_ci_95": [round(ci_v2[0] * 100.0, 2), round(ci_v2[1] * 100.0, 2)],
            "v3_success_count": v3_total_succ,
            "v3_success_pct": round(v3_total_succ / n_pairs * 100.0, 2),
            "v3_wilson_ci_95": [round(ci_v3[0] * 100.0, 2), round(ci_v3[1] * 100.0, 2)],
            "net_v3_improvement": round((v3_total_succ - v2_total_succ) / n_pairs * 100.0, 2),
            "mcnemar_p_value": round(p_mcnemar, 5),
            "is_significant_p05": p_mcnemar < 0.05,
            "discordant_count": len(discordant_in_tier),
            "discordant_seeds": discordant_in_tier,
        }

    # Overall Metrics across all 250 evaluation pairs
    total_pairs = len(paired_by_seed)
    tot_both_succ = sum(ta["both_succeed"] for ta in tier_analysis.values())
    tot_v2_only = sum(ta["v2_only_succeed"] for ta in tier_analysis.values())
    tot_v3_only = sum(ta["v3_only_succeed"] for ta in tier_analysis.values())
    tot_both_fail = sum(ta["both_fail"] for ta in tier_analysis.values())
    p_tot_mcnemar = mcnemar_exact(tot_v2_only, tot_v3_only)

    v2_all_succ = tot_both_succ + tot_v2_only
    v3_all_succ = tot_both_succ + tot_v3_only
    ci_v2_tot = compute_wilson_ci(v2_all_succ, total_pairs)
    ci_v3_tot = compute_wilson_ci(v3_all_succ, total_pairs)

    overall_analysis = {
        "total_pairs": total_pairs,
        "both_succeed": tot_both_succ,
        "v2_only_succeed": tot_v2_only,
        "v3_only_succeed": tot_v3_only,
        "both_fail": tot_both_fail,
        "v2_total_success": v2_all_succ,
        "v2_success_pct": round(v2_all_succ / total_pairs * 100.0, 2),
        "v2_wilson_ci_95": [round(ci_v2_tot[0] * 100.0, 2), round(ci_v2_tot[1] * 100.0, 2)],
        "v3_total_success": v3_all_succ,
        "v3_success_pct": round(v3_all_succ / total_pairs * 100.0, 2),
        "v3_wilson_ci_95": [round(ci_v3_tot[0] * 100.0, 2), round(ci_v3_tot[1] * 100.0, 2)],
        "net_improvement_pct": round((v3_all_succ - v2_all_succ) / total_pairs * 100.0, 2),
        "mcnemar_exact_p_value": round(p_tot_mcnemar, 6),
        "total_discordant_pairs": len(all_discordant),
    }

    results = {
        "overall": overall_analysis,
        "by_tier": tier_analysis,
        "discordant_seed_inventory": all_discordant,
    }

    out_json = path.parent / "exhaustive_paired_analysis.json"
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)

    print("\n" + "=" * 80)
    print("PAIRED REGRESSION STATISTICAL SUMMARY")
    print("=" * 80)
    print(f"Total Paired Seeds Evaluated: {total_pairs}")
    print(f"Overall v2 Success Rate: {v2_all_succ}/{total_pairs} ({overall_analysis['v2_success_pct']}%) [95% CI: {overall_analysis['v2_wilson_ci_95']}]")
    print(f"Overall v3 Success Rate: {v3_all_succ}/{total_pairs} ({overall_analysis['v3_success_pct']}%) [95% CI: {overall_analysis['v3_wilson_ci_95']}]")
    print(f"Net Improvement: +{overall_analysis['net_improvement_pct']}% (p-value: {p_tot_mcnemar:.2e})")
    print(f"Paired Matrix: Both Succ={tot_both_succ}, Only v2={tot_v2_only}, Only v3={tot_v3_only}, Both Fail={tot_both_fail}")
    print("-" * 80)

    for t, ta in tier_analysis.items():
        print(f"{t:<34} | v2: {ta['v2_success_pct']:>5}% [{ta['v2_wilson_ci_95'][0]}-{ta['v2_wilson_ci_95'][1]}%] "
              f"| v3: {ta['v3_success_pct']:>5}% [{ta['v3_wilson_ci_95'][0]}-{ta['v3_wilson_ci_95'][1]}%] "
              f"| Net: {ta['net_v3_improvement']:>+5.1f}% | p={ta['mcnemar_p_value']:.4f}")

    print("\nDiscordant Seeds:")
    for d in all_discordant:
        print(f"  [{d['type']}] Seed {d['seed']} ({d['tier']}): v2={d['v2_status']} vs v3={d['v3_status']} | v2_c={d['v2_min_clearance_m']}m, v3_c={d['v3_min_clearance_m']}m")

    print(f"\nSaved analysis to {out_json}")


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv"
    analyze_sweep(target_csv)

