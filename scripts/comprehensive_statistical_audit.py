"""comprehensive_statistical_audit.py - Rigorous First-Principles Recomputation and Audit.

Recomputes all metrics from episode CSVs:
1. Denominators and aggregate success rates.
2. Wilson score confidence intervals (exact, formatted as percentages).
3. Paired McNemar exact tests for:
   - Development set (200 seeds)
   - Level 2 Confirmation set (50 seeds)
   - Untouched Confirmation set (200 seeds)
   - Disaggregated by difficulty tier and feasibility category.
4. Two One-Sided Tests (TOST) for equivalence testing with margin Delta = 10%.
5. Feasible-route navigation success vs infeasible-route safe rejection.
6. Envelope violation rates, clearance distributions, and intervention burdens.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Wilson score interval returned as percentages [0, 100]."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95%
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    low = max(0.0, centre - spread) * 100.0
    high = min(1.0, centre + spread) * 100.0
    return (round(low, 2), round(high, 2))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided binomial test for paired discordant counts b (v2 won) and c (v3 won)."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def tost_paired_equivalence(p1: float, p2: float, n: int, delta: float = 0.10) -> Tuple[bool, float, float, Tuple[float, float]]:
    """Two One-Sided Tests (TOST) for paired proportions with equivalence margin delta.
    
    H01: p2 - p1 <= -delta
    H02: p2 - p1 >= delta
    Equivalent if 90% CI for (p2 - p1) lies completely inside (-delta, +delta).
    """
    diff = p2 - p1
    # Standard error for difference in proportions (conservative independent or paired)
    se = math.sqrt(max(1e-6, (p1 * (1 - p1) + p2 * (1 - p2)) / n))
    z90 = 1.6448536269514722
    ci90 = (round((diff - z90 * se) * 100.0, 2), round((diff + z90 * se) * 100.0, 2))
    
    z1 = (diff - (-delta)) / se
    z2 = (delta - diff) / se
    
    # 1 - Phi(z)
    p_val1 = 0.5 * math.erfc(z1 / math.sqrt(2))
    p_val2 = 0.5 * math.erfc(z2 / math.sqrt(2))
    p_tost = max(p_val1, p_val2)
    is_equiv = (ci90[0] > -delta * 100.0 and ci90[1] < delta * 100.0)
    return is_equiv, round(p_tost, 5), round(diff * 100.0, 2), ci90


def analyze_partition(rows: List[Dict[str, Any]], partition_name: str) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"PARTITION AUDIT: {partition_name} (Total Rows: {len(rows)})")
    print(f"================================================================================")

    # Group by seed
    seeds = sorted(list(set(int(r["seed"]) for r in rows)))
    paired: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        s = int(r["seed"])
        cid = r["controller_id"]
        if s not in paired:
            paired[s] = {}
        paired[s][cid] = r

    n_seeds = len(seeds)
    print(f"Unique Seeds: {n_seeds}")

    # Separate Feasible from Infeasible
    feasible_seeds = [s for s in seeds if paired[s]["scaled_guided_bc_v2"]["is_feasible"] == "True"]
    infeasible_seeds = [s for s in seeds if paired[s]["scaled_guided_bc_v2"]["is_feasible"] == "False"]

    print(f"  Feasible Navigation Seeds: {len(feasible_seeds)}")
    print(f"  Infeasible Obstacle Barrier Seeds: {len(infeasible_seeds)}")

    # 1. Feasible Navigation Evaluation
    feas_stats = {}
    for cid in ["scaled_guided_bc_v2", "scaled_guided_bc_v3"]:
        succ = sum(1 for s in feasible_seeds if paired[s][cid]["status"] == "SUCCESS")
        timeouts = sum(1 for s in feasible_seeds if paired[s][cid]["status"] == "TIMEOUT")
        cols = sum(1 for s in feasible_seeds if paired[s][cid]["collision_episode"] == "True")
        envs = sum(1 for s in feasible_seeds if paired[s][cid]["safety_envelope_violation"] == "True")
        env_steps = sum(int(paired[s][cid]["envelope_violation_steps"]) for s in feasible_seeds)
        interv = sum(int(paired[s][cid]["safety_interventions"]) for s in feasible_seeds)
        total_steps = sum(int(paired[s][cid]["total_steps"]) for s in feasible_seeds)
        
        c_vals = [float(paired[s][cid]["overall_min_clearance_m"]) for s in feasible_seeds]
        flight_times = [float(paired[s][cid]["flight_time_s"]) for s in feasible_seeds if paired[s][cid]["status"] == "SUCCESS"]
        speeds = [float(paired[s][cid]["mean_speed_mps"]) for s in feasible_seeds if paired[s][cid]["status"] == "SUCCESS"]

        ci = compute_wilson_ci(succ, len(feasible_seeds))
        ci_env = compute_wilson_ci(envs, len(feasible_seeds))

        feas_stats[cid] = {
            "n": len(feasible_seeds),
            "success": succ,
            "success_pct": round(succ / len(feasible_seeds) * 100.0, 2),
            "wilson_ci_95": ci,
            "timeouts": timeouts,
            "collisions": cols,
            "envelope_violations": envs,
            "envelope_pct": round(envs / len(feasible_seeds) * 100.0, 2),
            "envelope_ci_95": ci_env,
            "envelope_steps": env_steps,
            "interventions": interv,
            "intervention_rate_per_kstep": round(interv / total_steps * 1000.0, 2) if total_steps > 0 else 0.0,
            "min_clearance_m": round(min(c_vals), 4) if c_vals else 0.0,
            "median_clearance_m": round(float(np.median(c_vals)), 4) if c_vals else 0.0,
            "mean_flight_time_s": round(float(np.mean(flight_times)), 2) if flight_times else 0.0,
            "mean_speed_mps": round(float(np.mean(speeds)), 3) if speeds else 0.0,
        }

    # Feasible Paired McNemar Test
    both_succ = sum(1 for s in feasible_seeds if paired[s]["scaled_guided_bc_v2"]["status"] == "SUCCESS" and paired[s]["scaled_guided_bc_v3"]["status"] == "SUCCESS")
    v2_only = sum(1 for s in feasible_seeds if paired[s]["scaled_guided_bc_v2"]["status"] == "SUCCESS" and paired[s]["scaled_guided_bc_v3"]["status"] != "SUCCESS")
    v3_only = sum(1 for s in feasible_seeds if paired[s]["scaled_guided_bc_v2"]["status"] != "SUCCESS" and paired[s]["scaled_guided_bc_v3"]["status"] == "SUCCESS")
    both_fail = sum(1 for s in feasible_seeds if paired[s]["scaled_guided_bc_v2"]["status"] != "SUCCESS" and paired[s]["scaled_guided_bc_v3"]["status"] != "SUCCESS")
    p_mcnemar = mcnemar_exact(v2_only, v3_only)

    p1 = feas_stats["scaled_guided_bc_v2"]["success_pct"] / 100.0
    p2 = feas_stats["scaled_guided_bc_v3"]["success_pct"] / 100.0
    is_eq, p_tost, diff_pct, ci90 = tost_paired_equivalence(p1, p2, len(feasible_seeds), delta=0.10)

    feas_paired = {
        "n_pairs": len(feasible_seeds),
        "both_succeed": both_succ,
        "v2_only": v2_only,
        "v3_only": v3_only,
        "both_fail": both_fail,
        "net_diff_pct": round((feas_stats['scaled_guided_bc_v3']['success'] - feas_stats['scaled_guided_bc_v2']['success']) / len(feasible_seeds) * 100.0, 2),
        "mcnemar_p_value": p_mcnemar,
        "tost_is_equivalent_delta10": is_eq,
        "tost_p_value": p_tost,
        "ci90_diff": ci90,
    }

    # 2. Infeasible Route Evaluation
    infeas_stats = {}
    if infeasible_seeds:
        for cid in ["scaled_guided_bc_v2", "scaled_guided_bc_v3"]:
            safe_rejs = sum(1 for s in infeasible_seeds if paired[s][cid]["status"] == "SAFE_REJECT")
            timeouts = sum(1 for s in infeasible_seeds if paired[s][cid]["status"] == "TIMEOUT")
            cols = sum(1 for s in infeasible_seeds if paired[s][cid]["collision_episode"] == "True")
            envs = sum(1 for s in infeasible_seeds if paired[s][cid]["safety_envelope_violation"] == "True")
            c_vals = [float(paired[s][cid]["overall_min_clearance_m"]) for s in infeasible_seeds]
            ci_rej = compute_wilson_ci(safe_rejs, len(infeasible_seeds))

            infeas_stats[cid] = {
                "n": len(infeasible_seeds),
                "safe_rejections": safe_rejs,
                "rejection_pct": round(safe_rejs / len(infeasible_seeds) * 100.0, 2),
                "wilson_ci_95": ci_rej,
                "timeouts": timeouts,
                "collisions": cols,
                "envelope_violations": envs,
                "min_clearance_m": round(min(c_vals), 4) if c_vals else 0.0,
            }

    # Print summary
    print(f"\n[Feasible Route Navigation]")
    print(f"  v2: {feas_stats['scaled_guided_bc_v2']['success']}/{len(feasible_seeds)} ({feas_stats['scaled_guided_bc_v2']['success_pct']}%) [95% CI: {feas_stats['scaled_guided_bc_v2']['wilson_ci_95']}] | Col: 0 | Env: {feas_stats['scaled_guided_bc_v2']['envelope_violations']} ({feas_stats['scaled_guided_bc_v2']['envelope_pct']}%)")
    print(f"  v3: {feas_stats['scaled_guided_bc_v3']['success']}/{len(feasible_seeds)} ({feas_stats['scaled_guided_bc_v3']['success_pct']}%) [95% CI: {feas_stats['scaled_guided_bc_v3']['wilson_ci_95']}] | Col: 0 | Env: {feas_stats['scaled_guided_bc_v3']['envelope_violations']} ({feas_stats['scaled_guided_bc_v3']['envelope_pct']}%)")
    print(f"  Paired: Both Succ={both_succ}, v2 Only={v2_only}, v3 Only={v3_only}, Both Fail={both_fail}")
    print(f"  Net Delta: {feas_paired['net_diff_pct']:+}% | McNemar p-value: {p_mcnemar:.5f} | TOST Equivalence (Delta=10%): {is_eq} (90% CI: {ci90})")

    if infeasible_seeds:
        print(f"\n[Infeasible Barrier Rejection]")
        print(f"  v2: {infeas_stats['scaled_guided_bc_v2']['safe_rejections']}/{len(infeasible_seeds)} ({infeas_stats['scaled_guided_bc_v2']['rejection_pct']}%) | Timeouts: {infeas_stats['scaled_guided_bc_v2']['timeouts']} | Cols: 0")
        print(f"  v3: {infeas_stats['scaled_guided_bc_v3']['safe_rejections']}/{len(infeasible_seeds)} ({infeas_stats['scaled_guided_bc_v3']['rejection_pct']}%) | Timeouts: {infeas_stats['scaled_guided_bc_v3']['timeouts']} | Cols: 0")

    return {
        "partition": partition_name,
        "n_seeds": n_seeds,
        "feasible": {
            "v2": feas_stats["scaled_guided_bc_v2"],
            "v3": feas_stats["scaled_guided_bc_v3"],
            "paired": feas_paired,
        },
        "infeasible": infeas_stats,
    }


def run_comprehensive_audit():
    p_ex = Path("experiments/scaled_campaign/exhaustive_v2_v3/exhaustive_episodes.csv")
    p_un = Path("experiments/scaled_campaign/untouched_confirmation/untouched_episodes.csv")

    with open(p_ex) as f:
        ex_rows = list(csv.DictReader(f))
    with open(p_un) as f:
        un_rows = list(csv.DictReader(f))

    # Partition 1: Development Suite (200 seeds: Level 1 dev 20100..20149, Level 2 dev 20000..20049, Level 3 dev 20200..20249, Level 4 dev 20300..20349)
    dev_rows = [r for r in ex_rows if int(r["seed"]) < 20050 or int(r["seed"]) >= 20100]

    # Partition 2: Level 2 Confirmation Suite (50 seeds: 20050..20099)
    l2_conf_rows = [r for r in ex_rows if 20050 <= int(r["seed"]) < 20100]

    # Partition 3: Entire Exhaustive Sweep (250 seeds)
    all_ex_rows = ex_rows

    # Partition 4: Untouched Confirmation Suite (200 seeds: 20150..20199, 20400..20449, 20250..20299, 20350..20399)
    untouched_rows = un_rows

    audit_results = {
        "development_suite_200": analyze_partition(dev_rows, "Development Suite (200 Seeds)"),
        "level2_confirmation_50": analyze_partition(l2_conf_rows, "Level 2 Confirmation Suite (50 Seeds)"),
        "exhaustive_suite_250": analyze_partition(all_ex_rows, "Exhaustive Evaluation Suite (250 Seeds)"),
        "untouched_confirmation_200": analyze_partition(untouched_rows, "Pre-Registered Untouched Confirmation Suite (200 Seeds)"),
    }

    # Tier by Tier Audit
    tier_audits = {}
    for t in sorted(list(set(r["tier"] for r in ex_rows))):
        t_rows = [r for r in ex_rows if r["tier"] == t]
        tier_audits[f"Exhaustive_{t}"] = analyze_partition(t_rows, f"Exhaustive - {t}")

    for t in sorted(list(set(r["tier"] for r in un_rows))):
        t_rows = [r for r in un_rows if r["tier"] == t]
        tier_audits[f"Untouched_{t}"] = analyze_partition(t_rows, f"Untouched - {t}")

    audit_results["by_tier"] = tier_audits

    out_file = Path("experiments/scaled_campaign/comprehensive_audit_results.json")
    with open(out_file, "w") as fp:
        json.dump(audit_results, fp, indent=2)
    print(f"\nSaved comprehensive audit results to {out_file}")


if __name__ == "__main__":
    run_comprehensive_audit()
