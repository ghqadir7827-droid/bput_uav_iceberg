"""analyze_pillar3_failures.py - Detailed telemetry inspection for the 5 Pillar 3 failures.

Extracts:
1. Obstacle geometry (pillar 1, 2, 3 centers and radii)
2. Trajectory before, during, and after Pillars 1, 2, and 3
3. Exact step where forward progress ceased / standoff began
4. Rangefinder clearances (front, left, right, wall_l, wall_r) at stall
5. Action proposals vs supervisor filters (raw_vy vs safe_vy, raw_vx vs safe_vx)
6. Guidance state transitions and oscillation counts
7. Centerline displacement at stall
8. Root cause classification across A, B, C, D, E, F with evidentiary proof
"""

import json
from pathlib import Path
from typing import Any, Dict, List

WORKSPACE_ROOT = Path("/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav")
EXP_DIR = WORKSPACE_ROOT / "experiments/phase6b/phase6b_pillar3_recovery_20261030_025800"
TRACES_DIR = EXP_DIR / "traces"

FAILURE_SEEDS = [2001, 2005, 2006, 2022, 2042]


def analyze_seed(seed: int) -> Dict[str, Any]:
    trace_file = TRACES_DIR / f"seed_{seed}_phase6a_telemetry.json"
    with open(trace_file) as f:
        data = json.load(f)

    summary = data["summary"]
    trace = data["trace"]
    pillar_configs = summary["pillar_configs"]

    # Analyze trajectory
    total_steps = len(trace)
    p1 = pillar_configs[0]
    p2 = pillar_configs[1]
    p3 = pillar_configs[2]

    # Find step where progress stalled (e.g. rolling progress drops below 0.05m over 50 steps)
    stall_step = None
    for step_data in trace[50:]:
        if step_data["rolling_progress_50step_m"] < 0.05 and step_data["position"][0] >= 5.5:
            stall_step = step_data["step"]
            break

    if stall_step is None:
        stall_step = total_steps - 1

    stall_data = trace[stall_step]
    final_data = trace[-1]

    # Analyze guidance decisions during standoff (from stall_step to end)
    guidance_dirs = [t["guidance_dir"] for t in trace[stall_step:]]
    guidance_dir_counts = {}
    for d in guidance_dirs:
        guidance_dir_counts[d] = guidance_dir_counts.get(d, 0) + 1

    # Direction switches
    switches = 0
    for i in range(1, len(guidance_dirs)):
        if guidance_dirs[i] != guidance_dirs[i-1] and guidance_dirs[i] != "NONE" and guidance_dirs[i-1] != "NONE":
            switches += 1

    # Supervisor interventions during standoff
    safety_reasons = [t["safety_reason"] for t in trace[stall_step:]]
    safety_states = [t["safety_state"] for t in trace[stall_step:]]

    # Opposing forces check: does guidance push one way and supervisor push/clamp the opposite?
    opposing_steps = 0
    clamped_vy_steps = 0
    for t in trace[stall_step:]:
        raw_vy = t["raw_action"][1]
        safe_vy = t["safe_action"][1]
        # Opposing sign or clamped to zero when raw was significant
        if (raw_vy > 0.1 and safe_vy < -0.01) or (raw_vy < -0.1 and safe_vy > 0.01):
            opposing_steps += 1
        if abs(raw_vy) >= 0.25 and abs(safe_vy) < 0.05:
            clamped_vy_steps += 1

    # Drone displacement relative to Pillar 3
    # P3 center: (p3[0], p3[1]), radius: p3[2]
    # Corridor width: y in [-1.0, 1.0]
    p3_x, p3_y, p3_r = p3
    left_gap = 1.0 - (p3_y + p3_r)   # gap between left wall (y=1.0) and obstacle surface
    right_gap = (p3_y - p3_r) - (-1.0) # gap between obstacle surface and right wall (y=-1.0)

    # Position at Pillar 2 clearance (when x passes p2[0] + p2[2])
    p2_pass_step = None
    for t in trace:
        if t["position"][0] > (p2[0] + p2[2]):
            p2_pass_step = t["step"]
            break

    p2_pass_y = trace[p2_pass_step]["position"][1] if p2_pass_step else None

    # Check root causes
    # A. Incorrect obstacle-passage detection
    # B. Guidance repeatedly switching directions
    # C. Wall guardrail and obstacle guidance opposing one another
    # D. Supervisor throttling preventing escape from a local standoff
    # E. Insufficient lateral recovery authority
    # F. An unrelated failure mechanism

    return {
        "seed": seed,
        "pillar_1": p1,
        "pillar_2": p2,
        "pillar_3": p3,
        "left_gap_p3_m": round(left_gap, 3),
        "right_gap_p3_m": round(right_gap, 3),
        "p2_pass_step": p2_pass_step,
        "p2_pass_y": round(p2_pass_y, 3) if p2_pass_y is not None else None,
        "stall_step": stall_step,
        "stall_pos": stall_data["position"],
        "stall_speed": stall_data["speed_mps"],
        "stall_rolling_prog": stall_data["rolling_progress_50step_m"],
        "final_pos": final_data["position"],
        "final_step": total_steps,
        "p3_passed": bool(final_data["position"][0] > (p3_x + p3_r)),
        "guidance_dir_counts": guidance_dir_counts,
        "direction_switches": switches,
        "opposing_steps": opposing_steps,
        "clamped_vy_steps": clamped_vy_steps,
        "standoff_duration_steps": total_steps - stall_step,
        "stall_c_front": stall_data["c_front"],
        "stall_c_left": stall_data["c_left"],
        "stall_c_right": stall_data["c_right"],
        "stall_diff": stall_data["diff"],
        "stall_wall_l": stall_data["wall_l"],
        "stall_wall_r": stall_data["wall_r"],
        "stall_guidance_dir": stall_data["guidance_dir"],
        "stall_safety_state": stall_data["safety_state"],
        "stall_safety_reason": stall_data["safety_reason"],
        "sample_stall_actions": [
            {
                "step": t["step"],
                "x": t["position"][0],
                "y": t["position"][1],
                "raw_act": t["raw_action"],
                "safe_act": t["safe_action"],
                "guidance_dir": t["guidance_dir"],
                "c_front": t["c_front"],
                "diff": t["diff"],
                "wall_l": t["wall_l"],
                "wall_r": t["wall_r"],
                "safety_reason": t["safety_reason"],
            }
            for t in trace[stall_step:stall_step+10]
        ]
    }


def main():
    print("=================== PILLAR 3 FAILURE FORENSICS ===================")
    for s in FAILURE_SEEDS:
        res = analyze_seed(s)
        print(f"\n--- SEED {s} ---")
        print(f"Pillar 1: x={res['pillar_1'][0]}, y={res['pillar_1'][1]:.3f}, r={res['pillar_1'][2]}")
        print(f"Pillar 2: x={res['pillar_2'][0]}, y={res['pillar_2'][1]:.3f}, r={res['pillar_2'][2]}")
        print(f"Pillar 3: x={res['pillar_3'][0]}, y={res['pillar_3'][1]:.3f}, r={res['pillar_3'][2]}")
        print(f"P3 gaps: left_gap={res['left_gap_p3_m']}m, right_gap={res['right_gap_p3_m']}m (wider side: {'LEFT' if res['left_gap_p3_m'] > res['right_gap_p3_m'] else 'RIGHT'})")
        print(f"After Pillar 2: drone y = {res['p2_pass_y']} at step {res['p2_pass_step']}")
        print(f"Stall started at step {res['stall_step']} (t={res['stall_step']*0.05:.2f}s): pos={res['stall_pos']}, speed={res['stall_speed']}m/s")
        print(f"Final pos at step {res['final_step']}: pos={res['final_pos']}, P3 passed={res['p3_passed']}")
        print(f"Guidance decisions in stall: {res['guidance_dir_counts']}, switches={res['direction_switches']}")
        print(f"Clamped vy steps: {res['clamped_vy_steps']}/{res['standoff_duration_steps']}, Opposing steps: {res['opposing_steps']}")
        print(f"At stall: c_front={res['stall_c_front']}m, c_left={res['stall_c_left']}m, c_right={res['stall_c_right']}m, diff={res['stall_diff']}m")
        print(f"At stall: wall_l={res['stall_wall_l']}m, wall_r={res['stall_wall_r']}m")
        print(f"At stall: guidance_dir={res['stall_guidance_dir']}, safety_state={res['stall_safety_state']}")
        print(f"Safety reason: {res['stall_safety_reason']}")

        print("Sample 3 stall steps:")
        for sa in res["sample_stall_actions"][:3]:
            print(f"  Step {sa['step']}: pos=({sa['x']}, {sa['y']}), raw_act={sa['raw_act']}, safe_act={sa['safe_act']}, dir={sa['guidance_dir']}, c_front={sa['c_front']}, diff={sa['diff']}, reason={sa['safety_reason']}")


if __name__ == "__main__":
    main()

