"""test_candidate_on_failures.py - Quick test of Phase 6B candidate on failure seeds."""

import numpy as np
import pybullet as p
from pathlib import Path

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.geometry import Corridor3D
from sim.corridor_env import CorridorSimEnv
from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.diagnose_phase6a_failures import run_episode
from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.pillar3_recovery_controller import PostEvasionRecoveryController

WORKSPACE_ROOT = Path("/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav")
FAILURE_SEEDS = [2001, 2005, 2006, 2022, 2042]

def main():
    corridor = Corridor3D.get_10m_benchmark()
    env = CorridorSimEnv(corridor=corridor)
    sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
    supervisor = SafetySupervisor(corridor=corridor, limits=env.action_limits, config=sup_config)

    controller = PostEvasionRecoveryController(
        base_model=str(WORKSPACE_ROOT / "models/checkpoints/bc_policy_best.pt"),
    )

    print("=== Testing Phase 6B Candidate on the 5 Known Failure Seeds ===")
    for s in FAILURE_SEEDS:
        summary, trace = run_episode(env, supervisor, controller, seed=s, record_trace=False)
        print(f"Seed {s}: outcome={summary['outcome']}, success={summary['success']}, steps={summary['steps']}, prog={summary['progress_m']}m, min_obs={summary['min_obs_clearance_m']}m, min_wall={summary['min_wall_clearance_m']}m, p_cleared={summary['pillars_cleared']}")

    env.close()

if __name__ == "__main__":
    main()

