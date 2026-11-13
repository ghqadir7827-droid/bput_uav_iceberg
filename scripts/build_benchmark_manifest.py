"""build_benchmark_manifest.py - Freezes Benchmark Checkpoints and Source Code Hashes.

Generates data/benchmark_manifest_v2_v3.json containing SHA-256 checksums and parameter manifests.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any

from core.guidance import GuidanceConfig
from controllers.safety_supervisor import SafetySupervisorConfig


def compute_sha256(filepath: str) -> str:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {filepath}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    print("=" * 80)
    print("PHASE 2: BUILDING BENCHMARK MANIFEST")
    print("=" * 80)

    tracked_files = [
        "models/checkpoints/bc_policy_scaled_v2.pt",
        "controllers/bc_policy.py",
        "controllers/safety_supervisor.py",
        "controllers/registry.py",
        "core/guidance.py",
        "core/contracts.py",
        "core/geometry.py",
        "core/physical_dimensions.py",
        "sim/corridor_env.py",
        "scripts/run_exhaustive_v2_vs_v3_evaluation.py",
        "scripts/run_untouched_confirmation_sweep.py",
        "scripts/analyze_exhaustive_sweep_statistics.py",
        "scripts/audit_existing_results.py",
        "tests/test_regression_harness_integrity.py",
    ]

    file_hashes: Dict[str, str] = {}
    for f in tracked_files:
        h = compute_sha256(f)
        file_hashes[f] = h
        print(f"  {f:<48} : {h}")

    # Verify Checkpoint SHA-256 matches frozen baseline
    expected_ckpt = "d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1"
    assert file_hashes["models/checkpoints/bc_policy_scaled_v2.pt"] == expected_ckpt, "Checkpoint SHA-256 altered!"

    v2_guidance = GuidanceConfig.for_65cm_uav_v2()
    v3_guidance = GuidanceConfig.for_65cm_uav_v3()
    sup_config = SafetySupervisorConfig.for_65cm_uav(drone_radius=0.325)

    manifest: Dict[str, Any] = {
        "benchmark_title": "Frozen UAV Navigation Benchmark: scaled_guided_bc_v2 vs scaled_guided_bc_v3",
        "timestamp_iso": "2026-10-31T01:30:00Z",
        "physical_drone_radius_m": 0.325,
        "physical_drone_diagonal_m": 0.65,
        "shared_model_checkpoint": {
            "path": "models/checkpoints/bc_policy_scaled_v2.pt",
            "sha256": file_hashes["models/checkpoints/bc_policy_scaled_v2.pt"],
            "observation_dim": 23,
            "action_dim": 4,
            "policy_architecture": "MLP [128, 128] ELU",
        },
        "guidance_configurations": {
            "scaled_guided_bc_v2": {
                "activation_threshold_m": v2_guidance.activation_threshold_m,
                "min_lateral_override_mps": v2_guidance.min_lateral_override_mps,
                "persistent_tie_breaking": v2_guidance.persistent_tie_breaking,
                "deadband_m": v2_guidance.deadband_m,
                "hysteresis_m": v2_guidance.hysteresis_m,
                "default_tie_breaker": v2_guidance.default_tie_breaker,
                "forward_ray_indices": list(v2_guidance.forward_ray_indices),
            },
            "scaled_guided_bc_v3": {
                "activation_threshold_m": v3_guidance.activation_threshold_m,
                "min_lateral_override_mps": v3_guidance.min_lateral_override_mps,
                "persistent_tie_breaking": v3_guidance.persistent_tie_breaking,
                "deadband_m": v3_guidance.deadband_m,
                "hysteresis_m": v3_guidance.hysteresis_m,
                "default_tie_breaker": v3_guidance.default_tie_breaker,
                "forward_ray_indices": list(v3_guidance.forward_ray_indices),
            },
        },
        "safety_supervisor_configuration": {
            "boundary_buffer_m": sup_config.boundary_buffer_m,
            "critical_obstacle_dist_m": sup_config.critical_obstacle_dist_m,
            "emergency_reverse_dist_m": sup_config.emergency_reverse_dist_m,
            "obstacle_caution_dist_m": sup_config.obstacle_caution_dist_m,
            "critical_lateral_dist_m": sup_config.critical_lateral_dist_m,
            "caution_lateral_dist_m": sup_config.caution_lateral_dist_m,
            "repulsion_gain": sup_config.repulsion_gain,
            "deceleration_limit_mps2": sup_config.deceleration_limit_mps2,
        },
        "source_file_hashes": file_hashes,
    }

    out_json = Path("data/benchmark_manifest_v2_v3.json")
    with open(out_json, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nSaved benchmark manifest to {out_json}")


if __name__ == "__main__":
    build_manifest()
