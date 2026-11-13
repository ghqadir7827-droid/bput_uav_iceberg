"""compile_phase4g_training_metrics.py - Consolidates Arm C and Arm D training logs.

Generates:
- reports/phase4g_training_metrics.json
"""

import hashlib
import json
from pathlib import Path


def get_sha256(filepath: str) -> str:
    p = Path(filepath)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compile_training_metrics():
    log_c_path = Path("experiments/phase4g/logs/arm_c_training_log.json")
    log_d_path = Path("experiments/phase4g/logs/arm_d_training_log.json")

    with open(log_c_path) as f:
        log_c = json.load(f)
    with open(log_d_path) as f:
        log_d = json.load(f)

    metrics = {
        "metadata": {
            "scenario": "10m",
            "step_budget_per_arm": 20480,
            "iterations_per_arm": 10,
            "rollout_steps_per_iter": 2048,
            "batch_size": 128,
            "ppo_epochs": 4,
            "learning_rate": 3e-4,
            "clip_eps": 0.20,
            "entropy_coef": 0.005,
            "value_coef": 0.50,
            "supervisor_active": True,
            "starting_checkpoint": "models/checkpoints/ppo_policy_best.pt",
            "starting_checkpoint_hash": get_sha256("models/checkpoints/ppo_policy_best.pt"),
        },
        "arm_c_standard_ppo": {
            "checkpoint": "experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt",
            "checkpoint_hash": get_sha256("experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt"),
            "reflection_augmentation": False,
            "pre_training_open_space_vy": log_c["pre_training_symmetry"]["open_space_vy"],
            "pre_training_open_space_yaw_rate": log_c["pre_training_symmetry"]["open_space_yaw_rate"],
            "final_open_space_vy": log_c["final_symmetry"]["open_space_vy"],
            "final_open_space_yaw_rate": log_c["final_symmetry"]["open_space_yaw_rate"],
            "final_val_success_rate_pct": log_c["final_validation"]["success_rate_pct"],
            "history": log_c["history"],
            "training_time_s": log_c["total_training_time_s"],
        },
        "arm_d_reflection_ppo": {
            "checkpoint": "experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt",
            "checkpoint_hash": get_sha256("experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt"),
            "reflection_augmentation": True,
            "pre_training_open_space_vy": log_d["pre_training_symmetry"]["open_space_vy"],
            "pre_training_open_space_yaw_rate": log_d["pre_training_symmetry"]["open_space_yaw_rate"],
            "final_open_space_vy": log_d["final_symmetry"]["open_space_vy"],
            "final_open_space_yaw_rate": log_d["final_symmetry"]["open_space_yaw_rate"],
            "final_val_success_rate_pct": log_d["final_validation"]["success_rate_pct"],
            "history": log_d["history"],
            "training_time_s": log_d["total_training_time_s"],
        },
        "comparison_summary": {
            "yaw_rate_bias_reduction_pct": round(
                (1.0 - abs(log_d["final_symmetry"]["open_space_yaw_rate"]) / abs(log_c["final_symmetry"]["open_space_yaw_rate"])) * 100.0, 2
            ),
            "lateral_velocity_arm_c": log_c["final_symmetry"]["open_space_vy"],
            "lateral_velocity_arm_d": log_d["final_symmetry"]["open_space_vy"],
        }
    }

    out_path = Path("reports/phase4g_training_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics compiled into {out_path}")


if __name__ == "__main__":
    compile_training_metrics()
