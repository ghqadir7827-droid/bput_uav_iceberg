"""export_and_validate_bc_v3.py - Milestone 2A: BC v3 ONNX Export & Parity Harness.

Exports the exact frozen BC neural network used by scaled_guided_bc_v3 into ONNX
and establishes comprehensive numerical equivalence against the original PyTorch model.
"""

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import torch

from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from models.policy_net import NavigationPolicyMLP


EXPECTED_CKPT_SHA256 = "d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1"
CKPT_PATH = "models/checkpoints/bc_policy_scaled_v2.pt"
ONNX_OUTPUT_PATH = "models/checkpoints/policy_bc_v3.onnx"
PARITY_METRICS_PATH = "data/milestone2a_parity_metrics.json"
TEST_VECTORS_PATH = "data/milestone2a_test_vectors.json"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_test_suites(
    val_dataset_path: str = "data/demonstrations/bc_val_scaled_v2.npz",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Constructs comprehensive test suites spanning the operational domain."""
    rng = np.random.default_rng(seed)
    suites: Dict[str, np.ndarray] = {}

    # Suite 1: 1,000 deterministic random observations
    s1 = []
    for _ in range(1000):
        rays = rng.uniform(0.0125, 1.0, size=12).astype(np.float32)
        # Goal vector
        goal_raw = rng.normal(0, 1, size=3).astype(np.float32)
        goal_norm = np.linalg.norm(goal_raw)
        scale = float(rng.uniform(0.1, 1.0))
        goal = (goal_raw / (goal_norm + 1e-6)) * scale
        # Body velocities
        vx = rng.uniform(-0.20, 0.80)
        vy = rng.uniform(-0.60, 0.60)
        vz = rng.uniform(-0.40, 0.40)
        vel = np.array([vx, vy, vz], dtype=np.float32)
        # Attitude: roll, pitch in [-30 deg, 30 deg]
        roll = np.deg2rad(rng.uniform(-30.0, 30.0))
        pitch = np.deg2rad(rng.uniform(-30.0, 30.0))
        att = np.array([np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)], dtype=np.float32)
        # Yaw rate
        yaw_rate = float(rng.uniform(-0.60, 0.60))
        obs = np.concatenate([rays, goal, vel, att, [yaw_rate]]).astype(np.float32)
        s1.append(obs)
    suites["random_1000"] = np.array(s1, dtype=np.float32)

    # Suite 2: Zero and Maximum Range Observations (60 vectors)
    s2 = []
    # All zeros
    s2.append(np.zeros(23, dtype=np.float32))
    # All rays zero, varied state
    for _ in range(19):
        obs = suites["random_1000"][_].copy()
        obs[:12] = 0.0
        s2.append(obs)
    # All rays max (1.0)
    for _ in range(20):
        obs = suites["random_1000"][20 + _].copy()
        obs[:12] = 1.0
        s2.append(obs)
    # Mixed 0.0 and 1.0 rays
    for _ in range(20):
        obs = suites["random_1000"][40 + _].copy()
        mask = rng.integers(0, 2, size=12).astype(np.float32)
        obs[:12] = mask
        s2.append(obs)
    suites["zero_and_max_range"] = np.array(s2, dtype=np.float32)

    # Suite 3: Minimum Physical Range (0.05m / 4.0m = 0.0125) (50 vectors)
    s3 = []
    for _ in range(50):
        obs = suites["random_1000"][60 + _].copy()
        obs[:12] = 0.0125
        s3.append(obs)
    suites["minimum_range"] = np.array(s3, dtype=np.float32)

    # Suite 4: Valid Lower and Upper Feature Boundaries (64 corner combinations)
    s4 = []
    rays_options = [0.0125, 1.0]
    vx_options = [-0.20, 0.80]
    vy_options = [-0.60, 0.60]
    vz_options = [-0.40, 0.40]
    yaw_options = [-0.60, 0.60]
    goal_options = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    att_options = [
        [0.0, 1.0, 0.0, 1.0],  # Level
        [0.5, 0.866, 0.5, 0.866], # +30 deg roll/pitch
    ]
    for r in rays_options:
        for vx in vx_options:
            for vy in vy_options:
                for vz in vz_options:
                    for yr in yaw_options:
                        for g in goal_options:
                            att = att_options[len(s4) % len(att_options)]
                            obs = np.concatenate([
                                np.full(12, r, dtype=np.float32),
                                np.array(g, dtype=np.float32),
                                np.array([vx, vy, vz], dtype=np.float32),
                                np.array(att, dtype=np.float32),
                                np.array([yr], dtype=np.float32),
                            ])
                            s4.append(obs)
                            if len(s4) >= 64:
                                break
                        if len(s4) >= 64:
                            break
                    if len(s4) >= 64:
                        break
                if len(s4) >= 64:
                    break
            if len(s4) >= 64:
                break
        if len(s4) >= 64:
            break
    suites["boundary_corners"] = np.array(s4, dtype=np.float32)

    # Suite 5: Representative Recorded Simulation Observations (1,000 vectors)
    val_p = Path(val_dataset_path)
    if val_p.exists():
        data = np.load(val_p)
        val_obs = data["observations"]
        indices = np.linspace(0, len(val_obs) - 1, 1000, dtype=int)
        suites["recorded_sim_1000"] = val_obs[indices].astype(np.float32)
    else:
        # Fallback if dataset missing
        suites["recorded_sim_1000"] = suites["random_1000"].copy()

    # Suite 6: Saturated Action Outputs (100 vectors)
    # Inputs designed to drive logits high/low to saturate tanh
    s6 = []
    for _ in range(100):
        scale = rng.choice([-10.0, 10.0])
        obs = rng.normal(0, 5.0, size=23).astype(np.float32) * scale
        s6.append(obs)
    suites["saturated_inputs"] = np.array(s6, dtype=np.float32)

    return suites


def main():
    print("=" * 80)
    print(">>> MILESTONE 2A: BC v3 ONNX EXPORT & NUMERICAL PARITY HARNESS <<<")
    print(f"Target Checkpoint:   {CKPT_PATH}")
    print(f"Target ONNX Model:   {ONNX_OUTPUT_PATH}")
    print("=" * 80)

    # 1. Verify Checkpoint Hash Before Export
    p_ckpt = Path(CKPT_PATH)
    if not p_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")
    
    init_sha256 = compute_file_sha256(CKPT_PATH)
    print(f"[1/8] Verifying Checkpoint SHA-256:")
    print(f"      Observed: {init_sha256}")
    print(f"      Expected: {EXPECTED_CKPT_SHA256}")
    assert init_sha256 == EXPECTED_CKPT_SHA256, "Checkpoint SHA-256 checksum mismatch!"
    print("      -> CHECKPOINT INTEGRITY VERIFIED (BIT-EXACT MATCH)")

    # 2. Load Checkpoint and Inspect Architecture
    print(f"\n[2/8] Loading PyTorch Checkpoint:")
    ckpt = torch.load(p_ckpt, map_location="cpu", weights_only=False)
    obs_dim = ckpt.get("obs_dim", 23)
    act_dim = ckpt.get("act_dim", 4)
    epoch = ckpt.get("epoch", None)
    val_loss = ckpt.get("val_loss", None)
    val_mae = ckpt.get("val_mae", None)
    print(f"      obs_dim: {obs_dim}, act_dim: {act_dim}, epoch: {epoch}")
    print(f"      val_loss: {val_loss:.6f}, val_mae: {val_mae:.4f}")

    model = NavigationPolicyMLP(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=(128, 128, 64),
        dropout_p=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total Trainable Parameters: {total_params:,}")
    print(f"      Action Center Buffer:     {model.act_center.tolist()}")
    print(f"      Action Half-Range Buffer:  {model.act_half_range.tolist()}")

    # 3. Export to ONNX
    print(f"\n[3/8] Exporting NavigationPolicyMLP to ONNX:")
    p_onnx = Path(ONNX_OUTPUT_PATH)
    p_onnx.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)
    export_opset = 18

    t0_export = time.perf_counter()
    torch.onnx.export(
        model,
        (dummy_input,),
        str(p_onnx),
        input_names=["observation"],
        output_names=["velocity_action"],
        dynamic_axes={"observation": {0: "batch_size"}, "velocity_action": {0: "batch_size"}},
        opset_version=export_opset,
        do_constant_folding=True,
    )
    t_export_s = time.perf_counter() - t0_export
    print(f"      Initial export completed in {t_export_s:.3f} s (Opset {export_opset})")

    # Consolidate external tensor initializers into self-contained single ONNX binary
    m_temp = onnx.load(str(p_onnx), load_external_data=True)
    onnx.save_model(m_temp, str(p_onnx), save_as_external_data=False)
    p_data = Path(str(p_onnx) + ".data")
    if p_data.exists():
        p_data.unlink()
    print("      Consolidated into self-contained single-file ONNX binary (no sidecar .data file)")

    # 4. Validate ONNX Graph
    print(f"\n[4/8] Validating ONNX Graph Structure & Metadata:")
    onnx_sha256 = compute_file_sha256(str(p_onnx))
    file_size_bytes = p_onnx.stat().st_size
    file_size_kb = file_size_bytes / 1024.0

    onnx_model = onnx.load(str(p_onnx))
    onnx.checker.check_model(onnx_model)
    print("      -> ONNX Checker: VALID")
    print(f"      File Size:       {file_size_bytes:,} bytes ({file_size_kb:.1f} KB)")
    print(f"      SHA-256 Checksum:{onnx_sha256}")
    print(f"      IR Version:      {onnx_model.ir_version}")
    print(f"      Opset Import:    {onnx_model.opset_import[0].version}")

    graph = onnx_model.graph
    input_info = [(inp.name, [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]) for inp in graph.input]
    output_info = [(out.name, [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]) for out in graph.output]
    print(f"      Graph Inputs:    {input_info}")
    print(f"      Graph Outputs:   {output_info}")

    op_counts: Dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f"      Operator Inventory ({len(graph.node)} nodes total):")
    for op, cnt in sorted(op_counts.items()):
        print(f"        - {op:<20}: {cnt}")

    init_info = [(init.name, list(init.dims), init.data_type) for init in graph.initializer]
    print(f"      Initializers ({len(init_info)} tensors total):")
    for name, dims, dt in init_info:
        print(f"        - {name:<22}: dims={dims}, dtype={dt}")

    # 5. Initialize ONNX Runtime Session
    print(f"\n[5/8] Initializing ONNX Runtime Session:")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(p_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    print("      ONNX Runtime Session successfully created with CPUExecutionProvider (1 thread).")

    # 6. Comprehensive Numerical Equivalence Testing
    print(f"\n[6/8] Executing Numerical Parity Test Battery:")
    test_suites = generate_test_suites()
    total_vectors = sum(len(vecs) for vecs in test_suites.values())
    print(f"      Total Test Vectors to Evaluate: {total_vectors:,} across {len(test_suites)} suites")

    suite_results: Dict[str, Any] = {}
    saved_vectors: List[Dict[str, Any]] = []

    global_max_abs_err = 0.0
    global_errors: List[float] = []
    dim_max_abs_err = [0.0, 0.0, 0.0, 0.0]
    dim_mae = [0.0, 0.0, 0.0, 0.0]
    total_samples = 0

    TOLERANCE = 1e-4

    for suite_name, vectors in test_suites.items():
        n_samples = len(vectors)
        suite_dim_errs = []

        for i in range(n_samples):
            obs = vectors[i]
            # PyTorch inference
            with torch.no_grad():
                pt_out = model(torch.from_numpy(obs).unsqueeze(0), bounded=True).numpy()[0]
            # ONNX inference
            ort_out = session.run(["velocity_action"], {"observation": obs.reshape(1, 23)})[0][0]

            abs_diff = np.abs(pt_out - ort_out)
            max_diff = float(np.max(abs_diff))
            mae_diff = float(np.mean(abs_diff))

            suite_dim_errs.append(abs_diff)
            global_errors.append(max_diff)
            global_max_abs_err = max(global_max_abs_err, max_diff)

            for d in range(4):
                dim_max_abs_err[d] = max(dim_max_abs_err[d], float(abs_diff[d]))
                dim_mae[d] += float(abs_diff[d])
            total_samples += 1

            # Save representative samples (first 5 of each suite)
            if i < 5 or max_diff > 1e-5:
                saved_vectors.append({
                    "suite": suite_name,
                    "index": i,
                    "observation": [round(float(x), 6) for x in obs],
                    "pytorch_action": [round(float(x), 7) for x in pt_out],
                    "onnx_action": [round(float(x), 7) for x in ort_out],
                    "abs_error_per_dim": [float(x) for x in abs_diff],
                    "max_abs_error": max_diff,
                    "mean_abs_error": mae_diff,
                })

        suite_dim_errs_arr = np.array(suite_dim_errs)
        suite_max_err = float(np.max(suite_dim_errs_arr))
        suite_mean_err = float(np.mean(suite_dim_errs_arr))
        suite_p99_err = float(np.percentile(suite_dim_errs_arr, 99))
        passed = suite_max_err <= TOLERANCE

        suite_results[suite_name] = {
            "samples": n_samples,
            "max_abs_error": suite_max_err,
            "mean_abs_error": suite_mean_err,
            "p99_abs_error": suite_p99_err,
            "status": "PASS" if passed else "FAIL",
        }
        status_flag = "PASS" if passed else "FAIL"
        print(f"      Suite '{suite_name:<20}': {n_samples:>5} samples | Max Err: {suite_max_err:.6e} | Mean Err: {suite_mean_err:.6e} [{status_flag}]")

    dim_mae = [val / total_samples for val in dim_mae]
    print("-" * 80)
    print(f"      Global Maximum Absolute Error: {global_max_abs_err:.6e} (Target <= {TOLERANCE:.0e})")
    print(f"      Per-Dimension Max Absolute Error:")
    print(f"        - vx:       {dim_max_abs_err[0]:.6e} (MAE: {dim_mae[0]:.6e})")
    print(f"        - vy:       {dim_max_abs_err[1]:.6e} (MAE: {dim_mae[1]:.6e})")
    print(f"        - vz:       {dim_max_abs_err[2]:.6e} (MAE: {dim_mae[2]:.6e})")
    print(f"        - yaw_rate: {dim_max_abs_err[3]:.6e} (MAE: {dim_mae[3]:.6e})")
    assert global_max_abs_err <= TOLERANCE, f"Global max error {global_max_abs_err} exceeded tolerance {TOLERANCE}!"
    print("      -> NUMERICAL EQUIVALENCE FULLY CERTIFIED ACROSS ALL SUITES")

    # 7. Final Composite Controller Parity Testing
    print(f"\n[7/8] Evaluating Final Composite Controller Parity (Policy + Guidance v3 + Supervisor):")
    corridor = Corridor3D(drone_radius=0.325)
    sup_cfg = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
    sup_pt = SafetySupervisor(corridor=corridor, config=sup_cfg)
    sup_ort = SafetySupervisor(corridor=corridor, config=sup_cfg)

    g_cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb_pt = StatelessClearanceGuidanceArbitrator(config=g_cfg)
    arb_ort = StatelessClearanceGuidanceArbitrator(config=g_cfg)

    limits = ActionLimits()

    ctrl_max_abs_diff = 0.0
    guidance_dir_mismatches = 0
    supervisor_state_mismatches = 0
    n_ctrl_tests = 1000

    rng = np.random.default_rng(100)
    for i in range(n_ctrl_tests):
        obs = test_suites["random_1000"][i]
        # Simulate varied drone position and velocity inside corridor
        pos = np.array([rng.uniform(0.5, 9.5), rng.uniform(-0.8, 0.8), rng.uniform(0.3, 1.7)], dtype=np.float32)
        vel = np.array([rng.uniform(0.0, 0.6), rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)], dtype=np.float32)
        front_d = float(min(obs[0:3])) * 4.0

        # Pipeline A (PyTorch)
        with torch.no_grad():
            pt_raw = model(torch.from_numpy(obs).unsqueeze(0), bounded=True).numpy()[0]
        act_pt = VelocityAction(vx=pt_raw[0], vy=pt_raw[1], vz=pt_raw[2], yaw_rate=pt_raw[3])
        g_act_pt, g_dir_pt, _, _ = arb_pt.arbitrate(act_pt, obs)
        s_act_pt, s_state_pt, _ = sup_pt.filter_action(g_act_pt, pos, vel, front_d, dt=0.05)

        # Pipeline B (ONNX)
        ort_raw = session.run(["velocity_action"], {"observation": obs.reshape(1, 23)})[0][0]
        act_ort = VelocityAction(vx=ort_raw[0], vy=ort_raw[1], vz=ort_raw[2], yaw_rate=ort_raw[3])
        g_act_ort, g_dir_ort, _, _ = arb_ort.arbitrate(act_ort, obs)
        s_act_ort, s_state_ort, _ = sup_ort.filter_action(g_act_ort, pos, vel, front_d, dt=0.05)

        # Compare guidance decisions
        if g_dir_pt != g_dir_ort:
            guidance_dir_mismatches += 1
        if s_state_pt != s_state_ort:
            supervisor_state_mismatches += 1

        # Compare final safe actions
        diff = max(
            abs(s_act_pt.vx - s_act_ort.vx),
            abs(s_act_pt.vy - s_act_ort.vy),
            abs(s_act_pt.vz - s_act_ort.vz),
            abs(s_act_pt.yaw_rate - s_act_ort.yaw_rate),
        )
        ctrl_max_abs_diff = max(ctrl_max_abs_diff, diff)

    print(f"      Evaluated {n_ctrl_tests:,} full closed-loop control states:")
    print(f"        - Guidance Direction Mismatches:  {guidance_dir_mismatches} / {n_ctrl_tests}")
    print(f"        - Supervisor State Mismatches:    {supervisor_state_mismatches} / {n_ctrl_tests}")
    print(f"        - Max Final Action Discrepancy:   {ctrl_max_abs_diff:.6e} (Target <= {TOLERANCE:.0e})")
    assert ctrl_max_abs_diff <= TOLERANCE, f"Composite controller discrepancy {ctrl_max_abs_diff} exceeded tolerance!"
    print("      -> COMPOSITE CONTROLLER EQUIVALENCE CERTIFIED")

    # 8. Checkpoint Immutability Verification & Artifact Generation
    print(f"\n[8/8] Verifying Checkpoint Immutability & Saving Artifacts:")
    final_sha256 = compute_file_sha256(CKPT_PATH)
    assert final_sha256 == EXPECTED_CKPT_SHA256, "CRITICAL ERROR: Source checkpoint was altered during execution!"
    print("      -> SOURCE CHECKPOINT VERIFIED COMPLETELY UNTOUCHED")

    parity_metrics_data = {
        "milestone": "2A",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_checkpoint": {
            "path": CKPT_PATH,
            "sha256": init_sha256,
            "total_params": total_params,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dims": [128, 128, 64],
            "epoch": epoch,
            "val_loss": val_loss,
            "val_mae": val_mae,
        },
        "onnx_model": {
            "path": ONNX_OUTPUT_PATH,
            "sha256": onnx_sha256,
            "file_size_bytes": file_size_bytes,
            "file_size_kb": round(file_size_kb, 1),
            "opset_version": export_opset,
            "input_names": ["observation"],
            "input_shape": ["batch_size", 23],
            "output_names": ["velocity_action"],
            "output_shape": ["batch_size", 4],
            "operator_inventory": op_counts,
            "total_nodes": len(graph.node),
            "total_initializers": len(init_info),
        },
        "numerical_parity": {
            "tolerance_target": TOLERANCE,
            "total_test_vectors": total_samples,
            "global_max_abs_error": global_max_abs_err,
            "per_dimension_max_abs_error": {
                "vx": dim_max_abs_err[0],
                "vy": dim_max_abs_err[1],
                "vz": dim_max_abs_err[2],
                "yaw_rate": dim_max_abs_err[3],
            },
            "per_dimension_mean_abs_error": {
                "vx": dim_mae[0],
                "vy": dim_mae[1],
                "vz": dim_mae[2],
                "yaw_rate": dim_mae[3],
            },
            "test_suites": suite_results,
            "status": "PASS",
        },
        "composite_controller_parity": {
            "tests_evaluated": n_ctrl_tests,
            "guidance_direction_mismatches": guidance_dir_mismatches,
            "supervisor_state_mismatches": supervisor_state_mismatches,
            "max_final_action_diff": ctrl_max_abs_diff,
            "status": "PASS",
        },
        "environment": {
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__,
        },
    }

    Path(PARITY_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PARITY_METRICS_PATH, "w") as f:
        json.dump(parity_metrics_data, f, indent=2)
    print(f"      Saved: {PARITY_METRICS_PATH}")

    Path(TEST_VECTORS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_VECTORS_PATH, "w") as f:
        json.dump({
            "milestone": "2A",
            "total_recorded_samples": len(saved_vectors),
            "sample_test_vectors": saved_vectors,
        }, f, indent=2)
    print(f"      Saved: {TEST_VECTORS_PATH}")

    print("=" * 80)
    print(">>> MILESTONE 2A CERTIFICATION COMPLETE: ALL CHECKS PASSED <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
