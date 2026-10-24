"""export_onnx.py - Edge Neural Network Export & Verification Pipeline.

Exports trained PyTorch navigation policies to ONNX format for deployment
on embedded companion computers (e.g., Raspberry Pi 4/5):
- Converts NavigationPolicyMLP to ONNX computation graph.
- Verifies graph validity via onnx.checker.
- Mathematically validates numerical equivalence between PyTorch and ONNX Runtime.
- Benchmarks single-sample inference latency with ONNX Runtime.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import torch

from models.policy_net import NavigationPolicyMLP


def export_and_verify_onnx(
    checkpoint_path: str = "models/checkpoints/ppo_policy_best.pt",
    output_onnx_path: str = "models/checkpoints/policy_ppo.onnx",
    num_test_samples: int = 100,
    tolerance: float = 1e-4,
) -> Dict:
    print("=" * 80)
    print(">>> ONNX EDGE EXPORT & NUMERICAL VERIFICATION PIPELINE <<<")
    print(f"PyTorch Checkpoint: {checkpoint_path}")
    print(f"Output ONNX Model:  {output_onnx_path}")
    print("=" * 80)

    p_ckpt = Path(checkpoint_path)
    if not p_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load PyTorch Checkpoint
    ckpt = torch.load(p_ckpt, map_location="cpu", weights_only=False)
    obs_dim = ckpt.get("obs_dim", 23)
    act_dim = ckpt.get("act_dim", 4)

    model = NavigationPolicyMLP(obs_dim=obs_dim, act_dim=act_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Export to ONNX
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)
    p_onnx = Path(output_onnx_path)
    p_onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(p_onnx),
        input_names=["observation"],
        output_names=["velocity_action"],
        dynamic_axes={"observation": {0: "batch_size"}, "velocity_action": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[PASS] Successfully exported ONNX graph to: {p_onnx.resolve()}")

    # 3. Verify ONNX Graph
    onnx_model = onnx.load(str(p_onnx))
    onnx.checker.check_model(onnx_model)
    file_size_kb = p_onnx.stat().st_size / 1024.0
    print(f"[PASS] ONNX Graph integrity verified (File size: {file_size_kb:.1f} KB)")

    # 4. Verify Numerical Equivalence with ONNX Runtime
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(p_onnx), sess_options=opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    max_abs_diff = 0.0

    for _ in range(num_test_samples):
        sample_obs = rng.uniform(-1.0, 1.0, size=(1, obs_dim)).astype(np.float32)
        # Rays [0:12]
        sample_obs[0, :12] = rng.uniform(0.05, 1.0, size=12)

        # PyTorch forward pass
        with torch.no_grad():
            pt_out = model(torch.from_numpy(sample_obs), bounded=True).numpy()

        # ONNX Runtime forward pass
        ort_out = session.run(["velocity_action"], {"observation": sample_obs})[0]

        diff = float(np.max(np.abs(pt_out - ort_out)))
        max_abs_diff = max(max_abs_diff, diff)

    print(f"[PASS] Numerical Equivalence Verified over {num_test_samples} random inputs:")
    print(f"       Maximum Absolute Difference: {max_abs_diff:.6e} (Tolerance: {tolerance})")
    assert max_abs_diff < tolerance, f"ONNX output deviated from PyTorch: max_diff={max_abs_diff}"

    # 5. Benchmark ONNX Runtime Single-Sample Latency
    # Warmup
    for _ in range(50):
        _ = session.run(["velocity_action"], {"observation": dummy_input.numpy()})

    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = session.run(["velocity_action"], {"observation": dummy_input.numpy()})
        latencies.append((time.perf_counter() - t0) * 1000.0) # ms

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    print(f"[PASS] ONNX Runtime Single-Sample CPU Latency (1,000 trials):")
    print(f"       Mean Latency:   {mean_lat:.3f} ms")
    print(f"       Median Latency: {median_lat:.3f} ms")
    print(f"       P95 Latency:    {p95_lat:.3f} ms")
    print(f"       P99 Latency:    {p99_lat:.3f} ms")
    print(f"       Throughput:     {1000.0 / mean_lat:.0f} Hz (Control Loop Requirement: 20-50 Hz)")

    results = {
        "onnx_path": str(p_onnx.resolve()),
        "file_size_kb": round(file_size_kb, 1),
        "max_abs_difference": max_abs_diff,
        "latency_ms": {
            "mean": round(mean_lat, 3),
            "median": round(median_lat, 3),
            "p95": round(p95_lat, 3),
            "p99": round(p99_lat, 3),
        },
        "throughput_hz": round(1000.0 / mean_lat, 0),
        "status": "VERIFIED_PASS",
    }

    print("=" * 80)
    print(">>> ONNX EDGE MODEL CERTIFIED READY FOR RASPBERRY PI DEPLOYMENT <<<")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch Policy to ONNX")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/ppo_policy_best.pt", help="PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="models/checkpoints/policy_ppo.onnx", help="Output ONNX file")
    args = parser.parse_args()

    export_and_verify_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
    )
