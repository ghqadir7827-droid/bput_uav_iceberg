"""onnx_runner.py - Embedded ONNX Runtime Inference Engine for Drone Companion Computer.

Runs high-frequency (50 Hz) standalone inference on edge companion hardware (e.g. Raspberry Pi):
- Zero PyTorch dependency: runs purely on lightweight ONNX Runtime + NumPy.
- Inputs: 23-float DeployableObservation from multi-beam rangefinders, IMU, and localization.
- Outputs: Continuous VelocityAction [vx, vy, vz, yaw_rate] in Body FLU frame.
- Integrated safety supervisor validation and hardware command abstraction.
"""

from pathlib import Path
import time
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort

from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D


class OnboardONNXPolicyRunner:
    """Standalone edge policy runner for Raspberry Pi companion computer."""

    def __init__(
        self,
        onnx_model_path: str = "models/checkpoints/policy_ppo.onnx",
        corridor: Optional[Corridor3D] = None,
        intra_threads: int = 1,
    ):
        self.onnx_path = Path(onnx_model_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {onnx_model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.corridor = corridor or Corridor3D.get_10m_benchmark()
        self.safety_supervisor = SafetySupervisor(corridor=self.corridor)
        self.action_limits = ActionLimits()

    def predict(self, observation: np.ndarray) -> VelocityAction:
        """Runs fast single-sample ONNX inference."""
        obs_input = np.ascontiguousarray(observation, dtype=np.float32)
        if obs_input.ndim == 1:
            obs_input = np.expand_dims(obs_input, axis=0)

        # Run ONNX session
        raw_output = self.session.run([self.output_name], {self.input_name: obs_input})[0]
        act = raw_output[0]

        return VelocityAction(
            vx=float(np.clip(act[0], -0.20, 0.80)),
            vy=float(np.clip(act[1], -0.60, 0.60)),
            vz=float(np.clip(act[2], -0.40, 0.40)),
            yaw_rate=float(np.clip(act[3], -0.60, 0.60)),
        )

    def step_onboard_control(
        self,
        deployable_obs: DeployableObservation,
        estimated_position: np.ndarray,
        estimated_velocity: np.ndarray,
        dt: float = 0.05,
    ) -> Tuple[VelocityAction, SafetyState, str]:
        """Complete edge control loop execution: ONNX inference + safety supervisor filter."""
        obs_vec = deployable_obs.to_vector()
        raw_action = self.predict(obs_vec)

        # Min front distance from forward rangefinder rays (rays 0, 1, 2) in meters
        front_dist = float(min(deployable_obs.distance_rays[0:3])) * 4.0

        # Filter action through Safety Supervisor
        safe_action, s_state, s_reason = self.safety_supervisor.filter_action(
            proposed_action=raw_action,
            current_position=estimated_position,
            current_velocity=estimated_velocity,
            min_front_range=front_dist,
            dt=dt,
            is_localization_healthy=True,
            telemetry_age_s=0.01,
        )

        return safe_action, s_state, s_reason
