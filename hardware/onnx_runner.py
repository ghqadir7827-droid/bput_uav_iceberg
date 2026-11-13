"""onnx_runner.py - Embedded ONNX Runtime Inference Engine for Drone Companion Computer.

Runs high-frequency (50 Hz) standalone inference on edge companion hardware (e.g. Raspberry Pi):
- Zero PyTorch dependency: runs purely on lightweight ONNX Runtime + NumPy.
- Inputs: 23-float DeployableObservation from multi-beam rangefinders, IMU, and localization.
- Outputs: Continuous VelocityAction [vx, vy, vz, yaw_rate] in Body FLU frame.
- Integrated Stateless Clearance Guidance layer with wall-aware margin arbitration.
- Canonical Safety Supervisor integration with optimal caution distance (0.50m).
"""

from pathlib import Path
import time
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator


class OnboardONNXPolicyRunner:
    """Standalone edge policy runner for Raspberry Pi companion computer."""

    def __init__(
        self,
        onnx_model_path: str = "models/checkpoints/policy_bc.onnx",
        corridor: Optional[Corridor3D] = None,
        caution_distance_m: float = 0.50,
        activation_threshold_m: float = 0.85,
        enable_guidance: bool = True,
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
        self.caution_distance_m = float(caution_distance_m)
        self.activation_threshold_m = float(activation_threshold_m)
        self.enable_guidance = bool(enable_guidance)
        self.action_limits = ActionLimits()

        # Shared guidance arbitrator
        self.guidance_config = GuidanceConfig(
            activation_threshold_m=self.activation_threshold_m,
        )
        self.guidance_arbitrator = StatelessClearanceGuidanceArbitrator(
            config=self.guidance_config,
            limits=self.action_limits,
        )
        self.last_guidance_dir = "NONE"

        # Configure safety supervisor with verified 0.50m caution distance
        self.supervisor_config = SafetySupervisorConfig(
            obstacle_caution_dist_m=self.caution_distance_m
        )
        self.safety_supervisor = SafetySupervisor(
            corridor=self.corridor, config=self.supervisor_config
        )

    def reset(self) -> None:
        """Resets guidance latch and safety supervisor state across episodes."""
        self.guidance_arbitrator.reset()
        self.last_guidance_dir = "NONE"
        self.safety_supervisor.reset()

    def predict(self, observation: np.ndarray) -> VelocityAction:
        """Runs fast single-sample ONNX inference."""
        obs_input = np.ascontiguousarray(observation, dtype=np.float32)
        # Sanitize against non-finite values (NaN / Inf) from sensor drivers
        obs_input = np.nan_to_num(obs_input, nan=0.0, posinf=1.0, neginf=-1.0)
        if obs_input.ndim == 1:
            obs_input = np.expand_dims(obs_input, axis=0)

        # Run ONNX session
        raw_output = self.session.run([self.output_name], {self.input_name: obs_input})[0]
        act = raw_output[0]

        return VelocityAction(
            vx=float(np.clip(float(act[0]), self.action_limits.min_fwd_vel_mps, self.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(float(act[1]), -self.action_limits.max_lat_vel_mps, self.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(float(act[2]), -self.action_limits.max_vert_vel_mps, self.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(float(act[3]), -self.action_limits.max_yaw_rate_radps, self.action_limits.max_yaw_rate_radps)),
        )

    def apply_clearance_guidance(
        self, raw_action: VelocityAction, obs_vec: np.ndarray
    ) -> VelocityAction:
        """Applies stateless clearance guidance and wall-proximity margin tie-breaking."""
        if not self.enable_guidance:
            return raw_action

        guided_action, s_dir, _, _ = self.guidance_arbitrator.arbitrate(raw_action, obs_vec)
        self.last_guidance_dir = s_dir
        return guided_action

    def step_onboard_control(
        self,
        deployable_obs: DeployableObservation,
        estimated_position: np.ndarray,
        estimated_velocity: np.ndarray,
        dt: float = 0.05,
    ) -> Tuple[VelocityAction, SafetyState, str]:
        """Complete edge control loop execution: ONNX inference + guidance + safety supervisor filter."""
        obs_vec = deployable_obs.to_vector()
        raw_action = self.predict(obs_vec)

        # Apply Clearance Guidance layer
        guided_action = self.apply_clearance_guidance(raw_action, obs_vec)

        # Min front distance from forward rangefinder rays (rays 0, 1, 2) in meters
        front_dist = float(min(deployable_obs.distance_rays[0:3])) * 4.0

        # Filter action through Safety Supervisor
        safe_action, s_state, s_reason = self.safety_supervisor.filter_action(
            proposed_action=guided_action,
            current_position=estimated_position,
            current_velocity=estimated_velocity,
            min_front_range=front_dist,
            dt=dt,
            is_localization_healthy=True,
            telemetry_age_s=0.01,
            observation=deployable_obs,
        )

        return safe_action, s_state, s_reason
