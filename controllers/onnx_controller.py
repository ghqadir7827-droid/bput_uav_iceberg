"""onnx_controller.py - Controller wrapper for OnboardONNXPolicyRunner.

Enables standalone ONNX runtime inference to be registered within ControllerRegistry,
evaluated via run_quantitative_eval.py, rendered in PyBullet GUI, and tested.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from controllers.registry import ControllerMetadata, IController
from core.contracts import VelocityAction
from hardware.onnx_runner import OnboardONNXPolicyRunner


class ONNXEdgePolicyController(IController):
    """Standalone edge ONNX controller wrapping OnboardONNXPolicyRunner."""

    def __init__(self, onnx_path: str = "models/checkpoints/policy_ppo.onnx"):
        self.onnx_path = Path(onnx_path)
        self.runner: Optional[OnboardONNXPolicyRunner] = None

        self._meta = ControllerMetadata(
            controller_id="onnx_ppo_edge",
            checkpoint_id="onnx_runtime_v1",
            display_name="ONNX PPO Edge Policy (Standalone)",
            is_implemented=False,
            milestone_tag="Milestone 5",
            description="Standalone ONNX Runtime CPU inference engine (5.3 KB, 0.016 ms latency).",
            file_path=str(self.onnx_path),
        )

        if self.onnx_path.exists():
            try:
                self.runner = OnboardONNXPolicyRunner(onnx_model_path=str(self.onnx_path))
                self._meta.is_implemented = True
                self._meta.checkpoint_id = "onnx_5.3kb_0.016ms"
            except Exception as e:
                print(f"Warning: Failed to load ONNX runner: {e}")

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def predict(self, observation: np.ndarray) -> VelocityAction:
        if self.runner is None:
            return VelocityAction(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
        return self.runner.predict(observation)
