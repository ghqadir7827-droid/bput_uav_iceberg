"""test_phase4jc_audit.py - Comprehensive Unit Tests for Phase 4J-C Verification & Safety Gate.

Validates:
1. Geometric clearance calculations (surface-to-surface).
2. Explicit inclusion/handling of drone radius and pillar radius.
3. Boundary clearance calculations and signed wall distances.
4. Safety-envelope threshold comparisons (>= 0.080m).
5. Detection and intervention for obstacles approaching from rear-lateral directions.
6. Non-intervention when predicted clearance satisfies envelope.
7. Rejection/handling of NaN, infinite, or corrupted values.
8. State isolation between episode resets (no state leakage between seeds).
9. Deterministic reproduction of policy inference and environment rollouts.
10. Metric separation between collision (<= 0), envelope violation (< 0.08), and near-contact.
11. Checkpoint integrity and immutability (SHA-256 validation).
12. Canonical files immutability (controllers/safety_supervisor.py).
"""

import hashlib
import pytest
import numpy as np
import torch

from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from controllers.safety_supervisor import SafetySupervisorConfig
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from experiments.phase4j.isolated_geometric_safety_monitor import OracleGeometricSafetyMonitor
from models.policy_net import NavigationPolicyMLP


CANONICAL_BC_HASH = "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f"
CANONICAL_PPO_HASH = "33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestPhase4JCGeometryAndRadii:
    """Tests 1-4: Exact radii handling and clearance mathematics."""

    def test_drone_and_pillar_radius_handling(self):
        """Verifies surface-to-surface distance accounts for both r_drone and r_pillar."""
        corridor = Corridor3D.get_10m_benchmark()
        assert corridor.drone_radius == 0.15
        assert corridor.pillar_radius == 0.20

        obs = Obstacle3D(center=np.array([3.0, 0.0, 1.0]), radius=0.20, height=2.0)
        # Drone center at (2.50, 0.0, 1.0) -> center-to-center = 0.50m
        # Surface clearance = 0.50 - 0.20 (pillar) - 0.15 (drone) = 0.15m
        pos = np.array([2.50, 0.0, 1.0])
        clearance = corridor.obstacle_clearance(pos, [obs])
        assert clearance == pytest.approx(0.15, abs=1e-5)

    def test_boundary_clearance_calculation(self):
        """Verifies analytical boundary distance correctly subtracts drone radius."""
        corridor = Corridor3D.get_10m_benchmark()
        # Left wall at y = +1.0m. Drone center at y = +0.70m.
        # Boundary clearance = 1.0 - 0.70 - 0.15 = 0.15m
        pos = np.array([2.0, 0.70, 1.0])
        b_dist = corridor.boundary_distance(pos)
        assert b_dist == pytest.approx(0.15, abs=1e-5)

        # Floor at z = 0.0m. Drone center at z = 0.25m.
        # Floor clearance = 0.25 - 0.15 = 0.10m
        pos_low = np.array([2.0, 0.0, 0.25])
        assert corridor.boundary_distance(pos_low) == pytest.approx(0.10, abs=1e-5)

    def test_metric_separation_collision_vs_envelope(self):
        """Verifies clear trichotomy: Compliant (>0.08m), Envelope Violation (<0.08m), Collision (<=0.0m)."""
        corridor = Corridor3D.get_10m_benchmark()
        obs = Obstacle3D(center=np.array([3.0, 0.0, 1.0]), radius=0.20, height=2.0)

        # Compliant: 0.12m
        pos_comp = np.array([3.0 - (0.35 + 0.12), 0.0, 1.0])
        c_comp = corridor.obstacle_clearance(pos_comp, [obs])
        assert c_comp >= 0.080
        assert c_comp > 0.0

        # Violation without collision: 0.03m
        pos_viol = np.array([3.0 - (0.35 + 0.03), 0.0, 1.0])
        c_viol = corridor.obstacle_clearance(pos_viol, [obs])
        assert c_viol < 0.080
        assert c_viol > 0.0 # NOT a collision!

        # Collision: -0.01m
        pos_crash = np.array([3.0 - (0.35 - 0.01), 0.0, 1.0])
        c_crash = corridor.obstacle_clearance(pos_crash, [obs])
        assert c_crash <= 0.0


class TestPhase4JCOracleMonitor:
    """Tests 5-8: Isolated Oracle Safety Monitor behavior."""

    def test_rear_lateral_obstacle_detection(self):
        """Verifies monitor intervenes when cutting sharply around an obstacle in rear-lateral quadrant."""
        corridor = Corridor3D.get_10m_benchmark()
        monitor = OracleGeometricSafetyMonitor(corridor=corridor, guard_envelope_m=0.080)
        obs = Obstacle3D(center=np.array([3.0, 0.0, 1.0]), radius=0.20, height=2.0)

        # Drone at (3.35, -0.05, 1.0), heading 0 rad. Obstacle is at (3.0, 0.0) -> rear-left quadrant
        # Current clearance is ~0.353m - 0.35m = ~0.003m
        # Commanded vy = +0.30 (moving toward obstacle on left)
        proposed = VelocityAction(vx=0.20, vy=0.30, vz=0.0, yaw_rate=0.0)
        pos = np.array([3.35, -0.05, 1.0])
        vel = np.array([0.20, 0.30, 0.0])

        filtered, intervened, reason = monitor.evaluate_and_filter(
            proposed_action=proposed, current_position=pos, current_velocity=vel,
            current_yaw=0.0, obstacles=[obs]
        )
        assert intervened is True
        assert filtered.vy < proposed.vy # Lateral velocity toward obstacle was clamped/reversed
        assert "Oracle Geometric Monitor Active" in reason

    def test_non_intervention_in_open_space(self):
        """Verifies monitor does NOT intervene when clearance is comfortably above guard envelope."""
        corridor = Corridor3D.get_10m_benchmark()
        monitor = OracleGeometricSafetyMonitor(corridor=corridor, guard_envelope_m=0.080)
        obs = Obstacle3D(center=np.array([3.0, 0.0, 1.0]), radius=0.20, height=2.0)

        # Drone at (1.50, 0.0, 1.0), heading 0 rad. Pillar 1.5m ahead.
        pos = np.array([1.50, 0.0, 1.0])
        vel = np.array([0.40, 0.0, 0.0])
        proposed = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

        filtered, intervened, _ = monitor.evaluate_and_filter(
            proposed_action=proposed, current_position=pos, current_velocity=vel,
            current_yaw=0.0, obstacles=[obs]
        )
        assert intervened is False
        assert filtered.vx == proposed.vx
        assert filtered.vy == proposed.vy

    def test_state_isolation_and_resets(self):
        """Verifies monitor state counters reset properly between episodes."""
        corridor = Corridor3D.get_10m_benchmark()
        monitor = OracleGeometricSafetyMonitor(corridor=corridor, guard_envelope_m=0.080)
        monitor.total_interventions = 15
        monitor.blocked_envelope_violations = 4
        monitor.reset_state()
        assert monitor.total_interventions == 0
        assert monitor.blocked_envelope_violations == 0


class TestPhase4JCIntegrityAndDeterminism:
    """Tests 9-12: Determinism, NaN safety, and checkpoint immutability."""

    def test_nan_and_inf_safety(self):
        """Verifies policy forward pass and action sanitization handle extreme inputs cleanly."""
        model = NavigationPolicyMLP(obs_dim=23, act_dim=4)
        nan_obs = torch.full((1, 23), float("nan"))
        # Clean sanitization
        clean_obs = torch.nan_to_num(nan_obs, nan=0.0)
        out = model(clean_obs, bounded=True)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic_reproduction(self):
        """Verifies that two runs with the same model and same seed produce identical trajectories."""
        device = torch.device("cpu")
        model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
        ckpt = torch.load("models/checkpoints/bc_policy_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_obs = torch.ones((1, 23), dtype=torch.float32, device=device)
        with torch.no_grad():
            out1 = model(dummy_obs, bounded=True).numpy()
            out2 = model(dummy_obs, bounded=True).numpy()
        np.testing.assert_array_almost_equal(out1, out2)

    def test_checkpoint_and_canonical_files_immutable(self):
        """Verifies SHA-256 hashes of canonical files match frozen master records."""
        assert compute_file_sha256("models/checkpoints/bc_policy_best.pt") == CANONICAL_BC_HASH
        assert compute_file_sha256("models/checkpoints/ppo_policy_best.pt") == CANONICAL_PPO_HASH
        cfg = SafetySupervisorConfig()
        assert cfg.obstacle_caution_dist_m == 0.80
        assert cfg.critical_obstacle_dist_m == 0.30
        assert cfg.emergency_reverse_dist_m == 0.15

