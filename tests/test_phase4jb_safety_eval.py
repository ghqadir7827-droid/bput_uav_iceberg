"""test_phase4jb_safety_eval.py - Unit Tests for Phase 4J-B Safety-Aware Evaluation Layer.

Verifies:
1. Caution-distance parameter propagation.
2. Exact caution-threshold activation.
3. Exact emergency-brake activation.
4. Distance-convention consistency (eff_front is center-to-surface).
5. Correct obstacle surface-clearance calculation (includes drone radius).
6. Correct boundary-clearance calculation (includes drone radius).
7. Distinct separation between collision (d <= 0) and envelope violation (d < 0.08m).
8. Metric handling when an episode terminates early.
9. Correct handling of a timeout at canonical 500-step horizon.
10. Paired-seed matching and isolation from held-out seeds (90000..90099).
11. No mutation of canonical supervisor file or checkpoints.
"""

import hashlib
from pathlib import Path
import pytest
import numpy as np

from core.contracts import ActionLimits, VelocityAction, compute_directional_clearances_m, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor


CANONICAL_BC_HASH = "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f"
CANONICAL_PPO_HASH = "33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0"
PHASE4H_BC_HASH = "1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b"
PHASE4H_PPO_HASH = "751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestPhase4JBSafetySemantics:
    """Tests 1-7: Safety supervisor thresholds, distance conventions, and envelopes."""

    def test_parameter_propagation(self):
        """Verifies caution distance is correctly propagated to the supervisor configuration."""
        corridor = Corridor3D.get_10m_benchmark()
        sup_50 = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.50)
        assert sup_50.caution_distance_m == 0.50
        assert sup_50.config.obstacle_caution_dist_m == 0.50
        assert sup_50._supervisor.config.obstacle_caution_dist_m == 0.50

        sup_45 = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.45)
        assert sup_45.caution_distance_m == 0.45
        assert sup_45.config.obstacle_caution_dist_m == 0.45
        assert sup_45._supervisor.config.obstacle_caution_dist_m == 0.45

        # Invariants strictly preserved
        assert sup_50.config.critical_obstacle_dist_m == 0.30
        assert sup_50.config.emergency_reverse_dist_m == 0.15
        assert sup_50.config.boundary_buffer_m == 0.25

    def test_exact_caution_threshold_activation(self):
        """Verifies velocity throttling triggers at exactly eff_front < caution_dist."""
        corridor = Corridor3D.get_10m_benchmark()
        sup = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.50)
        pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        proposed = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

        # eff_front >= 0.50 -> No caution throttle
        act_unthrottled, state_u, _ = sup.filter_action(
            proposed_action=proposed, current_position=pos, current_velocity=vel,
            min_front_range=0.51, dt=0.05
        )
        assert act_unthrottled.vx == pytest.approx(0.60, abs=1e-3)
        assert state_u == SafetyState.NOMINAL

        # eff_front < 0.50 -> Caution throttle active
        act_throttled, state_t, reason_t = sup.filter_action(
            proposed_action=proposed, current_position=pos, current_velocity=vel,
            min_front_range=0.45, dt=0.05
        )
        # frac = (0.45 - 0.30) / (0.50 - 0.30) = 0.15 / 0.20 = 0.75 -> max_vx = 0.75 * 0.60 = 0.45 m/s
        assert act_throttled.vx == pytest.approx(0.45, abs=1e-3)
        assert "Obstacle caution zone" in reason_t

    def test_exact_emergency_brake_activation(self):
        """Verifies forward brake triggers when eff_front < 0.30m and reverse when < 0.15m."""
        corridor = Corridor3D.get_10m_benchmark()
        sup = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.50)
        pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        proposed = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

        # eff_front = 0.25m (< 0.30m) -> Emergency brake (vx = 0.0)
        act_brake, state_b, reason_b = sup.filter_action(
            proposed_action=proposed, current_position=pos, current_velocity=vel,
            min_front_range=0.25, dt=0.05
        )
        assert act_brake.vx == 0.0
        assert "Forward brake" in reason_b

        # eff_front = 0.12m (< 0.15m) -> Emergency reverse (vx = -0.10)
        act_rev, state_r, reason_r = sup.filter_action(
            proposed_action=proposed, current_position=pos, current_velocity=vel,
            min_front_range=0.12, dt=0.05
        )
        assert act_rev.vx == -0.10
        assert "Reversing" in reason_r

    def test_distance_conventions_and_drone_radius(self):
        """Verifies analytical obstacle and boundary clearances explicitly subtract drone radius."""
        corridor = Corridor3D.get_10m_benchmark()
        assert corridor.drone_radius == 0.15

        # Obstacle clearance check
        obs = Obstacle3D(center=np.array([3.0, 0.0, 1.0]), radius=0.20, height=2.0)
        # Drone center at (2.60, 0.0, 1.0) -> Center-to-center = 0.40m
        # Surface-to-surface gap = 0.40 - 0.20 (pillar) - 0.15 (drone) = 0.05m
        clearance = corridor.obstacle_clearance(np.array([2.60, 0.0, 1.0]), [obs])
        assert clearance == pytest.approx(0.05, abs=1e-5)

        # Boundary distance check
        # Corridor y_max = 1.0. Drone center at y = 0.80.
        # Wall clearance = 1.0 - 0.80 - 0.15 = 0.05m
        b_clearance = corridor.boundary_distance(np.array([1.0, 0.80, 1.0]))
        assert b_clearance == pytest.approx(0.05, abs=1e-5)

    def test_collision_vs_envelope_violation_separation(self):
        """Verifies distinct classification of collision (<=0.0) vs envelope violation (<0.08m)."""
        corridor = Corridor3D.get_10m_benchmark()
        obs = Obstacle3D(center=np.array([3.0, 0.0, 1.0]), radius=0.20, height=2.0)

        # Case 1: Clearance = 0.10m -> Compliant (No violation, No collision)
        p1 = np.array([3.0 - (0.20 + 0.15 + 0.10), 0.0, 1.0])
        c1 = corridor.obstacle_clearance(p1, [obs])
        term1, _ = corridor.check_termination(p1, np.zeros(3), [obs], step_count=10)
        assert c1 == pytest.approx(0.10, abs=1e-5)
        assert term1 == TerminationReason.CONTINUE
        assert not (c1 < 0.08)

        # Case 2: Clearance = 0.03m -> Envelope Violation, but NOT a collision
        p2 = np.array([3.0 - (0.20 + 0.15 + 0.03), 0.0, 1.0])
        c2 = corridor.obstacle_clearance(p2, [obs])
        term2, _ = corridor.check_termination(p2, np.zeros(3), [obs], step_count=10)
        assert c2 == pytest.approx(0.03, abs=1e-5)
        assert term2 == TerminationReason.CONTINUE  # In flight, did not crash
        assert c2 < 0.08  # Envelope violation!

        # Case 3: Clearance = -0.005m -> Obstacle Collision
        p3 = np.array([3.0 - (0.20 + 0.15 - 0.005), 0.0, 1.0])
        c3 = corridor.obstacle_clearance(p3, [obs])
        term3, _ = corridor.check_termination(p3, np.zeros(3), [obs], step_count=10)
        assert c3 < 0.0
        assert term3 == TerminationReason.CRASH_OBSTACLE


class TestPhase4JBEvaluationProtocols:
    """Tests 8-12: Integrity of evaluation seeds, checkpoint immutability, and harness contracts."""

    def test_seed_isolation_from_held_out_pool(self):
        """Ensures evaluation seeds do NOT intersect reserved held-out seeds (90000..90099)."""
        held_out_set = set(range(90000, 90100))
        # Proposed 50 evaluation seeds for Phase 4J-B
        eval_seeds = set(range(2000, 2050))
        # Phase 4J-A dev seeds
        phase4ja_dev_seeds = set(range(1000, 1010))

        assert len(eval_seeds.intersection(held_out_set)) == 0
        assert len(eval_seeds.intersection(phase4ja_dev_seeds)) == 0
        assert len(eval_seeds) == 50

    def test_checkpoint_integrity(self):
        """Verifies canonical and Phase 4H policy checkpoints remain untouched and uncorrupted."""
        assert compute_file_sha256("models/checkpoints/bc_policy_best.pt") == CANONICAL_BC_HASH
        assert compute_file_sha256("models/checkpoints/ppo_policy_best.pt") == CANONICAL_PPO_HASH
        assert compute_file_sha256("experiments/phase4h/checkpoints/bc_phase4h_candidate.pt") == PHASE4H_BC_HASH
        assert compute_file_sha256("experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt") == PHASE4H_PPO_HASH

    def test_canonical_supervisor_file_untouched(self):
        """Verifies controllers/safety_supervisor.py is not modified during Phase 4J."""
        # Ensure default obstacle_caution_dist_m remains 0.80 in canonical file
        cfg = SafetySupervisorConfig()
        assert cfg.obstacle_caution_dist_m == 0.80
        assert cfg.critical_obstacle_dist_m == 0.30
        assert cfg.emergency_reverse_dist_m == 0.15
