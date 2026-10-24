"""test_localization.py - Unit Tests for Modular Localization Layer."""

import numpy as np
import pytest

from core.contracts import CoordinateFrame
from localization.interface import LocalizationManager, LocalizationPacket
from localization.mock_source import MockLocalizationSource
from localization.sim_source import SimGroundTruthSource


def test_source_switching_disabled_by_default():
    """Confirms dynamic source switching is strictly disabled by default."""
    sim_src = SimGroundTruthSource()
    mock_src = MockLocalizationSource()
    mgr = LocalizationManager(primary_source=sim_src, allow_dynamic_switching=False)

    mgr.register_source(mock_src)
    assert mgr.active_source_name == "sim_ground_truth"

    # Invalidate sim source
    sim_src.update_from_sim([0, 0, 0], [0, 0, 0], [0, 0, 0, 1], sim_time_s=1.0)
    # Check at time = 5.0 (stale)
    pkt, health = mgr.get_state(current_time_s=5.0)

    # Must NOT automatically switch to mock_src
    assert pkt is None
    assert not health.is_healthy
    assert mgr.active_source_name == "sim_ground_truth"


def test_explicit_source_selection():
    """Confirms that active source can be switched explicitly."""
    sim_src = SimGroundTruthSource()
    mock_src = MockLocalizationSource()
    mgr = LocalizationManager(primary_source=sim_src)
    mgr.register_source(mock_src)

    mgr.set_active_source_explicit("mock_tracker")
    assert mgr.active_source_name == "mock_tracker"

    with pytest.raises(ValueError):
        mgr.set_active_source_explicit("non_existent_source")


def test_staleness_detection():
    """Confirms that packets older than max_stale_age_s are flagged unhealthy."""
    src = SimGroundTruthSource()
    src.update_from_sim([1, 2, 3], [0, 0, 0], [0, 0, 0, 1], sim_time_s=10.0)

    # At t=10.1 (0.1s old), healthy
    h1 = src.check_health(current_time_s=10.10, max_stale_age_s=0.35)
    assert h1.is_healthy

    # At t=10.5 (0.5s old), stale
    h2 = src.check_health(current_time_s=10.50, max_stale_age_s=0.35)
    assert not h2.is_healthy
    assert "lagged" in h2.error_message


def test_frame_conversion_ned_to_enu():
    """Verifies that NED coordinates are accurately converted to ENU."""
    mock_src = MockLocalizationSource(frame=CoordinateFrame.NED)
    # NED position: X=North(2.0), Y=East(3.0), Z=Down(-1.5) -> Altitude is +1.5m
    mock_src.push_true_state(
        true_position=np.array([2.0, 3.0, -1.5]),
        true_velocity=np.array([0.5, 0.0, 0.0]),
        true_quat=np.array([0, 0, 0, 1]),
        timestamp_s=10.0
    )
    mock_src.advance_time(10.10)

    mgr = LocalizationManager(primary_source=mock_src, target_frame=CoordinateFrame.ENU)
    pkt, health = mgr.get_state(current_time_s=10.10)

    assert health.is_healthy
    assert pkt.frame == CoordinateFrame.ENU
    # In ENU: X = East = 3.0, Y = North = 2.0, Z = Up = 1.5
    assert np.isclose(pkt.position[0], 3.0, atol=0.1)
    assert np.isclose(pkt.position[1], 2.0, atol=0.1)
    assert np.isclose(pkt.position[2], 1.5, atol=0.1)
