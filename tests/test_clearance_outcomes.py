"""test_clearance_outcomes.py - Verification of 8-Category Outcome Taxonomy & Clearance Invariants."""

import pytest
from core.geometry import EpisodeOutcome, OutcomeClassifier, TerminationReason


def test_seed_20334_envelope_breach_is_not_clean_success():
    """Verifies that an arrival with 0.0892m clearance (Seed 20334) is strictly ENVELOPE_BREACH_ARRIVAL."""
    res = OutcomeClassifier.classify(
        termination_reason=TerminationReason.SUCCESS,
        min_obstacle_clearance_m=0.0892,
        min_boundary_clearance_m=0.450,
    )
    assert res.outcome == EpisodeOutcome.ENVELOPE_BREACH_ARRIVAL
    assert res.is_clean is False
    assert res.envelope_breach_detected is True
    assert not res.is_collision
    assert not res.is_timeout


def test_clean_success_requires_15cm_clearance():
    """Verifies that CLEAN_SUCCESS strictly requires min_obstacle_clearance >= 0.150m."""
    # Just below 0.15m: breach
    res_below = OutcomeClassifier.classify(
        termination_reason=TerminationReason.SUCCESS,
        min_obstacle_clearance_m=0.1499,
        min_boundary_clearance_m=0.500,
    )
    assert res_below.outcome == EpisodeOutcome.ENVELOPE_BREACH_ARRIVAL
    assert res_below.is_clean is False

    # At exactly 0.15m: clean
    res_exact = OutcomeClassifier.classify(
        termination_reason=TerminationReason.SUCCESS,
        min_obstacle_clearance_m=0.1500,
        min_boundary_clearance_m=0.500,
    )
    assert res_exact.outcome == EpisodeOutcome.CLEAN_SUCCESS
    assert res_exact.is_clean is True
    assert res_exact.envelope_breach_detected is False

    # Well above 0.15m: clean
    res_above = OutcomeClassifier.classify(
        termination_reason=TerminationReason.SUCCESS,
        min_obstacle_clearance_m=0.2850,
        min_boundary_clearance_m=0.620,
    )
    assert res_above.outcome == EpisodeOutcome.CLEAN_SUCCESS
    assert res_above.is_clean is True


def test_collision_cannot_be_overridden_by_goal_arrival():
    """Verifies that physical collision takes absolute precedence over destination arrival."""
    res = OutcomeClassifier.classify(
        termination_reason=TerminationReason.CRASH_OBSTACLE,
        min_obstacle_clearance_m=-0.005,
        min_boundary_clearance_m=0.500,
    )
    assert res.outcome == EpisodeOutcome.COLLISION
    assert res.is_collision is True
    assert res.is_clean is False

    # Negative clearance even if reported reason was string 'success'
    res_zero = OutcomeClassifier.classify(
        termination_reason="success",
        min_obstacle_clearance_m=0.000,
        min_boundary_clearance_m=0.500,
    )
    assert res_zero.outcome == EpisodeOutcome.COLLISION
    assert res_zero.is_collision is True


def test_timeout_cannot_be_reclassified_as_success():
    """Verifies that timeout remains TIMEOUT and is never silently counted as success."""
    res = OutcomeClassifier.classify(
        termination_reason=TerminationReason.TIMEOUT,
        min_obstacle_clearance_m=0.350,
        min_boundary_clearance_m=0.400,
    )
    assert res.outcome == EpisodeOutcome.TIMEOUT
    assert res.is_timeout is True
    assert res.is_clean is False


def test_safe_rejection_requires_stability_and_positive_clearance():
    """Verifies conditions distinguishing SAFE_REJECTION from UNSAFE_OR_UNSTABLE_ARREST."""
    # Stable dwell with positive clearance (> 0.05m)
    res_safe = OutcomeClassifier.classify(
        termination_reason=TerminationReason.CONTINUE,
        min_obstacle_clearance_m=0.250,
        min_boundary_clearance_m=0.300,
        safe_arrest_state="VERIFIED_INFEASIBLE",
        is_arrest_stable=True,
    )
    assert res_safe.outcome == EpisodeOutcome.SAFE_REJECTION
    assert res_safe.is_safe_rejection is True
    assert res_safe.is_clean is False

    # Infeasible passage reported, but arrest was UNSTABLE
    res_unstable = OutcomeClassifier.classify(
        termination_reason=TerminationReason.CONTINUE,
        min_obstacle_clearance_m=0.250,
        min_boundary_clearance_m=0.300,
        safe_arrest_state="VERIFIED_INFEASIBLE",
        is_arrest_stable=False,
    )
    assert res_unstable.outcome == EpisodeOutcome.UNSAFE_OR_UNSTABLE_ARREST
    assert res_unstable.is_safe_rejection is False

    # Infeasible passage reported, but clearance penetrated (< 0.05m)
    res_penetrated = OutcomeClassifier.classify(
        termination_reason=TerminationReason.CONTINUE,
        min_obstacle_clearance_m=0.030,
        min_boundary_clearance_m=0.300,
        safe_arrest_state="VERIFIED_INFEASIBLE",
        is_arrest_stable=True,
    )
    assert res_penetrated.outcome == EpisodeOutcome.UNSAFE_OR_UNSTABLE_ARREST


def test_unresolved_obstacle_classification():
    """Verifies UNRESOLVED_OBSTACLE for ambiguous routes where drone safely arrested."""
    res_unresolved = OutcomeClassifier.classify(
        termination_reason=TerminationReason.CONTINUE,
        min_obstacle_clearance_m=0.400,
        min_boundary_clearance_m=0.500,
        safe_arrest_state="UNRESOLVED",
        is_arrest_stable=True,
    )
    assert res_unresolved.outcome == EpisodeOutcome.UNRESOLVED_OBSTACLE

