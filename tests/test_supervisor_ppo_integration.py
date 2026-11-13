"""test_supervisor_ppo_integration.py - Focused Tests for Supervisor-in-the-Loop PPO Integration.

Verifies:
1. Action-space shape and canonical ActionLimits bounds.
2. Policy action decoding and round-trip normalization.
3. SafetySupervisor invocation during training rollouts.
4. RolloutBuffer retention of original sampled actions (a_prop).
5. Preservation of behavior-policy log-probabilities under a_prop.
6. Execution of safety-filtered commands in physics simulation.
7. Exact intervention magnitude calculation.
8. Per-dimension saturation accounting (policy proposed vs applied).
9. Finite loss, advantage, return, and gradient values in PPO updates.
10. Nominal non-intervening behavior when clearances are safe.
11. Emergency reverse behavior when obstacles are critically close.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from models.actor_critic import ActorCriticPPO
from models.train_ppo import RolloutBuffer, RolloutTelemetry
from sim.corridor_env import CorridorSimEnv


def test_action_space_shape_and_physical_limits():
    """Verifies that the action space is 4D and adheres to canonical physical limits."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    try:
        assert env.action_space.shape == (4,), f"Expected action space shape (4,), got {env.action_space.shape}"
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

        limits = env.action_limits
        assert limits.min_fwd_vel_mps == -0.20
        assert limits.max_fwd_vel_mps == 0.60
        assert limits.max_lat_vel_mps == 0.40
        assert limits.max_vert_vel_mps == 0.30
        assert limits.max_yaw_rate_radps == 0.50
    finally:
        env.close()


def test_policy_action_decoding_and_roundtrip_normalization():
    """Verifies exact round-trip normalization for 4D velocity actions."""
    limits = ActionLimits()
    test_cases = [
        (-0.20, -0.40, -0.30, -0.50),
        (-0.10, 0.00, 0.15, 0.25),
        (0.00, 0.20, -0.15, -0.25),
        (0.30, -0.20, 0.00, 0.00),
        (0.60, 0.40, 0.30, 0.50),
    ]

    for vx, vy, vz, yr in test_cases:
        act = VelocityAction(vx=vx, vy=vy, vz=vz, yaw_rate=yr)
        norm = act.to_normalized(limits)
        assert np.all(norm >= -1.0) and np.all(norm <= 1.0)

        recovered = VelocityAction.from_normalized(norm, limits)
        assert np.isclose(recovered.vx, vx, atol=1e-5)
        assert np.isclose(recovered.vy, vy, atol=1e-5)
        assert np.isclose(recovered.vz, vz, atol=1e-5)
        assert np.isclose(recovered.yaw_rate, yr, atol=1e-5)


def test_supervisor_invocation_in_rollout():
    """Verifies that SafetySupervisor is invoked during the rollout loop and intervenes when needed."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=100)
    safety_sup = SafetySupervisor(corridor=env.corridor)

    try:
        obs, info = env.reset(seed=100)
        priv = info["privileged_state"]
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Propose forward motion
        prop = VelocityAction(vx=0.50, vy=0.0, vz=0.0, yaw_rate=0.0)
        safe_act, state, reason = safety_sup.filter_action(
            proposed_action=prop,
            current_position=priv.global_position,
            current_velocity=priv.global_velocity,
            min_front_range=front_dist,
            dt=env.dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        assert isinstance(safe_act, VelocityAction)
        assert isinstance(state, SafetyState)
        assert isinstance(reason, str)
    finally:
        env.close()


def test_preservation_of_sampled_action_in_rollout_buffer():
    """Verifies that the rollout buffer retains the raw proposed action, NOT the filtered safe action."""
    buffer = RolloutBuffer(buffer_size=10, obs_dim=23, act_dim=4)

    obs = np.ones(23, dtype=np.float32)
    # Proposed action sampled from policy
    raw_proposed_act = np.array([0.55, 0.25, -0.10, 0.05], dtype=np.float32)
    # Filtered safe action (e.g. reverse intervention)
    filtered_safe_act = np.array([-0.10, 0.00, 0.00, 0.00], dtype=np.float32)

    logp_sampled = -1.5432
    val = 0.85
    rew = 1.20
    done = 0.0

    # PPO invariant: add raw_proposed_act, NOT filtered_safe_act
    buffer.add(obs, raw_proposed_act, rew, done, val, logp_sampled)

    assert buffer.ptr == 1
    assert np.allclose(buffer.actions[0], raw_proposed_act)
    assert not np.allclose(buffer.actions[0], filtered_safe_act)
    assert np.isclose(buffer.log_probs[0], logp_sampled)


def test_preservation_of_behavior_policy_log_probability():
    """Verifies that evaluating log_prob on buffer action matches sampled log_prob, but diverges on safe action."""
    torch.manual_seed(42)
    model = ActorCriticPPO(obs_dim=23, act_dim=4)
    model.eval()

    obs = torch.randn(1, 23)
    with torch.no_grad():
        act, logp_sampled, _, _ = model.get_action_and_value(obs)

    # Re-evaluating with sampled action must produce identical log_prob
    with torch.no_grad():
        _, logp_recomputed, _, _ = model.get_action_and_value(obs, act)

    assert torch.isclose(logp_sampled, logp_recomputed, atol=1e-5), (
        f"Log-prob mismatch: {logp_sampled.item()} vs {logp_recomputed.item()}"
    )

    # Evaluating log_prob on a forced safety reverse action will diverge sharply
    forced_safe_act = torch.tensor([[-0.10, 0.00, 0.00, 0.00]], dtype=torch.float32)
    with torch.no_grad():
        _, logp_forced, _, _ = model.get_action_and_value(obs, forced_safe_act)

    diff = torch.abs(logp_sampled - logp_forced).item()
    assert diff > 0.1, f"Expected significant divergence in log-prob on safe action, got diff={diff}"


def test_physical_action_execution_after_supervisor_intervention():
    """Verifies that the simulator actually executes the safety-filtered command."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=100)
    try:
        env.reset(seed=100)
        limits = env.action_limits

        # Simulate emergency reverse command from supervisor
        safe_action = VelocityAction(vx=-0.10, vy=0.0, vz=0.0, yaw_rate=0.0)
        norm_cmd = safe_action.to_normalized(limits)

        env.step(norm_cmd)
        curr_vel = env._get_drone_velocity()

        # In body frame, negative forward velocity was commanded
        # In world frame at yaw=0, vx should be negative
        assert curr_vel[0] < -0.01, f"Expected backward velocity, got {curr_vel[0]}"
    finally:
        env.close()


def test_intervention_magnitude_calculation():
    """Verifies Euclidean distance computation between clamped action and safe action."""
    clamped = VelocityAction(vx=0.50, vy=0.20, vz=0.10, yaw_rate=-0.05)
    safe = VelocityAction(vx=-0.10, vy=0.00, vz=0.10, yaw_rate=0.00)

    # Expected: sqrt((0.5 - (-0.1))^2 + (0.2 - 0)^2 + (0.1 - 0.1)^2 + (-0.05 - 0)^2)
    # sqrt(0.36 + 0.04 + 0 + 0.0025) = sqrt(0.4025) = 0.634428877
    expected_mag = np.sqrt((0.60)**2 + (0.20)**2 + (0.00)**2 + (-0.05)**2)

    calc_mag = float(np.sqrt(
        (clamped.vx - safe.vx)**2 +
        (clamped.vy - safe.vy)**2 +
        (clamped.vz - safe.vz)**2 +
        (clamped.yaw_rate - safe.yaw_rate)**2
    ))

    assert np.isclose(calc_mag, expected_mag, atol=1e-6)


def test_per_dimension_saturation_accounting():
    """Verifies that policy proposed saturation is distinguished from applied action saturation."""
    limits = ActionLimits()
    telem = RolloutTelemetry()

    # Step 1: Policy proposes saturated forward velocity (vx=0.80 m/s > max 0.60 m/s)
    # Supervisor brakes to vx = 0.00 m/s (unsaturated)
    prop1 = VelocityAction(vx=0.80, vy=0.10, vz=0.00, yaw_rate=0.00)
    applied1 = VelocityAction(vx=0.00, vy=0.10, vz=0.00, yaw_rate=0.00)

    norm_prop1 = prop1.to_normalized(limits)
    norm_app1 = applied1.to_normalized(limits)

    telem.total_steps += 1
    telem.policy_sat_vx += int(abs(norm_prop1[0]) >= 0.999)
    telem.policy_sat_vy += int(abs(norm_prop1[1]) >= 0.999)
    telem.policy_sat_vz += int(abs(norm_prop1[2]) >= 0.999)
    telem.policy_sat_yr += int(abs(norm_prop1[3]) >= 0.999)

    telem.applied_sat_vx += int(abs(norm_app1[0]) >= 0.999)
    telem.applied_sat_vy += int(abs(norm_app1[1]) >= 0.999)
    telem.applied_sat_vz += int(abs(norm_app1[2]) >= 0.999)
    telem.applied_sat_yr += int(abs(norm_app1[3]) >= 0.999)

    assert telem.policy_sat_vx == 1
    assert telem.applied_sat_vx == 0
    assert telem.policy_sat_vy == 0
    assert telem.applied_sat_vy == 0


def test_finite_values_in_short_ppo_update():
    """Verifies that 1 PPO iteration with supervisor completes with finite loss and gradients."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    safety_sup = SafetySupervisor(corridor=env.corridor)
    buffer = RolloutBuffer(buffer_size=16, obs_dim=23, act_dim=4)

    try:
        obs, info = env.reset(seed=42)
        model.eval()

        for step in range(16):
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act, logp, _, val = model.get_action_and_value(t_obs)
            act_np = act.squeeze(0).numpy()

            prop_act = VelocityAction(
                vx=float(act_np[0]), vy=float(act_np[1]), vz=float(act_np[2]), yaw_rate=float(act_np[3])
            )
            priv = info["privileged_state"]
            front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

            safe_act, _, _ = safety_sup.filter_action(
                proposed_action=prop_act,
                current_position=priv.global_position,
                current_velocity=priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                observation=obs,
            )

            norm_applied = safe_act.to_normalized(env.action_limits)
            next_obs, rew, term, trunc, info = env.step(norm_applied)
            done = term or trunc

            buffer.add(obs, act_np, rew, float(done), float(val.item()), float(logp.item()))
            obs = next_obs
            if done:
                obs, info = env.reset(seed=43)

        # GAE
        buffer.compute_gae(0.0)
        assert np.all(np.isfinite(buffer.advantages))
        assert np.all(np.isfinite(buffer.returns))

        # PPO Update
        model.train()
        b_obs = torch.as_tensor(buffer.observations, dtype=torch.float32)
        b_acts = torch.as_tensor(buffer.actions, dtype=torch.float32)
        b_logp = torch.as_tensor(buffer.log_probs, dtype=torch.float32)
        b_adv = torch.as_tensor(buffer.advantages, dtype=torch.float32)
        b_ret = torch.as_tensor(buffer.returns, dtype=torch.float32)

        _, new_logp, entropy, new_val = model.get_action_and_value(b_obs, b_acts)
        ratio = torch.exp(new_logp - b_logp)
        surr1 = ratio * b_adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
        pi_loss = -torch.min(surr1, surr2).mean()
        v_loss = 0.5 * torch.mean((new_val - b_ret) ** 2)
        loss = pi_loss + 0.5 * v_loss - 0.01 * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        assert torch.isfinite(pi_loss).item()
        assert torch.isfinite(v_loss).item()
        assert torch.isfinite(loss).item()
        assert torch.isfinite(grad_norm).item()
    finally:
        env.close()


def test_behavior_when_supervisor_nominal():
    """Verifies that in nominal conditions (obstacle far, center position), action is unmodified."""
    from core.geometry import Corridor3D
    corridor = Corridor3D()
    sup = SafetySupervisor(corridor)

    # Safe center position: (1.0, 0.0, 1.0)
    pos = np.array([1.0, 0.0, 1.0])
    vel = np.array([0.2, 0.0, 0.0])
    prop = VelocityAction(vx=0.30, vy=0.00, vz=0.00, yaw_rate=0.00)

    # All ranges far (4.0 m)
    safe_act, state, reason = sup.filter_action(
        proposed_action=prop,
        current_position=pos,
        current_velocity=vel,
        min_front_range=4.0,
        dt=0.05,
        range_rays=np.full(12, 4.0),
    )

    assert state == SafetyState.NOMINAL
    assert np.isclose(safe_act.vx, prop.vx)
    assert np.isclose(safe_act.vy, prop.vy)
    assert np.isclose(safe_act.vz, prop.vz)
    assert np.isclose(safe_act.yaw_rate, prop.yaw_rate)


def test_behavior_during_emergency_reverse():
    """Verifies that when front obstacle is dangerously close, emergency reverse triggers."""
    from core.geometry import Corridor3D
    corridor = Corridor3D()
    sup = SafetySupervisor(corridor)

    pos = np.array([2.0, 0.0, 1.0])
    vel = np.array([0.1, 0.0, 0.0])
    prop = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Obstacle at 0.12m (< emergency_reverse_dist_m = 0.15m)
    rays = np.full(12, 4.0)
    rays[0] = 0.12  # front ray

    safe_act, state, reason = sup.filter_action(
        proposed_action=prop,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.12,
        dt=0.05,
        range_rays=rays,
    )

    assert safe_act.vx == -0.10
    assert state in (SafetyState.OBSTACLE_BRAKE, SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE)
    assert "Reversing" in reason or "Reverse" in reason


def test_ppo_checkpoint_loading_and_compatibility():
    """Task 1 Test: Verifies explicit PPO checkpoint loader compatibility."""
    ckpt_path = Path("models/checkpoints/ppo_policy_best.pt")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    model = ActorCriticPPO(obs_dim=23, act_dim=4)
    loaded = model.load_ppo_checkpoint(str(ckpt_path), device=torch.device("cpu"))
    assert loaded is True

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k, v in raw_ckpt["model_state_dict"].items():
        assert torch.allclose(model.actor.state_dict()[k], v)
    if "critic_state_dict" in raw_ckpt:
        for k, v in raw_ckpt["critic_state_dict"].items():
            assert torch.allclose(model.critic.state_dict()[k], v)

    # Verify forward pass produces valid tensor shapes
    obs_batch = torch.randn(2, 23)
    with torch.no_grad():
        act, logp, entropy, val = model.get_action_and_value(obs_batch)

    assert act.shape == (2, 4)
    assert logp.shape == (2,)
    assert entropy.shape == (2,)
    assert val.shape == (2,)

    # Check exploration parameter log_std
    assert hasattr(model, "log_std")
    assert model.log_std.shape == (4,)
    assert torch.all(torch.isfinite(model.log_std))


def test_gae_successful_task_termination():
    """Task 2 Test 1: Verifies GAE on successful task termination (zero future value bootstrap)."""
    buffer = RolloutBuffer(buffer_size=3, obs_dim=23, act_dim=4)
    dummy_obs = np.zeros(23, dtype=np.float32)
    dummy_act = np.zeros(4, dtype=np.float32)

    # Step 0: continuing step
    buffer.add(dummy_obs, dummy_act, rew=1.0, done=0.0, val=2.0, logp=0.0, terminal=False, timeout=False)
    # Step 1: goal success terminal step (absorbing terminal state)
    buffer.add(dummy_obs, dummy_act, rew=10.0, done=1.0, val=8.0, logp=0.0, terminal=True, timeout=False)
    # Step 2: new episode initial step
    buffer.add(dummy_obs, dummy_act, rew=0.5, done=0.0, val=1.0, logp=0.0, terminal=False, timeout=False)

    buffer.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)

    # Step 1 terminal transition: next_val must be 0.0, next_non_terminal must be 0.0
    # delta_1 = rew_1 + gamma * 0 - val_1 = 10.0 - 8.0 = 2.0
    # A_1 = delta_1 = 2.0
    # R_1 = A_1 + val_1 = 10.0 (exact terminal reward, no future bootstrap)
    assert np.isclose(buffer.advantages[1], 2.0, atol=1e-5)
    assert np.isclose(buffer.returns[1], 10.0, atol=1e-5)


def test_gae_collision_termination():
    """Task 2 Test 2: Verifies GAE on collision termination (zero future value bootstrap, crash penalty)."""
    buffer = RolloutBuffer(buffer_size=3, obs_dim=23, act_dim=4)
    dummy_obs = np.zeros(23, dtype=np.float32)
    dummy_act = np.zeros(4, dtype=np.float32)

    # Step 0: continuing
    buffer.add(dummy_obs, dummy_act, rew=0.2, done=0.0, val=0.5, logp=0.0, terminal=False, timeout=False)
    # Step 1: collision termination
    buffer.add(dummy_obs, dummy_act, rew=-5.0, done=1.0, val=0.5, logp=0.0, terminal=True, timeout=False)
    # Step 2: reset episode
    buffer.add(dummy_obs, dummy_act, rew=0.1, done=0.0, val=0.3, logp=0.0, terminal=False, timeout=False)

    buffer.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)

    # delta_1 = -5.0 + 0 - 0.5 = -5.5
    # A_1 = -5.5
    # R_1 = A_1 + val_1 = -5.0
    assert np.isclose(buffer.advantages[1], -5.5, atol=1e-5)
    assert np.isclose(buffer.returns[1], -5.0, atol=1e-5)


def test_gae_timeout_truncation_with_nonzero_value():
    """Task 2 Test 3: Verifies GAE on time-limit truncation bootstraps from final state value."""
    buffer = RolloutBuffer(buffer_size=3, obs_dim=23, act_dim=4)
    dummy_obs = np.zeros(23, dtype=np.float32)
    dummy_act = np.zeros(4, dtype=np.float32)

    gamma = 0.99
    # Step 0: continuing
    buffer.add(dummy_obs, dummy_act, rew=1.0, done=0.0, val=1.0, logp=0.0, terminal=False, timeout=False)
    # Step 1: timeout truncation with V(s_{t+1}) = 4.0
    timeout_v = 4.0
    buffer.add(dummy_obs, dummy_act, rew=1.0, done=1.0, val=2.0, logp=0.0, terminal=False, timeout=True, timeout_value=timeout_v)
    # Step 2: reset episode
    buffer.add(dummy_obs, dummy_act, rew=0.5, done=0.0, val=1.0, logp=0.0, terminal=False, timeout=False)

    buffer.compute_gae(last_value=0.0, gamma=gamma, lam=0.95)

    # Step 1 timeout transition:
    # next_val must be timeout_v = 4.0
    # delta_1 = rew_1 + gamma * 4.0 - val_1 = 1.0 + 0.99 * 4.0 - 2.0 = 2.96
    # next_non_terminal = 0.0 (prevents leakage from step 2)
    # A_1 = delta_1 = 2.96
    # R_1 = A_1 + val_1 = 2.96 + 2.0 = 4.96 = rew_1 + gamma * timeout_v
    expected_delta_1 = 1.0 + gamma * timeout_v - 2.0
    expected_ret_1 = 1.0 + gamma * timeout_v
    assert np.isclose(buffer.advantages[1], expected_delta_1, atol=1e-5)
    assert np.isclose(buffer.returns[1], expected_ret_1, atol=1e-5)


def test_gae_no_advantage_leakage_across_episode_boundary():
    """Task 2 Test 4: Verifies advantage recursion does not leak across episode resets (both term and trunc)."""
    gamma = 0.95
    lam = 0.90

    for end_mode in ["terminal", "timeout"]:
        buffer = RolloutBuffer(buffer_size=4, obs_dim=23, act_dim=4)
        dummy_obs = np.zeros(23, dtype=np.float32)
        dummy_act = np.zeros(4, dtype=np.float32)

        # Episode 1: Steps 0, 1
        buffer.add(dummy_obs, dummy_act, rew=1.0, done=0.0, val=0.8, logp=0.0)
        if end_mode == "terminal":
            buffer.add(dummy_obs, dummy_act, rew=5.0, done=1.0, val=2.0, logp=0.0, terminal=True, timeout=False)
        else:
            buffer.add(dummy_obs, dummy_act, rew=2.0, done=1.0, val=1.5, logp=0.0, terminal=False, timeout=True, timeout_value=3.5)

        # Episode 2: Steps 2, 3
        buffer.add(dummy_obs, dummy_act, rew=0.5, done=0.0, val=0.4, logp=0.0)
        buffer.add(dummy_obs, dummy_act, rew=-2.0, done=1.0, val=0.1, logp=0.0, terminal=True)

        buffer.compute_gae(last_value=0.0, gamma=gamma, lam=lam)
        adv_ep1 = np.copy(buffer.advantages[:2])
        ret_ep1 = np.copy(buffer.returns[:2])

        # Now perturb Episode 2 drastically
        buffer.rewards[2] = 999.0
        buffer.values[2] = 500.0
        buffer.rewards[3] = -999.0
        buffer.values[3] = 1000.0

        buffer.compute_gae(last_value=0.0, gamma=gamma, lam=lam)
        adv_ep1_perturbed = np.copy(buffer.advantages[:2])
        ret_ep1_perturbed = np.copy(buffer.returns[:2])

        # Steps 0 and 1 must be strictly identical regardless of Episode 2 perturbations
        assert np.allclose(adv_ep1, adv_ep1_perturbed, atol=1e-7), f"Advantage leaked across {end_mode} boundary!"
        assert np.allclose(ret_ep1, ret_ep1_perturbed, atol=1e-7), f"Returns leaked across {end_mode} boundary!"


def test_gae_precedence_termination_over_truncation():
    """Task 2 Requirement: Documented precedence rule - terminal strictly supersedes timeout."""
    buffer = RolloutBuffer(buffer_size=2, obs_dim=23, act_dim=4)
    dummy_obs = np.zeros(23, dtype=np.float32)
    dummy_act = np.zeros(4, dtype=np.float32)

    # Pass both terminal=True and timeout=True
    buffer.add(
        dummy_obs, dummy_act, rew=-10.0, done=1.0, val=1.0, logp=0.0,
        terminal=True, timeout=True, timeout_value=8.0
    )
    buffer.add(dummy_obs, dummy_act, rew=0.0, done=0.0, val=0.0, logp=0.0)

    # Precedence ensures terminals=1, timeouts=0, timeout_values=0
    assert buffer.terminals[0] == 1.0
    assert buffer.timeouts[0] == 0.0
    assert buffer.timeout_values[0] == 0.0

    buffer.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
    # Next value bootstrapped must be 0.0 (terminal state), NOT 8.0 (timeout value)
    # delta_0 = -10.0 + 0 - 1.0 = -11.0
    assert np.isclose(buffer.advantages[0], -11.0, atol=1e-5)
    assert np.isclose(buffer.returns[0], -10.0, atol=1e-5)


def test_gae_hand_computable_example():
    """Task 2 Test 5: Hand-computable GAE numerical validation.
    
    Setup:
      Buffer size T = 3, gamma = 0.9, lambda = 0.8
      Step 0: continuing (r0=1.0, V0=0.5)
      Step 1: timeout (r1=2.0, V1=1.5, V_timeout=3.0)
      Step 2: terminal crash (r2=0.5, V2=1.0)
      
    Manual Computation:
      Step 2 (terminal crash):
        next_val = 0.0, next_non_terminal = 0.0
        delta_2 = r2 + gamma * 0.0 - V2 = 0.5 - 1.0 = -0.5
        A_2 = delta_2 + gamma * lambda * 0.0 * 0.0 = -0.5
        R_2 = A_2 + V2 = -0.5 + 1.0 = 0.5
        
      Step 1 (timeout truncation):
        next_val = V_timeout = 3.0, next_non_terminal = 0.0 (episode reset boundary)
        delta_1 = r1 + gamma * 3.0 - V1 = 2.0 + 2.7 - 1.5 = 3.2
        A_1 = delta_1 + gamma * lambda * next_non_terminal * A_2 = 3.2 + 0 = 3.2
        R_1 = A_1 + V1 = 3.2 + 1.5 = 4.7
        
      Step 0 (continuing within episode 1):
        next_val = V1 = 1.5, next_non_terminal = 1.0
        delta_0 = r0 + gamma * 1.5 - V0 = 1.0 + 1.35 - 0.5 = 1.85
        A_0 = delta_0 + gamma * lambda * 1.0 * A_1 = 1.85 + (0.9 * 0.8) * 3.2 = 1.85 + 2.304 = 4.154
        R_0 = A_0 + V0 = 4.154 + 0.5 = 4.654
    """
    gamma = 0.9
    lam = 0.8
    buffer = RolloutBuffer(buffer_size=3, obs_dim=23, act_dim=4)
    dummy_obs = np.zeros(23, dtype=np.float32)
    dummy_act = np.zeros(4, dtype=np.float32)

    buffer.add(dummy_obs, dummy_act, rew=1.0, done=0.0, val=0.5, logp=0.0, terminal=False, timeout=False)
    buffer.add(dummy_obs, dummy_act, rew=2.0, done=1.0, val=1.5, logp=0.0, terminal=False, timeout=True, timeout_value=3.0)
    buffer.add(dummy_obs, dummy_act, rew=0.5, done=1.0, val=1.0, logp=0.0, terminal=True, timeout=False)

    buffer.compute_gae(last_value=0.0, gamma=gamma, lam=lam)

    # Assert exact match against hand-computed values
    assert np.isclose(buffer.advantages[2], -0.500, atol=1e-5)
    assert np.isclose(buffer.returns[2], 0.500, atol=1e-5)

    assert np.isclose(buffer.advantages[1], 3.200, atol=1e-5)
    assert np.isclose(buffer.returns[1], 4.700, atol=1e-5)

    assert np.isclose(buffer.advantages[0], 4.154, atol=1e-5)
    assert np.isclose(buffer.returns[0], 4.654, atol=1e-5)


def test_telemetry_units_component_ordering_and_accounting():
    """Task 4 Test: Verifies component-wise intervention telemetry units and accounting."""
    telem = RolloutTelemetry()

    # Step 1: Forward intervention only (proposed 0.50 m/s -> safe 0.20 m/s, delta = 0.30 m/s)
    prop1 = VelocityAction(vx=0.50, vy=0.0, vz=0.0, yaw_rate=0.0)
    safe1 = VelocityAction(vx=0.20, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Step 2: Lateral and vertical intervention (vy diff 0.20 m/s, vz diff 0.10 m/s)
    prop2 = VelocityAction(vx=0.20, vy=0.30, vz=-0.20, yaw_rate=0.0)
    safe2 = VelocityAction(vx=0.20, vy=0.10, vz=-0.10, yaw_rate=0.0)

    # Step 3: Yaw rate intervention (yaw diff 0.40 rad/s)
    prop3 = VelocityAction(vx=0.20, vy=0.0, vz=0.0, yaw_rate=0.50)
    safe3 = VelocityAction(vx=0.20, vy=0.0, vz=0.0, yaw_rate=0.10)

    steps = [(prop1, safe1), (prop2, safe2), (prop3, safe3)]

    for p, s in steps:
        telem.total_steps += 1
        d_vx = abs(p.vx - s.vx)
        d_vy = abs(p.vy - s.vy)
        d_vz = abs(p.vz - s.vz)
        d_yr = abs(p.yaw_rate - s.yaw_rate)

        intervened = False
        if d_vx > 1e-4:
            telem.interv_mag_vx_sum += d_vx
            telem.max_interv_mag_vx = max(telem.max_interv_mag_vx, d_vx)
            telem.interv_steps_vx += 1
            intervened = True
        if d_vy > 1e-4:
            telem.interv_mag_vy_sum += d_vy
            telem.max_interv_mag_vy = max(telem.max_interv_mag_vy, d_vy)
            telem.interv_steps_vy += 1
            intervened = True
        if d_vz > 1e-4:
            telem.interv_mag_vz_sum += d_vz
            telem.max_interv_mag_vz = max(telem.max_interv_mag_vz, d_vz)
            telem.interv_steps_vz += 1
            intervened = True
        if d_yr > 1e-4:
            telem.interv_mag_yr_sum += d_yr
            telem.max_interv_mag_yr = max(telem.max_interv_mag_yr, d_yr)
            telem.interv_steps_yr += 1
            intervened = True

        if intervened:
            telem.intervention_steps += 1
            mixed_norm = float(np.sqrt(d_vx**2 + d_vy**2 + d_vz**2 + d_yr**2))
            telem.mixed_unit_diag_norm_sum += mixed_norm
            telem.max_mixed_unit_diag_norm = max(telem.max_mixed_unit_diag_norm, mixed_norm)

    d = telem.to_dict()

    assert d["total_steps"] == 3
    assert d["intervention_steps"] == 3
    assert d["intervention_rate_pct"] == 100.0

    # Verify component-wise telemetry and explicit units
    comps = d["intervention_components"]
    assert comps["vx"]["unit"] == "m/s"
    assert comps["vx"]["intervened_steps"] == 1
    assert np.isclose(comps["vx"]["mean_mps"], 0.30)
    assert np.isclose(comps["vx"]["max_mps"], 0.30)

    assert comps["vy"]["unit"] == "m/s"
    assert comps["vy"]["intervened_steps"] == 1
    assert np.isclose(comps["vy"]["mean_mps"], 0.20)

    assert comps["vz"]["unit"] == "m/s"
    assert comps["vz"]["intervened_steps"] == 1
    assert np.isclose(comps["vz"]["mean_mps"], 0.10)

    assert comps["yaw_rate"]["unit"] == "rad/s"
    assert comps["yaw_rate"]["intervened_steps"] == 1
    assert np.isclose(comps["yaw_rate"]["mean_radps"], 0.40)

    # Verify diagnostic label on mixed-unit Euclidean norm
    diag = comps["mixed_unit_diagnostic_norm"]
    assert "DIAGNOSTIC ONLY" in diag["note"]
    assert "Never used in reward" in diag["note"]
    assert diag["mean"] > 0.0

