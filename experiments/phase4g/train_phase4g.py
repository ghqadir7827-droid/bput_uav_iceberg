"""train_phase4g.py - Isolated Experimental PPO Training Pipeline for Phase 4G.

Supports controlled experimental comparison between:
- Arm C: PPO fine-tuning without reflection augmentation (baseline replication).
- Arm D: PPO fine-tuning with bilateral reflection augmentation (50% alternating episodes).

Strict Safety and Integrity Rules:
1. Canonical checkpoints in models/checkpoints/ are never modified or overwritten.
2. Canonical contracts and supervisor remain untouched.
3. Training seeds strictly exclude VALIDATION_SEEDS (100..109) and HELD_OUT_SEEDS (50000..50099, 80000..80099).
4. All candidate checkpoints, logs, and evaluation artifacts are stored in experiments/phase4g/.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler

from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.reflection import (
    reflect_action_np,
    reflect_observation_np,
    reflect_velocity_action,
)
from models.actor_critic import ActorCriticPPO
from models.train_ppo import (
    RolloutBuffer,
    TRAINING_HARD_NEGATIVE_SEEDS,
    VALIDATION_SEEDS,
)
from sim.corridor_env import CorridorSimEnv


# Strictly reserved seed pools
PHASE4G_HELD_OUT_RANGE = (80000, 80099)
PAST_HELD_OUT_RANGE = (50000, 50099)


def sample_training_seed(hard_negative_prob: float = 0.60) -> int:
    """Samples a valid training seed ensuring zero leakage into val or held-out pools."""
    if random.random() < hard_negative_prob and TRAINING_HARD_NEGATIVE_SEEDS:
        s = random.choice(TRAINING_HARD_NEGATIVE_SEEDS)
    else:
        s = random.randint(3000, 9999)

    while (
        s in VALIDATION_SEEDS
        or (PAST_HELD_OUT_RANGE[0] <= s <= PAST_HELD_OUT_RANGE[1])
        or (PHASE4G_HELD_OUT_RANGE[0] <= s <= PHASE4G_HELD_OUT_RANGE[1])
    ):
        s = random.randint(3000, 9999)

    assert s not in VALIDATION_SEEDS, f"LEAKAGE: validation seed {s} in training!"
    assert not (PAST_HELD_OUT_RANGE[0] <= s <= PAST_HELD_OUT_RANGE[1]), f"LEAKAGE: past held-out {s} in training!"
    assert not (PHASE4G_HELD_OUT_RANGE[0] <= s <= PHASE4G_HELD_OUT_RANGE[1]), f"LEAKAGE: Phase 4G held-out {s} in training!"
    return s


def probe_policy_symmetry(model: ActorCriticPPO, device: torch.device) -> Dict[str, float]:
    """Measures open-space lateral velocity and yaw rate at corridor center Y=0.0."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=1)
    env.reset(seed=1)
    # Position at (1.0, 0.0, 1.0) with forward speed 0.60 m/s
    p.resetBasePositionAndOrientation(env.drone_id, [1.0, 0.0, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, [0.60, 0.0, 0.0], [0, 0, 0], physicsClientId=env.client_id)

    deploy_obs, _ = env._get_observations()
    obs_vec = deploy_obs.to_vector()

    model.eval()
    with torch.no_grad():
        t_obs = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        act = model.actor(t_obs, bounded=True).squeeze(0).cpu().numpy()

    env.close()
    return {
        "open_space_vx": float(act[0]),
        "open_space_vy": float(act[1]),
        "open_space_vz": float(act[2]),
        "open_space_yaw_rate": float(act[3]),
    }


def evaluate_on_validation_seeds(
    model: ActorCriticPPO,
    seeds: List[int] = VALIDATION_SEEDS,
    scenario: str = "10m",
    device: torch.device = torch.device("cpu"),
    use_supervisor: bool = True,
) -> Dict[str, Any]:
    """Evaluates candidate policy deterministically over the fixed validation seed pool."""
    model.eval()
    env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
    supervisor = SafetySupervisor(corridor=env.corridor) if use_supervisor else None

    total_episodes = len(seeds)
    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = 0
    total_interventions = 0
    min_obs_clearances = []
    min_wall_clearances = []
    forward_progresses = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        if supervisor is not None:
            supervisor.loss_timer_s = 0.0
            supervisor.stalled_timer_s = 0.0
            supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

        ep_start_x = env._get_drone_position()[0]
        done = False
        steps = 0
        min_obs = 10.0
        min_wall = 10.0

        current_info = info
        while not done and steps < 200:
            priv = current_info["privileged_state"]
            min_obs = min(min_obs, float(priv.obstacle_clearance))
            min_wall = min(min_wall, float(priv.corridor_boundary_distance))

            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act = model.actor(t_obs, bounded=True).squeeze(0).cpu().numpy()

            prop_act = VelocityAction(vx=float(act[0]), vy=float(act[1]), vz=float(act[2]), yaw_rate=float(act[3]))
            clamped_act = VelocityAction(
                vx=float(np.clip(prop_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
                vy=float(np.clip(prop_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
                vz=float(np.clip(prop_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
                yaw_rate=float(np.clip(prop_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
            )

            if supervisor is not None:
                front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
                safe_act, s_state, _ = supervisor.filter_action(
                    proposed_action=clamped_act,
                    current_position=priv.global_position,
                    current_velocity=priv.global_velocity,
                    min_front_range=front_dist,
                    dt=env.dt,
                    is_localization_healthy=True,
                    telemetry_age_s=0.0,
                    observation=obs,
                )
                if s_state != SafetyState.NOMINAL:
                    total_interventions += 1
            else:
                safe_act = clamped_act

            norm_act = safe_act.to_normalized(env.action_limits)
            obs, rew, term, trunc, current_info = env.step(norm_act)
            steps += 1
            done = term or trunc

        term_reason = current_info.get("termination_reason", "")
        if term_reason == "success":
            successes += 1
        elif "crash" in term_reason:
            collisions += 1
        elif term_reason == "timeout":
            timeouts += 1

        total_steps += steps
        final_x = env._get_drone_position()[0]
        forward_progresses.append(float(final_x - ep_start_x))
        min_obs_clearances.append(min_obs)
        min_wall_clearances.append(min_wall)

    env.close()
    return {
        "success_rate_pct": round((successes / max(1, total_episodes)) * 100.0, 1),
        "success_count": successes,
        "total_episodes": total_episodes,
        "collision_count": collisions,
        "timeout_count": timeouts,
        "mean_steps": round(total_steps / max(1, total_episodes), 1),
        "mean_forward_progress_m": round(float(np.mean(forward_progresses)), 3),
        "mean_min_obs_clearance_m": round(float(np.mean(min_obs_clearances)), 3),
        "mean_min_wall_clearance_m": round(float(np.mean(min_wall_clearances)), 3),
        "intervention_rate_pct": round((total_interventions / max(1, total_steps)) * 100.0, 2),
    }


def train_phase4g_arm(
    arm: str = "arm_d", # "arm_c" (no aug) or "arm_d" (reflection aug)
    scenario: str = "10m",
    init_mode: str = "ppo",
    ppo_checkpoint: str = "models/checkpoints/ppo_policy_best.pt",
    total_iterations: int = 10,
    rollout_steps: int = 2048,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    clip_eps: float = 0.20,
    entropy_coef: float = 0.005,
    value_coef: float = 0.50,
    hard_negative_prob: float = 0.60,
    use_supervisor: bool = True,
    exp_dir: str = "experiments/phase4g",
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes isolated PPO training for Arm C or Arm D."""
    assert arm in ["arm_c", "arm_d"], f"Unknown arm: {arm}. Must be 'arm_c' or 'arm_d'."

    # Random seed initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_reflection = (arm == "arm_d")
    arm_tag = "Arm D (Reflection Augmentation)" if use_reflection else "Arm C (Standard PPO)"
    checkpoint_name = "ppo_arm_d_reflection.pt" if use_reflection else "ppo_arm_c_baseline.pt"

    exp_path = Path(exp_dir)
    ckpt_dir = exp_path / "checkpoints"
    logs_dir = exp_path / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f">>> PHASE 4G EXPERIMENTAL TRAINING: {arm_tag.upper()} <<<")
    print(f"Arm:                  {arm}")
    print(f"Reflection Aug:       {use_reflection}")
    print(f"Warmstart Checkpoint: {ppo_checkpoint}")
    print(f"Total Iterations:     {total_iterations} ({total_iterations * rollout_steps:,} env steps)")
    print(f"Rollout Length:       {rollout_steps} steps per iteration")
    print(f"Batch Size:           {batch_size} (epochs: {ppo_epochs})")
    print(f"Learning Rate:        {lr}")
    print(f"Safety Supervisor:    {'ACTIVE' if use_supervisor else 'DISABLED'}")
    print(f"Target Checkpoint:    {ckpt_dir / checkpoint_name}")
    print("=" * 80)

    device = torch.device("cpu") # Highest stability for PyBullet interaction

    # Model initialization
    model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    if init_mode == "ppo":
        success = model.load_ppo_checkpoint(ppo_checkpoint, device=device)
        assert success, f"Failed to load PPO warmstart from {ppo_checkpoint}"
    else:
        raise ValueError(f"Unsupported init_mode for Phase 4G: {init_mode}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = RolloutBuffer(buffer_size=rollout_steps, obs_dim=23, act_dim=4)

    # Pre-training benchmark evaluation
    pre_eval = evaluate_on_validation_seeds(model, device=device, use_supervisor=use_supervisor)
    pre_symm = probe_policy_symmetry(model, device=device)
    print(f"\n[Pre-Training Validation Success]: {pre_eval['success_rate_pct']}% ({pre_eval['success_count']}/{pre_eval['total_episodes']})")
    print(f"[Pre-Training Center Probe]: vy = {pre_symm['open_space_vy']:+.4f} m/s, yaw_rate = {pre_symm['open_space_yaw_rate']:+.4f} rad/s\n")

    env = CorridorSimEnv(scenario=scenario, gui=False, seed=100)
    safety_sup = SafetySupervisor(corridor=env.corridor) if use_supervisor else None

    history = []
    best_success_rate = pre_eval["success_rate_pct"]
    best_symm_err = abs(pre_symm["open_space_yaw_rate"]) + abs(pre_symm["open_space_vy"])
    best_checkpoint_path = ckpt_dir / checkpoint_name

    # Initial candidate checkpoint
    torch.save({
        "iteration": 0,
        "model_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "eval_success_rate": pre_eval["success_rate_pct"],
        "open_space_vy": pre_symm["open_space_vy"],
        "open_space_yaw_rate": pre_symm["open_space_yaw_rate"],
        "mean_reward": 0.0,
        "obs_dim": 23,
        "act_dim": 4,
        "arm": arm,
        "use_reflection": use_reflection,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, best_checkpoint_path)

    # Episode tracking state
    curr_ep_idx = 0
    curr_seed = sample_training_seed(hard_negative_prob)
    obs_raw, info = env.reset(seed=curr_seed)
    if safety_sup is not None:
        safety_sup.loss_timer_s = 0.0
        safety_sup.stalled_timer_s = 0.0
        safety_sup.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    # Determine reflection state for initial episode
    is_ep_mirrored = use_reflection and (curr_ep_idx % 2 == 1)

    prev_pos = env._get_drone_position()
    prev_dist_goal = env.corridor.distance_to_goal(prev_pos)
    prev_act = np.zeros(4, dtype=np.float32)

    t_train_start = time.time()

    for iteration in range(1, total_iterations + 1):
        t_iter_start = time.time()
        buffer.reset()
        episode_rewards = []
        curr_ep_reward = 0.0
        episodes_in_iter = 0
        mirrored_steps_in_iter = 0

        model.eval()
        for step in range(rollout_steps):
            # 1. Observation presented to policy
            if is_ep_mirrored:
                obs_policy = reflect_observation_np(obs_raw)
                mirrored_steps_in_iter += 1
            else:
                obs_policy = obs_raw.copy()

            # 2. Sample action from policy
            with torch.no_grad():
                t_obs = torch.as_tensor(obs_policy, dtype=torch.float32, device=device).unsqueeze(0)
                act, log_prob, _, val = model.get_action_and_value(t_obs)
                act_policy_np = act.squeeze(0).cpu().numpy()
                logp_np = float(log_prob.item())
                val_np = float(val.item())

            # 3. Transform to physical action if mirrored
            if is_ep_mirrored:
                act_phys_np = reflect_action_np(act_policy_np)
            else:
                act_phys_np = act_policy_np.copy()

            proposed_action = VelocityAction(
                vx=float(act_phys_np[0]),
                vy=float(act_phys_np[1]),
                vz=float(act_phys_np[2]),
                yaw_rate=float(act_phys_np[3])
            )

            # 4. Canonical physical limit clamping
            clamped_vx = float(np.clip(proposed_action.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps))
            clamped_vy = float(np.clip(proposed_action.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps))
            clamped_vz = float(np.clip(proposed_action.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps))
            clamped_yr = float(np.clip(proposed_action.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps))
            clamped_action = VelocityAction(vx=clamped_vx, vy=clamped_vy, vz=clamped_vz, yaw_rate=clamped_yr)

            # 5. Safety Supervisor filtering (applied to physical action)
            priv = info["privileged_state"]
            if safety_sup is not None:
                front_dist = compute_forward_clearance_m(obs_raw, max_range_m=env.sensor_config.max_range_m)
                safe_action, s_state, s_reason = safety_sup.filter_action(
                    proposed_action=clamped_action,
                    current_position=priv.global_position,
                    current_velocity=priv.global_velocity,
                    min_front_range=front_dist,
                    dt=env.dt,
                    is_localization_healthy=True,
                    telemetry_age_s=0.0,
                    observation=obs_raw,
                )
            else:
                safe_action = clamped_action
                s_state = SafetyState.NOMINAL

            # 6. Apply to simulator
            norm_applied = safe_action.to_normalized(env.action_limits)
            next_obs_raw, base_rew, term, trunc, info = env.step(norm_applied)

            curr_pos = env._get_drone_position()
            curr_dist_goal = env.corridor.distance_to_goal(curr_pos)
            obs_clear = float(info.get("obstacle_clearance", 10.0))
            wall_dist = float(info.get("boundary_distance", 10.0))

            # Dense Shaped Reward Function (IDENTICAL across all arms)
            r_progress = float(prev_dist_goal - curr_dist_goal) * 4.0
            r_obs = -2.5 * float((0.40 - obs_clear) ** 2) if obs_clear < 0.40 else 0.0
            r_wall = -2.5 * float((0.35 - wall_dist) ** 2) if wall_dist < 0.35 else 0.0
            r_smooth = -0.015 * float(np.sum((act_policy_np - prev_act) ** 2))

            r_term = 0.0
            term_reason = info.get("termination_reason", "")
            if term_reason == "success":
                r_term = 20.0
            elif "crash" in term_reason:
                r_term = -20.0

            shaped_reward = r_progress + r_obs + r_wall + r_smooth + r_term
            curr_ep_reward += shaped_reward

            done = term or trunc

            # Value bootstrapping on timeout
            timeout_val = 0.0
            if trunc and not term:
                next_obs_policy = reflect_observation_np(next_obs_raw) if is_ep_mirrored else next_obs_raw
                with torch.no_grad():
                    t_next_obs = torch.as_tensor(next_obs_policy, dtype=torch.float32, device=device).unsqueeze(0)
                    timeout_val = float(model.critic(t_next_obs).item())

            # Store in PPO buffer (observations and actions in policy's coordinate frame)
            buffer.add(
                obs=obs_policy,
                act=act_policy_np,
                rew=shaped_reward,
                done=float(done),
                val=val_np,
                logp=logp_np,
                terminal=bool(term),
                timeout=bool(trunc and not term),
                timeout_value=timeout_val,
            )

            obs_raw = next_obs_raw
            prev_dist_goal = curr_dist_goal
            prev_act = act_policy_np

            if done:
                episode_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
                episodes_in_iter += 1

                # Advance episode and sample next seed
                curr_ep_idx += 1
                curr_seed = sample_training_seed(hard_negative_prob)
                obs_raw, info = env.reset(seed=curr_seed)
                if safety_sup is not None:
                    safety_sup.loss_timer_s = 0.0
                    safety_sup.stalled_timer_s = 0.0
                    safety_sup.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

                is_ep_mirrored = use_reflection and (curr_ep_idx % 2 == 1)
                prev_pos = env._get_drone_position()
                prev_dist_goal = env.corridor.distance_to_goal(prev_pos)
                prev_act = np.zeros(4, dtype=np.float32)

        # Compute GAE
        last_obs_policy = reflect_observation_np(obs_raw) if is_ep_mirrored else obs_raw
        with torch.no_grad():
            t_last = torch.as_tensor(last_obs_policy, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, last_val = model.get_action_and_value(t_last)
            last_val_f = float(last_val.item())
        buffer.compute_gae(last_val_f)

        # PPO Learning Phase
        model.train()
        b_obs = torch.as_tensor(buffer.observations, dtype=torch.float32, device=device)
        b_acts = torch.as_tensor(buffer.actions, dtype=torch.float32, device=device)
        b_logp = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(buffer.advantages, dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(buffer.returns, dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        update_count = 0

        for _ in range(ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(rollout_steps)), batch_size=batch_size, drop_last=False)
            for batch_indices in sampler:
                mb_obs = b_obs[batch_indices]
                mb_acts = b_acts[batch_indices]
                mb_logp = b_logp[batch_indices]
                mb_adv = b_adv[batch_indices]
                mb_ret = b_ret[batch_indices]

                _, new_logp, entropy, new_val = model.get_action_and_value(mb_obs, mb_acts)

                ratio = torch.exp(new_logp - mb_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * torch.mean((new_val - mb_ret) ** 2)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                total_pi_loss += float(policy_loss.item())
                total_v_loss += float(value_loss.item())
                total_ent += float(entropy.mean().item())
                update_count += 1

        avg_pi_loss = total_pi_loss / max(1, update_count)
        avg_v_loss = total_v_loss / max(1, update_count)
        avg_ent = total_ent / max(1, update_count)
        mean_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0

        # Periodic Validation & Symmetry Probe
        val_eval = evaluate_on_validation_seeds(model, device=device, use_supervisor=use_supervisor)
        symm_probe = probe_policy_symmetry(model, device=device)

        # Checkpoint selection rule:
        # Primary: Higher validation success rate
        # Secondary: If success rate ties, lower combined symmetry error |yaw_rate| + |vy|
        curr_symm_err = abs(symm_probe["open_space_yaw_rate"]) + abs(symm_probe["open_space_vy"])
        is_best = (val_eval["success_rate_pct"] > best_success_rate) or (
            val_eval["success_rate_pct"] == best_success_rate and curr_symm_err < best_symm_err
        )
        if is_best:
            best_success_rate = val_eval["success_rate_pct"]
            best_symm_err = curr_symm_err
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.actor.state_dict(),
                "critic_state_dict": model.critic.state_dict(),
                "eval_success_rate": val_eval["success_rate_pct"],
                "open_space_vy": symm_probe["open_space_vy"],
                "open_space_yaw_rate": symm_probe["open_space_yaw_rate"],
                "mean_reward": mean_rew,
                "obs_dim": 23,
                "act_dim": 4,
                "arm": arm,
                "use_reflection": use_reflection,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, best_checkpoint_path)
            print(f"  >>> New Best Checkpoint Saved at Iteration {iteration}: {best_success_rate:.1f}% success, vy={symm_probe['open_space_vy']:+.4f} m/s, yaw_rate={symm_probe['open_space_yaw_rate']:+.4f} rad/s")

        iter_record = {
            "iteration": iteration,
            "mean_reward": round(mean_rew, 3),
            "policy_loss": round(avg_pi_loss, 4),
            "value_loss": round(avg_v_loss, 4),
            "entropy": round(avg_ent, 4),
            "episodes_completed": episodes_in_iter,
            "mirrored_step_fraction": round(mirrored_steps_in_iter / rollout_steps, 3),
            "val_success_rate_pct": val_eval["success_rate_pct"],
            "val_mean_progress_m": val_eval["mean_forward_progress_m"],
            "val_intervention_rate_pct": val_eval["intervention_rate_pct"],
            "open_space_vy": round(symm_probe["open_space_vy"], 4),
            "open_space_yaw_rate": round(symm_probe["open_space_yaw_rate"], 4),
            "iteration_time_s": round(time.time() - t_iter_start, 2),
        }
        history.append(iter_record)

        print(
            f"Iter {iteration:2d} | Rew: {mean_rew:6.2f} | PiLoss: {avg_pi_loss:6.4f} | "
            f"ValSucc: {val_eval['success_rate_pct']:4.1f}% | OpenSpace vy: {symm_probe['open_space_vy']:+6.4f} m/s | "
            f"yaw_rate: {symm_probe['open_space_yaw_rate']:+6.4f} rad/s"
        )

    env.close()

    final_checkpoint_path = ckpt_dir / f"ppo_{arm}_final.pt"
    torch.save({
        "iteration": total_iterations,
        "model_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "eval_success_rate": val_eval["success_rate_pct"],
        "open_space_vy": symm_probe["open_space_vy"],
        "open_space_yaw_rate": symm_probe["open_space_yaw_rate"],
        "obs_dim": 23,
        "act_dim": 4,
        "arm": arm,
        "use_reflection": use_reflection,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, final_checkpoint_path)

    # Save complete training log
    log_file = logs_dir / f"{arm}_training_log.json"
    log_data = {
        "arm": arm,
        "use_reflection": use_reflection,
        "init_checkpoint": ppo_checkpoint,
        "saved_checkpoint": str(best_checkpoint_path),
        "total_iterations": total_iterations,
        "rollout_steps_per_iter": rollout_steps,
        "total_env_steps": total_iterations * rollout_steps,
        "pre_training_eval": pre_eval,
        "pre_training_symmetry": pre_symm,
        "history": history,
        "final_validation": val_eval,
        "final_symmetry": symm_probe,
        "total_training_time_s": round(time.time() - t_train_start, 2),
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nTraining log saved to {log_file}")
    return log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4G Isolated Experimental Training")
    parser.add_argument("--arm", type=str, choices=["arm_c", "arm_d"], default="arm_d", help="Experimental arm")
    parser.add_argument("--iterations", type=int, default=10, help="Number of PPO iterations")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout buffer size")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-iteration smoke test")
    args = parser.parse_args()

    iters = 1 if args.smoke_test else args.iterations
    rollout = 256 if args.smoke_test else args.rollout_steps
    train_phase4g_arm(arm=args.arm, total_iterations=iters, rollout_steps=rollout)
