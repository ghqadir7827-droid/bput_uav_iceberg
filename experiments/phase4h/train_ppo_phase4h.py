"""train_ppo_phase4h.py - Phase 4H PPO Candidate Training Pipeline.

Principal Intervention: Full 500-Step Canonical Horizon with Bilateral Reflection Augmentation.

Key enhancements over Phase 4G:
1. Uses the canonical 500-step episode horizon (eliminates the artificial 200-step cutoff).
2. Maintains bilateral reflection augmentation to guarantee symmetry and eliminate lateral bias.
3. Incorporates terminal arrival braking incentive (rewarding v_term <= 0.35 m/s upon goal arrival).
4. Strictly preserves canonical checkpoints: saves to experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt.
5. Seeds strictly drawn from training range, avoiding validation seeds (100..109) and held-out seeds (90000..90099).
"""

import argparse
import hashlib
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
)
from models.actor_critic import ActorCriticPPO
from models.train_ppo import RolloutBuffer, TRAINING_HARD_NEGATIVE_SEEDS, VALIDATION_SEEDS
from sim.corridor_env import CorridorSimEnv


RESERVED_HELD_OUT_POOLS = [
    (50000, 50099), # Phase 4C held-out
    (80000, 80099), # Phase 4G held-out
    (90000, 90099), # Phase 4H reserved held-out
]


def sample_training_seed(hard_negative_prob: float = 0.50) -> int:
    """Samples a valid training seed ensuring zero leakage into val or held-out pools."""
    if random.random() < hard_negative_prob and TRAINING_HARD_NEGATIVE_SEEDS:
        s = random.choice(TRAINING_HARD_NEGATIVE_SEEDS)
    else:
        s = random.randint(3000, 9999)

    while (
        s in VALIDATION_SEEDS
        or any(low <= s <= high for low, high in RESERVED_HELD_OUT_POOLS)
    ):
        s = random.randint(3000, 9999)

    return s


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def probe_policy_symmetry(model: ActorCriticPPO, device: torch.device) -> Dict[str, float]:
    """Measures open-space lateral velocity and yaw rate at corridor center Y=0.0."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=1)
    env.reset(seed=1)
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


def evaluate_on_validation_seeds_500s(
    model: ActorCriticPPO,
    seeds: List[int] = VALIDATION_SEEDS,
    scenario: str = "10m",
    device: torch.device = torch.device("cpu"),
    use_supervisor: bool = True,
) -> Dict[str, Any]:
    """Evaluates candidate policy deterministically over the fixed validation seed pool under 500-step horizon."""
    model.eval()
    env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
    supervisor = SafetySupervisor(corridor=env.corridor) if use_supervisor else None

    total_episodes = len(seeds)
    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = 0
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
        while not done and steps < 500:
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
            else:
                safe_act = clamped_act

            norm_act = safe_act.to_normalized(env.action_limits)
            obs, rew, term, trunc, current_info = env.step(norm_act)
            steps += 1
            done = term or trunc

        final_pos = env._get_drone_position()
        fwd_progress = float(final_pos[0] - ep_start_x)
        term_reason = current_info.get("termination_reason", "unknown")

        if term_reason == "success":
            successes += 1
        elif "crash" in term_reason:
            collisions += 1
        elif steps >= 500 or term_reason == "timeout":
            timeouts += 1

        total_steps += steps
        min_obs_clearances.append(min_obs)
        min_wall_clearances.append(min_wall)
        forward_progresses.append(fwd_progress)

    env.close()
    return {
        "success_rate_pct": round((successes / total_episodes) * 100.0, 1),
        "success_count": successes,
        "collision_count": collisions,
        "timeout_count": timeouts,
        "total_episodes": total_episodes,
        "mean_steps": round(total_steps / total_episodes, 1),
        "mean_fwd_progress_m": round(float(np.mean(forward_progresses)), 3),
        "min_obs_clearance_m": round(float(np.min(min_obs_clearances)), 3),
        "min_wall_clearance_m": round(float(np.min(min_wall_clearances)), 3),
    }


def train_ppo_candidate(
    warmstart_checkpoint: str = "experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt",
    output_checkpoint: str = "experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt",
    total_iterations: int = 15,
    rollout_steps: int = 1024,
    ppo_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    clip_eps: float = 0.20,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Dict[str, Any]:
    print("=" * 80)
    print(">>> PHASE 4H PPO CANDIDATE TRAINING: CANONICAL 500S HORIZON + REFLECTION <<<")
    print(f"Warmstart Checkpoint: {warmstart_checkpoint}")
    print(f"Output Target:        {output_checkpoint}")
    print(f"Total Iterations:     {total_iterations} ({total_iterations * rollout_steps:,} env steps)")
    print(f"Rollout Length:       {rollout_steps} steps per iteration")
    print(f"Horizon per Episode:  500 steps (25 s)")
    print(f"Batch Size:           {batch_size} (epochs: {ppo_epochs})")
    print(f"Learning Rate:        {lr}")
    print("=" * 80)

    device = torch.device("cpu")
    model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    success = model.load_ppo_checkpoint(warmstart_checkpoint, device=device)
    assert success, f"Failed to load warmstart checkpoint: {warmstart_checkpoint}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = RolloutBuffer(buffer_size=rollout_steps, obs_dim=23, act_dim=4)

    # Pre-training benchmark
    pre_eval = evaluate_on_validation_seeds_500s(model, device=device)
    pre_symm = probe_policy_symmetry(model, device=device)
    print(f"\n[Pre-Training Validation Success (500s)]: {pre_eval['success_rate_pct']}% ({pre_eval['success_count']}/{pre_eval['total_episodes']})")
    print(f"[Pre-Training Center Probe]: vy = {pre_symm['open_space_vy']:+.4f} m/s, yaw_rate = {pre_symm['open_space_yaw_rate']:+.4f} rad/s\n")

    ckpt_path = Path(output_checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_success_rate = pre_eval["success_rate_pct"]
    best_fwd_progress = pre_eval["mean_fwd_progress_m"]

    # Initial checkpoint save
    torch.save({
        "iteration": 0,
        "model_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "eval_success_rate": pre_eval["success_rate_pct"],
        "open_space_vy": pre_symm["open_space_vy"],
        "open_space_yaw_rate": pre_symm["open_space_yaw_rate"],
        "obs_dim": 23,
        "act_dim": 4,
        "use_reflection": True,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ckpt_path)

    env = CorridorSimEnv(scenario="10m", gui=False, seed=3000)
    safety_sup = SafetySupervisor(corridor=env.corridor)

    curr_ep_idx = 0
    curr_seed = sample_training_seed()
    obs_raw, info = env.reset(seed=curr_seed)
    safety_sup.loss_timer_s = 0.0
    safety_sup.stalled_timer_s = 0.0
    safety_sup.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    is_ep_mirrored = (curr_ep_idx % 2 == 1)
    prev_pos = env._get_drone_position()
    prev_dist_goal = env.corridor.distance_to_goal(prev_pos)
    prev_act = np.zeros(4, dtype=np.float32)

    history = []
    t_train_start = time.time()

    for iteration in range(1, total_iterations + 1):
        t_iter_start = time.time()
        buffer.reset()
        episode_rewards = []
        curr_ep_reward = 0.0
        episodes_in_iter = 0

        model.eval()
        for step in range(rollout_steps):
            if is_ep_mirrored:
                obs_policy = reflect_observation_np(obs_raw)
            else:
                obs_policy = obs_raw.copy()

            with torch.no_grad():
                t_obs = torch.as_tensor(obs_policy, dtype=torch.float32, device=device).unsqueeze(0)
                act, log_prob, _, val = model.get_action_and_value(t_obs)
                act_policy_np = act.squeeze(0).cpu().numpy()
                logp_np = float(log_prob.item())
                val_np = float(val.item())

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

            clamped_vx = float(np.clip(proposed_action.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps))
            clamped_vy = float(np.clip(proposed_action.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps))
            clamped_vz = float(np.clip(proposed_action.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps))
            clamped_yr = float(np.clip(proposed_action.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps))
            clamped_action = VelocityAction(vx=clamped_vx, vy=clamped_vy, vz=clamped_vz, yaw_rate=clamped_yr)

            priv = info["privileged_state"]
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

            norm_applied = safe_action.to_normalized(env.action_limits)
            next_obs_raw, base_rew, term, trunc, info = env.step(norm_applied)

            curr_pos = env._get_drone_position()
            curr_dist_goal = env.corridor.distance_to_goal(curr_pos)
            obs_clear = float(info.get("obstacle_clearance", 10.0))
            wall_dist = float(info.get("boundary_distance", 10.0))

            # Reward shaping
            r_progress = float(prev_dist_goal - curr_dist_goal) * 4.0
            r_obs = -2.5 * float((0.40 - obs_clear) ** 2) if obs_clear < 0.40 else 0.0
            r_wall = -2.5 * float((0.35 - wall_dist) ** 2) if wall_dist < 0.35 else 0.0
            r_smooth = -0.015 * float(np.sum((act_policy_np - prev_act) ** 2))

            r_term = 0.0
            term_reason = info.get("termination_reason", "")
            if term_reason == "success":
                # Check arrival speed braking requirement (v <= 0.35 m/s)
                v_term = float(np.linalg.norm(info["privileged_state"].global_velocity))
                if v_term <= 0.35:
                    r_term = 25.0 # Full bonus for clean, safe arrival
                else:
                    r_term = 10.0 # Reduced bonus if overspeeding
            elif "crash" in term_reason:
                r_term = -20.0

            shaped_reward = r_progress + r_obs + r_wall + r_smooth + r_term
            curr_ep_reward += shaped_reward

            done = term or trunc

            timeout_val = 0.0
            if trunc and not term:
                next_obs_policy = reflect_observation_np(next_obs_raw) if is_ep_mirrored else next_obs_raw
                with torch.no_grad():
                    t_next_obs = torch.as_tensor(next_obs_policy, dtype=torch.float32, device=device).unsqueeze(0)
                    timeout_val = float(model.critic(t_next_obs).item())

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

                curr_ep_idx += 1
                curr_seed = sample_training_seed()
                obs_raw, info = env.reset(seed=curr_seed)
                safety_sup.loss_timer_s = 0.0
                safety_sup.stalled_timer_s = 0.0
                safety_sup.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

                is_ep_mirrored = (curr_ep_idx % 2 == 1)
                prev_pos = env._get_drone_position()
                prev_dist_goal = env.corridor.distance_to_goal(prev_pos)
                prev_act = np.zeros(4, dtype=np.float32)

        last_obs_policy = reflect_observation_np(obs_raw) if is_ep_mirrored else obs_raw
        with torch.no_grad():
            t_last = torch.as_tensor(last_obs_policy, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, last_val = model.get_action_and_value(t_last)
            last_val_f = float(last_val.item())
        buffer.compute_gae(last_val_f, gamma=gamma, lam=gae_lambda)

        # Optimization
        model.train()
        b_obs = torch.as_tensor(buffer.observations, dtype=torch.float32, device=device)
        b_acts = torch.as_tensor(buffer.actions, dtype=torch.float32, device=device)
        b_logp = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(buffer.advantages, dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(buffer.returns, dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for epoch in range(ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(rollout_steps)), batch_size=batch_size, drop_last=False)
            for batch_indices in sampler:
                mb_obs = b_obs[batch_indices]
                mb_act = b_acts[batch_indices]
                mb_ret = b_ret[batch_indices]
                mb_adv = b_adv[batch_indices]
                mb_old_logp = b_logp[batch_indices]

                _, new_logp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)
                ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * torch.mean((new_val.squeeze(-1) - mb_ret) ** 2)
                entropy_loss = -0.01 * entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        mean_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        iter_time = time.time() - t_iter_start

        # Benchmark every 3 iterations or final
        if iteration % 3 == 0 or iteration == total_iterations:
            val_eval = evaluate_on_validation_seeds_500s(model, device=device)
            symm = probe_policy_symmetry(model, device=device)

            is_best = (val_eval["success_rate_pct"] > best_success_rate) or (
                val_eval["success_rate_pct"] == best_success_rate and val_eval["mean_fwd_progress_m"] > best_fwd_progress
            )
            if is_best:
                best_success_rate = val_eval["success_rate_pct"]
                best_fwd_progress = val_eval["mean_fwd_progress_m"]
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": model.actor.state_dict(),
                    "critic_state_dict": model.critic.state_dict(),
                    "eval_success_rate": best_success_rate,
                    "mean_fwd_progress": best_fwd_progress,
                    "open_space_vy": symm["open_space_vy"],
                    "open_space_yaw_rate": symm["open_space_yaw_rate"],
                    "obs_dim": 23,
                    "act_dim": 4,
                    "use_reflection": True,
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, ckpt_path)

            flag = " *" if is_best else ""
            print(f"Iter {iteration:02d}/{total_iterations} | Rew: {mean_rew:+6.2f} | Val Succ: {val_eval['success_rate_pct']:4.1f}% | Progress: {val_eval['mean_fwd_progress_m']:4.2f}m | vy: {symm['open_space_vy']:+.4f} | Time: {iter_time:4.1f}s{flag}")
        else:
            print(f"Iter {iteration:02d}/{total_iterations} | Rew: {mean_rew:+6.2f} | Eps: {episodes_in_iter:02d} | Time: {iter_time:4.1f}s")

    env.close()
    total_train_time = time.time() - t_train_start
    ckpt_hash = sha256_file(ckpt_path)

    print("\n" + "=" * 80)
    print(">>> PHASE 4H PPO CANDIDATE TRAINING COMPLETE <<<")
    print(f"Total Training Time:    {total_train_time:.1f} s")
    print(f"Best Val Success Rate:  {best_success_rate:.1f}%")
    print(f"Saved Checkpoint:       {ckpt_path.resolve()}")
    print(f"SHA-256 Checksum:       {ckpt_hash}")
    print("=" * 80)

    log_path = Path("experiments/phase4h/logs/ppo_training_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "status": "TRAINING_COMPLETE",
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "total_iterations": total_iterations,
            "best_validation_success_pct": best_success_rate,
            "total_train_time_s": round(total_train_time, 2),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    return {"status": "SUCCESS", "checkpoint_sha256": ckpt_hash}


if __name__ == "__main__":
    train_ppo_candidate()
