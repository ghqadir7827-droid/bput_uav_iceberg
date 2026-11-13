"""reflection.py - Bilateral Reflection Transformation across Lateral Center Plane.

Formalizes the mathematical reflection transformation across the corridor's
lateral symmetry plane (Y = 0) for the 23-dim DeployableObservation and
4-dim VelocityAction vectors.

Coordinate Conventions:
- World Frame: ENU (X = Corridor length [0, 10] m, Y = Corridor width [-1, 1] m, Z = Height [0, 2] m).
- Body Frame: FLU (X = Forward, Y = Left, Z = Up).
- Reflection Plane: Corridor longitudinal center plane (Y = 0).

Transformation Semantics:
1. Rangefinder Rays (12 channels, normalized in [0, 1]):
   - Ray 0 (Azimuth 0 deg, Pitch 0 deg): Front center -> unchanged
   - Ray 1 (Azimuth +30 deg) <-> Ray 2 (Azimuth -30 deg): Front-Left <-> Front-Right swapped
   - Ray 3 (Azimuth +60 deg) <-> Ray 4 (Azimuth -60 deg): Side-Front-Left <-> Side-Front-Right swapped
   - Ray 5 (Azimuth +90 deg) <-> Ray 6 (Azimuth -90 deg): Left 90 deg <-> Right 90 deg swapped
   - Ray 7 (Azimuth 180 deg, Pitch 0 deg): Rear center -> unchanged
   - Ray 8 (Pitch +45 deg): Pitch up 45 deg -> unchanged
   - Ray 9 (Pitch -45 deg): Pitch down 45 deg -> unchanged
   - Ray 10 (Pitch +90 deg): Zenith ceiling -> unchanged
   - Ray 11 (Pitch -90 deg): Nadir floor -> unchanged
2. Relative Goal Vector in Body FLU (3 features):
   - g_x: Forward goal direction -> unchanged
   - g_y: Lateral goal direction (+Left, -Right) -> negated (-g_y)
   - g_z: Vertical goal direction (+Up, -Down) -> unchanged
3. Body Velocity in Body FLU (3 features):
   - v_x: Forward velocity -> unchanged
   - v_y: Lateral velocity (+Left, -Right) -> negated (-v_y)
   - v_z: Vertical velocity (+Up, -Down) -> unchanged
4. Trigonometric Attitude (4 features: [sin(roll), cos(roll), sin(pitch), cos(pitch)]):
   - roll phi -> -phi => sin(-phi) = -sin(phi), cos(-phi) = cos(phi)
   - pitch theta -> theta => sin(theta), cos(theta) unchanged
   - att_trig -> [-sin(roll), cos(roll), sin(pitch), cos(pitch)]
5. Yaw Angular Rate (1 feature):
   - yaw_rate psi_dot -> -psi_dot (CCW positive becomes CW negative)

Action Semantics (4 continuous dimensions):
- vx: Forward velocity -> unchanged
- vy: Lateral velocity -> negated (-vy)
- vz: Vertical velocity -> unchanged
- yaw_rate: Angular rate -> negated (-yaw_rate)
"""

from typing import Union
import numpy as np
import torch

from core.contracts import DeployableObservation, VelocityAction


# Ray swap index mapping for 12 rangefinder rays:
# [0, 2, 1, 4, 3, 6, 5, 7, 8, 9, 10, 11]
RAY_REFLECTION_PERMUTATION = np.array([0, 2, 1, 4, 3, 6, 5, 7, 8, 9, 10, 11], dtype=np.int64)

# Sign flip mask for the 23-dim observation vector:
# Positive (+1) for invariant features, Negative (-1) for mirrored lateral/yaw features
# Rays (0..11): all +1 (handled by permutation)
# Goal (12..14): [1, -1, 1]
# Velocity (15..17): [1, -1, 1]
# Attitude trig (18..21): [-1, 1, 1, 1]
# Yaw rate (22): -1
OBSERVATION_SIGN_FLIPS = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 0..11 rays
    1.0, -1.0, 1.0,                                               # 12..14 goal
    1.0, -1.0, 1.0,                                               # 15..17 vel
    -1.0, 1.0, 1.0, 1.0,                                          # 18..21 attitude trig
    -1.0                                                          # 22 yaw rate
], dtype=np.float32)

# Action sign flips for [vx, vy, vz, yaw_rate]: [1, -1, 1, -1]
ACTION_SIGN_FLIPS = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)


def reflect_observation_np(obs: np.ndarray) -> np.ndarray:
    """Reflects a 23-dim observation vector or (N, 23) batch across Y=0 plane.
    
    Parameters:
        obs: 1D array of shape (23,) or 2D array of shape (N, 23).
        
    Returns:
        Reflected observation array of identical shape and dtype.
    """
    arr = np.array(obs, copy=True)
    if arr.ndim == 1:
        assert arr.shape == (23,), f"Expected shape (23,), got {arr.shape}"
        # Permute rays
        arr[:12] = arr[RAY_REFLECTION_PERMUTATION]
        # Flip signs of lateral/yaw features
        arr[12:] *= OBSERVATION_SIGN_FLIPS[12:]
    elif arr.ndim == 2:
        assert arr.shape[1] == 23, f"Expected shape (N, 23), got {arr.shape}"
        arr[:, :12] = arr[:, RAY_REFLECTION_PERMUTATION]
        arr[:, 12:] *= OBSERVATION_SIGN_FLIPS[12:]
    else:
        raise ValueError(f"Unsupported observation shape: {arr.shape}")
    return arr


def reflect_observation_torch(obs: torch.Tensor) -> torch.Tensor:
    """Reflects a 23-dim PyTorch observation tensor or (N, 23) batch across Y=0 plane."""
    perm = torch.as_tensor(RAY_REFLECTION_PERMUTATION, device=obs.device)
    flips = torch.as_tensor(OBSERVATION_SIGN_FLIPS, dtype=obs.dtype, device=obs.device)

    if obs.ndim == 1:
        assert obs.shape[0] == 23
        reflected_rays = obs[:12][perm]
        reflected_rest = obs[12:] * flips[12:]
        return torch.cat([reflected_rays, reflected_rest], dim=0)
    elif obs.ndim == 2:
        assert obs.shape[1] == 23
        reflected_rays = obs[:, perm]
        reflected_rest = obs[:, 12:] * flips[12:]
        return torch.cat([reflected_rays, reflected_rest], dim=1)
    else:
        raise ValueError(f"Unsupported tensor shape: {obs.shape}")


def reflect_observation(obs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Polymorphic reflection function supporting NumPy array and PyTorch tensor."""
    if isinstance(obs, torch.Tensor):
        return reflect_observation_torch(obs)
    return reflect_observation_np(np.asarray(obs))


def reflect_action_np(act: np.ndarray) -> np.ndarray:
    """Reflects a 4-dim action vector [vx, vy, vz, yaw_rate] across Y=0 plane."""
    arr = np.array(act, copy=True)
    if arr.ndim == 1:
        assert arr.shape == (4,), f"Expected shape (4,), got {arr.shape}"
        arr *= ACTION_SIGN_FLIPS
    elif arr.ndim == 2:
        assert arr.shape[1] == 4, f"Expected shape (N, 4), got {arr.shape}"
        arr *= ACTION_SIGN_FLIPS
    else:
        raise ValueError(f"Unsupported action shape: {arr.shape}")
    return arr


def reflect_action_torch(act: torch.Tensor) -> torch.Tensor:
    """Reflects a 4-dim PyTorch action tensor across Y=0 plane."""
    flips = torch.as_tensor(ACTION_SIGN_FLIPS, dtype=act.dtype, device=act.device)
    return act * flips


def reflect_action(act: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Polymorphic reflection function for action vectors."""
    if isinstance(act, torch.Tensor):
        return reflect_action_torch(act)
    return reflect_action_np(np.asarray(act))


def reflect_velocity_action(action: VelocityAction) -> VelocityAction:
    """Reflects a VelocityAction dataclass instance across Y=0 plane."""
    return VelocityAction(
        vx=float(action.vx),
        vy=float(-action.vy),
        vz=float(action.vz),
        yaw_rate=float(-action.yaw_rate)
    )


def reflect_deployable_observation(obs: DeployableObservation) -> DeployableObservation:
    """Reflects a DeployableObservation dataclass instance across Y=0 plane."""
    vec = obs.to_vector()
    reflected_vec = reflect_observation_np(vec)
    return DeployableObservation(
        distance_rays=reflected_vec[:12],
        rel_goal_direction_body=reflected_vec[12:15],
        body_velocity=reflected_vec[15:18],
        attitude_trig=reflected_vec[18:22],
        yaw_rate=float(reflected_vec[22])
    )
