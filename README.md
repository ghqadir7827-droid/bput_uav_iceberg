# Autonomous UAV Navigation in GPS-Denied 3D Corridors

Autonomous 3D corridor navigation system for physical unmanned aerial vehicles (UAVs) operating in GPS-denied environments. Designed for a **scaled 65 cm physical quadrotor platform** powered by an onboard companion computer (**Raspberry Pi 5, ARM64**) interfaced to a flight controller (**RadioLink CrossFlight / PX4**).

---

## System Overview & Architecture

The system enables collision-free, high-speed transit through constrained 3D corridors with internal obstacles, boundary walls, and variable corridor geometries under strict zero-GPS constraints.

```
                           +------------------------------------------+
                           |  Physical Sensors (Rangefinders, IMU)   |
                           +------------------------------------------+
                                                |
                                                v
                           +------------------------------------------+
                           |   Deployable Observation (23 Floats)    |
                           +------------------------------------------+
                                                |
                                                v
                           +------------------------------------------+
                           | Neural Policy: NavigationPolicyMLP       |
                           | (28,740 params, LayerNorm, GELU, Tanh)   |
                           +------------------------------------------+
                                                |
                                                v  Raw Velocity Action
                           +------------------------------------------+
                           | Stateless Clearance Guidance Arbitrator  |
                           | (Directional Persistence Tie-Breaker)    |
                           +------------------------------------------+
                                                |
                                                v  Guided Velocity Action
                           +------------------------------------------+
                           | ActionLimits Clipper                     |
                           | (Physical Actuation Bounds)              |
                           +------------------------------------------+
                                                |
                                                v  Bounded Action
                           +------------------------------------------+
                           | Deterministic Safety Supervisor          |
                           | (Velocity Projection & Dynamic Braking)  |
                           +------------------------------------------+
                                                |
                                                v  Safe Velocity Setpoint
                           +------------------------------------------+
                           | Flight Controller (PX4 / CrossFlight)    |
                           +------------------------------------------+
```

### Key Pillars

1. **Perception Contract (23-Float Student Observation):**
   - **Rangefinder Rays [0:12]:** 12 multi-beam distance rays normalized to $[0.0, 1.0]$ via $d / 4.0\text{ m}$.
   - **Goal Direction [12:15]:** Unit direction vector in body FLU (Forward-Left-Up) frame, scaled proportionally within terminal range ($< 2.0\text{ m}$).
   - **Body Velocity [15:18]:** Estimated linear body velocities $[v_x, v_y, v_z]\text{ m/s}$.
   - **Attitude Features [18:22]:** Trigonometric orientation features $[\sin(\phi), \cos(\phi), \sin(\theta), \cos(\theta)]$.
   - **Yaw Rate [22]:** Body z-axis angular velocity $\dot{\psi}\text{ rad/s}$.

2. **Neural Navigation Policy (`NavigationPolicyMLP`):**
   - Supervised Imitation Learning (Behavior Cloning) augmented with recovery perturbation trajectories.
   - Bounded output head with scaled $\tanh$ strictly guaranteeing velocity limits ($v_x \in [-0.2, 0.8]\text{ m/s}$, $v_y \in [-0.6, 0.6]\text{ m/s}$, $v_z \in [-0.4, 0.4]\text{ m/s}$, $\dot{\psi} \in [-0.6, 0.6]\text{ rad/s}$).
   - 28,740 trainable parameters ($\approx 115\text{ KB}$ raw float32 weights, $146.7\text{ KB}$ single-file ONNX binary).

3. **Stateless Clearance Guidance Layer (v3 Persistent Guidance):**
   - Sits between the neural network policy and the Safety Supervisor.
   - Forward obstacle detection across 5 forward rays with dynamic wall-margin evaluation.
   - Directional persistence latching suppresses high-frequency symmetric tie-breaking chattering in wide corridors.
   - Soft terminal landing release yields control to the raw policy within $1.0\text{ m}$ of the destination.

4. **Deterministic Safety Supervisor:**
   - Independent verification barrier enforcing corridor boundaries and obstacle caution distances ($0.50\text{ m}$).
   - Dynamic velocity projection guarantees that forward velocity never exceeds stopping distance $d_{\text{stop}} = \frac{v_x^2}{2 a_{\text{max}}} + r_{\text{drone}} + d_{\text{buffer}}$.

5. **Embedded Edge Inference Engine (Raspberry Pi 5):**
   - High-frequency ($50\text{ Hz}$) companion computer inference runner (`hardware/onnx_runner.py`).
   - Pure ONNX Runtime + NumPy execution with **zero PyTorch dependency**.
   - Single-step inference latency: $< 0.18\text{ ms}$ on ARM Cortex-A76 ($< 0.9\%$ CPU core load).
   - Numerical equivalence against original PyTorch model certified to $\max |\Delta| \le 1.40 \times 10^{-6}\text{ m/s}$.

---

## Repository Structure

```
├── controllers/              # Control logic & policy wrappers
│   ├── apf_expert.py         # Scripted Artificial Potential Field expert demonstrator
│   ├── bc_policy.py          # Supervised Behavior Cloning policy wrapper
│   ├── guidance_controller.py# Stateless Clearance Guidance Controller
│   ├── onnx_controller.py    # ONNX Runtime edge controller
│   ├── registry.py           # Unified controller & checkpoint discovery registry
│   └── safety_supervisor.py  # Deterministic safety supervisor & envelope filter
├── core/                     # Core system contracts & geometry
│   ├── contracts.py          # Data classes (DeployableObservation, VelocityAction, etc.)
│   ├── geometry.py           # 3D rectangular corridor segment models & coordinate math
│   ├── guidance.py           # Pure-NumPy Stateless Clearance Guidance Arbitrator
│   └── physical_dimensions.py# Physical UAV footprint & rotor radius invariants
├── hardware/                 # Embedded edge deployment
│   └── onnx_runner.py        # Standalone ONNX companion computer runner (Raspberry Pi)
├── models/                   # Neural network architectures & training pipelines
│   ├── actor_critic.py       # PPO Actor-Critic network architecture
│   ├── checkpoints/          # Model weights (.pt) and exported ONNX graphs (.onnx)
│   ├── dataset.py            # PyTorch dataset loader for expert demonstrations
│   ├── export_onnx.py        # Generic ONNX export & latency verification pipeline
│   ├── policy_net.py         # Lightweight NavigationPolicyMLP architecture
│   └── train_ppo.py          # Curriculum PPO fine-tuning pipeline
├── sim/                      # Simulation physics engine
│   └── corridor_env.py       # PyBullet-based 3D corridor simulation environment
├── viewer/                   # Interactive 3D Web Cockpit & Evaluation Service
│   ├── eval_engine.py        # Replayable evaluation engine & telemetry recorder
│   ├── server.py             # HTTP service hosting Three.js viewer on port 8050
│   └── static/               # HTML5 / Three.js 3D viewport, telemetry HUD & controls
├── scripts/                  # Evaluation, benchmarking, and export utilities
│   ├── export_and_validate_bc_v3.py  # Deterministic ONNX export & parity harness
│   ├── run_pybullet_gui.py           # Native OpenGL graphical PyBullet viewer
│   ├── run_quantitative_eval.py      # Batch scenario evaluation suite
│   ├── smoke_test.py                 # Fast 50-step end-to-end integration test
│   └── train_scaled_bc.py            # Supervised BC training on scaled demonstrations
├── tests/                    # Comprehensive unit & regression test suite (229 tests)
│   ├── test_contracts.py     # Sensor normalization & action bounds tests
│   ├── test_guidance_parity.py # Parity between NumPy and PyTorch guidance
│   ├── test_onnx_runner.py   # ONNX edge runner latency & boundary tests
│   ├── test_safety_supervisor.py # Safety supervisor braking & wall buffer tests
│   └── ...
├── data/                     # Evaluation manifests, test vectors & metrics JSONs
└── reports/                  # Detailed engineering & forensic audit reports
```

---

## Quickstart Guide

### 1. Prerequisites

- Python 3.10 or 3.11
- macOS or Linux (x86_64 or aarch64 / ARM64)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<username>/uav-bput-nav.git
cd uav-bput-nav

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Fast Smoke Test

Verify environment physics, sensor ray casting, localization, and safety supervisor rollouts:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

### 4. Run Full Unit Test Battery (229 Tests)

```bash
PYTHONPATH=. pytest tests/ -v
```

### 5. Launch Interactive 3D Web Simulation Cockpit

Start the simulation viewer server:

```bash
PYTHONPATH=. python -m viewer.server --port 8050
```

Open [http://localhost:8050](http://localhost:8050) in your browser to access:
- **Interactive Three.js 3D Viewport:** Orbit, pan, and zoom through the 3D corridor.
- **Model Selector:** Switch between baseline controllers, BC v1–v5, and ONNX models live.
- **Real-Time Telemetry HUD:** View active 23-float observation inputs, velocity setpoints, attitude, and rangefinder rays.
- **Playback Controls:** Play, pause, step, and evaluate multi-episode batches.

### 6. Export ONNX Edge Policy & Validate Parity

Run the deterministic ONNX export and parity harness:

```bash
PYTHONPATH=. python scripts/export_and_validate_bc_v3.py
```

This verifies the source checkpoint SHA-256 (`d02d5712...`), generates `models/checkpoints/policy_bc_v3.onnx`, and validates numerical parity across 2,274 test vectors ($\max |\Delta| \le 1.40 \times 10^{-6}\text{ m/s}$).

---

## Physical Platform Specifications

| Parameter | Scaled 65 cm Physical Platform |
| :--- | :--- |
| **Vehicle Footprint** | $0.65\text{ m}$ tip-to-tip diameter ($r_{\text{drone}} = 0.325\text{ m}$) |
| **Mass** | $1.80\text{ kg}$ |
| **Max Forward Speed** | $0.80\text{ m/s}$ (Policy) / $0.60\text{ m/s}$ (ActionLimits) |
| **Max Lateral Speed** | $0.60\text{ m/s}$ (Policy) / $0.40\text{ m/s}$ (ActionLimits) |
| **Max Vertical Speed** | $0.40\text{ m/s}$ (Policy) / $0.30\text{ m/s}$ (ActionLimits) |
| **Max Yaw Rate** | $0.60\text{ rad/s}$ (Policy) / $0.50\text{ rad/s}$ (ActionLimits) |
| **Companion Computer** | Raspberry Pi 5 (4 GB LPDDR4X, Quad Cortex-A76 @ 2.4 GHz) |
| **Inference Runtime** | ONNX Runtime 1.31.0 (ARM64 Linux, 1 thread, NEON SIMD) |
| **Flight Controller** | RadioLink CrossFlight / PX4 via MAVLink |
| **Control Loop Rate** | $20\text{ Hz}$ to $50\text{ Hz}$ ($\Delta t = 0.02\text{--}0.05\text{ s}$) |

---

## License

This project is developed for BPUT autonomous robotics and UAV navigation research. All rights reserved.
