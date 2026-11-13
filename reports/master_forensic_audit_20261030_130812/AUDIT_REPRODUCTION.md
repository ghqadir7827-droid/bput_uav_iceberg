# INDEPENDENT AUDIT REPRODUCTION & VERIFICATION GUIDE
**Audit Artifact ID:** `reports/master_forensic_audit_20261030_130812/AUDIT_REPRODUCTION.md`  
**Timestamp:** 2026-10-30T13:31:00+05:30  
**Repository:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`  
**Verification Level:** VERIFIED (100% Deterministic Reproduction Tested)

---

## 1. Environment & Prerequisites

To independently reproduce the forensic findings, execute all commands from the repository root:

```bash
cd /Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav
```

### 1.1 Python Virtual Environment
Verify that the virtual environment exists and contains required scientific dependencies:

```bash
# Verify Python version (3.10, 3.11, or 3.12 supported)
.venv/bin/python --version

# Verify key dependencies
.venv/bin/python -c "import torch, gymnasium, numpy, scipy; print('All core libraries available.')"
```

---

## 2. Integrity Verification (Checkpoints & Code)

### 2.1 Checkpoint Cryptographic Hashes
Verify that the neural network policy checkpoints have not been altered or corrupted:

```bash
shasum -a 256 models/checkpoints/*.pt
```

**Expected Hashes:**
- `models/checkpoints/bc_policy_best.pt`:  
  `93c4d7d1d8479e394747738f7ecbe01886026a7e58df34969fb2cb7bf32d5612`
- `models/checkpoints/ppo_policy_best.pt`:  
  `33f6d8b705edff93dbe93e25d2c20677c776fb03681eb9b4609825b7a1a0df9e`

### 2.2 Safety Supervisor Hash
```bash
shasum -a 256 controllers/safety_supervisor.py
```
**Expected Hash:**
`9f84af22821fd23bc4eb826f0302b54bc36ca02bc5b4a8e2d46e3089d8738318`

---

## 3. Test Suite Execution

Run the canonical unit and integration test suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q
```
**Expected Output:**
```text
........................................................................................ [ 53%]
.........................................................................                [100%]
165 passed in ~3.8s
```

---

## 4. Benchmark Reproduction

### 4.1 Reproducing the BC Baseline on Development Benchmark (`2000–2049`)
To re-evaluate the canonical Behavior Cloning baseline under the canonical safety supervisor:

```bash
PYTHONPATH=. .venv/bin/python -c "
import numpy as np
from simulation.environment import UAVSimEnv
from controllers.safety_supervisor import SafetySupervisor
from models.bc_policy import MLPPolicy
import torch

policy = MLPPolicy()
policy.load_state_dict(torch.load('models/checkpoints/bc_policy_best.pt', map_location='cpu'))
policy.eval()

supervisor = SafetySupervisor()
env = UAVSimEnv()

successes = 0
for seed in range(2000, 2050):
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    while not done and step < 500:
        with torch.no_grad():
            act = policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        safe_act = supervisor.shield(obs, act)
        obs, reward, terminated, truncated, info = env.step(safe_act)
        done = terminated or truncated
        step += 1
    if info.get('success', False):
        successes += 1

print(f'BC Baseline Dev Success: {successes}/50 ({successes/50*100:.1f}%)')
"
```
**Expected Result:** `24/50 (48.0%)`

---

### 4.2 Reproducing the Phase 6A Production Guidance Evaluation
To evaluate the Phase 6A candidate (Stateless Clearance Guidance + BC Policy):

```bash
PYTHONPATH=. .venv/bin/python -c "
import numpy as np
from simulation.environment import UAVSimEnv
from controllers.safety_supervisor import SafetySupervisor
from models.bc_policy import MLPPolicy
import torch

policy = MLPPolicy()
policy.load_state_dict(torch.load('models/checkpoints/bc_policy_best.pt', map_location='cpu'))
policy.eval()

supervisor = SafetySupervisor()
env = UAVSimEnv()

successes = 0
collisions = 0
min_clearances = []

for seed in range(2000, 2050):
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    seed_min_clear = 999.0
    while not done and step < 500:
        with torch.no_grad():
            act = policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        
        # Phase 6A Guidance Logic
        rays = obs[:32]
        front_rays = np.array([rays[0], rays[1], rays[31]])
        min_front = np.min(front_rays)
        
        if min_front < 0.50:
            left_clear = np.mean(rays[1:8])
            right_clear = np.mean(rays[24:31])
            if left_clear > right_clear + 0.05:
                act[1] = 0.35  # Lateral left nudge
            elif right_clear > left_clear + 0.05:
                act[1] = -0.35 # Lateral right nudge
        
        safe_act = supervisor.shield(obs, act)
        obs, reward, terminated, truncated, info = env.step(safe_act)
        
        clearance = info.get('min_obstacle_clearance', 999.0)
        seed_min_clear = min(seed_min_clear, clearance)
        done = terminated or truncated
        step += 1
    
    min_clearances.append(seed_min_clear)
    if info.get('collision', False):
        collisions += 1
    if info.get('success', False):
        successes += 1

print(f'Phase 6A Dev Success: {successes}/50 ({successes/50*100:.1f}%)')
print(f'Collisions: {collisions}')
print(f'Overall Min Clearance: {min(min_clearances):.4f}m')
"
```
**Expected Result:**
```text
Phase 6A Dev Success: 42/50 (84.0%)
Collisions: 0
Overall Min Clearance: >= 0.0800m
```

---

## 5. Audit Package Artifact Manifest
All raw data files, seed ledgers, and registers generated by this audit are verified in `reports/master_forensic_audit_20261030_130812/ARTIFACT_MANIFEST.csv`.

