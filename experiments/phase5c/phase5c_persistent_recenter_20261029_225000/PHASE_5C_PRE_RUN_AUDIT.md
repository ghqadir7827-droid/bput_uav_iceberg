# PHASE 5C — PRE-RUN INTEGRITY AUDIT

- **Date:** October 29, 2026
- **Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Isolation Directory:** `experiments/phase5c/phase5c_persistent_recenter_20261029_225000/`

---

## 1. Environment and Checksum Verification

Before executing any Phase 5C experiments, repository state and cryptographic checksums were verified:

| Component | File Path | Verified SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- |
| **Safety Supervisor** | `controllers/safety_supervisor.py` | `9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f` | **VERIFIED - UNTOUCHED** |
| **Frozen BC Baseline** | `models/checkpoints/bc_policy_best.pt` | `93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f` | **VERIFIED - UNTOUCHED** |
| **Phase 5B Artifacts** | `experiments/phase5b/phase5b_clearance_audit_20261029_220000/` | Complete paired results, traces, CSVs present | **VERIFIED - PRESERVED** |

---

## 2. Mandatory Boundary Constraints

1. **Safety Supervisor Preservation:** `controllers/safety_supervisor.py` remains immutable. No thresholds or rules will be modified.
2. **Frozen Baseline Preservation:** `models/checkpoints/bc_policy_best.pt` is strictly frozen and serves as reference Controller A.
3. **Canonical Benchmark Horizon:** Canonical 500 steps ($25.0\text{ s}$ at $\Delta t = 0.05\text{ s}$) is strictly enforced across all controllers.
4. **Sealed Held-Out Seeds:** Seeds `90000–90099` remain strictly sealed and untouched across Phase 5C.
5. **Evaluation Domain:** Matched development seeds `2000–2049` (50 episodes per controller).
6. **Hardware Safety:** No physical actuators, flight controllers, or RadioLink interfaces are accessed.
7. **No Model Training:** No neural network training will be conducted.
8. **Isolation:** All new state machines, evaluation scripts, logs, and traces are strictly contained within `experiments/phase5c/phase5c_persistent_recenter_20261029_225000/`.

