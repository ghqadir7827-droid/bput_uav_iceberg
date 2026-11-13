# Milestone 2B — BC v3 Raspberry Pi File Transfer & Integrity Report

**Date:** November 9, 2026  
**Source Host:** Mac Development Workstation  
**Destination Target:** Raspberry Pi 5 (`10.101.16.70`, user `pi`)  
**Task Scope:** File Transfer & Checksum Verification Only  
**Overall Status:** **TRANSFER BLOCKED — INTERACTIVE AUTHENTICATION REQUIRED**

---

## 1. Source Workspace & File Inventory

- **Canonical Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`
- **Working Tree Integrity:** Preserved without modification.

### Pre-Transfer Verification

| Artifact Filename | Relative Path | Size (Bytes) | SHA-256 Checksum | Pre-Transfer Check |
| :--- | :--- | :---: | :--- | :---: |
| **BC v3 ONNX Model** | `models/checkpoints/policy_bc_v3.onnx` | 150,205 | `beaaaf5ac2a90997811452c35ec085b27c7c344adf1d2acb04d74a38a435e0e7` | **MATCH (Exact)** |
| **Parity Test Vectors** | `data/milestone2a_test_vectors.json` | 28,489 | `06c6f3fd869127218fa81dd0e5bfc1b0c7d5d547fde086d7384fedc58406f31c` | **MATCH (Exact)** |

The source ONNX checksum matches the expected specification (`beaaaf5ac2a90997811452c35ec085b27c7c344adf1d2acb04d74a38a435e0e7`) bit-for-bit.

---

## 2. Raspberry Pi Connectivity & Authentication Audit

- **Target IP Address:** `10.101.16.70`
- **Target User:** `pi`
- **Port 22 (SSH):** **OPEN** (`Connection to 10.101.16.70 port 22 [tcp/ssh] succeeded`)
- **Port 5900 (VNC):** **OPEN** (`Connection to 10.101.16.70 port 5900 [tcp/rfb] succeeded`)
- **SSH Authentication Status:**
  - Automated non-interactive SSH test (`ssh -o BatchMode=yes pi@10.101.16.70`) returned `Permission denied (publickey,password)`.
  - The local Mac environment does not have a configured SSH private key matching an authorized key on the Raspberry Pi.
  - The Raspberry Pi requires interactive password authentication or an installed SSH public key.

---

## 3. Strict Compliance with Safety Boundaries

In accordance with project safety guidelines:
1. No attempt was made to bypass authentication or execute unauthorized privilege elevation.
2. No firewall, network, or router configurations were modified.
3. No passwords were asked for, stored, or exposed.
4. Neither the original model nor the test vectors were altered in the source workspace.

---

## 4. Artifact Transfer & Verification Summary

| Artifact | Planned Destination Path on Pi | Transfer Status | Verification Classification |
| :--- | :--- | :---: | :---: |
| `policy_bc_v3.onnx` | `~/bc-v3-deployment/models/checkpoints/policy_bc_v3.onnx` | Not Completed | **NOT TRANSFERRED** |
| `milestone2a_test_vectors.json` | `~/bc-v3-deployment/data/milestone2a_test_vectors.json` | Not Completed | **NOT TRANSFERRED** |

---

## 5. Recommended Safe Transfer Alternatives

### Alternative 1: Interactive SCP Transfer (Terminal)
Execute the following commands in your Mac terminal, entering your Raspberry Pi password when prompted:

```bash
# 1. Create target directories on Raspberry Pi
ssh pi@10.101.16.70 "mkdir -p ~/bc-v3-deployment/models/checkpoints ~/bc-v3-deployment/data"

# 2. Transfer ONNX model and test vectors
scp /Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/models/checkpoints/policy_bc_v3.onnx pi@10.101.16.70:~/bc-v3-deployment/models/checkpoints/
scp /Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav/data/milestone2a_test_vectors.json pi@10.101.16.70:~/bc-v3-deployment/data/

# 3. Verify destination checksums on Raspberry Pi
ssh pi@10.101.16.70 "sha256sum ~/bc-v3-deployment/models/checkpoints/policy_bc_v3.onnx ~/bc-v3-deployment/data/milestone2a_test_vectors.json"
```

Expected destination output:
```
beaaaf5ac2a90997811452c35ec085b27c7c344adf1d2acb04d74a38a435e0e7  /home/pi/bc-v3-deployment/models/checkpoints/policy_bc_v3.onnx
06c6f3fd869127218fa81dd0e5bfc1b0c7d5d547fde086d7384fedc58406f31c  /home/pi/bc-v3-deployment/data/milestone2a_test_vectors.json
```

### Alternative 2: Enable Passwordless SSH Key Authentication
To allow future automated agent tasks to interface with the Raspberry Pi without password prompts:
```bash
# On Mac: copy public key to Raspberry Pi
ssh-copy-id pi@10.101.16.70
```

### Alternative 3: RealVNC File Transfer
Since port 5900 is open and RealVNC Server is running:
1. Connect to `10.101.16.70` via RealVNC Viewer.
2. Use the RealVNC toolbar **File Transfer** utility.
3. Transfer `models/checkpoints/policy_bc_v3.onnx` and `data/milestone2a_test_vectors.json` directly into `~/bc-v3-deployment/`.

