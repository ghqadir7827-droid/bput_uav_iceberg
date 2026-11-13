# Phase 4F-1: Viewer Reset Defect Correction & State Integrity Report

**Date:** October 28, 2026  
**Status:** COMPLETED & VERIFIED  
**Investigation Mode:** Interface & Reset Integrity Correction (Strictly Non-Training)  
**Target Workspace:** `/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav`

---

## 1. Executive Summary

During Phase 4F-0 and user interactive testing, a critical interface defect was reported: changing the seed in the interactive simulation cockpit and clicking "Play" did not dispatch an episode reset, causing the viewer to continue running the previous episode/seed. Furthermore, inspection revealed that episode-dependent state in the safety supervisor survived across environment resets in `viewer/eval_engine.py`.

In Phase 4F-1, these defects were independently reproduced, isolated, and resolved with minimal, robust, regression-tested modifications.

---

## 2. Confirmed Defects Before Fix

### 2.1 UI Play and Step Bypassed Seed Reset
- **Location:** `viewer/static/app.js` (L634–650)
- **Defect:** `btn-play` and `btn-step` event handlers simply invoked `fetch('/api/play')` and `fetch('/api/step')`. Neither inspected the `#seed-input` field nor checked if the entered seed differed from the active episode seed. If an episode was already running, changing the number in `#seed-input` and clicking Play or Step silently continued the old episode. If the episode was terminated, clicking Play did nothing because the backend play loop exited upon encountering a terminal state.
- **User Impact:** When the user entered seeds such as `1, 20, 100, 2000, 8000` and clicked Play, the simulation silently replayed or continued the prior seed, creating the false visual impression of a broken seed mechanism or invariant routes.

### 2.2 Lack of Seed Input Validation
- **Location:** `viewer/static/app.js` (L657), `viewer/server.py` (L126–127)
- **Defect:** In `app.js`, `parseInt(value, 10)` was called without checking for `NaN`, zero, or negative numbers. In `server.py`, if a non-integer or negative seed was supplied, `int(raw_seed)` either threw an uncaught `ValueError` or silently substituted the previous seed without reporting an error.
- **Requirement:** Requirement 8 dictates that invalid seed input must produce a clear error rather than silently substituting another seed.

### 2.3 Absence of Active Seed Status Indicator
- **Location:** `viewer/static/index.html`, `viewer/static/app.js`
- **Defect:** The UI only displayed the editable `<input id="seed-input">`. There was no readout in the top HUD bar or sidebar indicating which seed was *actively running* in the physics engine.

### 2.4 Supervisor Episode-State Leakage Across Resets
- **Location:** `viewer/eval_engine.py` (L112–136)
- **Defect:** `EvaluationEngine.reset()` reset PyBullet simulation bodies and initial kinematics, but failed to reset `self.safety_supervisor` internal timers (`loss_timer_s`, `stalled_timer_s`) and `last_safe_action`. If a previous episode ended in an obstacle-stalled recovery state (`stalled_timer_s > 0`), those timers leaked into the subsequent episode.

---

## 3. Minimal Technical Corrections

### 3.1 `viewer/eval_engine.py`
In `EvaluationEngine.reset()`, added explicit episode-state restoration for the safety supervisor without modifying canonical supervisor code:
```python
# Reset supervisor episode-dependent state
if self.safety_supervisor is not None:
    self.safety_supervisor.loss_timer_s = 0.0
    self.safety_supervisor.stalled_timer_s = 0.0
    self.safety_supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
```

### 3.2 `viewer/server.py`
In `ViewerHandler.do_POST()`, hardened `/api/reset` to validate input seeds and return HTTP 400 with a descriptive JSON error on invalid inputs:
```python
elif path == "/api/reset":
    sim_server.pause_play()
    raw_seed = data.get("seed")
    if raw_seed is None or (isinstance(raw_seed, str) and raw_seed.strip() == ""):
        self._send_json({"error": "Seed parameter is required and cannot be empty."}, status=400)
        return
    try:
        seed = int(raw_seed)
        if seed < 1 or seed > 999999999:
            self._send_json({"error": f"Invalid seed '{raw_seed}': must be a positive integer between 1 and 999999999."}, status=400)
            return
    except (ValueError, TypeError):
        self._send_json({"error": f"Invalid seed value '{raw_seed}': must be a valid integer."}, status=400)
        return
    snap = sim_server.engine.reset(seed=seed)
    self._send_json(snap)
```

### 3.3 `viewer/static/index.html`
Added two synchronized active seed displays:
1. **Top HUD Bar:** `<div class="hud-item"><span class="hud-label">ACTIVE SEED</span><span id="hud-active-seed" class="hud-value mono text-blue">50001</span></div>`
2. **Sidebar Scenario Card:** `<div class="meta-row mt-6"><span class="meta-label">Active Simulation Seed:</span><span id="active-seed-display" class="meta-value mono text-blue">50001</span></div>`

### 3.4 `viewer/static/app.js`
1. **Unified Reset Execution (`executeReset`):**
   Created a single, centralized asynchronous function that:
   - Validates the seed (positive integer $\ge 1$).
   - Manages an asynchronous sequence counter (`currentResetSeq`) to discard superseded in-flight responses and prevent race conditions.
   - Automatically pauses simulation before resetting.
   - Updates `activeEpisodeSeed`, the HUD display, and the sidebar badge upon receipt of server confirmation.
   - Clears stale banner warnings and scrubs replay buffers.
2. **Preset Chips & Reset Button Routing:**
   Both `btn-reset` and `setSeed(seedVal)` route through `executeReset()`.
3. **Play and Step Auto-Reset Guard:**
   `btn-play` and `btn-step` event listeners inspect `#seed-input`. If the entered seed differs from `activeEpisodeSeed` or if the simulation has completed (`lastEpisodeStatus !== 'running'`), they automatically call `await executeReset(enteredSeed)` before beginning playback or advancing steps.

---

## 4. Behavior Comparison: Before vs After

| Interaction Scenario | Behavior Before Fix | Behavior After Fix |
| :--- | :--- | :--- |
| **User types new seed, clicks Play** | Silently continued prior episode with old seed. | Detects seed change, executes reset with new seed, updates UI displays, then starts playback. |
| **User types new seed, clicks Step** | Silently stepped prior episode with old seed. | Detects seed change, executes reset with new seed, then executes step 1 of new episode. |
| **User enters empty or text seed** | Silently substituted previous seed or crashed with 500 error. | UI displays error banner; server returns HTTP 400 with descriptive error message. Simulation does not reset or corrupt state. |
| **Rapid consecutive reset clicks** | Older async responses could arrive out of order and overwrite newer seed. | Sequence guard (`currentResetSeq`) discards superseded stale responses. |
| **Active Seed Visibility** | Only the editable text input existed; user could not see active physics seed. | Top HUD and Sidebar prominently display synchronized `Active Seed`. |
| **Supervisor Timers on Reset** | Lingering `stalled_timer_s` or `loss_timer_s` survived reset. | Guaranteed nominal zero initialization on every reset. |

---

## 5. Regression Testing & Verification

A dedicated regression test suite was created in `tests/test_viewer_reset_regression.py`:
1. `test_eval_engine_resets_supervisor_state`: Verifies that corrupted `stalled_timer_s`, `loss_timer_s`, and `last_safe_action` are reset to zero on `reset()`.
2. `test_eval_engine_seed_determinism`: Verifies that resetting the same seed reproduces identical obstacle placements and positions.
3. `test_eval_engine_seed_diversity`: Verifies that different seeds produce distinct obstacle configurations.
4. `test_server_reset_validation`: Tests `/api/reset` with valid seeds, missing seeds, empty strings, negative integers, zero, and alphanumeric strings.

### Test Execution Results
```
PYTHONPATH=. .venv/bin/pytest tests/test_viewer_reset_regression.py -v
============================== 4 passed in 1.48s ===============================
```
Full repository test suite verification:
```
PYTHONPATH=. .venv/bin/pytest
============================== 99 passed in 2.30s ==============================
```

All 99 unit, contract, and regression tests passed with zero failures. Checkpoint SHA-256 hashes remain 100% intact.
