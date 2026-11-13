"""test_viewer_reset_regression.py - Regression Tests for Viewer Reset & Seed Integrity.

Verifies:
1. /api/reset validates seed parameters (positive integer requirement, rejection of invalid/missing/negative seeds).
2. EvaluationEngine.reset() clears all episode-dependent safety supervisor state (loss_timer_s, stalled_timer_s, last_safe_action).
3. Deterministic reset reproducibility under identical seeds.
4. Distinct obstacle geometry generation across distinct seeds.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from controllers.safety_supervisor import SafetySupervisor, VelocityAction
from viewer.eval_engine import EvaluationEngine
from viewer.server import SimulationServer, create_handler


def test_eval_engine_resets_supervisor_state():
    """Verify that EvaluationEngine.reset() clears any lingering supervisor timers or actions."""
    engine = EvaluationEngine(seed=100)
    
    # Intentionally corrupt/dirty the supervisor state to simulate a pinned or failing previous episode
    engine.safety_supervisor.loss_timer_s = 1.25
    engine.safety_supervisor.stalled_timer_s = 0.45
    engine.safety_supervisor.last_safe_action = VelocityAction(vx=-0.15, vy=0.40, vz=0.10, yaw_rate=0.20)
    
    assert engine.safety_supervisor.loss_timer_s == 1.25
    assert engine.safety_supervisor.stalled_timer_s == 0.45
    assert engine.safety_supervisor.last_safe_action.vx == -0.15
    
    # Execute reset
    snap = engine.reset(seed=200)
    
    # State must be cleanly restored to nominal zero
    assert engine.safety_supervisor.loss_timer_s == 0.0
    assert engine.safety_supervisor.stalled_timer_s == 0.0
    assert engine.safety_supervisor.last_safe_action.vx == 0.0
    assert engine.safety_supervisor.last_safe_action.vy == 0.0
    assert engine.safety_supervisor.last_safe_action.vz == 0.0
    assert engine.safety_supervisor.last_safe_action.yaw_rate == 0.0
    assert snap["seed"] == 200
    assert snap["safety_state"] == "nominal"
    assert engine.current_seed == 200
    engine.env.close()


def test_eval_engine_seed_determinism():
    """Verify that same seed reset produces identical obstacle geometry and initial state."""
    engine = EvaluationEngine(seed=1)
    
    snap1 = engine.reset(seed=50001)
    obs1 = [[float(c) for c in o["center"]] for o in snap1["privileged_simulation_ground_truth"]["obstacles"]]
    pos1 = list(snap1["position"])
    
    # Advance a few steps to alter state
    for _ in range(5):
        engine.step()
        
    # Re-reset with same seed
    snap2 = engine.reset(seed=50001)
    obs2 = [[float(c) for c in o["center"]] for o in snap2["privileged_simulation_ground_truth"]["obstacles"]]
    pos2 = list(snap2["position"])
    
    assert obs1 == obs2, "Obstacles must be identical for identical seed"
    assert pos1 == pos2, "Initial position must be identical for identical seed"
    engine.env.close()


def test_eval_engine_seed_diversity():
    """Verify that different seeds produce measurably different obstacle configurations."""
    engine = EvaluationEngine(seed=1)
    
    snap_a = engine.reset(seed=20)
    obs_a = [o["center"][1] for o in snap_a["privileged_simulation_ground_truth"]["obstacles"]]
    
    snap_b = engine.reset(seed=100)
    obs_b = [o["center"][1] for o in snap_b["privileged_simulation_ground_truth"]["obstacles"]]
    
    assert obs_a != obs_b, "Different seeds must produce different obstacle lateral offsets"
    engine.env.close()


class DummyRequest:
    """Mock socket request object for testing ViewerHandler POST routing directly."""
    def __init__(self, body_bytes: bytes):
        self.body_bytes = body_bytes
        self.rfile = self
        self.wfile = self
        self.headers = {"Content-Length": str(len(body_bytes))}
        self.written_bytes = b""

    def read(self, n):
        return self.body_bytes[:n]

    def write(self, b):
        self.written_bytes += b

    def flush(self):
        pass

    def makefile(self, *args, **kwargs):
        return self


def test_server_reset_validation():
    """Directly test server handler reset endpoint validation."""
    sim_server = SimulationServer(port=8099)
    handler_class = create_handler(sim_server)
    
    # Helper to invoke do_POST on mock handler
    def post_reset(payload_dict):
        body = json.dumps(payload_dict).encode("utf-8")
        req = DummyRequest(body)
        
        # Instantiate handler with mocks
        handler = handler_class.__new__(handler_class)
        handler.headers = req.headers
        handler.rfile = req
        handler.wfile = req
        handler.path = "/api/reset"
        
        response_data = {}
        def mock_send_response(status):
            response_data["status"] = status
        def mock_send_header(k, v):
            pass
        def mock_end_headers():
            pass
            
        handler.send_response = mock_send_response
        handler.send_header = mock_send_header
        handler.end_headers = mock_end_headers
        
        handler.do_POST()
        response_data["body"] = json.loads(req.written_bytes.decode("utf-8")) if req.written_bytes else {}
        return response_data

    try:
        # 1. Valid seed
        r1 = post_reset({"seed": 100})
        assert r1["status"] == 200
        assert r1["body"]["seed"] == 100
        
        # 2. Missing seed
        r2 = post_reset({})
        assert r2["status"] == 400
        assert "error" in r2["body"]
        
        # 3. Empty string seed
        r3 = post_reset({"seed": ""})
        assert r3["status"] == 400
        assert "error" in r3["body"]
        
        # 4. Negative seed
        r4 = post_reset({"seed": -5})
        assert r4["status"] == 400
        assert "error" in r4["body"]
        
        # 5. Non-numeric string seed
        r5 = post_reset({"seed": "invalid_abc"})
        assert r5["status"] == 400
        assert "error" in r5["body"]
        
        # 6. Zero seed
        r6 = post_reset({"seed": 0})
        assert r6["status"] == 400
        assert "error" in r6["body"]
    finally:
        sim_server.pause_play()
        sim_server.engine.env.close()
