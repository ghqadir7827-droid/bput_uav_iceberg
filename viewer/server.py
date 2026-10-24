"""server.py - Reusable 3D Simulation Viewer & Evaluation HTTP Service.

Hosts the simulation environment, serves the interactive Three.js cockpit UI,
and provides REST endpoints for live telemetry, model switching, repeatable evaluation,
and episode replay.
"""

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from viewer.eval_engine import EvaluationEngine


class SimulationServer:
    """HTTP and background execution server for the simulation viewer."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8050, static_dir: str = "viewer/static"):
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir).resolve()
        self.engine = EvaluationEngine(seed=100)

        self.is_playing = False
        self.play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_play(self) -> None:
        if not self.is_playing:
            self.is_playing = True
            self._stop_event.clear()
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def pause_play(self) -> None:
        self.is_playing = False
        self._stop_event.set()

    def _play_loop(self) -> None:
        """Background loop advancing simulation at real-time or scaled speed."""
        while not self._stop_event.is_set() and self.is_playing:
            t0 = time.time()
            with self.engine.lock:
                if self.engine.current_episode_status == "running":
                    self.engine.step()
                else:
                    # Episode completed; auto-pause
                    self.is_playing = False
                    break

            target_dt = self.engine.env.dt / max(0.1, self.engine.playback_speed)
            elapsed = time.time() - t0
            sleep_time = max(0.001, target_dt - elapsed)
            time.sleep(sleep_time)


def create_handler(sim_server: SimulationServer):
    """Factory creating request handler bound to the simulation server."""

    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(sim_server.static_dir), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/scene":
                self._send_json(sim_server.engine.env.get_scene_spec())
            elif path == "/api/state":
                with sim_server.engine.lock:
                    snap = sim_server.engine._capture_snapshot()
                    snap["trajectory_trail"] = sim_server.engine.trajectory_trail[-500:] # Latest 500 points
                    snap["is_playing"] = sim_server.is_playing
                    snap["playback_speed"] = sim_server.engine.playback_speed
                self._send_json(snap)
            elif path == "/api/controllers":
                metadata_list = sim_server.engine.registry.list_all_metadata()
                self._send_json({
                    "active_controller_id": sim_server.engine.active_controller_id,
                    "active_checkpoint_id": sim_server.engine.active_controller.metadata.checkpoint_id,
                    "controllers": metadata_list
                })
            elif path == "/api/eval_summary":
                self._send_json(sim_server.engine.get_summary_metrics())
            elif path == "/api/replay":
                qs = parse_qs(parsed.query)
                ep_id = int(qs.get("episode_id", [sim_server.engine.episode_counter])[0])
                replay_snaps = sim_server.engine.get_replay_data(ep_id)
                if replay_snaps is not None:
                    self._send_json({"episode_id": ep_id, "snapshots": replay_snaps})
                else:
                    self._send_json({"error": f"Episode {ep_id} not found in buffer"}, status=404)
            else:
                # Serve static files (index.html, styles, scripts)
                if path == "/":
                    self.path = "/index.html"
                return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                data = {}

            if path == "/api/step":
                sim_server.pause_play()
                snap = sim_server.engine.step()
                self._send_json(snap)
            elif path == "/api/play":
                sim_server.start_play()
                self._send_json({"status": "playing"})
            elif path == "/api/pause":
                sim_server.pause_play()
                self._send_json({"status": "paused"})
            elif path == "/api/reset":
                sim_server.pause_play()
                raw_seed = data.get("seed")
                seed = int(raw_seed) if (raw_seed is not None and str(raw_seed).strip() != "") else sim_server.engine.current_seed
                snap = sim_server.engine.reset(seed=seed)
                self._send_json(snap)
            elif path == "/api/select_controller":
                sim_server.pause_play()
                ctrl_id = data.get("controller_id", "nominal_baseline")
                ok, msg = sim_server.engine.select_controller(ctrl_id)
                self._send_json({"success": ok, "message": msg}, status=200 if ok else 400)
            elif path == "/api/select_scenario":
                sim_server.pause_play()
                scenario_id = data.get("scenario_id", "10m")
                ok, msg = sim_server.engine.select_scenario(scenario_id)
                self._send_json({"success": ok, "message": msg}, status=200 if ok else 400)
            elif path == "/api/speed":
                speed = float(data.get("speed", 1.0))
                sim_server.engine.playback_speed = max(0.1, min(10.0, speed))
                self._send_json({"playback_speed": sim_server.engine.playback_speed})
            elif path == "/api/eval_batch":
                sim_server.pause_play()
                seeds = data.get("seeds", [1000, 1001, 1002, 1003, 1004])
                summary = sim_server.engine.run_eval_batch(seeds=[int(s) for s in seeds])
                self._send_json(summary)
            else:
                self._send_json({"error": "Unknown endpoint"}, status=404)

        def _send_json(self, data: Any, status: int = 200):
            response_bytes = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response_bytes)

        def log_message(self, format, *args):
            # Suppress default noisy access logs to keep stdout clean
            pass

    return ViewerHandler


def run_viewer_server(port: int = 8050):
    """Entry point to launch the 3D simulation viewer."""
    sim_server = SimulationServer(port=port)
    handler = create_handler(sim_server)
    httpd = ThreadingHTTPServer((sim_server.host, sim_server.port), handler)
    print(f"=================================================================")
    print(f"[*] Reusable 3D Simulation Viewer running at: http://localhost:{port}")
    print(f"[*] Features: Model Selector, Live Trajectory, Telemetry HUD, Replay")
    print(f"=================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulation viewer...")
    finally:
        sim_server.pause_play()
        httpd.server_close()
        sim_server.engine.env.close()


if __name__ == "__main__":
    run_viewer_server()
