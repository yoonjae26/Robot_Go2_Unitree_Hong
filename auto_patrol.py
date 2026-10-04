#!/usr/bin/env python3
"""
Autonomous Reconnaissance (자율 정찰) — fixed-route patrol combining the
robot's existing move/turn control with the face-detection pipeline already
built in camera_vision_download/camera_vision_test2.py, plus a spoken
reconnaissance report (GPT-4o-mini scene analysis) whenever it encounters a
person.

Previously this stopped and spoke a short personalized *greeting* instead
(see git history / _build_greeting if that behavior is ever wanted back) --
changed on request to a full report of the scene instead, reusing
dashboard_server.py's PipelineWorker._analyze_camera_scene() (the same
capture -> InsightFace -> GPT-4o-mini pipeline behind the dashboard's
manual "what do you see" button/voice command) rather than maintaining a
second GPT-4o-mini prompt here that only produced a greeting sentence.

Explicitly a v1/simplified version (as scoped with the user): a fixed
there-and-back route, no SLAM/localization -- see PATROL_ROUTE below. A
precise closed-loop route (e.g. a square via 4x 90-degree turns) would need
odometry feedback to avoid drifting off course over time; small heading
error on open-loop turns compounds with every corner. A there-and-back path
self-corrects (retraces the same line) without needing that.

Reuses rather than duplicates:
  - camera_vision_test2.py's initialize_face_model()/load_people_database()
    (InsightFace) -- imported directly, not copied.
  - The dashboard's own CameraStreamer.get_frame() for the video feed
    (JPEG bytes, decoded here) instead of opening a second video connection
    the way camera_vision_test2.py's standalone DDS VideoClient did (LAN-only
    and a redundant connection on WiFi -- see CameraStreamer/TelemetryStreamer
    docstrings for why a second connection is avoided throughout this project).
  - PipelineWorker's already-lazy-initialized robot controller
    (._ensure_robot()/._robot.executor.sport_client), TTS (._speak()), and
    now also its vision-analysis pipeline (._analyze_camera_scene()) instead
    of building a separate RobotController or a second scene-analysis prompt.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VISION_DIR = Path(__file__).parent / "camera_vision_download"

# Fixed back-and-forth route -- edit vx/omega/duration to match the actual
# space the robot patrols in. Each leg is a raw Move(vx, vy, omega) held for
# `duration` seconds, refreshed every MOVE_REFRESH_S like everywhere else in
# this project that uses continuous (non-schema) control (see dance/
# motion_executor.py's _sway()/_rock(), behavior_executor.py's "move" branch).
PATROL_ROUTE = [
    {"vx": 0.25, "vy": 0.0, "omega": 0.0, "duration": 4.0},   # forward
    {"vx": 0.0, "vy": 0.0, "omega": 0.8, "duration": 3.9},    # turn ~180°
    {"vx": 0.25, "vy": 0.0, "omega": 0.0, "duration": 4.0},   # back
    {"vx": 0.0, "vy": 0.0, "omega": 0.8, "duration": 3.9},    # turn ~180° (face original way)
]

MOVE_REFRESH_S = 0.3           # Move() decays after ~1s unrefreshed
FACE_CHECK_INTERVAL_S = 1.5    # InsightFace is CPU-bound (hundreds of ms) -- don't run every tick
REPORT_COOLDOWN_S = 15.0       # don't re-report the same encounter repeatedly
FACE_CONFIRM_HITS = 2          # consecutive detections required before reporting (avoid motion-blur false positives)

# Report/TTS output language -- separate from the STT language selector
# (dashboard_server.py's language buttons / voice_control_go2.py's
# _LANGUAGE_NAMES): patrol doesn't parse speech, it only speaks a report, so
# only languages we can plausibly generate+TTS one in are listed. Also
# imported by dashboard_server.py's _analyze_camera_scene() -- shared
# between the manual "what do you see" path and this automatic one so both
# validate/display language names the same way.
SUPPORTED_TTS_LANGUAGES = {"ko": "한국어", "vi": "베트남어"}


class AutoPatrol:
    def __init__(self, camera, worker, model: str = "gpt-4o-mini"):
        self._camera = camera
        self._worker = worker
        self._model = model
        self._face_model = None
        self._people = []
        self._vision_lock = threading.Lock()  # guards _load_vision()/reload_people() vs. vision_people.xlsx writes
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state = "idle"  # "idle" | "patrolling" | "reporting"
        self._language = "ko"
        self._last_reported_at = 0.0
        self._consecutive_hits = 0
        self._last_error: Optional[str] = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def set_language(self, language: str):
        self._language = language if language in SUPPORTED_TTS_LANGUAGES else "ko"

    def start(self, language: str = "ko"):
        if self._running:
            return
        self.set_language(language)
        self._last_error = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="auto-patrol")
        self._thread.start()

    def stop(self):
        self._running = False  # thread notices on its next loop tick and stops the robot itself

    # ── Main loop ────────────────────────────────────────────────────────

    def _run(self):
        try:
            self._load_vision()
        except Exception as e:
            logger.error(f"Auto Patrol: vision init failed, aborting: {e}", exc_info=True)
            self._last_error = f"Vision init failed: {e}"
            self._running = False
            self._state = "idle"
            return

        try:
            self._worker._ensure_robot()
        except Exception as e:
            logger.error(f"Auto Patrol: robot init failed, aborting: {e}", exc_info=True)
            self._last_error = f"Robot init failed: {e}"
            self._running = False
            self._state = "idle"
            return

        sc = self._worker._robot.executor.sport_client
        if sc is None or not self._worker._robot.executor.connected:
            # _ensure_robot() can "succeed" (construct the objects) while the
            # underlying DDS/WebRTC connection itself failed -- e.g. robot
            # off/unreachable -- leaving sport_client as None. Calling
            # Move() on that would just spam warnings forever with the UI
            # stuck showing "patrolling" while nothing actually happens.
            err = self._worker._robot.executor.last_error or "robot not connected"
            logger.error(f"Auto Patrol: aborting — {err}")
            self._last_error = f"Robot not connected: {err}"
            self._running = False
            self._state = "idle"
            return

        logger.info("Auto Patrol: started")
        self._state = "patrolling"

        last_face_check = 0.0
        leg_index = 0
        leg_started_at = time.monotonic()

        while self._running:
            now = time.monotonic()
            leg = PATROL_ROUTE[leg_index % len(PATROL_ROUTE)]

            if now - leg_started_at > leg["duration"]:
                leg_index += 1
                leg_started_at = now
                leg = PATROL_ROUTE[leg_index % len(PATROL_ROUTE)]

            try:
                sc.Move(leg["vx"], leg["vy"], leg["omega"])
            except Exception as e:
                logger.warning(f"Auto Patrol: Move failed: {e}")

            if now - last_face_check > FACE_CHECK_INTERVAL_S:
                last_face_check = now
                if self._check_for_person():
                    self._consecutive_hits += 1
                    if (
                        self._consecutive_hits >= FACE_CONFIRM_HITS
                        and now - self._last_reported_at > REPORT_COOLDOWN_S
                    ):
                        self._report_and_pause(sc)
                        leg_started_at = time.monotonic()  # resume this leg fresh after reporting
                else:
                    self._consecutive_hits = 0

            time.sleep(MOVE_REFRESH_S)

        try:
            sc.Move(0.0, 0.0, 0.0)
            sc.StopMove()
        except Exception:
            pass
        self._state = "idle"
        logger.info("Auto Patrol: stopped")

    # ── Vision ───────────────────────────────────────────────────────────

    def _load_vision(self):
        if self._face_model is not None:
            return
        with self._vision_lock:
            if self._face_model is not None:  # re-check -- another thread may have loaded it while we waited
                return
            vpath = str(VISION_DIR)
            if vpath not in sys.path:
                sys.path.insert(0, vpath)
            from camera_vision_test2 import initialize_face_model, load_people_database
            logger.info("Auto Patrol: loading face model + people database...")
            self._face_model = initialize_face_model()
            self._people = load_people_database(self._face_model)
            logger.info(f"Auto Patrol: {len(self._people)} registered people loaded")

    def reload_people(self):
        """Re-read vision_people.xlsx + faces/ with the already-loaded
        InsightFace model (cheap -- the model itself isn't reinitialized) so
        a newly registered person is recognizable without restarting the
        server. Call after appending to the workbook (see
        dashboard_server.py's PipelineWorker.register_face)."""
        if self._face_model is None:
            return
        with self._vision_lock:
            from camera_vision_test2 import load_people_database
            self._people = load_people_database(self._face_model)
            logger.info(f"Auto Patrol: people DB reloaded -- {len(self._people)} registered")

    def _get_frame(self):
        jpeg = self._camera.get_frame() if self._camera else None
        if not jpeg:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _check_for_person(self) -> bool:
        frame = self._get_frame()
        if frame is None:
            return False
        try:
            faces = self._face_model.get(frame)
        except Exception as e:
            logger.warning(f"Auto Patrol: face detection failed: {e}")
            return False
        return len(faces) > 0

    # ── Reconnaissance report ────────────────────────────────────────────

    def _report_and_pause(self, sc):
        self._state = "reporting"
        self._consecutive_hits = 0
        self._last_reported_at = time.monotonic()
        try:
            sc.Move(0.0, 0.0, 0.0)
            sc.StopMove()
        except Exception:
            pass

        # Same capture -> InsightFace -> GPT-4o-mini pipeline as the
        # dashboard's manual "what do you see" button/voice command -- see
        # PipelineWorker._analyze_camera_scene() in dashboard_server.py.
        # Reused rather than re-prompted here so there's one description of
        # "what the robot is looking at" instead of two that can drift.
        report_error = None
        try:
            report_text, tts_lang, _llm_ms = self._worker._analyze_camera_scene(self._language)
            logger.info(f"Auto Patrol: reconnaissance report — {report_text}")
            self._worker._speak(report_text, lang=tts_lang)
        except Exception as e:
            report_error = str(e)
            logger.warning(f"Auto Patrol: reconnaissance report failed: {e}")
        self._worker._record_analytics(
            source="patrol_report", language=self._language, understood=True,
            success=report_error is None, actions=["reconnaissance_report"], error=report_error,
        )

        time.sleep(1.0)  # brief settle before resuming the route
        if self._running:
            self._state = "patrolling"
