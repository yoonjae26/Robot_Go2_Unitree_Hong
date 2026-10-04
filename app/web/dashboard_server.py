#!/usr/bin/env python3
"""
Go2 Robot Dashboard Server with Person Tracking
Run:  python3 dashboard_server.py --robot-ip 192.168.123.18
Open: http://localhost:8080
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import asyncio
import base64
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO

from unitree_sdk2py.go2.video.video_client import VideoClient

# .env may live above this project's own root (e.g. /media/hong/data/.env) —
# search upward from cwd rather than assuming a fixed relative path.
load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _SuppressVendorDataChannelNoise(logging.Filter):
    """vendor/unitree_webrtc_connect/unitree_webrtc_connect/webrtc_datachannel.py's
    on_message handler logs literally every data-channel message via a bare
    logging.info() call (webrtc_audio.py's frame receiver does the same for
    every audio frame) -- these go through the true root logger, so
    record.name == "root", unlike every one of this project's own loggers
    (logging.getLogger(__name__), e.g. "dashboard_server", "auto_patrol").
    At low traffic this was harmless background noise; once slam_navigator.py
    started subscribing to rt/utlidar/robot_pose (~18-20Hz, continuous), the
    added volume of synchronous log calls was enough to starve the shared
    asyncio event loop and break telemetry/Move() -- confirmed live (see
    slam_navigator.py's module docstring for the full incident). Only
    suppresses INFO-and-below from "root" specifically -- this project's own
    INFO logs (named loggers) and anything WARNING+ (including from the
    vendored library, e.g. real connection errors) still print normally."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name == "root" and record.levelno <= logging.INFO)


logging.getLogger().addFilter(_SuppressVendorDataChannelNoise())

ROBOT_IP = "192.168.123.18"
SERVER_PORT = 8080
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"

# ─── Person Tracking Constants ────────────────────────────────────────────────

MODEL_NAME = "yolo11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.16
YOLO_IOU = 0.50
YOLO_DEVICE = "cpu"
PERSON_CLASS_ID = 0

MIN_BOX_WIDTH = 24
MIN_BOX_HEIGHT = 38
MIN_BOX_AREA_RATIO = 0.0015

LOST_FRAME_LIMIT = 6
VISION_STALE_TIMEOUT = 1.0
MOTION_LOOP_INTERVAL = 0.02

# Voice commands already turn at omega=1.0 rad/s (see voice_control_go2.py's
# "왼쪽/오른쪽으로 N도 돌기" examples, calibrated against real hardware) --
# match that instead of the much gentler 0.45-0.62 this used before, which
# made tracking's turn-to-center feel far slower than manual/voice control.
YAW_KP = 1.1
MIN_YAW_SPEED = 0.8
YAW_DIRECTION = -1.0

TURN_STOP_ERROR_RATIO = 0.045
TURN_RESTART_ERROR_RATIO = 0.075
# Measured YOLO CPU inference at ~32ms/frame (warm) on this machine -- far
# faster than the camera's own frame rate, so detection speed isn't why
# tracking falls behind a fast-moving person. The real cause is this dead
# time: after every turn the state machine refuses to react at all for
# TURN_SETTLING_TIME, so a fast walker can cross most of the frame during
# that blind window alone. Shortened from 0.60s (kept non-zero -- still lets
# the body's residual motion/blur settle briefly before trusting vision
# again) and TURN_MAX_DURATION raised so one continuous turn can actually
# catch up instead of being cut off mid-correction and forced through
# another settle cycle.
TURN_SETTLING_TIME = 0.20
TURN_MAX_DURATION = 1.80

# Move() decays after ~1s if not refreshed (see behavior_executor.py's "move"
# handler for the same issue on the voice-command path) -- _motion_worker
# used to call Move() exactly once on entering TURNING and never again until
# stopping, so any turn longer than ~1s (now common since TURN_MAX_DURATION
# above is 1.8s) silently decayed and the robot stopped moving mid-turn while
# the state machine still believed it was turning for up to another ~0.8s.
TURN_REFRESH_S = 0.3

# Fixes a limit-cycle oscillation ("wobbling") in tracking: MIN_YAW_SPEED
# above means nearly every correction ran at that same fixed 0.8 rad/s no
# matter how small the error was (the proportional term YAW_KP*error only
# exceeds 0.8 once error is >72% of half-frame width, i.e. almost never) --
# this was a bang-bang controller in disguise, not a real P-controller, so
# the robot never decelerated on approach to center. Combined with
# TURN_STOP_ERROR_RATIO's narrow 4.5% deadband and real turn/vision
# latency, it routinely overshot dead center, then immediately restarted a
# fresh 0.8 rad/s turn the other way on the next IDLE tick -- oscillating
# instead of converging. CREEP_* adds a slow third tier for small errors so
# the final approach happens at a speed the loop can actually stop inside
# the deadband, while large corrections still use the fast 0.8 rad/s tier
# above for responsiveness.
CREEP_ERROR_RATIO = 0.15
CREEP_YAW_SPEED = 0.35

POSITION_SMOOTHING_ALPHA = 0.55

COLOR_TARGET = (0, 255, 255)
COLOR_OTHER = (0, 165, 255)
COLOR_CROSSHAIR = (0, 0, 255)
COLOR_CENTER_LINE = (255, 0, 255)
COLOR_TEXT = (245, 245, 245)
COLOR_OK = (0, 255, 0)
COLOR_WARNING = (0, 200, 255)
COLOR_SETTLING = (255, 255, 0)
COLOR_ERROR = (0, 0, 255)
COLOR_OUTLINE = (0, 0, 0)
COLOR_LABEL_BACKGROUND = (20, 20, 20)


# ─── Person Tracking Data Classes ─────────────────────────────────────────────

class MotionMode(str, Enum):
    IDLE = "IDLE"
    TURNING = "TURNING"
    SETTLING = "SETTLING"


@dataclass
class PersonDetection:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class VisionState:
    detections: List[PersonDetection]
    target: Optional[PersonDetection]
    requested_yaw: float
    error_ratio: float
    aligned: bool
    inference_ms: float
    vision_fps: float
    updated_at: float
    error: Optional[str]
    lost_frames: int
    target_held: bool


@dataclass
class MotionState:
    mode: MotionMode
    current_yaw: float
    turn_started_at: float
    settling_until: float
    last_result: Optional[object]
    updated_at: float
    error: Optional[str]


@dataclass
class DisplayState:
    frame: Optional[np.ndarray]
    updated_at: float


# ─── Person Tracking Helper Functions ─────────────────────────────────────────

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def parse_yolo_result(
    result,
    frame_width: int,
    frame_height: int,
) -> List[PersonDetection]:
    detections: List[PersonDetection] = []

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()

    if boxes.id is not None:
        ids = boxes.id.detach().cpu().numpy().astype(int)
    else:
        ids = np.arange(len(xyxy), dtype=int) * -1 - 1

    classes = boxes.cls.detach().cpu().numpy().astype(int)
    frame_area = max(1, frame_width * frame_height)

    for box, confidence, track_id, class_id in zip(
        xyxy,
        confidences,
        ids,
        classes,
    ):
        if int(class_id) != PERSON_CLASS_ID:
            continue

        x1 = int(clamp(float(box[0]), 0, frame_width - 1))
        y1 = int(clamp(float(box[1]), 0, frame_height - 1))
        x2 = int(clamp(float(box[2]), 0, frame_width - 1))
        y2 = int(clamp(float(box[3]), 0, frame_height - 1))

        width = x2 - x1
        height = y2 - y1
        area_ratio = (width * height) / frame_area

        if width < MIN_BOX_WIDTH:
            continue
        if height < MIN_BOX_HEIGHT:
            continue
        if area_ratio < MIN_BOX_AREA_RATIO:
            continue

        detections.append(
            PersonDetection(
                track_id=int(track_id),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(confidence),
            )
        )

    return detections


def choose_target(
    detections: List[PersonDetection],
    current_target_id: Optional[int],
    previous_center_x: Optional[float],
    frame_width: int,
) -> Optional[PersonDetection]:
    if not detections:
        return None

    if current_target_id is not None:
        for detection in detections:
            if detection.track_id == current_target_id:
                return detection

    if previous_center_x is not None:
        nearest = min(
            detections,
            key=lambda detection: abs(
                detection.center_x - previous_center_x
            ),
        )

        if abs(nearest.center_x - previous_center_x) <= frame_width * 0.30:
            return nearest

    return max(
        detections,
        key=lambda detection: detection.area
        * (0.70 + 0.30 * detection.confidence),
    )


def calculate_requested_yaw(
    center_x: float,
    frame_width: int,
) -> Tuple[float, bool, float]:
    image_center_x = frame_width / 2.0
    normalized_error = (
        center_x - image_center_x
    ) / max(image_center_x, 1.0)

    error_ratio = abs(normalized_error)

    if error_ratio <= TURN_STOP_ERROR_RATIO:
        return 0.0, True, error_ratio

    if error_ratio <= CREEP_ERROR_RATIO:
        return math.copysign(CREEP_YAW_SPEED, YAW_DIRECTION * normalized_error), False, error_ratio

    requested_yaw = YAW_DIRECTION * YAW_KP * normalized_error

    if abs(requested_yaw) < MIN_YAW_SPEED:
        requested_yaw = math.copysign(
            MIN_YAW_SPEED,
            requested_yaw,
        )

    return requested_yaw, False, error_ratio


def draw_outlined_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    outline_thickness = max(thickness + 2, 3)

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_OUTLINE,
        outline_thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_label(
    frame: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    x = max(0, anchor[0])
    y = max(text_height + baseline + 4, anchor[1])
    padding_x = 5
    padding_y = 4

    box_x1 = x
    box_y1 = y - text_height - baseline - padding_y
    box_x2 = min(frame.shape[1] - 1, x + text_width + padding_x * 2)
    box_y2 = min(frame.shape[0] - 1, y + padding_y)

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        COLOR_LABEL_BACKGROUND,
        -1,
    )

    draw_outlined_text(
        frame,
        text,
        (x + padding_x, y - baseline),
        font_scale,
        color,
        thickness,
    )


def draw_crosshair(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    cx = width // 2
    cy = height // 2

    length = max(28, min(width, height) // 15)
    gap = max(9, length // 4)

    segments = [
        ((cx - length, cy), (cx - gap, cy)),
        ((cx + gap, cy), (cx + length, cy)),
        ((cx, cy - length), (cx, cy - gap)),
        ((cx, cy + gap), (cx, cy + length)),
    ]

    for start, end in segments:
        cv2.line(frame, start, end, COLOR_OUTLINE, 5, cv2.LINE_AA)
        cv2.line(frame, start, end, COLOR_CROSSHAIR, 2, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), gap, COLOR_OUTLINE, 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), gap, COLOR_CROSSHAIR, 1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections: List[PersonDetection],
    target: Optional[PersonDetection],
    requested_yaw: float,
    error_ratio: float,
    aligned: bool,
    inference_ms: float,
    vision_fps: float,
    lost_frames: int,
    target_held: bool,
    motion_state: MotionState,
) -> None:
    draw_crosshair(frame)

    target_id = target.track_id if target is not None else None

    for detection in detections:
        is_target = (
            target is not None
            and detection.track_id == target_id
        )

        color = COLOR_TARGET if is_target else COLOR_OTHER
        thickness = 3 if is_target else 2
        outline_thickness = thickness + 3

        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            COLOR_OUTLINE,
            outline_thickness,
        )
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            thickness,
        )

        label = (
            f"TARGET ID:{detection.track_id} {detection.confidence:.2f}"
            if is_target
            else f"PERSON ID:{detection.track_id} {detection.confidence:.2f}"
        )

        label_y = max(28, detection.y1 - 7)
        draw_label(
            frame,
            label,
            (detection.x1, label_y),
            color,
            0.52,
            2 if is_target else 1,
        )

    if target is not None:
        target_point = (
            int(target.center_x),
            int(target.center_y),
        )
        image_center = (
            frame.shape[1] // 2,
            frame.shape[0] // 2,
        )

        cv2.circle(frame, target_point, 9, COLOR_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame, target_point, 6, COLOR_CENTER_LINE, -1, cv2.LINE_AA)
        cv2.line(
            frame,
            image_center,
            target_point,
            COLOR_OUTLINE,
            4,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            image_center,
            target_point,
            COLOR_CENTER_LINE,
            1,
            cv2.LINE_AA,
        )

    if target is None:
        status = "SEARCHING PERSON"
        status_color = COLOR_WARNING
    elif target_held:
        status = f"TARGET HOLD {lost_frames}/{LOST_FRAME_LIMIT}"
        status_color = COLOR_WARNING
    elif motion_state.mode == MotionMode.SETTLING:
        status = "SETTLING POSTURE"
        status_color = COLOR_SETTLING
    elif aligned:
        status = "PERSON CENTERED"
        status_color = COLOR_OK
    elif motion_state.mode == MotionMode.TURNING:
        status = "TURNING TO PERSON"
        status_color = COLOR_TARGET
    else:
        status = "TRACKING PERSON"
        status_color = COLOR_TARGET

    draw_outlined_text(
        frame,
        status,
        (18, 35),
        0.84,
        status_color,
        2,
    )

    draw_outlined_text(
        frame,
        (
            f"Persons:{len(detections)}  "
            f"Req:{requested_yaw:+.3f}  "
            f"Sent:{motion_state.current_yaw:+.3f}  "
            f"Err:{error_ratio:.3f}  "
            f"Mode:{motion_state.mode.value}"
        ),
        (18, 65),
        0.49,
        COLOR_TEXT,
        1,
    )

    draw_outlined_text(
        frame,
        (
            f"YOLO:{inference_ms:.0f}ms  "
            f"VisionFPS:{vision_fps:.1f}  "
            f"Lost:{lost_frames}"
        ),
        (18, 90),
        0.49,
        COLOR_TEXT,
        1,
    )

    draw_outlined_text(
        frame,
        "Person Tracking Active",
        (18, frame.shape[0] - 18),
        0.49,
        COLOR_TEXT,
        1,
    )


# ─── Person Tracker ──────────────────────────────────────────────────────────

class PersonTracker:
    """Manages person detection and tracking with automatic robot orientation.

    Reuses rather than duplicates -- same rationale as AutoPatrol (see
    auto_patrol.py's module docstring): a second independent DDS/WebRTC video
    connection is unreliable on the WiFi/hotspot path (Go2's local WebRTC
    bridge handles only one connection well at a time -- see
    CameraStreamer.webrtc_conn's docstring), and a second independent
    SportClient would let this thread's continuous Move() calls fight with
    voice/text/patrol commands issued through the primary one. So tracking
    pulls frames from the dashboard's existing CameraStreamer and drives
    motion through PipelineWorker's already-connected RobotController
    (._robot.executor.sport_client), exactly like AutoPatrol does.
    """

    def __init__(self, camera: "CameraStreamer", worker: "PipelineWorker"):
        self._camera = camera
        self._worker = worker
        self._running = False
        self._enabled = False
        self.yolo_model = None
        self._last_error: Optional[str] = None

        self._vision_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._display_lock = threading.Lock()

        self.vision_state = VisionState(
            detections=[],
            target=None,
            requested_yaw=0.0,
            error_ratio=0.0,
            aligned=False,
            inference_ms=0.0,
            vision_fps=0.0,
            updated_at=0.0,
            error=None,
            lost_frames=0,
            target_held=False,
        )

        self.motion_state = MotionState(
            mode=MotionMode.IDLE,
            current_yaw=0.0,
            turn_started_at=0.0,
            settling_until=0.0,
            last_result=None,
            updated_at=0.0,
            error=None,
        )

        self.display_state = DisplayState(
            frame=None,
            updated_at=0.0,
        )

        self._stop_event = threading.Event()
        self._vision_thread = None
        self._motion_thread = None
        self._sport_client = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self, enable_tracking: bool = True):
        """Start the person tracker. Connection/model-loading happens on a
        background thread (mirrors AutoPatrol.start()/WakeWordListener.start())
        so the calling FastAPI request handler never blocks the event loop."""
        if self._running:
            return

        self._running = True
        self._enabled = enable_tracking
        self._last_error = None
        self._stop_event.clear()

        threading.Thread(target=self._start_worker, daemon=True, name="tracking-init").start()

    def _start_worker(self):
        try:
            logger.info(f"Loading YOLO model: {MODEL_NAME}")
            self.yolo_model = YOLO(MODEL_NAME)
        except Exception as e:
            logger.error(f"Person tracker: YOLO load failed, aborting: {e}", exc_info=True)
            self._last_error = f"YOLO load failed: {e}"
            self._running = False
            return

        try:
            self._worker._ensure_robot()
        except Exception as e:
            logger.error(f"Person tracker: robot init failed, aborting: {e}", exc_info=True)
            self._last_error = f"Robot init failed: {e}"
            self._running = False
            return

        sc = self._worker._robot.executor.sport_client
        if sc is None or not self._worker._robot.executor.connected:
            # _ensure_robot() can "succeed" (construct the objects) while the
            # underlying DDS/WebRTC connection itself failed -- see the same
            # guard in auto_patrol.py's _run() for why this must be checked
            # explicitly instead of letting Move() spam warnings forever.
            err = self._worker._robot.executor.last_error or "robot not connected"
            logger.error(f"Person tracker: aborting — {err}")
            self._last_error = f"Robot not connected: {err}"
            self._running = False
            return
        self._sport_client = sc

        result = sc.StandUp()
        logger.info(f"StandUp result: {result}")
        time.sleep(3.0)

        try:
            result = sc.FreeAvoid(False)
            logger.info(f"FreeAvoid(False) result: {result}")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"FreeAvoid not available: {e}")

        # Start vision and motion threads
        self._vision_thread = threading.Thread(
            target=self._vision_worker,
            daemon=True,
            name="tracking-vision",
        )
        self._motion_thread = threading.Thread(
            target=self._motion_worker,
            daemon=True,
            name="tracking-motion",
        )

        self._vision_thread.start()
        self._motion_thread.start()

        logger.info("Person tracker started")

    def stop(self):
        """Stop the person tracker."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._vision_thread:
            self._vision_thread.join(timeout=3.0)
        if self._motion_thread:
            self._motion_thread.join(timeout=3.0)

        # Send stop commands
        try:
            if self._sport_client is not None:
                for _ in range(10):
                    self._sport_client.Move(0.0, 0.0, 0.0)
                    time.sleep(0.05)
                self._sport_client.StopMove()
        except Exception as e:
            logger.warning(f"Error stopping sport client: {e}")

        logger.info("Person tracker stopped")

    def get_display_frame(self) -> Optional[np.ndarray]:
        """Get the current display frame with overlay."""
        with self._display_lock:
            return self.display_state.frame.copy() if self.display_state.frame is not None else None

    def _get_frame(self) -> Optional[np.ndarray]:
        """Pull the latest camera frame from the dashboard's shared CameraStreamer
        instead of opening a second video connection (see class docstring)."""
        jpeg = self._camera.get_frame() if self._camera else None
        if not jpeg:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _vision_worker(self):
        """Vision thread: YOLO detection, tracking, and frame overlay."""
        selected_target_id: Optional[int] = None
        smoothed_center_x: Optional[float] = None
        last_target: Optional[PersonDetection] = None
        last_requested_yaw = 0.0
        last_error_ratio = 0.0
        last_aligned = False
        lost_frames = 0

        fps_counter = 0
        fps_window_start = time.monotonic()
        vision_fps = 0.0

        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._get_frame()

                    if frame is None:
                        self._stop_event.wait(0.02)
                        continue

                    frame_height, frame_width = frame.shape[:2]
                    started = time.monotonic()

                    results = self.yolo_model.track(
                        source=frame,
                        persist=True,
                        tracker=TRACKER_CONFIG,
                        classes=[PERSON_CLASS_ID],
                        conf=YOLO_CONFIDENCE,
                        iou=YOLO_IOU,
                        imgsz=YOLO_IMAGE_SIZE,
                        device=YOLO_DEVICE,
                        verbose=False,
                    )
                    
                    inference_ms = (time.monotonic() - started) * 1000.0
                    
                    detections = (
                        parse_yolo_result(results[0], frame_width, frame_height)
                        if results
                        else []
                    )
                    
                    target = choose_target(
                        detections,
                        selected_target_id,
                        smoothed_center_x,
                        frame_width,
                    )
                    
                    target_held = False
                    
                    if target is not None:
                        lost_frames = 0
                        
                        if target.track_id >= 0:
                            selected_target_id = target.track_id
                        
                        if smoothed_center_x is None:
                            smoothed_center_x = target.center_x
                        else:
                            smoothed_center_x = (
                                POSITION_SMOOTHING_ALPHA * target.center_x
                                + (1.0 - POSITION_SMOOTHING_ALPHA) * smoothed_center_x
                            )
                        
                        (requested_yaw, aligned, error_ratio) = calculate_requested_yaw(
                            smoothed_center_x, frame_width
                        )
                        
                        last_target = target
                        last_requested_yaw = requested_yaw
                        last_error_ratio = error_ratio
                        last_aligned = aligned
                    
                    else:
                        lost_frames += 1
                        
                        if last_target is not None and lost_frames <= LOST_FRAME_LIMIT:
                            target = last_target
                            requested_yaw = last_requested_yaw
                            error_ratio = last_error_ratio
                            aligned = last_aligned
                            target_held = True
                        else:
                            selected_target_id = None
                            smoothed_center_x = None
                            last_target = None
                            last_requested_yaw = 0.0
                            last_error_ratio = 0.0
                            last_aligned = False
                            requested_yaw = 0.0
                            error_ratio = 0.0
                            aligned = False
                    
                    now = time.monotonic()
                    fps_counter += 1
                    elapsed = now - fps_window_start
                    
                    if elapsed >= 1.0:
                        vision_fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_window_start = now
                    
                    # Update vision state
                    with self._vision_lock:
                        self.vision_state = VisionState(
                            detections=detections,
                            target=target,
                            requested_yaw=requested_yaw,
                            error_ratio=error_ratio,
                            aligned=aligned,
                            inference_ms=inference_ms,
                            vision_fps=vision_fps,
                            updated_at=now,
                            error=None,
                            lost_frames=lost_frames,
                            target_held=target_held,
                        )
                    
                    # Draw overlay
                    motion_snapshot = self._get_motion_snapshot()
                    display_frame = frame.copy()
                    draw_overlay(
                        frame=display_frame,
                        detections=detections,
                        target=target,
                        requested_yaw=requested_yaw,
                        error_ratio=error_ratio,
                        aligned=aligned,
                        inference_ms=inference_ms,
                        vision_fps=vision_fps,
                        lost_frames=lost_frames,
                        target_held=target_held,
                        motion_state=motion_snapshot,
                    )
                    
                    with self._display_lock:
                        self.display_state = DisplayState(
                            frame=display_frame,
                            updated_at=now,
                        )
                
                except Exception as e:
                    logger.warning(f"Vision worker error: {e}")
                    now = time.monotonic()
                    with self._vision_lock:
                        self.vision_state = VisionState(
                            detections=[],
                            target=None,
                            requested_yaw=0.0,
                            error_ratio=0.0,
                            aligned=False,
                            inference_ms=0.0,
                            vision_fps=vision_fps,
                            updated_at=now,
                            error=str(e),
                            lost_frames=LOST_FRAME_LIMIT + 1,
                            target_held=False,
                        )
                    self._stop_event.wait(0.05)
        
        except Exception as e:
            logger.error(f"Vision worker crashed: {e}")

    def _motion_worker(self):
        """Motion thread: state machine for auto-tracking."""
        mode = MotionMode.IDLE
        current_yaw = 0.0
        turn_started_at = 0.0
        last_move_refresh = 0.0
        settling_until = 0.0
        last_result = None

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                
                with self._vision_lock:
                    command_target = self.vision_state.target
                    command_yaw = self.vision_state.requested_yaw
                    command_error = self.vision_state.error_ratio
                    command_aligned = self.vision_state.aligned
                    command_updated_at = self.vision_state.updated_at
                    command_error_text = self.vision_state.error
                
                state_age = (
                    now - command_updated_at
                    if command_updated_at > 0.0
                    else float("inf")
                )
                
                target_valid = (
                    command_target is not None
                    and command_error_text is None
                    and state_age <= VISION_STALE_TIMEOUT
                    and self._enabled
                    and not self._worker._busy  # don't fight an active voice/text/self-intro/patrol command
                )
                
                if mode == MotionMode.IDLE:
                    current_yaw = 0.0
                    
                    if (
                        target_valid
                        and not command_aligned
                        and command_error >= TURN_RESTART_ERROR_RATIO
                        and command_yaw != 0.0
                    ):
                        last_result = self._sport_client.Move(0.0, 0.0, float(command_yaw))
                        current_yaw = float(command_yaw)
                        turn_started_at = now
                        last_move_refresh = now
                        mode = MotionMode.TURNING

                elif mode == MotionMode.TURNING:
                    turn_elapsed = now - turn_started_at

                    # Track live vision instead of the speed frozen at
                    # turn-entry, so the commanded speed decelerates (e.g.
                    # into the CREEP tier) as the person approaches center
                    # instead of slamming a fixed fast speed all the way to
                    # the stop check every time. See CREEP_* above.
                    if target_valid and command_yaw != 0.0:
                        current_yaw = float(command_yaw)

                    if now - last_move_refresh >= TURN_REFRESH_S:
                        last_result = self._sport_client.Move(0.0, 0.0, current_yaw)
                        last_move_refresh = now

                    direction_reversed = (
                        command_yaw != 0.0
                        and current_yaw != 0.0
                        and math.copysign(1.0, command_yaw) != math.copysign(1.0, current_yaw)
                    )
                    
                    should_stop = (
                        not target_valid
                        or command_aligned
                        or command_error <= TURN_STOP_ERROR_RATIO
                        or direction_reversed
                        or turn_elapsed >= TURN_MAX_DURATION
                    )
                    
                    if should_stop:
                        last_result = self._sport_client.Move(0.0, 0.0, 0.0)
                        current_yaw = 0.0
                        settling_until = now + TURN_SETTLING_TIME
                        mode = MotionMode.SETTLING
                
                elif mode == MotionMode.SETTLING:
                    current_yaw = 0.0
                    if now >= settling_until:
                        mode = MotionMode.IDLE
                
                with self._motion_lock:
                    self.motion_state = MotionState(
                        mode=mode,
                        current_yaw=current_yaw,
                        turn_started_at=turn_started_at,
                        settling_until=settling_until,
                        last_result=last_result,
                        updated_at=now,
                        error=None,
                    )
                
                self._stop_event.wait(MOTION_LOOP_INTERVAL)
        
        except Exception as e:
            logger.error(f"Motion worker crashed: {e}")
            try:
                self._sport_client.Move(0.0, 0.0, 0.0)
            except Exception:
                pass

    def _get_motion_snapshot(self) -> MotionState:
        """Get a copy of current motion state."""
        with self._motion_lock:
            return MotionState(
                mode=self.motion_state.mode,
                current_yaw=self.motion_state.current_yaw,
                turn_started_at=self.motion_state.turn_started_at,
                settling_until=self.motion_state.settling_until,
                last_result=self.motion_state.last_result,
                updated_at=self.motion_state.updated_at,
                error=self.motion_state.error,
            )


# ─── Camera Streamer ──────────────────────────────────────────────────────────

class CameraStreamer:
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self.connected = False
        # Exposed so sport commands (BehaviorExecutor) can reuse this same WebRTC
        # peer connection instead of opening a second one -- Go2's local WebRTC
        # bridge handles two simultaneous connections unreliably (both can stall
        # negotiating at once).
        self.webrtc_conn = None
        self.webrtc_loop: Optional[asyncio.AbstractEventLoop] = None
        self.webrtc_ready = threading.Event()

    def start(self):
        self._running = True
        if self._is_wifi():
            threading.Thread(target=self._webrtc_loop, daemon=True, name="camera-webrtc").start()
        else:
            threading.Thread(target=self._loop, daemon=True, name="camera-dds").start()

    def stop(self):
        self._running = False

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def _is_wifi(self) -> bool:
        # Go2's own WiFi hotspot subnet (192.168.12.x). The DDS VideoClient service
        # lives on the internal companion computer (192.168.123.x) and isn't reachable
        # through the hotspot, so WebRTC is required there instead.
        return self.robot_ip.startswith("192.168.12.")

    def _webrtc_loop(self):
        try:
            # unitree_webrtc_connect is pip-installed editable (see
            # vendor/unitree_webrtc_connect/), so no sys.path hack is needed.
            from unitree_webrtc_connect.webrtc_driver import (
                UnitreeWebRTCConnection, WebRTCConnectionMethod,
            )

            logger.info(f"Camera: robot={self.robot_ip} method=webrtc")
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.robot_ip)

            async def recv_frames(track):
                while self._running:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    with self._lock:
                        self._frame = jpeg.tobytes()
                    self.connected = True

            async def setup():
                await conn.connect()
                self.webrtc_conn = conn
                self.webrtc_loop = loop
                self.webrtc_ready.set()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_frames)
                while self._running:
                    await asyncio.sleep(0.5)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(setup())
        except Exception as e:
            logger.error(f"Camera (WebRTC) crashed: {e}", exc_info=True)
            self.connected = False
            self.webrtc_ready.set()  # unblock any waiter — caller falls back to its own connection

    def _detect_interface(self) -> str:
        subnet = ".".join(self.robot_ip.split(".")[:3]) + "."  # trailing dot: "192.168.12." won't match "192.168.123."
        try:
            out = subprocess.check_output(["ip", "-o", "-4", "addr", "show"], text=True, timeout=3)
            for line in out.splitlines():
                if subnet in line:
                    return line.split()[1]
        except Exception:
            pass
        return ""

    def _loop(self):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.video.video_client import VideoClient

            iface = self._detect_interface()
            logger.info(f"Camera: robot={self.robot_ip} iface={iface!r}")

            # Go2 WiFi hotspot blocks DDS multicast between AP and clients.
            # Use unicast peer discovery to connect directly to the robot IP.
            if "CYCLONEDDS_URI" not in os.environ:
                os.environ["CYCLONEDDS_URI"] = (
                    "<CycloneDDS><Domain>"
                    "<General><AllowMulticast>false</AllowMulticast></General>"
                    f"<Discovery><Peers><Peer address='{self.robot_ip}'/></Peers></Discovery>"
                    "</Domain></CycloneDDS>"
                )
                logger.info(f"Set CycloneDDS unicast peer (no multicast): {self.robot_ip}")

            try:
                ChannelFactoryInitialize(0, iface)
            except Exception:
                pass

            client = VideoClient()
            client.SetTimeout(5.0)
            client.Init()

            fail = 0
            while self._running:
                code, data = client.GetImageSample()
                if code == 0 and data:
                    arr = np.frombuffer(bytes(data), dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        with self._lock:
                            self._frame = jpeg.tobytes()
                        self.connected = True
                        fail = 0
                else:
                    fail += 1
                    if fail > 50:
                        self.connected = False
                    time.sleep(0.02)
        except Exception as e:
            logger.error(f"Camera crashed: {e}", exc_info=True)
            self.connected = False


# ─── Telemetry Streamer ───────────────────────────────────────────────────────

class TelemetryStreamer:
    """Real battery % and per-motor temperature, read straight from the robot's
    LowState topic — no simulated/estimated numbers. Two transports, same as
    CameraStreamer: WebRTC (`rt/lf/lowstate`) over the Go2's own WiFi hotspot,
    DDS (`rt/lowstate`) over LAN. On WiFi this reuses the camera's already-open
    WebRTC connection (see CameraStreamer/WebRTCSportClient docstrings — a
    second simultaneous connection is unreliable), so it only starts once the
    camera is connected.
    """

    def __init__(self, robot_ip: str, camera: Optional["CameraStreamer"] = None):
        self.robot_ip = robot_ip
        self._camera = camera
        self._lock = threading.Lock()
        self._data = {
            "connected": False,
            "battery_pct": None,
            "power_v": None,
            "power_a": None,
            "motor_temp_max": None,
            "motor_temp_avg": None,
            "updated_at": None,
        }

    def start(self):
        if self.robot_ip.startswith("192.168.12."):
            threading.Thread(target=self._webrtc_loop, daemon=True, name="telemetry-webrtc").start()
        else:
            threading.Thread(target=self._dds_loop, daemon=True, name="telemetry-dds").start()

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def _store(self, battery_pct, power_v, power_a, temps):
        temps = [t for t in temps if t is not None]
        with self._lock:
            self._data.update({
                "connected": True,
                "battery_pct": battery_pct,
                "power_v": power_v,
                "power_a": power_a,
                "motor_temp_max": max(temps) if temps else None,
                "motor_temp_avg": round(sum(temps) / len(temps), 1) if temps else None,
                "updated_at": time.time(),
            })

    def _webrtc_loop(self):
        try:
            if self._camera is None or not self._camera.webrtc_ready.wait(timeout=20):
                logger.warning("Telemetry: camera WebRTC not ready — no telemetry available")
                return
            conn, loop = self._camera.webrtc_conn, self._camera.webrtc_loop
            if conn is None or loop is None:
                logger.warning("Telemetry: camera WebRTC connection missing — no telemetry available")
                return

            # unitree_webrtc_connect is pip-installed editable (see
            # vendor/unitree_webrtc_connect/), so no sys.path hack is needed.
            from unitree_webrtc_connect.constants import RTC_TOPIC

            def on_lowstate(message):
                try:
                    data = message.get("data") or {}
                    bms = data.get("bms_state") or {}
                    motors = (data.get("motor_state") or [])[:12]
                    self._store(
                        battery_pct=bms.get("soc"),
                        power_v=data.get("power_v"),
                        power_a=data.get("power_a"),
                        temps=[m.get("temperature") for m in motors],
                    )
                except Exception as e:
                    logger.warning(f"Telemetry parse error: {e}")

            # subscribe() sends over the datachannel synchronously — must run
            # on the connection's own event-loop thread, not this one.
            loop.call_soon_threadsafe(
                conn.datachannel.pub_sub.subscribe, RTC_TOPIC["LOW_STATE"], on_lowstate
            )
            logger.info("Telemetry: subscribed to LOW_STATE over shared WebRTC connection")
        except Exception as e:
            logger.error(f"Telemetry (WebRTC) crashed: {e}", exc_info=True)

    def _dds_loop(self):
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

            iface = self._camera._detect_interface() if self._camera else ""
            try:
                ChannelFactoryInitialize(0, iface)
            except Exception as e:
                logger.info(f"ChannelFactory already initialized or minor error (continuing): {e}")

            def handler(msg: "LowState_"):
                temps = [m.temperature for m in msg.motor_state[:12]]
                self._store(
                    battery_pct=msg.bms_state.soc,
                    power_v=msg.power_v,
                    power_a=msg.power_a,
                    temps=temps,
                )

            sub = ChannelSubscriber("rt/lowstate", LowState_)
            sub.Init(handler, 10)
            logger.info("Telemetry: subscribed to rt/lowstate over DDS")
            while True:
                time.sleep(3600)
        except Exception as e:
            logger.error(f"Telemetry (DDS) crashed: {e}", exc_info=True)


# ─── WebSocket Manager ────────────────────────────────────────────────────────

class WSManager:
    def __init__(self):
        self._clients: List = []
        self._lock = threading.Lock()

    async def connect(self, ws):
        await ws.accept()
        with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws):
        with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, ensure_ascii=False)
        with self._lock:
            snapshot = list(self._clients)
        dead = []
        for ws in snapshot:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._clients:
                        self._clients.remove(ws)

    def emit(self, data: dict, loop: asyncio.AbstractEventLoop):
        asyncio.run_coroutine_threadsafe(self.broadcast(data), loop)


# ─── Self-intro demo script ────────────────────────────────────────────────────
# One-shot opening sequence for a live demo: stand up, wave, bilingual TTS
# self-introduction (alternating Korean/Vietnamese, matching the language
# selector work elsewhere), then a short showcase of a few representative
# moves. Pure orchestration of already-existing pieces (execute_action +
# _speak) -- no new capability, just a scripted sequence.
SELF_INTRO_SCRIPT = [
    ("안녕하세요! 저는 Unitree Go2 로봇입니다.", "ko"),
    ("Xin chào! Tôi là robot Unitree Go2.", "vi"),
    ("음성 명령, 자율 정찰, 감정 인식, 춤까지 다양한 기능을 갖추고 있습니다.", "ko"),
    ("Tôi có thể nhận lệnh giọng nói, trinh sát tự động, nhận diện cảm xúc và nhảy múa.", "vi"),
    ("지금부터 몇 가지 동작을 보여드리겠습니다!", "ko"),
    ("Bây giờ tôi sẽ trình diễn một vài động tác!", "vi"),
]
SELF_INTRO_DEMO_ACTIONS = ["stretch", "trot_run", "dance1"]
SELF_INTRO_CLOSING = [
    ("감사합니다!", "ko"),
    ("Cảm ơn các bạn!", "vi"),
]


# ─── Pipeline Worker ──────────────────────────────────────────────────────────

class PipelineWorker:
    # Direction -> (vx, vy, omega) unit vector, scaled by self._speed at fire
    # time. Matches the dashboard's D-pad: forward/back translate, turn
    # left/right rotate in place (no strafe buttons in this layout).
    _MOVE_DIRS = {
        "forward": (1, 0, 0),
        "backward": (-1, 0, 0),
        "turn_left": (0, 0, 1),
        "turn_right": (0, 0, -1),
    }

    # Rotation feels much slower than a physical controller if omega shares
    # the same 0.1-0.6 range used for forward/back translation -- voice
    # commands already turn at omega=1.0 rad/s (see voice_control_go2.py's
    # "왼쪽/오른쪽으로 N도 돌기" examples, calibrated against real hardware),
    # so scale the D-pad's turn omega up to roughly match that instead of
    # reusing the linear speed value directly.
    _TURN_OMEGA_SCALE = 3.0

    def __init__(self, ws: WSManager, loop: asyncio.AbstractEventLoop):
        self._ws = ws
        self._loop = loop
        self._busy = False
        self._whisper = None
        self._openai = None
        self._robot = None
        # Analyze-then-confirm staging for the text-command panel (mirrors
        # the dashboard mockup's "AI 분석 결과" -> 실행/취소 flow) -- a typed
        # command is parsed by the LLM and shown before anything moves, so a
        # misread/ambiguous command can be caught before it reaches the robot.
        # The voice/mic path is intentionally left as its existing
        # auto-execute behavior (already tested extensively) rather than
        # retrofitted, to avoid regressing a working flow.
        self._pending = None  # {"actions": [(name, params), ...], "response": str}
        self._speed = 0.3
        self._move_thread: Optional[threading.Thread] = None
        self._move_stop_evt = threading.Event()
        # Set by emergency_stop(), checked between steps of every multi-step
        # sequence (self-intro, multi-action LLM commands) so a stop actually
        # halts remaining queued actions/TTS instead of running to completion
        # regardless -- cleared whenever a new sequence starts.
        self._cancel_evt = threading.Event()

    def _emit(self, t: str, **kw):
        self._ws.emit({"type": t, **kw}, self._loop)

    def _ensure_openai(self):
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=OPENAI_API_KEY)

    def _ensure_robot(self):
        if self._robot is None:
            from app.core.robot_controller import RobotController
            webrtc_conn = webrtc_loop = None
            if ROBOT_IP.startswith("192.168.12.") and _camera is not None:
                if _camera.webrtc_ready.wait(timeout=20):
                    webrtc_conn = _camera.webrtc_conn
                    webrtc_loop = _camera.webrtc_loop
                else:
                    logger.warning("Camera WebRTC not ready after 20s — sport client will open its own connection")
            self._robot = RobotController(
                robot_ip=ROBOT_IP, webrtc_conn=webrtc_conn, webrtc_loop=webrtc_loop, telemetry=_telemetry,
                person_gate=_person_gate,
            )

    def _lazy_init(self):
        if self._whisper is None:
            # Root partition is nearly full — keep the Whisper model download
            # (and any other HF cache) on the data disk instead.
            os.environ.setdefault("HF_HOME", "/media/hong/data/hf_cache")

            # pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 ship their .so files
            # under site-packages instead of a system CUDA install. Setting
            # LD_LIBRARY_PATH alone isn't enough -- ctranslate2 only dlopens cuBLAS
            # lazily on the first real inference call (not at model construction),
            # and that internal dlopen doesn't reliably pick up a path change made
            # from Python. Preloading the .so files directly via ctypes registers
            # them in the process so ctranslate2's own dlopen-by-soname finds the
            # already-loaded library instead of searching paths.
            import ctypes
            import nvidia.cublas, nvidia.cudnn
            cublas_dir = os.path.join(list(nvidia.cublas.__path__)[0], "lib")
            cudnn_dir = os.path.join(list(nvidia.cudnn.__path__)[0], "lib")
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                [cublas_dir, cudnn_dir, os.environ.get("LD_LIBRARY_PATH", "")]
            )
            for fname in ("libcublasLt.so.12", "libcublas.so.12"):
                ctypes.CDLL(os.path.join(cublas_dir, fname), mode=ctypes.RTLD_GLOBAL)
            for fname in ("libcudnn.so.9",):
                ctypes.CDLL(os.path.join(cudnn_dir, fname), mode=ctypes.RTLD_GLOBAL)

            from faster_whisper import WhisperModel
            logger.info("Loading Whisper medium (GPU)...")
            self._whisper = WhisperModel("medium", device="cuda", compute_type="float16")
        self._ensure_openai()
        self._ensure_robot()

    def process_audio(self, audio_bytes: bytes, language: str = "ko"):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        self._cancel_evt.clear()
        threading.Thread(target=self._pipeline, args=(audio_bytes, language), daemon=True).start()

    def process_audio_array(self, audio_f32_16k):
        """Same auto-execute pipeline as process_audio(), but for audio that's
        already decoded PCM (16kHz mono float32) instead of a browser upload —
        used by the wake-word listener, which captures straight from the
        robot's own mic over WebRTC (see wake_word_listener.py) so there's no
        webm/ogg container to decode."""
        if self._busy:
            logger.info("Wake-word: pipeline busy — dropping this utterance")
            return
        self._busy = True
        self._cancel_evt.clear()
        threading.Thread(target=self._pipeline_from_array, args=(audio_f32_16k,), daemon=True).start()

    def _pipeline(self, audio_bytes: bytes, language: str = "ko"):
        from app.voice.voice_control_go2 import WHISPER_INITIAL_PROMPTS
        try:
            self._emit("status", step="init", message="초기화 중...")
            self._lazy_init()

            # STT
            self._emit("status", step="stt", message="음성 인식 중...")
            t_stt = time.monotonic()
            raw_tmp = wav_tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                    f.write(audio_bytes)
                    raw_tmp = f.name
                wav_tmp = raw_tmp.replace(".webm", ".wav")
                conv = subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_tmp,
                     "-ar", "16000", "-ac", "1", wav_tmp],
                    capture_output=True, timeout=15,
                )
                if conv.returncode != 0 or not os.path.exists(wav_tmp):
                    raise RuntimeError("ffmpeg conversion failed: " + conv.stderr.decode()[-200:])
                # Explicit language (not auto-detect) -- forcing the correct
                # language is far more reliable than letting Whisper guess,
                # especially on short commands where auto-detect is shakiest.
                # initial_prompt biases decoding toward robot-command
                # vocabulary in that language instead of generic speech.
                # "ko-jeju" isn't a real Whisper language code -- Jeju is
                # phonetically still Korean, only the *vocabulary* differs
                # (see voice_control_go2.WHISPER_INITIAL_PROMPTS) -- so Whisper
                # itself always gets "ko"; the dialect distinction is carried
                # by initial_prompt here and by get_system_prompt() downstream.
                whisper_lang = "ko" if language.startswith("ko") else language
                segs, _ = self._whisper.transcribe(
                    wav_tmp, language=whisper_lang, beam_size=5,
                    initial_prompt=WHISPER_INITIAL_PROMPTS.get(language),
                )
                text = " ".join(s.text for s in segs).strip()
            finally:
                for p in [raw_tmp, wav_tmp]:
                    if p:
                        try:
                            os.unlink(p)
                        except Exception:
                            pass

            stt_ms = (time.monotonic() - t_stt) * 1000
            if not text:
                self._emit("error", message="음성이 인식되지 않았습니다.")
                self._record_analytics(source="mic", language=language, understood=False, success=False, stt_ms=stt_ms, error="empty STT result")
                return
            self._emit("stt_result", text=text)
            self._llm_and_execute(text, language=language, source="mic", stt_ms=stt_ms)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self._emit("error", message=str(e))
            self._record_analytics(source="mic", language=language, understood=False, success=False, error=str(e))
        finally:
            self._busy = False

    def _pipeline_from_array(self, audio_f32_16k):
        t_stt = time.monotonic()
        try:
            self._emit("status", step="init", message="초기화 중...")
            self._lazy_init()

            self._emit("status", step="stt", message="음성 인식 중...")
            segs, _ = self._whisper.transcribe(audio_f32_16k, language="ko", beam_size=5)
            text = " ".join(s.text for s in segs).strip()
            stt_ms = (time.monotonic() - t_stt) * 1000

            if not text:
                self._emit("error", message="음성이 인식되지 않았습니다.")
                self._record_analytics(source="wake_word", language="ko", understood=False, success=False, stt_ms=stt_ms, error="empty STT result")
                return
            self._emit("stt_result", text=text)
            self._llm_and_execute(text, source="wake_word", stt_ms=stt_ms)

        except Exception as e:
            logger.error(f"Pipeline (wake-word) error: {e}", exc_info=True)
            self._emit("error", message=str(e))
            self._record_analytics(source="wake_word", language="ko", understood=False, success=False, error=str(e))
        finally:
            self._busy = False

    def _llm_and_execute(self, text: str, language: str = "ko", source: str = "mic", stt_ms: float = 0.0):
        """LLM-analyze then immediately auto-execute — shared by the browser
        mic upload and the robot's own wake-word-triggered mic capture. Kept
        separate from the text-command panel's analyze/confirm/cancel flow
        (see analyze_text()) intentionally: both mic paths already behave
        this way and this preserves that (tested) behavior as-is."""
        import json as _j, re

        self._emit("status", step="llm", message="명령 분석 중...")
        from app.voice.voice_control_go2 import get_system_prompt, apply_tone_to_moves

        t_llm = time.monotonic()
        resp = self._openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt(language)},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=512,
        )
        llm_ms = (time.monotonic() - t_llm) * 1000
        raw = resp.choices[0].message.content
        try:
            cmd = _j.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            cmd = _j.loads(m.group()) if m else {}

        tts_text = cmd.get("response", "")
        tone = cmd.get("tone", "neutral")
        self._emit("llm_result", response=tts_text, actions=cmd.get("actions") or [], tone=tone)

        if not cmd.get("understood", False):
            self._emit("status", step="done", message="명령을 이해하지 못했습니다.")
            if tts_text:
                threading.Thread(target=self._speak, args=(tts_text,), daemon=True).start()
            self._record_analytics(
                source=source, language=language, tone=tone, understood=False,
                success=False, stt_ms=stt_ms, llm_ms=llm_ms,
            )
            return

        # "지금 뭐가 보이나요?" etc. -- a camera question, not a robot action.
        # cmd["response"] here is only a short filler ("확인해볼게요!", already
        # shown above via llm_result) -- don't speak it, go straight to the
        # real vision pipeline and speak *that* result instead, same as the
        # dedicated "📷 지금 뭐가 보이나요?" button (_vision_analysis_sequence).
        if any((a or {}).get("name") == "vision_analyze" for a in (cmd.get("actions") or [])):
            self._emit("status", step="vision", message="📷 카메라 화면을 분석하는 중...")
            try:
                result_text, tts_lang, vision_ms = self._analyze_camera_scene(language)
                self._emit("llm_result", response=result_text, actions=[])
                self._speak(result_text, lang=tts_lang)
                self._emit("status", step="done", message="분석 완료")
                self._record_analytics(
                    source=source, language=language, tone=tone, understood=True, success=True,
                    actions=["vision_analyze"], stt_ms=stt_ms, llm_ms=llm_ms + vision_ms,
                )
            except Exception as e:
                logger.error(f"Vision analysis ({source}) error: {e}", exc_info=True)
                self._emit("error", message=f"화면 분석 실패: {e}")
                self._record_analytics(
                    source=source, language=language, tone=tone, understood=True, success=False,
                    actions=["vision_analyze"], stt_ms=stt_ms, llm_ms=llm_ms, error=str(e),
                )
            return

        # "저기로 가줘" etc. -- an autonomous, many-second SLAM drive, not a
        # schema action -- see slam_navigator.py. Doesn't go through
        # RobotController.execute_action() like everything else below,
        # same reason vision_analyze is special-cased above it.
        nav_actions = [a for a in (cmd.get("actions") or []) if (a or {}).get("name") == "navigate_to"]
        if nav_actions:
            location = nav_actions[0].get("location", "")
            if tts_text:
                threading.Thread(target=self._speak, args=(tts_text,), daemon=True).start()
            self._emit("status", step="navigate", message=f"🧭 '{location}'(으)로 이동 중...")
            try:
                if _navigator is None:
                    raise RuntimeError("내비게이션 모듈을 사용할 수 없습니다")
                t_nav = time.monotonic()
                arrived = _navigator.navigate_to(location)
                nav_ms = (time.monotonic() - t_nav) * 1000
                if arrived:
                    self._emit("status", step="done", message=f"✅ '{location}' 도착")
                else:
                    self._emit("error", message=f"❌ 이동 실패: {_navigator.last_error}")
                self._record_analytics(
                    source=source, language=language, tone=tone, understood=True, success=arrived,
                    actions=["navigate_to"], stt_ms=stt_ms, llm_ms=llm_ms, exec_ms=nav_ms,
                    error=None if arrived else _navigator.last_error,
                )
            except Exception as e:
                logger.error(f"Navigate ({source}) error: {e}", exc_info=True)
                self._emit("error", message=f"이동 실패: {e}")
                self._record_analytics(
                    source=source, language=language, tone=tone, understood=True, success=False,
                    actions=["navigate_to"], stt_ms=stt_ms, llm_ms=llm_ms, error=str(e),
                )
            return

        from app.core.action_merge import merge_consecutive_moves
        merged = merge_consecutive_moves(
            cmd.get("actions") or [], cmd, self._robot.registry.get_action_name
        )
        merged = apply_tone_to_moves(merged, tone)

        # Runtime verification (V_s -> V_i -> V_c) -- runs once for the whole
        # sequence, before any action or the original TTS response is spoken,
        # since speaking cmd["response"] ("I'll jump!") and then rejecting the
        # command would itself be misleading. This is the auto-execute voice
        # path -- unlike the text-panel's confirm step, nothing here is a
        # substitute for this check, so it is not optional on this path.
        from app.safety.runtime_verification import verify_sequence
        from app.safety.verification_config import get_verification_config
        robot_state = self._robot.executor.robot_state
        verification = verify_sequence(
            text, merged, robot_state, get_verification_config(), telemetry=_telemetry, person_gate=_person_gate,
        )
        if not verification.accepted:
            reject_msg = f"⚠️ 안전 검증 실패 [{verification.reject_layer}]: {verification.reject_reason}"
            logger.warning(reject_msg)
            self._emit("error", message=reject_msg)
            self._emit("status", step="done", message=reject_msg)
            threading.Thread(
                target=self._speak,
                args=("죄송해요, 안전상의 이유로 이 명령을 실행할 수 없어요.",),
                daemon=True,
            ).start()
            self._record_analytics(
                source=source, language=language, tone=tone, understood=True, success=False,
                actions=verification.resolved_actions, stt_ms=stt_ms, llm_ms=llm_ms,
                error=verification.reject_reason,
                reject_layer=verification.reject_layer, reject_reason=verification.reject_reason,
                battery_pct=(verification.live_battery_pct if verification.live_battery_pct is not None else robot_state.battery_level),
                is_standing=robot_state.is_standing,
                obstacle_distance_m=robot_state.obstacle_distance_m, raw_llm_output=raw,
                verify_schema_ms=verification.schema_ms, verify_intent_ms=verification.intent_ms,
                verify_context_ms=verification.context_ms,
            )
            return

        # Execute + TTS in parallel
        status_msg = "실행 중..." if tone == "neutral" else f"실행 중... (tone: {tone})"
        self._emit("status", step="executing", message=status_msg)
        tts_t = threading.Thread(target=self._speak, args=(tts_text,), daemon=True)
        tts_t.start()

        t_exec = time.monotonic()
        ok = True
        executed_names = []
        last_error = None
        for name, params in merged:
            if self._cancel_evt.is_set():
                last_error = "긴급 정지로 중단됨"
                ok = False
                break
            self._emit("action", name=name, params=params)
            executed_names.append(name)
            result = self._robot.execute_action(name, **params)
            if not result.get("success"):
                last_error = result.get("error") or "원인 불명 — 서버 로그 확인"
                self._emit("error", message=f"❌ {name} 실패: {last_error}")
                ok = False
                break
        exec_ms = (time.monotonic() - t_exec) * 1000

        tts_t.join()
        if ok:
            self._emit("status", step="done", message="완료!")
        self._record_analytics(
            source=source, language=language, tone=tone, understood=True, success=ok,
            actions=executed_names, stt_ms=stt_ms, llm_ms=llm_ms, exec_ms=exec_ms, error=last_error,
            battery_pct=(verification.live_battery_pct if verification.live_battery_pct is not None else robot_state.battery_level),
                is_standing=robot_state.is_standing,
            obstacle_distance_m=robot_state.obstacle_distance_m, raw_llm_output=raw,
            verify_schema_ms=verification.schema_ms, verify_intent_ms=verification.intent_ms,
            verify_context_ms=verification.context_ms,
        )

    def _record_analytics(self, **kwargs):
        try:
            from app.analytics.analytics import get_store
            get_store().record(**kwargs)
        except Exception as e:
            logger.warning(f"Analytics record failed: {e}")

    def _speak(self, text: str, lang: str = "ko"):
        if not text.strip():
            return
        tmp = None
        try:
            from gtts import gTTS
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            gTTS(text=text, lang=lang, slow=False).save(tmp)

            if self._play_on_robot_speaker(tmp):
                return

            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp],
                check=True, timeout=30,
            )
        except Exception as e:
            logger.warning(f"TTS: {e}")
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def _play_on_robot_speaker(self, mp3_path: str) -> bool:
        """Play through the Go2's own speaker over the shared WebRTC data
        channel (see webrtc_audio_hub.py) instead of the local machine's
        speaker. Sends the whole reply as one clip (see play_wav() in
        webrtc_audio_hub.py; WebRTCAudioHub also supports playing sentence
        by sentence via enter()/play_in_session(), tried and reverted --
        the single-clip version is simpler and was preferred). Returns
        False -- meaning "fall back to local ffplay playback" -- whenever
        WiFi/WebRTC isn't the active transport, the shared connection isn't
        up yet, or anything about the robot-speaker path fails; this must
        never be the only way TTS can be heard."""
        if _camera is None or not _camera._is_wifi():
            return False
        if _camera.webrtc_conn is None or _camera.webrtc_loop is None:
            # Most common right after startup: the camera's WebRTC handshake
            # (ICE negotiation etc.) can take several seconds, and there's no
            # reconnect loop if it later drops mid-session -- see
            # CameraStreamer._webrtc_loop()'s docstring. Either way this TTS
            # call falls back to the laptop; a later one may not, once/if the
            # connection comes up. That's the "sometimes X sometimes Y" you're
            # seeing -- it's connection-health-dependent, not random.
            self._emit("status", step="tts_fallback",
                       message="🔈 로봇 WebRTC 연결 없음 — 이번 응답은 노트북 스피커로 재생")
            return False
        wav_path = None
        try:
            from pydub import AudioSegment
            from app.voice.webrtc_audio_hub import WebRTCAudioHub
            audio = AudioSegment.from_mp3(mp3_path).set_frame_rate(44100)
            wav_path = mp3_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav", parameters=["-ar", "44100"])
            # The megaphone transfer can legitimately take several seconds to
            # tens of seconds for a long reply (see webrtc_audio_hub.py's
            # docstring -- it's the chunked-upload RTT, not something this
            # status message fixes) -- without this, the UI shows the text
            # response then goes quiet with no signal that audio is still on
            # its way, which reads as broken/hung rather than "in progress".
            self._emit("status", step="speaking", message="🔊 로봇 스피커로 음성 전송 중... (문장이 길수록 오래 걸려요)")
            WebRTCAudioHub(_camera.webrtc_conn, _camera.webrtc_loop).play_wav(wav_path)
            return True
        except Exception as e:
            logger.warning(f"Robot speaker playback failed, falling back to local speaker: {e}")
            self._emit("status", step="tts_fallback",
                       message=f"🔈 로봇 스피커 재생 실패 ({e}) — 이번 응답은 노트북 스피커로 재생")
            return False
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def run_action(self, action_name: str, **params):
        threading.Thread(target=self._direct, args=(action_name,), kwargs=params, daemon=True).start()

    def _direct(self, name: str, **params):
        try:
            self._lazy_init()
            self._emit("action", name=name, params=params)
            result = self._robot.execute_action(name, **params)
            if result.get("success"):
                self._emit("status", step="done", message=f"{name} 완료!")
            else:
                err = result.get("error") or "실행 실패 (원인 불명 — 서버 로그 확인)"
                self._emit("error", message=f"❌ {name} 실패: {err}")
            self._record_analytics(
                source="direct", language="n/a", understood=True,
                success=bool(result.get("success")), actions=[name],
                error=result.get("error"),
            )
        except Exception as e:
            self._emit("error", message=str(e))
            self._record_analytics(
                source="direct", language="n/a", understood=True,
                success=False, actions=[name], error=str(e),
            )

    # ── Text command: analyze -> stage -> explicit confirm/cancel ──────────
    def analyze_text(self, text: str):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        threading.Thread(target=self._analyze_text_worker, args=(text,), daemon=True).start()

    def _analyze_text_worker(self, text: str):
        import json as _j, re
        try:
            self._emit("status", step="init", message="초기화 중...")
            self._ensure_openai()
            self._ensure_robot()
            self._emit("stt_result", text=text)

            self._emit("status", step="llm", message="명령 분석 중...")
            from app.voice.voice_control_go2 import KOREAN_SYSTEM_PROMPT, apply_tone_to_moves

            t_llm = time.monotonic()
            resp = self._openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": KOREAN_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=512,
            )
            llm_ms = (time.monotonic() - t_llm) * 1000
            raw = resp.choices[0].message.content
            try:
                cmd = _j.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                cmd = _j.loads(m.group()) if m else {}

            tts_text = cmd.get("response", "")
            tone = cmd.get("tone", "neutral")
            actions = cmd.get("actions") or []
            self._emit("llm_result", response=tts_text, actions=actions, tone=tone)

            if not cmd.get("understood", False):
                self._pending = None
                self._emit("status", step="done", message="명령을 이해하지 못했습니다.")
                self._emit("analysis_ready", ready=False, response=tts_text, actions=[])
                self._record_analytics(
                    source="text", language="ko", tone=tone, understood=False,
                    success=False, llm_ms=llm_ms,
                )
                return

            # Camera question, not a robot movement -- nothing physical to
            # confirm, so run it immediately instead of staging into
            # self._pending like every other (movement) action (see
            # _llm_and_execute's identical branch for why cmd["response"]
            # here is a filler, not spoken).
            if any((a or {}).get("name") == "vision_analyze" for a in actions):
                self._pending = None
                self._emit("analysis_ready", ready=False, response=tts_text, actions=[])
                self._emit("status", step="vision", message="📷 카메라 화면을 분석하는 중...")
                try:
                    result_text, tts_lang, vision_ms = self._analyze_camera_scene("ko")
                    self._emit("llm_result", response=result_text, actions=[])
                    self._speak(result_text, lang=tts_lang)
                    self._emit("status", step="done", message="분석 완료")
                    self._record_analytics(
                        source="text", language="ko", tone=tone, understood=True, success=True,
                        actions=["vision_analyze"], llm_ms=llm_ms + vision_ms,
                    )
                except Exception as e:
                    logger.error(f"Vision analysis (text) error: {e}", exc_info=True)
                    self._emit("error", message=f"화면 분석 실패: {e}")
                    self._record_analytics(
                        source="text", language="ko", tone=tone, understood=True, success=False,
                        actions=["vision_analyze"], llm_ms=llm_ms, error=str(e),
                    )
                return

            from app.core.action_merge import merge_consecutive_moves
            merged = merge_consecutive_moves(actions, cmd, self._robot.registry.get_action_name)
            merged = apply_tone_to_moves(merged, tone)
            self._pending = {
                "actions": merged, "response": tts_text, "tone": tone,
                "llm_ms": llm_ms, "instruction": text,
            }
            self._emit(
                "analysis_ready", ready=True, response=tts_text, tone=tone,
                actions=[{"name": n, "params": p} for n, p in merged],
            )
            self._emit("status", step="done", message="분석 완료 — 실행을 눌러 확인하세요")

        except Exception as e:
            logger.error(f"Text analysis error: {e}", exc_info=True)
            self._emit("error", message=str(e))
            self._record_analytics(source="text", language="ko", understood=False, success=False, error=str(e))
        finally:
            self._busy = False

    def confirm_execute(self):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        pending = self._pending
        if not pending:
            self._emit("error", message="확인할 명령이 없습니다.")
            return
        self._pending = None
        self._busy = True
        self._cancel_evt.clear()
        threading.Thread(target=self._execute_pending, args=(pending,), daemon=True).start()

    def _execute_pending(self, pending: dict):
        tone = pending.get("tone", "neutral")
        try:
            # Runtime verification also runs here, not just in the
            # auto-execute voice path -- the text panel's confirm click is a
            # human eyeballing the parsed actions, not a substitute for the
            # same automated schema/intent/context checks the thesis
            # evaluates. Run before speaking pending["response"], for the
            # same reason as _llm_and_execute: don't say "I'll do X" and
            # then refuse to do it.
            from app.safety.runtime_verification import verify_sequence
            from app.safety.verification_config import get_verification_config
            robot_state = self._robot.executor.robot_state
            verification = verify_sequence(
                pending.get("instruction", ""), pending["actions"], robot_state,
                get_verification_config(), telemetry=_telemetry, person_gate=_person_gate,
            )
            if not verification.accepted:
                reject_msg = f"⚠️ 안전 검증 실패 [{verification.reject_layer}]: {verification.reject_reason}"
                logger.warning(reject_msg)
                self._emit("error", message=reject_msg)
                self._emit("status", step="done", message=reject_msg)
                threading.Thread(
                    target=self._speak,
                    args=("죄송해요, 안전상의 이유로 이 명령을 실행할 수 없어요.",),
                    daemon=True,
                ).start()
                self._record_analytics(
                    source="text", language="ko", tone=tone, understood=True, success=False,
                    actions=verification.resolved_actions, llm_ms=pending.get("llm_ms", 0.0),
                    error=verification.reject_reason,
                    reject_layer=verification.reject_layer, reject_reason=verification.reject_reason,
                    battery_pct=(verification.live_battery_pct if verification.live_battery_pct is not None else robot_state.battery_level),
                is_standing=robot_state.is_standing,
                    obstacle_distance_m=robot_state.obstacle_distance_m,
                    verify_schema_ms=verification.schema_ms, verify_intent_ms=verification.intent_ms,
                    verify_context_ms=verification.context_ms,
                )
                return

            status_msg = "실행 중..." if tone == "neutral" else f"실행 중... (tone: {tone})"
            self._emit("status", step="executing", message=status_msg)
            tts_t = threading.Thread(target=self._speak, args=(pending["response"],), daemon=True)
            tts_t.start()

            t_exec = time.monotonic()
            ok = True
            executed_names = []
            last_error = None
            for name, params in pending["actions"]:
                if self._cancel_evt.is_set():
                    last_error = "긴급 정지로 중단됨"
                    ok = False
                    break
                self._emit("action", name=name, params=params)
                executed_names.append(name)
                if name == "navigate_to":
                    # Not a schema action (see slam_navigator.py) -- doesn't
                    # go through RobotController.execute_action().
                    location = params.get("location", "")
                    self._emit("status", step="navigate", message=f"🧭 '{location}'(으)로 이동 중...")
                    if _navigator is None:
                        result = {"success": False, "error": "내비게이션 모듈을 사용할 수 없습니다"}
                    else:
                        arrived = _navigator.navigate_to(location)
                        result = {"success": arrived, "error": _navigator.last_error}
                else:
                    result = self._robot.execute_action(name, **params)
                if not result.get("success"):
                    last_error = result.get("error") or "원인 불명 — 서버 로그 확인"
                    self._emit("error", message=f"❌ {name} 실패: {last_error}")
                    ok = False
                    break
            exec_ms = (time.monotonic() - t_exec) * 1000

            tts_t.join()
            if ok:
                self._emit("status", step="done", message="완료!")
            self._record_analytics(
                source="text", language="ko", tone=tone, understood=True, success=ok,
                actions=executed_names, llm_ms=pending.get("llm_ms", 0.0),
                exec_ms=exec_ms, error=last_error,
            )
        except Exception as e:
            logger.error(f"Execute error: {e}", exc_info=True)
            self._emit("error", message=str(e))
            self._record_analytics(source="text", language="ko", tone=tone, understood=True, success=False, error=str(e))
        finally:
            self._busy = False

    def cancel_pending(self):
        self._pending = None
        self._emit("analysis_ready", ready=False, response="", actions=[])
        self._emit("status", step="done", message="취소됨")

    # ── Manual control: continuous D-pad move + speed + emergency stop ─────
    def adjust_speed(self, delta: float):
        self._speed = round(max(0.1, min(0.6, self._speed + delta)), 2)
        self._emit("speed", value=self._speed)

    def start_move(self, direction: str):
        vec = self._MOVE_DIRS.get(direction)
        if vec is None:
            self._emit("error", message=f"Unknown direction: {direction}")
            return
        self.stop_move_continuous()
        self._move_stop_evt.clear()
        dx, dy, dz = vec

        def _loop():
            try:
                self._ensure_robot()
            except Exception as e:
                self._emit("error", message=str(e))
                return
            sc = self._robot.executor.sport_client
            try:
                # Plain Move() has no obstacle awareness at all -- FreeAvoid
                # is a separate gait-mode toggle that makes the robot's own
                # onboard sensing refuse to walk into something it detects.
                # Off by default (PersonTracker explicitly disables it too,
                # since it otherwise treats the tracked person as an
                # obstacle and won't finish turning toward them) -- only
                # switched on for manual D-pad driving specifically, where a
                # human not looking at the dashboard's own (laggy) video feed
                # is the one steering. Best-effort: never block driving if
                # this call fails or isn't supported on this firmware.
                sc.FreeAvoid(True)
            except Exception as e:
                logger.warning(f"FreeAvoid(True) failed: {e}")
            while not self._move_stop_evt.is_set():
                try:
                    sc.Move(dx * self._speed, dy * self._speed, dz * self._speed * self._TURN_OMEGA_SCALE)
                except Exception as e:
                    logger.warning(f"Move failed: {e}")
                self._move_stop_evt.wait(0.3)
            try:
                sc.Move(0, 0, 0)
            except Exception:
                pass
            try:
                sc.FreeAvoid(False)
            except Exception as e:
                logger.warning(f"FreeAvoid(False) failed: {e}")

        self._move_thread = threading.Thread(target=_loop, daemon=True, name="dpad-move")
        self._move_thread.start()
        self._emit("action", name="move", params={"direction": direction, "speed": self._speed})
        # One row per press, not per Move() tick (the loop above resends every
        # 0.3s while the button is held).
        self._record_analytics(source="direct", language="n/a", understood=True, success=True, actions=[f"move:{direction}"])

    def stop_move_continuous(self):
        self._move_stop_evt.set()
        if self._move_thread and self._move_thread.is_alive():
            self._move_thread.join(timeout=1.0)
        self._move_thread = None

    def emergency_stop(self):
        # Stops remaining steps of any in-progress sequence (self-intro's
        # script/actions, a multi-action LLM command still executing) --
        # checked between steps in _self_intro_sequence()/_llm_and_execute()/
        # _execute_pending(). Doesn't interrupt a single TTS utterance already
        # mid-playback, but no further lines/actions will start after it.
        self._cancel_evt.set()
        self.stop_move_continuous()
        self._pending = None
        if _tracker is not None:
            # Person tracking drives its own continuous rotation loop -- without
            # this it would keep issuing Move() a few ms later and undo the stop.
            _tracker.stop()
        if _patrol is not None:
            # Same reason -- Auto Patrol's route loop is otherwise untouched
            # by emergency_stop() and would keep moving on its next tick.
            _patrol.stop()
        if _navigator is not None:
            # Same reason -- SlamNavigator's drive-to-point loop is otherwise
            # untouched by emergency_stop() and would keep moving.
            _navigator.stop()
        try:
            self._ensure_robot()
            self._robot.executor.sport_client.StopMove()
            self._emit("analysis_ready", ready=False, response="", actions=[])
            self._emit("status", step="done", message="🛑 긴급 정지 실행됨")
        except Exception as e:
            self._emit("error", message=f"긴급 정지 실패: {e}")

    # ── Self-intro demo ─────────────────────────────────────────────────────
    def run_self_intro(self):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        self._cancel_evt.clear()
        threading.Thread(target=self._self_intro_sequence, daemon=True).start()

    def _self_intro_sequence(self):
        try:
            self._ensure_robot()  # no Whisper/LLM needed for a scripted sequence

            self._emit("status", step="executing", message="자기소개 데모 시작")
            for name in ("stand_up", "hello"):
                if self._cancel_evt.is_set():
                    break
                self._emit("action", name=name, params={})
                result = self._robot.execute_action(name)
                if not result.get("success"):
                    logger.warning(f"Self-intro: {name} failed: {result.get('error')}")

            for text, lang in SELF_INTRO_SCRIPT:
                if self._cancel_evt.is_set():
                    break
                self._emit("llm_result", response=text, actions=[])
                self._speak(text, lang=lang)

            for name in SELF_INTRO_DEMO_ACTIONS:
                if self._cancel_evt.is_set():
                    break
                self._emit("action", name=name, params={})
                result = self._robot.execute_action(name)
                if not result.get("success"):
                    logger.warning(f"Self-intro: {name} failed: {result.get('error')}")

            for text, lang in SELF_INTRO_CLOSING:
                if self._cancel_evt.is_set():
                    break
                self._emit("llm_result", response=text, actions=[])
                self._speak(text, lang=lang)

            if self._cancel_evt.is_set():
                self._emit("status", step="done", message="자기소개 데모 중단됨 (긴급 정지)")
            else:
                self._emit("status", step="done", message="자기소개 데모 완료!")
        except Exception as e:
            logger.error(f"Self-intro error: {e}", exc_info=True)
            self._emit("error", message=str(e))
        finally:
            self._busy = False

    # ── Vision analysis: "what do you see right now" ───────────────────────
    # Ports camera_vision_download/camera_vision_test2.py's analyze_frame()
    # (InsightFace person ID + GPT-4o-mini vision scene description), reusing
    # the dashboard's own camera feed instead of a second VideoClient, the
    # face model/people DB already loaded by AutoPatrol instead of loading
    # InsightFace's buffalo_l a second time, and _speak() (robot-speaker-
    # first, laptop fallback) instead of camera_vision_test2.py's own
    # pygame-only speak_text().
    def run_vision_analysis(self, language: str = "ko"):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        self._cancel_evt.clear()
        threading.Thread(target=self._vision_analysis_sequence, args=(language,), daemon=True).start()

    def _vision_analysis_sequence(self, language: str = "ko"):
        try:
            self._emit("status", step="vision", message="📷 카메라 화면을 분석하는 중...")
            result_text, tts_lang, llm_ms = self._analyze_camera_scene(language)
            self._emit("llm_result", response=result_text, actions=[])
            self._speak(result_text, lang=tts_lang)
            self._emit("status", step="done", message="분석 완료")
            self._record_analytics(
                source="vision_analyze", language=language, understood=True,
                success=True, llm_ms=llm_ms,
            )
        except Exception as e:
            logger.error(f"Vision analysis error: {e}", exc_info=True)
            self._emit("error", message=f"화면 분석 실패: {e}")
            self._record_analytics(
                source="vision_analyze", language=language, understood=True,
                success=False, error=str(e),
            )
        finally:
            self._busy = False

    def _analyze_camera_scene(self, language: str = "ko") -> Tuple[str, str, float]:
        """Core of "what do you see": capture -> InsightFace -> GPT-4o-mini
        vision -> description text. Raises RuntimeError on failure. Returns
        (description, tts_lang, llm_ms) -- shared by the dedicated button
        (_vision_analysis_sequence) and the voice/text pipeline's
        "vision_analyze" pseudo-action (_llm_and_execute) so both paths stay
        in sync instead of maintaining two copies of the prompt/pipeline."""
        if _camera is None or not _camera.connected:
            raise RuntimeError("카메라가 연결되어 있지 않습니다")
        jpeg = _camera.get_frame()
        if not jpeg:
            raise RuntimeError("카메라 프레임을 가져올 수 없습니다")
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("카메라 이미지를 디코딩할 수 없습니다")

        if _patrol is None:
            raise RuntimeError("Vision 모듈을 사용할 수 없습니다")
        _patrol._load_vision()  # idempotent -- shares AutoPatrol's InsightFace model/people DB

        vpath = str(Path(__file__).parent / "camera_vision_download")
        if vpath not in sys.path:
            sys.path.insert(0, vpath)
        from app.vision.camera_vision_download.camera_vision_test2 import recognize_faces, make_people_context
        from app.navigation.auto_patrol import SUPPORTED_TTS_LANGUAGES

        # gTTS only accepts real language codes -- "ko-jeju" (an internal
        # STT dialect marker, not a TTS language) must not reach _speak().
        tts_lang = language if language in SUPPORTED_TTS_LANGUAGES else "ko"

        face_results = recognize_faces(frame, _patrol._people, _patrol._face_model)
        people_context = make_people_context(face_results)
        lang_name = SUPPORTED_TTS_LANGUAGES.get(language, "한국어")

        self._ensure_openai()
        prompt = f"""이 이미지는 Unitree Go2 로봇의 전면 카메라 영상입니다.

현재 장면을 자연스러운 {lang_name} 구어체로 설명하세요.

아래에는 별도의 얼굴 인식 시스템이 현재 화면에서 분석한 등록 인물 정보가 있습니다.
사람의 이름과 신원은 반드시 아래의 등록 인물 판정 결과만 사용하세요.
이미지만 보고 사람의 이름이나 신원을 추측하지 마세요.

[등록 인물 판정 결과]
{people_context}

응답 규칙:
1. 이미지에서 실제로 명확하게 보이는 주요 물체와 장면을 짧게 설명하세요.
2. 물체/사람의 위치를 설명할 때는 반드시 "보는 사람(카메라) 기준"으로 왼쪽, 중앙, 오른쪽을
   판단하세요 — 이미지 프레임에서 실제로 왼쪽에 있으면 "왼쪽"입니다. 사람이 카메라를 마주보고
   있는 경우 그 사람 자신의 왼손/오른손 기준과는 좌우가 반대가 됩니다 — 절대 사람 자신의
   왼손/오른손 기준으로 말하지 마세요 (카메라를 마주보는 사람 기준 왼손 쪽은 이미지에서는
   오른쪽에 보입니다). 좌우 판단에 자신이 없으면 왼쪽/오른쪽을 단정하지 말고 "나란히 있다"처럼
   방향을 특정하지 않는 표현을 쓰세요.
3. 위치가 확실하지 않으면 위치를 말하지 마세요.
4. 거리나 이동 가능 여부 또는 이동 경로는 추측하지 마세요.
5. 등록 인물이 있는 경우 반드시 이름과 직함을 함께 사용하여 소개하세요.
6. 등록된 소개글을 그대로 읽지 말고 사람에게 말하듯 자연스럽게 한두 문장으로 다듬어 소개하세요.
7. 등록되지 않은 사람이 있는 경우에는 이름을 추측하지 말고 모르는 사람이라는 취지로 자연스럽게 표현하세요.
8. 장면 설명과 인물 소개를 하나의 자연스러운 설명처럼 이어서 말하세요.
9. 반드시 3문장 이내로, 핵심만 간결하게 설명하세요 (음성으로 로봇 스피커에 전송되는데
   응답이 길수록 실제로 소리가 나오기까지 오래 걸립니다 — 짧게 답하는 것이 매우 중요합니다).
10. 번호나 목록 형식으로 답하지 마세요.
11. 반드시 {lang_name}로만 답변하세요 (다른 언어를 섞지 마세요)."""

        image_base64 = base64.b64encode(jpeg).decode("utf-8")
        t_llm = time.monotonic()
        resp = self._openai.responses.create(
            model=LLM_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_base64}"},
                ],
            }],
            # Hard backstop on top of rule 9's 3-sentence instruction -- a
            # long reply here isn't just verbose, it directly costs tens of
            # extra seconds to transfer over the megaphone RPC (see
            # webrtc_audio_hub.py's docstring), so this is a latency control,
            # not just a style preference.
            max_output_tokens=220,
        )
        llm_ms = (time.monotonic() - t_llm) * 1000
        result_text = (resp.output_text or "").strip()
        if not result_text:
            raise RuntimeError("빈 응답")
        return result_text, tts_lang, llm_ms

    # ── Face registration: add a person to camera_vision_download's DB ─────
    # Captures the current camera frame (no separate upload flow -- stand in
    # front of the robot and register, matching how recognition itself
    # works), saves it into camera_vision_download/faces/, appends a row to
    # vision_people.xlsx (same name/title/photo/introduction columns
    # load_people_database() already expects), then reloads AutoPatrol's
    # cached people list so the person is recognizable immediately --
    # without this last step the new row would sit unused until the whole
    # server restarted.
    def register_face(self, name: str, title: str = "", introduction: str = ""):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        threading.Thread(
            target=self._register_face_sequence, args=(name, title, introduction), daemon=True
        ).start()

    def _register_face_sequence(self, name: str, title: str, introduction: str):
        try:
            name = (name or "").strip()
            if not name:
                raise RuntimeError("이름을 입력하세요")

            self._emit("status", step="register", message=f"📸 {name} 등록 중...")

            if _camera is None or not _camera.connected:
                raise RuntimeError("카메라가 연결되어 있지 않습니다")
            jpeg = _camera.get_frame()
            if not jpeg:
                raise RuntimeError("카메라 프레임을 가져올 수 없습니다")
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("카메라 이미지를 디코딩할 수 없습니다")

            if _patrol is None:
                raise RuntimeError("Vision 모듈을 사용할 수 없습니다")
            _patrol._load_vision()  # idempotent

            faces = _patrol._face_model.get(frame)
            if not faces:
                raise RuntimeError("화면에서 얼굴을 찾지 못했습니다 — 카메라를 정면으로 봐주세요")

            vision_dir = Path(__file__).parent / "camera_vision_download"
            faces_dir = vision_dir / "faces"
            faces_dir.mkdir(exist_ok=True)
            # UUID-based filename -- independent of what characters are in
            # `name` (Korean/Vietnamese and stray punctuation all fine on
            # disk, but this sidesteps collisions/sanitization entirely) --
            # the *display* name lives in the spreadsheet, not the filename.
            filename = f"person_{uuid.uuid4().hex[:8]}.jpg"
            photo_path = faces_dir / filename
            if not cv2.imwrite(str(photo_path), frame):
                raise RuntimeError("사진 파일을 저장하지 못했습니다")

            from openpyxl import load_workbook
            excel_path = vision_dir / "vision_people.xlsx"
            with _patrol._vision_lock:
                wb = load_workbook(excel_path)
                sheet = wb.active
                sheet.append([name, title.strip(), filename, introduction.strip()])
                wb.save(excel_path)

            _patrol.reload_people()

            self._emit(
                "status", step="done",
                message=f"✅ {name} 등록 완료! (등록 인원 {len(_patrol._people)}명)",
            )
            self._record_analytics(
                source="direct", language="n/a", understood=True, success=True,
                actions=["register_face"],
            )
        except Exception as e:
            logger.error(f"Face registration error: {e}", exc_info=True)
            self._emit("error", message=f"등록 실패: {e}")
            self._record_analytics(
                source="direct", language="n/a", understood=True, success=False,
                actions=["register_face"], error=str(e),
            )
        finally:
            self._busy = False

    # ── SLAM waypoint navigation -- see slam_navigator.py's docstring ──────
    def save_waypoint(self, name: str):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        threading.Thread(target=self._save_waypoint_sequence, args=(name,), daemon=True).start()

    def _save_waypoint_sequence(self, name: str):
        # Lidar is only switched on for the few seconds it takes to get a
        # fresh pose, then off again -- see PoseTracker.enable_lidar()'s
        # docstring (leaving it on floods the shared WebRTC channel and
        # breaks telemetry/Move() elsewhere on the dashboard).
        lidar_enabled = False
        try:
            name = (name or "").strip()
            if not name:
                raise RuntimeError("위치 이름을 입력하세요")
            if _pose_tracker is None:
                raise RuntimeError("SLAM 위치 정보가 없습니다 (lidar 연결을 확인하세요)")
            self._emit("status", step="locate", message="📡 위치 확인 중...")
            _pose_tracker.enable_lidar()
            lidar_enabled = True
            if not _pose_tracker.wait_for_fresh(timeout=5.0):
                raise RuntimeError("SLAM 위치 정보가 없습니다 (lidar 연결을 확인하세요)")
            pose = _pose_tracker.get()
            _waypoints.save(name, pose["x"], pose["y"], pose["yaw"])
            self._emit("status", step="done", message=f"📍 현재 위치를 '{name}'(으)로 저장했습니다")
            self._record_analytics(
                source="direct", language="n/a", understood=True, success=True,
                actions=["save_waypoint"],
            )
        except Exception as e:
            logger.error(f"Waypoint save error: {e}", exc_info=True)
            self._emit("error", message=f"위치 저장 실패: {e}")
            self._record_analytics(
                source="direct", language="n/a", understood=True, success=False,
                actions=["save_waypoint"], error=str(e),
            )
        finally:
            if lidar_enabled:
                _pose_tracker.disable_lidar()
            self._busy = False

    def navigate_to(self, location: str, language: str = "ko", source: str = "direct"):
        if self._busy:
            self._emit("error", message="Pipeline đang bận, thử lại sau.")
            return
        self._busy = True
        self._cancel_evt.clear()
        threading.Thread(
            target=self._navigate_sequence, args=(location, language, source), daemon=True
        ).start()

    def _navigate_sequence(self, location: str, language: str, source: str):
        t0 = time.monotonic()
        try:
            if _navigator is None:
                raise RuntimeError("내비게이션 모듈을 사용할 수 없습니다")
            self._emit("status", step="navigate", message=f"🧭 '{location}'(으)로 이동 중...")
            arrived = _navigator.navigate_to(location)
            exec_ms = (time.monotonic() - t0) * 1000
            if arrived:
                self._emit("status", step="done", message=f"✅ '{location}' 도착")
            else:
                self._emit("error", message=f"❌ 이동 실패: {_navigator.last_error}")
            self._record_analytics(
                source=source, language=language, understood=True, success=arrived,
                actions=["navigate_to"], exec_ms=exec_ms, error=None if arrived else _navigator.last_error,
            )
        except Exception as e:
            logger.error(f"Navigate error: {e}", exc_info=True)
            self._emit("error", message=f"이동 실패: {e}")
            self._record_analytics(
                source=source, language=language, understood=True, success=False,
                actions=["navigate_to"], error=str(e),
            )
        finally:
            self._busy = False


# ─── Dashboard HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GO2 AI 통합 관제 대시보드</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0f1a;--panel:#111827;--card:#161f30;--card2:#1b2740;--border:#26324a;
  --accent:#3b82f6;--accent2:#1d4ed8;
  --green:#22c55e;--red:#ef4444;--yellow:#eab308;--blue:#60a5fa;
  --text:#e7ecf5;--dim:#8a95ab;
}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size:14px;overflow:hidden}
button,input{font-family:inherit}

.shell{display:grid;grid-template-columns:212px 1fr;height:100vh}

/* Sidebar */
.sidebar{background:var(--panel);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden}
.brand{padding:16px 16px;font-weight:700;font-size:14.5px;display:flex;gap:9px;
  align-items:center;border-bottom:1px solid var(--border)}
.brand .sub{display:block;font-size:10px;font-weight:500;color:var(--dim);margin-top:1px}
.nav{flex:1;padding:10px 8px;display:flex;flex-direction:column;gap:2px;overflow-y:auto}
.nav-item{padding:10px 12px;border-radius:8px;color:var(--dim);cursor:pointer;
  display:flex;gap:10px;align-items:center;font-size:13px;user-select:none;
  transition:background .12s,color .12s}
.nav-item:hover{background:#1a2438;color:var(--text)}
.nav-item.active{background:var(--accent2);color:#fff}
.sidebar-foot{padding:10px;border-top:1px solid var(--border)}
.sidebar-foot button{width:100%;background:none;border:1px solid var(--border);color:var(--dim);
  border-radius:6px;padding:8px;cursor:pointer;font-size:12px}
.sidebar-foot button:hover{color:var(--text);border-color:var(--accent)}

/* Content */
.content{display:grid;grid-template-rows:56px 1fr;overflow:hidden}

.topbar{background:var(--panel);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;padding:0 18px;gap:14px}
.tb-left{display:flex;align-items:center;gap:16px;font-size:13px;color:var(--dim);min-width:0}
.tb-left b{color:var(--text);font-size:14.5px;font-weight:700}
.badge{display:flex;align-items:center;gap:6px;white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:var(--red);flex-shrink:0}
.dot.on{background:var(--green);box-shadow:0 0 6px var(--green)}
.tb-right{display:flex;align-items:center;gap:16px}
#clock{font-size:12.5px;color:var(--dim);min-width:56px}
.estop-top{background:var(--red);color:#fff;border:none;padding:8px 14px;border-radius:6px;
  font-weight:700;font-size:12px;cursor:pointer}
.estop-top:hover{filter:brightness(1.12)}
.intro-top{background:var(--accent2);color:#fff;border:none;padding:8px 14px;border-radius:6px;
  font-weight:700;font-size:12px;cursor:pointer}
.intro-top:hover{filter:brightness(1.15)}

.body{display:grid;grid-template-columns:1fr 360px;overflow:hidden}
.left{overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px}
.right{background:var(--panel);border-left:1px solid var(--border);
  overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px}

.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;align-items:center;
  padding:10px 14px;border-bottom:1px solid var(--border);
  font-size:12px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}

/* Camera */
.cam-body{position:relative;background:#000;aspect-ratio:16/9}
#cam-img,#trk-img{width:100%;height:100%;object-fit:contain;display:block}
.cam-badge{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.7);
  border:1px solid rgba(255,255,255,.12);border-radius:4px;padding:3px 10px;
  font-size:11px;font-weight:600;color:var(--green)}
.cam-badge.off{color:var(--red)}

/* Status cards */
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.card-head{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim);margin-bottom:9px}
.card-val{font-size:21px;font-weight:700;line-height:1}
.card-val.g{color:var(--green)}.card-val.y{color:var(--yellow)}.card-val.r{color:var(--red)}
.card-sub{font-size:10.5px;color:var(--dim);margin-top:6px}

/* Alerts */
#alerts{max-height:230px;overflow-y:auto}
.al-row{display:flex;gap:9px;align-items:flex-start;padding:9px 14px;
  border-bottom:1px solid rgba(38,50,74,.5);font-size:12px}
.al-t{color:var(--dim);font-size:10.5px;min-width:52px;padding-top:1px}
.al-tag{font-size:9.5px;font-weight:700;padding:2px 6px;border-radius:4px;flex-shrink:0}
.al-tag.in{background:rgba(96,165,250,.15);color:var(--blue)}
.al-tag.ok{background:rgba(34,197,94,.15);color:var(--green)}
.al-tag.wa{background:rgba(234,179,8,.15);color:var(--yellow)}
.al-tag.er{background:rgba(239,68,68,.15);color:var(--red)}
.al-m{flex:1;color:var(--text)}

/* AI panel */
.mic-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
#mic{width:44px;height:44px;border-radius:50%;background:var(--card2);
  border:2px solid var(--border);color:var(--text);font-size:18px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  transition:all .15s;user-select:none}
#mic:hover{border-color:var(--accent)}
#mic.rec{background:#3d0808;border-color:var(--red);
  box-shadow:0 0 0 3px rgba(239,68,68,.25);animation:mp 1.2s ease-in-out infinite}
@keyframes mp{0%,100%{box-shadow:0 0 0 3px rgba(239,68,68,.25)}
  50%{box-shadow:0 0 0 7px rgba(239,68,68,.08)}}
.mic-hint{font-size:11px;color:var(--dim)}
.ww-row{display:flex;align-items:center;justify-content:space-between;gap:10px;
  background:var(--card2);border:1px solid var(--border);border-radius:8px;
  padding:9px 12px;margin-bottom:10px;font-size:11.5px}
.ww-label{color:var(--dim)}
.ww-toggle{width:38px;height:20px;border-radius:10px;background:var(--border);
  border:none;cursor:pointer;position:relative;flex-shrink:0}
.ww-toggle::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;
  border-radius:50%;background:#fff;transition:transform .15s}
.ww-toggle.on{background:var(--green)}
.ww-toggle.on::after{transform:translateX(18px)}
.ww-toggle:disabled{opacity:.4;cursor:not-allowed}
.lang-row{display:flex;gap:8px;margin-bottom:10px}
.lang-btn{flex:1;background:var(--card2);border:1px solid var(--border);color:var(--dim);
  border-radius:6px;padding:7px 4px;font-size:11.5px;cursor:pointer;text-align:center}
.lang-btn.active{background:var(--accent2);border-color:var(--accent);color:#fff;font-weight:700}
.txt-row{display:flex;gap:8px;margin-bottom:12px}
#txtcmd{flex:1;background:var(--card2);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:9px 10px;font-size:12.5px}
#txtcmd:focus{outline:none;border-color:var(--accent)}
#txtsend{background:var(--accent2);border:none;color:#fff;border-radius:6px;
  padding:0 14px;font-size:12px;font-weight:600;cursor:pointer}
.ai-box{background:var(--card2);border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;font-size:12px;line-height:1.7;min-height:64px;color:var(--dim);
  white-space:pre-wrap}
.ai-actions{display:flex;gap:8px;margin-top:10px}
.ai-actions button{flex:1;border:none;border-radius:6px;padding:9px;font-size:12.5px;
  font-weight:700;cursor:pointer}
#btnExec{background:var(--accent2);color:#fff}
#btnExec:disabled{background:#20304f;color:#5a6a85;cursor:not-allowed}
#btnCancel{background:var(--card2);color:var(--dim);border:1px solid var(--border)}
#pst{font-size:11.5px;color:var(--dim);margin-top:8px;text-align:right}
#pst.busy{color:var(--yellow)}#pst.done{color:var(--green)}#pst.err{color:var(--red)}

/* Manual control */
.dpad{display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(2,1fr);
  gap:8px;margin-bottom:12px}
.dbtn{background:var(--card2);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:14px 0;font-size:17px;cursor:pointer;user-select:none;
  display:flex;align-items:center;justify-content:center;transition:background .1s}
.dbtn:hover{border-color:var(--accent)}
.dbtn:active,.dbtn.active{background:var(--accent2);border-color:var(--accent)}
.dbtn.stop{color:var(--red);font-weight:700;font-size:12px}
.posture-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px}
.pbtn{background:var(--card2);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:10px 4px;font-size:11px;cursor:pointer;text-align:center}
.pbtn:hover{border-color:var(--accent)}
.pbtn i{display:block;font-style:normal;font-size:17px;margin-bottom:3px}
.speed-row{display:flex;align-items:center;gap:10px;margin-bottom:12px;font-size:12px;color:var(--dim)}
.speed-row button{width:30px;height:30px;border-radius:6px;background:var(--card2);
  border:1px solid var(--border);color:var(--text);cursor:pointer;font-size:15px}
#speedVal{flex:1;text-align:center;color:var(--text);font-weight:700}
.estop-big{width:100%;background:var(--red);color:#fff;border:none;border-radius:8px;
  padding:13px;font-weight:700;font-size:13px;cursor:pointer}
.estop-big:hover{filter:brightness(1.1)}

/* Trick grid (로봇 제어 view) */
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.abtn{background:var(--card2);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:12px 4px 8px;cursor:pointer;font-size:11px;
  text-align:center;transition:background .1s,border-color .1s;user-select:none}
.abtn:hover{border-color:var(--accent)}
.abtn:active{background:var(--accent2)}
.abtn i{display:block;font-style:normal;font-size:19px;margin-bottom:4px}

.view{display:none;flex-direction:column;gap:14px}
.view.active{display:flex}
.hint{font-size:11.5px;color:var(--dim);line-height:1.6;padding:2px 2px 0}

/* Analytics (운행 기록) -- categorical slots follow the dataviz skill's
   validated dark-mode order (blue/orange/aqua), fixed per language, never
   reassigned/cycled. Sequential (single-series ranked) bars use one hue. */
.viz-root{
  --series-1:#3987e5;   /* categorical slot 1 -- ko */
  --series-2:#d95926;   /* categorical slot 2 -- vi */
  --series-3:#199e70;   /* categorical slot 3 -- ko-jeju */
  --seq-hue:#3987e5;    /* sequential (action-frequency ranking) -- one hue */
}
.bar-row{display:flex;align-items:center;gap:9px;margin-bottom:7px;font-size:11.5px}
.bar-label{width:92px;flex-shrink:0;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{flex:1;height:10px;background:var(--card2);border-radius:5px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;min-width:2px}
.bar-val{width:34px;flex-shrink:0;text-align:right;color:var(--text);font-variant-numeric:tabular-nums}
.legend-row{display:flex;gap:16px;margin-bottom:12px;font-size:11px;color:var(--dim);flex-wrap:wrap}
.legend-dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
.rt-table{width:100%;border-collapse:collapse;font-size:11px}
.rt-table th{text-align:left;padding:6px 8px;color:var(--dim);border-bottom:1px solid var(--border);font-weight:600}
.rt-table td{padding:6px 8px;border-bottom:1px solid rgba(38,50,74,.5);color:var(--text)}
.rt-ok{color:var(--green)}.rt-er{color:var(--red)}
.empty-note{color:var(--dim);font-size:12px;text-align:center;padding:24px 0}

::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<div class="shell">

<aside class="sidebar">
  <div class="brand">🤖<div>GO2 AI 통합 관제<span class="sub">대시보드</span></div></div>
  <nav class="nav">
    <div class="nav-item active" data-tab="dashboard">🏠&nbsp; 대시보드</div>
    <div class="nav-item" data-tab="control">🎮&nbsp; 로봇 제어</div>
    <div class="nav-item" data-tab="patrol">🛰️&nbsp; 자율 정찰</div>
    <div class="nav-item" data-tab="tracking">👁️&nbsp; 사람 추적</div>
    <div class="nav-item" data-tab="analytics">📋&nbsp; 운행 기록</div>
    <div class="nav-item" data-tab="soon">🛰️&nbsp; 장치 상태</div>
    <div class="nav-item" data-tab="soon">⚙️&nbsp; 시스템 설정</div>
  </nav>
  <div class="sidebar-foot"><button id="collapseBtn">≪ 접기</button></div>
</aside>

<div class="content">
  <header class="topbar">
    <div class="tb-left">
      <b id="robotName">GO2-01</b>
      <div class="badge"><div class="dot" id="rdot"></div><span id="rip">Robot</span></div>
      <div class="badge"><div class="dot on"></div><span>서버 정상</span></div>
    </div>
    <div class="tb-right">
      <span id="clock">--:--:--</span>
      <button class="intro-top" onclick="selfIntro()">🎬 자기소개 데모</button>
      <button class="estop-top" onclick="emergencyStop()">⏻ 긴급 정지</button>
    </div>
  </header>

  <div class="body">
    <div class="left">

      <!-- Dashboard view -->
      <div class="view active" id="view-dashboard">
        <div class="panel">
          <div class="panel-head"><span>실시간 카메라</span><span id="cbadge" style="text-transform:none;font-weight:600">연결 중...</span></div>
          <div class="cam-body">
            <img id="cam-img" src="/video_feed" alt="camera">
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">얼굴 등록</div>
          <div style="padding:14px">
            <div class="hint" style="margin-bottom:10px">카메라를 정면으로 보고 정보를 입력한 뒤 등록하면, 다음부터 화면 분석·순찰 인사에서 이름으로 인식됩니다.</div>
            <div class="txt-row"><input id="regName" placeholder="이름"></div>
            <div class="txt-row"><input id="regTitle" placeholder="직함 (예: 교수님)"></div>
            <div class="txt-row"><input id="regIntro" placeholder="소개 (AI가 소개할 때 참고할 정보)"></div>
            <button class="pbtn" style="width:100%;padding:10px" onclick="registerFace()">📸 현재 화면으로 등록</button>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head"><span>위치 이동 (SLAM)</span><span id="poseBadge" style="text-transform:none;font-weight:600">—</span></div>
          <div style="padding:14px">
            <div class="hint" id="poseReadout" style="margin-bottom:10px">lidar 꺼짐 (정상) — 위치 저장/이동 시에만 잠깐 켜집니다</div>
            <div class="txt-row">
              <input id="wpName" placeholder="위치 이름 (예: 문 앞)" style="flex:1">
              <button class="pbtn" style="padding:9px 12px" onclick="saveWaypoint()">📍 저장</button>
            </div>
            <div id="waypointList" style="margin-top:10px"></div>

            <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--border)">
              <div class="hint" style="margin-bottom:8px">🧭 경로 (문처럼 직선으로 못 가는 곳은 저장된 지점을 순서대로 연결)</div>
              <div class="txt-row">
                <select id="routeLegPicker" style="flex:1"></select>
                <button class="pbtn" style="padding:9px 12px" onclick="addRouteLeg()">+ 추가</button>
              </div>
              <div id="routeLegs" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px"></div>
              <div class="txt-row" style="margin-top:8px">
                <input id="routeName" placeholder="경로 이름 (예: 연구실)" style="flex:1">
                <button class="pbtn" style="padding:9px 12px" onclick="saveRoute()">🧭 경로 저장</button>
              </div>
            </div>

            <div class="hint" id="navStatus" style="margin-top:10px"></div>
          </div>
        </div>

        <div class="cards">
          <div class="card">
            <div class="card-head">🔋 배터리</div>
            <div class="card-val" id="c-batt">—</div>
            <div class="card-sub" id="c-batt-sub">텔레메트리 연결 중...</div>
          </div>
          <div class="card">
            <div class="card-head">🌡️ 모터 온도</div>
            <div class="card-val" id="c-temp">—</div>
            <div class="card-sub" id="c-temp-sub">—</div>
          </div>
          <div class="card">
            <div class="card-head">📶 네트워크</div>
            <div class="card-val" id="c-net">—</div>
            <div class="card-sub" id="c-net-sub">—</div>
          </div>
          <div class="card">
            <div class="card-head">🚶 현재 상태</div>
            <div class="card-val" id="c-state" style="font-size:15px">대기 중</div>
            <div class="card-sub" id="c-state-sub">—</div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">최근 알림</div>
          <div id="alerts"></div>
        </div>
      </div>

      <!-- Robot control view (trick grid) -->
      <div class="view" id="view-control">
        <div class="panel">
          <div class="panel-head">동작 명령</div>
          <div style="padding:14px">
            <div class="grid">
              <button class="abtn" onclick="act('stand_up')"><i>🦴</i>일어서기</button>
              <button class="abtn" onclick="act('stand_down')"><i>💤</i>엎드리기</button>
              <button class="abtn" onclick="act('stop_move')"><i>🛑</i>정지</button>
              <button class="abtn" onclick="act('recovery')"><i>🔁</i>회복</button>
              <button class="abtn" onclick="act('hello')"><i>👋</i>인사</button>
              <button class="abtn" onclick="act('stretch')"><i>🤸</i>스트레칭</button>
              <button class="abtn" onclick="act('scrape')"><i>🐾</i>긁기</button>
              <button class="abtn" onclick="act('pose')"><i>🧍</i>포즈</button>
              <button class="abtn" onclick="act('front_jump')"><i>⬆️</i>점프</button>
              <button class="abtn" onclick="act('front_pounce')"><i>🐆</i>덮치기</button>
              <button class="abtn" onclick="act('cross_step')"><i>✂️</i>크로스스텝</button>
              <button class="abtn" onclick="act('trot_run')"><i>🐎</i>트롯런</button>
              <button class="abtn" onclick="act('walk_upright')"><i>🚶</i>직립보행</button>
              <button class="abtn" onclick="act('static_walk')"><i>🐢</i>천천히걷기</button>
              <button class="abtn" onclick="act('classic_walk')"><i>🚶‍♂️</i>기본걸음</button>
              <button class="abtn" onclick="act('sit')"><i>🐕</i>앉기</button>
              <button class="abtn" onclick="act('rise_sit')"><i>🦵</i>앉은데서일어나기</button>
              <button class="abtn" onclick="act('content')"><i>😊</i>애교</button>
              <button class="abtn" onclick="act('dance1')"><i>💃</i>댄스</button>
              <button class="abtn" onclick="act('hand_stand')"><i>🤾</i>물구나무</button>
              <button class="abtn" onclick="act('back_flip')"><i>🔄</i>백플립</button>
              <button class="abtn" onclick="act('left_flip')"><i>🔃</i>레프트플립</button>
            </div>
            <div class="hint" style="margin-top:12px">⚠️ 백플립/레프트플립/물구나무는 고난도 동작입니다 — 넓고 평평한 공간에서만 사용하세요.</div>
          </div>
        </div>
      </div>

      <!-- Autonomous Reconnaissance view -->
      <div class="view" id="view-patrol">
        <div class="panel">
          <div class="panel-head">자율 정찰 (Autonomous Reconnaissance)</div>
          <div style="padding:14px">
            <div class="hint" style="margin-bottom:12px">
              고정 왕복 경로로 이동하며 사람을 감지하면 멈추고 카메라 화면을 분석해 정찰 보고를 음성으로
              전달합니다 (SLAM/경로 계획 없는 단순 버전 — PATROL_ROUTE를 직접 수정해 실제 공간에 맞게
              조정하세요). 시작 전 충분히 넓고 장애물 없는 공간인지 확인하세요.
            </div>
            <div class="lang-row" style="margin-bottom:14px">
              <div class="lang-btn active" data-plang="ko" onclick="setPatrolLang('ko')">🇰🇷 한국어 보고</div>
              <div class="lang-btn" data-plang="vi" onclick="setPatrolLang('vi')">🇻🇳 Báo cáo Tiếng Việt</div>
            </div>
            <div class="ww-row" style="margin-bottom:14px">
              <span class="ww-label" id="patrolStatus">🛰️ 정찰 상태: 확인 중...</span>
            </div>
            <div class="posture-row" style="grid-template-columns:1fr 1fr">
              <button class="pbtn" onclick="patrolStart()"><i>▶️</i>정찰 시작</button>
              <button class="pbtn" onclick="patrolStop()"><i>⏹️</i>정찰 정지</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Person Tracking view -->
      <div class="view" id="view-tracking">
        <div class="panel">
          <div class="panel-head"><span>사람 추적 (카메라)</span><span id="trkbadge" style="text-transform:none;font-weight:600">꺼짐</span></div>
          <div class="cam-body">
            <img id="trk-img" src="" alt="tracking camera" style="display:none">
            <div id="trk-placeholder" style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--dim);font-size:13px">추적을 시작하면 영상이 표시됩니다</div>
          </div>
        </div>
        <div class="panel">
          <div style="padding:14px">
            <div class="hint" style="margin-bottom:12px">
              YOLO로 사람을 인식해 화면 중앙에 오도록 몸을 회전시킵니다 (전후 이동 없음, CPU 추론이라 다소
              느릴 수 있습니다). 음성/텍스트 명령이 실행되는 동안에는 잠시 회전을 멈춥니다. 시작 전 충분히
              넓고 장애물 없는 공간인지 확인하세요.
            </div>
            <div class="ww-row" style="margin-bottom:14px">
              <span class="ww-label" id="trackingStatus">👁️ 추적 상태: 확인 중...</span>
            </div>
            <div class="posture-row" style="grid-template-columns:1fr 1fr">
              <button class="pbtn" onclick="trackingStart()"><i>▶️</i>추적 시작</button>
              <button class="pbtn" onclick="trackingStop()"><i>⏹️</i>추적 정지</button>
            </div>
            <button class="pbtn" style="width:100%;padding:10px;margin-top:10px;background:var(--red);border-color:var(--red)" onclick="fireShot()">🔫 발사 시뮬레이션 (Space)</button>
            <div class="hint" style="margin-top:8px">이 탭이 열려 있을 때 Space를 누르면 총소리 효과음이 재생됩니다 (브라우저에서만 재생 — 로봇에는 아무 명령도 전송되지 않습니다).</div>
          </div>
        </div>
      </div>

      <!-- Analytics (운행 기록) view -->
      <div class="view viz-root" id="view-analytics">
        <div class="cards" id="kpiRow">
          <div class="card"><div class="card-head">📊 총 명령 수</div><div class="card-val" id="kpi-total">—</div><div class="card-sub">전체 누적</div></div>
          <div class="card"><div class="card-head">✅ 성공률</div><div class="card-val" id="kpi-success">—</div><div class="card-sub">success / total</div></div>
          <div class="card"><div class="card-head">🎤 평균 STT</div><div class="card-val" id="kpi-stt">—</div><div class="card-sub">음성 인식 지연시간</div></div>
          <div class="card"><div class="card-head">🧠 평균 LLM+실행</div><div class="card-val" id="kpi-llm">—</div><div class="card-sub" id="kpi-exec-sub">—</div></div>
        </div>

        <div class="panel">
          <div class="panel-head">언어별 명령 수</div>
          <div style="padding:14px" id="langChart"><div class="empty-note">데이터가 아직 없습니다.</div></div>
        </div>

        <div class="panel">
          <div class="panel-head">가장 많이 사용된 동작</div>
          <div style="padding:14px" id="actionChart"><div class="empty-note">데이터가 아직 없습니다.</div></div>
        </div>

        <div class="panel">
          <div class="panel-head">최근 명령 기록</div>
          <div style="padding:14px;overflow-x:auto" id="recentTable"><div class="empty-note">데이터가 아직 없습니다.</div></div>
        </div>
      </div>

      <!-- Placeholder view for unbuilt sections -->
      <div class="view" id="view-soon">
        <div class="panel">
          <div class="panel-head">준비 중</div>
          <div style="padding:30px 20px;text-align:center;color:var(--dim);font-size:13px">
            이 기능은 아직 구현되지 않았습니다.<br>현재는 대시보드 · 로봇 제어 · 자율 정찰만 동작합니다.
          </div>
        </div>
      </div>

    </div>

    <div class="right">
      <div class="panel">
        <div class="panel-head">AI 명령 패널</div>
        <div style="padding:14px">
          <div class="ww-row">
            <span class="ww-label" id="wwLabel">🎙️ 웨이크워드 "윤재야" (로봇 마이크) — 확인 중...</span>
            <button class="ww-toggle" id="wwToggle" onclick="toggleWakeWord()" disabled></button>
          </div>
          <div class="lang-row">
            <div class="lang-btn active" data-lang="ko" onclick="setSttLang('ko')">🇰🇷 한국어</div>
            <div class="lang-btn" data-lang="vi" onclick="setSttLang('vi')">🇻🇳 Tiếng Việt</div>
            <div class="lang-btn" data-lang="ko-jeju" onclick="setSttLang('ko-jeju')">🍊 제주어</div>
          </div>
          <div class="mic-row">
            <button id="mic" title="눌러서 녹음">🎤</button>
            <div class="mic-hint">누르고 있으면 녹음, 손을 떼면 전송<br>(음성은 분석 즉시 자동 실행)</div>
          </div>
          <div class="txt-row">
            <input id="txtcmd" placeholder="텍스트 명령을 입력하세요" onkeydown="if(event.key==='Enter')sendText()">
            <button id="txtsend" onclick="sendText()">전송</button>
          </div>
          <div class="txt-row">
            <button id="visionBtn" class="pbtn" style="width:100%;padding:10px" onclick="visionAnalyze()">📷 지금 뭐가 보이나요? (화면 분석)</button>
          </div>
          <div class="ai-box" id="aiBox">AI 분석 결과가 여기에 표시됩니다.</div>
          <div class="ai-actions">
            <button id="btnExec" onclick="confirmExec()" disabled>실행</button>
            <button id="btnCancel" onclick="cancelCmd()" disabled>취소</button>
          </div>
          <div id="pst">Sẵn sàng</div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">수동 제어 패널</div>
        <div style="padding:14px">
          <div class="dpad">
            <div></div>
            <div class="dbtn" data-dir="forward">↑</div>
            <div></div>
            <div class="dbtn" data-dir="turn_left">↺</div>
            <div class="dbtn stop" onclick="moveStop(true)">정지</div>
            <div class="dbtn" data-dir="turn_right">↻</div>
            <div></div>
            <div class="dbtn" data-dir="backward">↓</div>
            <div></div>
          </div>
          <div class="posture-row">
            <div class="pbtn" onclick="act('stand_down')"><i>💤</i>엎드리기</div>
            <div class="pbtn" onclick="act('sit')"><i>🐕</i>앉기</div>
            <div class="pbtn" onclick="act('stand_up')"><i>🧍</i>일어서기</div>
            <div class="pbtn" onclick="act('recovery')"><i>🔁</i>자세 복귀</div>
          </div>
          <div class="speed-row">
            <button onclick="speedAdj(-0.1)">−</button>
            <span id="speedVal">속도 0.3</span>
            <button onclick="speedAdj(0.1)">+</button>
          </div>
          <button class="estop-big" onclick="emergencyStop()">🛑 긴급 정지</button>
        </div>
      </div>
    </div>
  </div>
</div>

</div>
<script>
// ── Utilities ──────────────────────────────────────────────────────────────
function ts(){
  return new Date().toLocaleTimeString('ko-KR',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function alertRow(msg,tag='in'){
  const el=document.getElementById('alerts');
  const d=document.createElement('div');
  d.className='al-row';
  const tagLabel={in:'정보',ok:'성공',wa:'주의',er:'오류'}[tag]||'정보';
  d.innerHTML='<span class="al-t">'+ts()+'</span><span class="al-tag '+tag+'">'+tagLabel+'</span><span class="al-m"></span>';
  d.querySelector('.al-m').textContent=msg;
  el.prepend(d);
  while(el.children.length>60)el.removeChild(el.lastChild);
}
function setPST(txt,cls=''){
  const el=document.getElementById('pst');
  el.textContent=txt; el.className=cls;
}
setInterval(()=>{document.getElementById('clock').textContent=ts();},1000);

// ── Tabs ───────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item=>{
  item.addEventListener('click',()=>{
    document.querySelectorAll('.nav-item').forEach(i=>i.classList.remove('active'));
    item.classList.add('active');
    const tab=item.dataset.tab;
    document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
    if(tab==='dashboard')document.getElementById('view-dashboard').classList.add('active');
    else if(tab==='control')document.getElementById('view-control').classList.add('active');
    else if(tab==='patrol')document.getElementById('view-patrol').classList.add('active');
    else if(tab==='tracking')document.getElementById('view-tracking').classList.add('active');
    else if(tab==='analytics'){document.getElementById('view-analytics').classList.add('active');loadAnalytics();}
    else{document.getElementById('view-soon').classList.add('active');alertRow('이 메뉴는 아직 준비 중입니다.','wa');}
  });
});
document.getElementById('collapseBtn').addEventListener('click',()=>{
  document.querySelector('.shell').style.gridTemplateColumns=
    document.querySelector('.shell').style.gridTemplateColumns==='56px 1fr'?'212px 1fr':'56px 1fr';
});

// ── WebSocket ──────────────────────────────────────────────────────────────
let ws;
function initWS(){
  ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onopen=()=>alertRow('서버에 연결되었습니다','ok');
  ws.onclose=()=>{alertRow('연결이 끊어졌습니다 — 재연결 중...','er');setTimeout(initWS,2000)};
  ws.onmessage=(e)=>onMsg(JSON.parse(e.data));
}
let lastAction='';
function toneLabel(t){
  return {urgent:'⚡ 급함',tired:'😮‍💨 피곤함/배려',calm:'😌 차분함',happy:'😄 기쁨',neutral:'🙂 보통'}[t]||'';
}
function onMsg(d){
  switch(d.type){
    case 'stt_result':
      alertRow('🎤 인식: '+d.text,'in'); setPST('분석 중...','busy'); break;
    case 'llm_result':{
      const tl=d.tone&&d.tone!=='neutral'?' ['+toneLabel(d.tone)+']':'';
      alertRow('🧠 '+(d.response||'—')+tl,'in'); break;}
    case 'analysis_ready':{
      const box=document.getElementById('aiBox');
      if(d.ready){
        const lines=(d.actions||[]).map(a=>{
          const p=Object.entries(a.params||{}).map(([k,v])=>k+'='+(typeof v==='number'?v.toFixed(2):v)).join(', ');
          return '· '+a.name+(p?' ('+p+')':'');
        }).join('\\n');
        const toneLine=d.tone&&d.tone!=='neutral'?('톤: '+toneLabel(d.tone)+'\\n'):'';
        box.textContent=toneLine+'응답: '+(d.response||'-')+'\\n\\n동작:\\n'+(lines||'(없음)');
        document.getElementById('btnExec').disabled=false;
        document.getElementById('btnCancel').disabled=false;
      }else{
        box.textContent=d.response?('이해하지 못했습니다: '+d.response):'AI 분석 결과가 여기에 표시됩니다.';
        document.getElementById('btnExec').disabled=true;
        document.getElementById('btnCancel').disabled=true;
      }
      break;}
    case 'action':{
      const p=Object.entries(d.params||{}).map(([k,v])=>k+'='+(typeof v==='number'?v.toFixed(2):v)).join(' ');
      lastAction=d.name+(p?' ('+p+')':'');
      document.getElementById('c-state').textContent=d.name;
      document.getElementById('c-state-sub').textContent=lastAction;
      alertRow('🤖 '+lastAction,'in'); break;}
    case 'status':
      alertRow(d.message,'in');
      if(d.step==='done'){
        setPST('완료','done');document.getElementById('c-state').textContent='대기 중';
        setTimeout(()=>setPST('Sẵn sàng'),3000);
      }
      break;
    case 'speed':
      document.getElementById('speedVal').textContent='속도 '+d.value.toFixed(1); break;
    case 'error':
      alertRow(d.message,'er'); setPST('오류','err'); setTimeout(()=>setPST('Sẵn sàng'),4000); break;
  }
}
initWS();

// ── STT language selector ────────────────────────────────────────────────
let sttLang='ko';
function setSttLang(lang){
  sttLang=lang;
  document.querySelectorAll('.lang-btn').forEach(b=>b.classList.toggle('active',b.dataset.lang===lang));
}

// ── Microphone (auto-executes, same as before) ──────────────────────────────
let mr=null,chunks=[],recActive=false;
const micBtn=document.getElementById('mic');
async function startRec(){
  if(recActive)return;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    chunks=[];
    const mime=MediaRecorder.isTypeSupported('audio/webm')?'audio/webm':
               MediaRecorder.isTypeSupported('audio/ogg')?'audio/ogg':'';
    mr=new MediaRecorder(stream,mime?{mimeType:mime}:{});
    mr.ondataavailable=(e)=>{if(e.data.size)chunks.push(e.data);};
    mr.start(100);
    recActive=true;
    micBtn.classList.add('rec');micBtn.textContent='⏹';
    setPST('🔴 녹음 중...','busy');
  }catch(e){alertRow('마이크 오류: '+e.message,'er');}
}
async function stopRec(){
  if(!recActive||!mr)return;
  recActive=false;
  await new Promise(res=>{mr.onstop=res;mr.stop();mr.stream.getTracks().forEach(t=>t.stop());});
  micBtn.classList.remove('rec');micBtn.textContent='🎤';
  const blob=new Blob(chunks,{type:mr.mimeType||'audio/webm'});
  if(blob.size<500){alertRow('녹음이 너무 짧습니다','wa');setPST('Sẵn sàng');return;}
  setPST('처리 중...','busy');
  const fd=new FormData();
  const ext=mr.mimeType&&mr.mimeType.includes('ogg')?'.ogg':'.webm';
  fd.append('audio',blob,'cmd'+ext);
  fd.append('language',sttLang);
  try{await fetch('/voice_command',{method:'POST',body:fd});}
  catch(e){alertRow('전송 실패: '+e.message,'er');setPST('오류','err');}
}
let holdMode=false,holdTimer=null;
micBtn.addEventListener('mousedown',e=>{e.preventDefault();holdTimer=setTimeout(()=>{holdMode=true;startRec();},150);});
micBtn.addEventListener('mouseup',async e=>{
  e.preventDefault();clearTimeout(holdTimer);
  if(holdMode){holdMode=false;await stopRec();}
  else{if(!recActive)startRec();else await stopRec();}
});
micBtn.addEventListener('mouseleave',async()=>{if(holdMode){holdMode=false;await stopRec();}});
micBtn.addEventListener('touchstart',e=>{e.preventDefault();startRec();},{passive:false});
micBtn.addEventListener('touchend',async e=>{e.preventDefault();await stopRec();},{passive:false});

// ── Text command: analyze -> confirm/cancel ─────────────────────────────────
async function sendText(){
  const el=document.getElementById('txtcmd');
  const text=el.value.trim();
  if(!text)return;
  alertRow('⌨️ '+text,'in');
  setPST('분석 중...','busy');
  el.value='';
  try{await fetch('/text_command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});}
  catch(e){alertRow('전송 실패: '+e.message,'er');}
}
async function confirmExec(){
  document.getElementById('btnExec').disabled=true;
  document.getElementById('btnCancel').disabled=true;
  try{await fetch('/confirm_execute',{method:'POST'});}catch(e){alertRow('실행 실패: '+e.message,'er');}
}
async function cancelCmd(){
  document.getElementById('btnExec').disabled=true;
  document.getElementById('btnCancel').disabled=true;
  try{await fetch('/cancel_command',{method:'POST'});}catch(e){}
}

// ── Trick buttons (instant, no confirm) ─────────────────────────────────────
async function act(name){
  alertRow('🎮 '+name,'in');
  try{await fetch('/action/'+name,{method:'POST'});}
  catch(e){alertRow(e.message,'er');}
}

// ── Manual D-pad (press-and-hold continuous move) ───────────────────────────
document.querySelectorAll('.dbtn[data-dir]').forEach(btn=>{
  const dir=btn.dataset.dir;
  const start=e=>{e.preventDefault();btn.classList.add('active');fetch('/move/'+dir,{method:'POST'}).catch(()=>{});};
  const stop=e=>{e.preventDefault();btn.classList.remove('active');moveStop(false);};
  btn.addEventListener('mousedown',start);
  btn.addEventListener('mouseup',stop);
  btn.addEventListener('mouseleave',stop);
  btn.addEventListener('touchstart',start,{passive:false});
  btn.addEventListener('touchend',stop,{passive:false});
});

// ── WASD keyboard shortcuts for the D-pad (buttons above still work too) ──
const WASD_DIR={w:'forward',s:'backward',a:'turn_left',d:'turn_right'};
const wasdHeld=new Set();
function wasdTarget(e){
  const t=e.target;
  const typing=t && (t.tagName==='INPUT' || t.tagName==='TEXTAREA' || t.isContentEditable);
  if(typing)return null;
  const dir=WASD_DIR[e.key && e.key.toLowerCase()];
  return dir||null;
}
document.addEventListener('keydown',e=>{
  const dir=wasdTarget(e);
  if(!dir || wasdHeld.has(dir))return;  // ignore focus-in-input and OS key-repeat
  wasdHeld.add(dir);
  const btn=document.querySelector('.dbtn[data-dir="'+dir+'"]');
  if(btn)btn.classList.add('active');
  fetch('/move/'+dir,{method:'POST'}).catch(()=>{});
});
document.addEventListener('keyup',e=>{
  const dir=WASD_DIR[e.key && e.key.toLowerCase()];
  if(!dir || !wasdHeld.has(dir))return;
  wasdHeld.delete(dir);
  const btn=document.querySelector('.dbtn[data-dir="'+dir+'"]');
  if(btn)btn.classList.remove('active');
  if(wasdHeld.size>0){
    // Another WASD key is still held (e.g. briefly overlapped switching
    // W->S) -- keep driving that direction instead of stopping outright.
    const remaining=wasdHeld.values().next().value;
    const rbtn=document.querySelector('.dbtn[data-dir="'+remaining+'"]');
    if(rbtn)rbtn.classList.add('active');
    fetch('/move/'+remaining,{method:'POST'}).catch(()=>{});
  }else{
    moveStop(false);
  }
});
// Releasing focus (alt-tab, clicking elsewhere) without a keyup firing would
// otherwise leave the robot driving indefinitely -- stop on blur too.
window.addEventListener('blur',()=>{
  if(wasdHeld.size===0)return;
  wasdHeld.forEach(dir=>{
    const btn=document.querySelector('.dbtn[data-dir="'+dir+'"]');
    if(btn)btn.classList.remove('active');
  });
  wasdHeld.clear();
  moveStop(false);
});
async function moveStop(log_it){
  try{await fetch('/move_stop',{method:'POST'});}catch(e){}
  if(log_it)alertRow('⏹ 정지','in');
}
async function speedAdj(delta){
  try{await fetch('/speed/'+delta,{method:'POST'});}catch(e){}
}
async function emergencyStop(){
  alertRow('🛑 긴급 정지 요청','wa');
  try{await fetch('/emergency_stop',{method:'POST'});}catch(e){alertRow(e.message,'er');}
}
async function selfIntro(){
  alertRow('🎬 자기소개 데모 시작','ok');
  try{await fetch('/self_intro',{method:'POST'});}catch(e){alertRow(e.message,'er');}
}

// ── Vision analysis ("what do you see") ──────────────────────────────────
async function visionAnalyze(){
  alertRow('📷 화면 분석 요청됨','ok');
  try{
    await fetch('/vision_analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({language:sttLang}),
    });
  }catch(e){alertRow(e.message,'er');}
}

// ── Face registration ────────────────────────────────────────────────────
async function registerFace(){
  const name=document.getElementById('regName').value.trim();
  if(!name){alertRow('이름을 입력하세요','er');return;}
  const title=document.getElementById('regTitle').value.trim();
  const introduction=document.getElementById('regIntro').value.trim();
  alertRow('📸 '+name+' 등록 요청됨','ok');
  try{
    await fetch('/vision/register',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,title,introduction}),
    });
  }catch(e){alertRow(e.message,'er');}
}

// ── SLAM waypoint navigation ─────────────────────────────────────────────
async function saveWaypoint(){
  const name=document.getElementById('wpName').value.trim();
  if(!name){alertRow('위치 이름을 입력하세요','er');return;}
  alertRow('📍 '+name+' 저장 요청됨','ok');
  try{
    await fetch('/waypoint/save',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name}),
    });
    document.getElementById('wpName').value='';
    loadWaypoints();
  }catch(e){alertRow(e.message,'er');}
}
async function goToWaypoint(name){
  alertRow(`🧭 ${name}(으)로 이동 요청됨`,'in');
  try{
    await fetch('/navigate_to',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({location:name}),
    });
  }catch(e){alertRow(e.message,'er');}
}
async function deleteWaypoint(name){
  try{
    await fetch('/waypoint/delete',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name}),
    });
    loadWaypoints();
  }catch(e){alertRow(e.message,'er');}
}
// ── SLAM routes (ordered chains of existing point waypoints) ────────────
// See slam_navigator.py's WaypointStore/SlamNavigator -- navigate_to() only
// drives a straight line to a single target, so a route through a doorway
// needs a human to pick the intermediate points (e.g. "door" then
// "room_center") rather than the robot planning one itself.
let routeLegBuilder=[];
function renderRouteLegs(){
  const el=document.getElementById('routeLegs');
  el.innerHTML='';
  routeLegBuilder.forEach((n,i)=>{
    const chip=document.createElement('span');
    chip.style.cssText='display:inline-flex;align-items:center;gap:5px;background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:4px 8px;font-size:12px';
    const label=document.createElement('span');
    label.textContent=(i+1)+'. '+n;
    const x=document.createElement('span');
    x.textContent='✕';
    x.style.cssText='cursor:pointer;opacity:.7';
    x.addEventListener('click',()=>{routeLegBuilder.splice(i,1);renderRouteLegs();});
    chip.appendChild(label);chip.appendChild(x);
    el.appendChild(chip);
  });
}
function addRouteLeg(){
  const sel=document.getElementById('routeLegPicker');
  if(sel && sel.value){routeLegBuilder.push(sel.value);renderRouteLegs();}
}
async function saveRoute(){
  const name=document.getElementById('routeName').value.trim();
  if(!name){alertRow('경로 이름을 입력하세요','er');return;}
  if(routeLegBuilder.length===0){alertRow('경로에 지점을 1개 이상 추가하세요','er');return;}
  try{
    const r=await(await fetch('/route/save',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,legs:routeLegBuilder}),
    })).json();
    if(r.status!=='ok'){alertRow(r.message||'경로 저장 실패','er');return;}
    alertRow(`🧭 경로 '${name}' 저장됨`,'ok');
    document.getElementById('routeName').value='';
    routeLegBuilder=[];
    renderRouteLegs();
    waypointNames=null;
    loadWaypoints();
  }catch(e){alertRow(e.message,'er');}
}

let waypointNames=null;
async function loadWaypoints(){
  try{
    const wps=await(await fetch('/waypoints')).json();
    const names=Object.keys(wps).sort();
    const pointNames=names.filter(n=>(wps[n].type||'point')==='point');

    const picker=document.getElementById('routeLegPicker');
    if(picker){
      const pickerNames=pointNames.join(',');
      if(picker.dataset.names!==pickerNames){
        picker.dataset.names=pickerNames;
        picker.innerHTML='';
        pointNames.forEach(n=>{
          const opt=document.createElement('option');
          opt.value=n;opt.textContent=n;
          picker.appendChild(opt);
        });
      }
    }

    if(waypointNames!==null && JSON.stringify(names)===JSON.stringify(waypointNames))return;
    waypointNames=names;
    const el=document.getElementById('waypointList');
    if(names.length===0){el.innerHTML='<div class="hint">저장된 위치가 없습니다</div>';return;}
    // Built via DOM nodes + data-name (not inline onclick with the name
    // embedded in a string) so waypoint names never need JS-string escaping
    // at all -- avoids a whole class of quoting bugs.
    el.innerHTML='';
    names.forEach(n=>{
      const isRoute=(wps[n].type||'point')==='route';
      const row=document.createElement('div');
      row.style.cssText='display:flex;gap:6px;align-items:center;margin-bottom:6px';
      const span=document.createElement('span');
      span.style.cssText='flex:1;font-size:12.5px';
      span.textContent=(isRoute?'🧭 ':'📍 ')+n+(isRoute?' ('+wps[n].legs.length+'단계)':'');
      const goBtn=document.createElement('button');
      goBtn.className='pbtn';
      goBtn.style.cssText='padding:6px 10px';
      goBtn.textContent='이동';
      goBtn.dataset.name=n;
      goBtn.addEventListener('click',()=>goToWaypoint(goBtn.dataset.name));
      const delBtn=document.createElement('button');
      delBtn.className='pbtn';
      delBtn.style.cssText='padding:6px 10px';
      delBtn.textContent='✕';
      delBtn.dataset.name=n;
      delBtn.addEventListener('click',()=>deleteWaypoint(delBtn.dataset.name));
      row.appendChild(span);row.appendChild(goBtn);row.appendChild(delBtn);
      el.appendChild(row);
    });
  }catch(e){}
}
loadWaypoints();

// ── Wake word toggle ─────────────────────────────────────────────────────
let wwEnabled=false;
async function toggleWakeWord(){
  const next=!wwEnabled;
  try{
    const r=await(await fetch('/wake_word/'+(next?'on':'off'),{method:'POST'})).json();
    wwEnabled=!!r.enabled;
    document.getElementById('wwToggle').classList.toggle('on',wwEnabled);
    alertRow(wwEnabled?'🎙️ 웨이크워드 켜짐 — "윤재야"라고 부르면 듣기 시작합니다':'🎙️ 웨이크워드 꺼짐','in');
  }catch(e){alertRow('웨이크워드 전환 실패: '+e.message,'er');}
}

// ── Auto Patrol ──────────────────────────────────────────────────────────
let patrolLang='ko';
let trkRunning=false;
function setPatrolLang(lang){
  patrolLang=lang;
  document.querySelectorAll('.lang-btn[data-plang]').forEach(b=>b.classList.toggle('active',b.dataset.plang===lang));
}
async function patrolStart(){
  try{
    const r=await(await fetch('/patrol/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({language:patrolLang})})).json();
    if(r.status==='unavailable'){alertRow('정찰 기능을 사용할 수 없습니다 (카메라 미연결)','er');return;}
    alertRow('🛰️ 정찰 시작','ok');
  }catch(e){alertRow('정찰 시작 실패: '+e.message,'er');}
}
async function patrolStop(){
  try{await fetch('/patrol/stop',{method:'POST'});alertRow('🛰️ 정찰 정지 요청','in');}
  catch(e){alertRow('정찰 정지 실패: '+e.message,'er');}
}
function patrolStateLabel(s){
  return {idle:'대기 중',patrolling:'정찰 중 🚶',reporting:'보고 중 🗣️'}[s]||s;
}

// ── Person Tracking ──────────────────────────────────────────────────────
async function trackingStart(){
  try{
    const r=await(await fetch('/tracking/start',{method:'POST'})).json();
    if(r.status==='unavailable'){alertRow('사람 추적 기능을 사용할 수 없습니다','er');return;}
    alertRow('👁️ 사람 추적 시작 요청 (YOLO 모델 로딩 중일 수 있습니다)','ok');
  }catch(e){alertRow('추적 시작 실패: '+e.message,'er');}
}
async function trackingStop(){
  try{await fetch('/tracking/stop',{method:'POST'});alertRow('👁️ 사람 추적 정지 요청','in');}
  catch(e){alertRow('추적 정지 실패: '+e.message,'er');}
}

// ── Person Tracking: simulated gunshot effect (Space) ────────────────────
// Purely a local browser sound effect (Web Audio API, synthesized -- no
// audio file asset, no server round-trip, no robot command) to punctuate
// "attack" during a demo. Synthesized instead of an audio file so this
// stays a single self-contained dashboard file like everywhere else here.
let audioCtx=null;
function playGunshotSound(){
  try{
    if(!audioCtx)audioCtx=new (window.AudioContext||window.webkitAudioContext)();
    if(audioCtx.state==='suspended')audioCtx.resume();
    const now=audioCtx.currentTime;

    // Sharp noise burst ("crack") -- random noise through a highpass
    // filter with a fast exponential decay.
    const bufferSize=Math.floor(audioCtx.sampleRate*0.25);
    const buffer=audioCtx.createBuffer(1,bufferSize,audioCtx.sampleRate);
    const data=buffer.getChannelData(0);
    for(let i=0;i<bufferSize;i++)data[i]=Math.random()*2-1;
    const noise=audioCtx.createBufferSource();
    noise.buffer=buffer;
    const noiseFilter=audioCtx.createBiquadFilter();
    noiseFilter.type='highpass';
    noiseFilter.frequency.value=800;
    const noiseGain=audioCtx.createGain();
    noiseGain.gain.setValueAtTime(1,now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001,now+0.18);
    noise.connect(noiseFilter);noiseFilter.connect(noiseGain);noiseGain.connect(audioCtx.destination);
    noise.start(now);noise.stop(now+0.2);

    // Low thump underneath for body/impact.
    const osc=audioCtx.createOscillator();
    osc.type='sine';
    osc.frequency.setValueAtTime(150,now);
    osc.frequency.exponentialRampToValueAtTime(40,now+0.15);
    const oscGain=audioCtx.createGain();
    oscGain.gain.setValueAtTime(0.9,now);
    oscGain.gain.exponentialRampToValueAtTime(0.001,now+0.15);
    osc.connect(oscGain);oscGain.connect(audioCtx.destination);
    osc.start(now);osc.stop(now+0.16);
  }catch(e){}
}
function fireShot(){
  playGunshotSound();
  alertRow('🔫 발사 시뮬레이션 (효과음만 재생 — 로봇에는 명령이 전송되지 않습니다)','in');
}
document.addEventListener('keydown',(e)=>{
  if(e.code!=='Space')return;
  const activeTag=(document.activeElement&&document.activeElement.tagName)||'';
  if(activeTag==='INPUT'||activeTag==='TEXTAREA'||activeTag==='SELECT')return;  // don't hijack Space while typing
  const trackingView=document.getElementById('view-tracking');
  if(!trackingView||!trackingView.classList.contains('active'))return;  // only while the tracking tab is open
  e.preventDefault();  // Space normally scrolls the page
  fireShot();
});

// ── Analytics (운행 기록) ─────────────────────────────────────────────────
const LANG_COLORS={'ko':'var(--series-1)','vi':'var(--series-2)','ko-jeju':'var(--series-3)'};
const LANG_LABELS={'ko':'한국어','vi':'Tiếng Việt','ko-jeju':'제주어'};
function escapeHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}

async function loadAnalytics(){
  try{
    const s=await(await fetch('/analytics')).json();
    document.getElementById('kpi-total').textContent=s.total;
    document.getElementById('kpi-success').textContent=s.total?s.success_rate+'%':'—';
    document.getElementById('kpi-stt').textContent=s.avg_stt_ms?Math.round(s.avg_stt_ms)+'ms':'—';
    document.getElementById('kpi-llm').textContent=s.avg_llm_ms?Math.round(s.avg_llm_ms)+'ms':'—';
    document.getElementById('kpi-exec-sub').textContent=s.avg_exec_ms?'평균 실행 '+Math.round(s.avg_exec_ms)+'ms':'—';

    const langEl=document.getElementById('langChart');
    if(!s.by_language.length){
      langEl.innerHTML='<div class="empty-note">데이터가 아직 없습니다.</div>';
    }else{
      const maxCount=Math.max(...s.by_language.map(l=>l.count));
      const legend='<div class="legend-row">'+s.by_language.map(l=>
        '<span><span class="legend-dot" style="background:'+(LANG_COLORS[l.language]||'var(--dim)')+'"></span>'+(LANG_LABELS[l.language]||l.language)+'</span>'
      ).join('')+'</div>';
      const bars=s.by_language.map(l=>{
        const pct=maxCount?(l.count/maxCount*100):0;
        const color=LANG_COLORS[l.language]||'var(--dim)';
        return '<div class="bar-row" title="'+(LANG_LABELS[l.language]||l.language)+': '+l.count+'건">'+
          '<div class="bar-label">'+(LANG_LABELS[l.language]||l.language)+'</div>'+
          '<div class="bar-track"><div class="bar-fill" style="width:'+pct+'%;background:'+color+'"></div></div>'+
          '<div class="bar-val">'+l.count+'</div></div>';
      }).join('');
      langEl.innerHTML=legend+bars;
    }

    const actEl=document.getElementById('actionChart');
    if(!s.top_actions.length){
      actEl.innerHTML='<div class="empty-note">데이터가 아직 없습니다.</div>';
    }else{
      const maxA=Math.max(...s.top_actions.map(a=>a.count));
      actEl.innerHTML=s.top_actions.map(a=>{
        const pct=maxA?(a.count/maxA*100):0;
        return '<div class="bar-row" title="'+a.action+': '+a.count+'회">'+
          '<div class="bar-label">'+a.action+'</div>'+
          '<div class="bar-track"><div class="bar-fill" style="width:'+pct+'%;background:var(--seq-hue)"></div></div>'+
          '<div class="bar-val">'+a.count+'</div></div>';
      }).join('');
    }

    const tEl=document.getElementById('recentTable');
    if(!s.recent.length){
      tEl.innerHTML='<div class="empty-note">데이터가 아직 없습니다.</div>';
    }else{
      const rows=s.recent.map(r=>{
        const t=new Date(r.ts*1000).toLocaleTimeString('ko-KR',{hour12:false});
        const total=Math.round((r.stt_ms||0)+(r.llm_ms||0)+(r.exec_ms||0));
        const statusCls=r.success?'rt-ok':'rt-er';
        const statusTxt=r.success?'성공':'실패';
        return '<tr><td>'+t+'</td><td>'+(LANG_LABELS[r.language]||r.language)+'</td><td>'+r.source+'</td>'+
          '<td>'+escapeHtml(r.actions||'-')+'</td><td class="'+statusCls+'">'+statusTxt+'</td><td>'+total+'ms</td></tr>';
      }).join('');
      tEl.innerHTML='<table class="rt-table"><thead><tr><th>시간</th><th>언어</th><th>출처</th><th>동작</th><th>결과</th><th>총 지연</th></tr></thead><tbody>'+rows+'</tbody></table>';
    }
  }catch(e){alertRow('통계 로드 실패: '+e.message,'er');}
}

// ── Status + telemetry polling ──────────────────────────────────────────────
function tempClass(t){return t==null?'':t<45?'g':t<55?'y':'r';}
function tempLabel(t){return t==null?'—':t<45?'정상':t<55?'주의':'위험';}
async function poll(){
  try{
    const d=await(await fetch('/status')).json();
    const rd=document.getElementById('rdot');
    const cb=document.getElementById('cbadge');
    cb.textContent=d.camera?'📷 LIVE':'📷 신호 없음';
    cb.className=d.camera?'':'off';
    rd.className=d.camera?'dot on':'dot';
    document.getElementById('rip').textContent=d.robot_ip;
    document.getElementById('robotName').textContent=d.robot_name||'GO2';

    const t=d.telemetry||{};
    const battEl=document.getElementById('c-batt'),battSub=document.getElementById('c-batt-sub');
    if(t.connected&&t.battery_pct!=null){
      battEl.textContent=t.battery_pct+'%';
      battEl.className='card-val '+(t.battery_pct<20?'r':t.battery_pct<40?'y':'g');
      battSub.textContent=(t.power_v!=null?t.power_v.toFixed(1)+'V':'—')+' · '+(t.power_a!=null?t.power_a.toFixed(0)+'mA':'—');
    }else{battEl.textContent='—';battEl.className='card-val';battSub.textContent='텔레메트리 연결 중...';}

    const tempEl=document.getElementById('c-temp'),tempSub=document.getElementById('c-temp-sub');
    if(t.connected&&t.motor_temp_max!=null){
      tempEl.textContent=t.motor_temp_max+'°C';
      tempEl.className='card-val '+tempClass(t.motor_temp_max);
      tempSub.textContent='평균 '+(t.motor_temp_avg!=null?t.motor_temp_avg+'°C':'—')+' · '+tempLabel(t.motor_temp_max);
    }else{tempEl.textContent='—';tempEl.className='card-val';tempSub.textContent='텔레메트리 연결 중...';}

    const netEl=document.getElementById('c-net'),netSub=document.getElementById('c-net-sub');
    const connected=!!t.connected;
    netEl.textContent=connected?'연결됨':'끊김';
    netEl.className='card-val '+(connected?'g':'r');
    netSub.textContent=(d.robot_ip||'').startsWith('192.168.12.')?'WebRTC (WiFi)':'DDS (LAN)';

    const ww=d.wake_word||{};
    const wwBtn=document.getElementById('wwToggle'),wwLbl=document.getElementById('wwLabel');
    wwEnabled=!!ww.enabled;
    wwBtn.disabled=!ww.available;
    wwBtn.classList.toggle('on',wwEnabled);
    wwLbl.textContent=!ww.available?'🎙️ 웨이크워드 "윤재야" — 사용 불가 (WiFi 모드 전용)':
      (wwEnabled?'🎙️ 웨이크워드 "윤재야" — 듣는 중':'🎙️ 웨이크워드 "윤재야" — 꺼짐');

    const patrol=d.patrol||{};
    const pLbl=document.getElementById('patrolStatus');
    if(pLbl){
      pLbl.textContent=!patrol.available?'🛰️ 정찰 상태: 사용 불가 (카메라 미연결)':
        '🛰️ 정찰 상태: '+patrolStateLabel(patrol.state)+(patrol.error?' — 오류: '+patrol.error:'');
    }

    const tracker=d.tracker||{};
    const tLbl=document.getElementById('trackingStatus');
    if(tLbl){
      tLbl.textContent=!tracker.available?'👁️ 추적 상태: 사용 불가':
        '👁️ 추적 상태: '+(tracker.running?'추적 중':'대기 중')+(tracker.error?' — 오류: '+tracker.error:'');
    }
    const slam=d.slam||{};
    const poseBadge=document.getElementById('poseBadge'),poseReadout=document.getElementById('poseReadout');
    if(poseBadge&&poseReadout){
      poseBadge.textContent=slam.pose_fresh?'🟢 LIVE':(slam.pose?'⚪ lidar 꺼짐':'—');
      if(slam.pose){
        const p=slam.pose;
        const yawDeg=(p.yaw*180/Math.PI).toFixed(0);
        poseReadout.textContent=(slam.pose_fresh?'':'마지막 확인 위치: ')+'x='+p.x.toFixed(2)+'m, y='+p.y.toFixed(2)+'m, 방향='+yawDeg+'°';
      }else{
        poseReadout.textContent='lidar 꺼짐 (정상) — 위치 저장/이동 시에만 잠깐 켜집니다';
      }
    }
    const navStatus=document.getElementById('navStatus');
    if(navStatus){
      const st=slam.navigator_state;
      const stLabel={idle:'대기 중',navigating:'이동 중...',arrived:'도착함',failed:'실패',unavailable:'사용 불가'}[st]||st;
      const progressText=(st==='navigating' && slam.navigator_progress)?' ('+slam.navigator_progress+')':'';
      navStatus.textContent='🧭 내비게이션: '+stLabel+progressText+(slam.navigator_error?' — 오류: '+slam.navigator_error:'');
    }
    loadWaypoints();

    const trkImg=document.getElementById('trk-img'),trkPh=document.getElementById('trk-placeholder'),trkBadge=document.getElementById('trkbadge');
    if(trkImg&&tracker.running!==trkRunning){
      trkRunning=!!tracker.running;
      if(trkRunning){
        trkImg.src='/tracking_feed?_='+Date.now();
        trkImg.style.display='block'; trkPh.style.display='none';
        if(trkBadge){trkBadge.textContent='📷 LIVE';}
      }else{
        trkImg.removeAttribute('src');
        trkImg.style.display='none'; trkPh.style.display='flex';
        if(trkBadge){trkBadge.textContent='꺼짐';}
      }
    }
  }catch(e){}
}
setInterval(poll,2000);poll();
alertRow('대시보드가 준비되었습니다','ok');
</script>
</body>
</html>"""


# ─── FastAPI App ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

app = FastAPI()
_camera: Optional[CameraStreamer] = None
_telemetry: Optional[TelemetryStreamer] = None
_person_gate: Optional["PersonSafetyGate"] = None
_wake_word: Optional["WakeWordListener"] = None
_patrol: Optional["AutoPatrol"] = None
_tracker: Optional[PersonTracker] = None
_pose_tracker: Optional["PoseTracker"] = None
_navigator: Optional["SlamNavigator"] = None
_waypoints: Optional["WaypointStore"] = None
_ws_manager = WSManager()
_worker: Optional[PipelineWorker] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
ROBOT_NAME = "GO2-01"


@app.on_event("startup")
async def _startup():
    global _loop, _worker, _patrol, _tracker, _pose_tracker, _navigator, _waypoints
    _loop = asyncio.get_event_loop()
    _worker = PipelineWorker(_ws_manager, _loop)
    if _camera:
        _camera.start()
    if _camera and _worker:
        from app.navigation.auto_patrol import AutoPatrol
        _patrol = AutoPatrol(camera=_camera, worker=_worker, model=LLM_MODEL)
    if _telemetry:
        _telemetry.start()
    if _wake_word:
        _wake_word.start()
    # Initialize tracker but don't start it yet -- reuses _camera/_worker's
    # already-managed connections (see PersonTracker's docstring), so both
    # must exist first.
    if _camera and _worker:
        _tracker = PersonTracker(camera=_camera, worker=_worker)
    # SLAM pose + waypoint navigation -- see slam_navigator.py's docstring.
    # _pose_tracker starts immediately (passive subscribe, like _telemetry);
    # _navigator is only ever driven on-demand (navigate_to()), nothing to
    # start eagerly.
    from app.navigation.slam_navigator import PoseTracker, SlamNavigator, WaypointStore
    _waypoints = WaypointStore()
    if _camera:
        _pose_tracker = PoseTracker(camera=_camera)
        _pose_tracker.start()
    if _worker and _pose_tracker:
        _navigator = SlamNavigator(camera=_camera, worker=_worker, pose_tracker=_pose_tracker, waypoints=_waypoints)


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.get("/video_feed")
async def video_feed():
    async def gen():
        while True:
            frame = _camera.get_frame() if _camera else None
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await _ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await _ws_manager.disconnect(ws)


@app.post("/voice_command")
async def voice_command(audio: UploadFile = File(...), language: str = Form("ko")):
    data = await audio.read()
    if _worker:
        _worker.process_audio(data, language=language)
    return JSONResponse({"status": "processing"})


@app.post("/action/{action_name}")
async def direct_action(action_name: str):
    if _worker:
        _worker.run_action(action_name)
    return JSONResponse({"status": "ok"})


@app.post("/text_command")
async def text_command(payload: dict = Body(...)):
    text = (payload or {}).get("text", "").strip()
    if text and _worker:
        _worker.analyze_text(text)
    return JSONResponse({"status": "processing"})


@app.post("/confirm_execute")
async def confirm_execute():
    if _worker:
        _worker.confirm_execute()
    return JSONResponse({"status": "ok"})


@app.post("/cancel_command")
async def cancel_command():
    if _worker:
        _worker.cancel_pending()
    return JSONResponse({"status": "ok"})


@app.post("/move/{direction}")
async def move_start(direction: str):
    if _worker:
        _worker.start_move(direction)
    return JSONResponse({"status": "ok"})


@app.post("/move_stop")
async def move_stop():
    if _worker:
        _worker.stop_move_continuous()
    return JSONResponse({"status": "ok"})


@app.post("/speed/{delta}")
async def speed_adjust(delta: float):
    if _worker:
        _worker.adjust_speed(delta)
    return JSONResponse({"status": "ok", "speed": _worker._speed if _worker else None})


@app.post("/emergency_stop")
async def emergency_stop():
    if _worker:
        _worker.emergency_stop()
    return JSONResponse({"status": "ok"})


@app.post("/self_intro")
async def self_intro():
    if _worker:
        _worker.run_self_intro()
    return JSONResponse({"status": "ok"})


@app.post("/vision_analyze")
async def vision_analyze(payload: dict = Body(default={})):
    language = (payload or {}).get("language", "ko")
    if _worker:
        _worker.run_vision_analysis(language=language)
    return JSONResponse({"status": "ok"})


@app.post("/vision/register")
async def vision_register(payload: dict = Body(...)):
    name = (payload or {}).get("name", "")
    title = (payload or {}).get("title", "")
    introduction = (payload or {}).get("introduction", "")
    if _worker:
        _worker.register_face(name, title, introduction)
    return JSONResponse({"status": "ok"})


@app.get("/analytics")
async def analytics_summary():
    from app.analytics.analytics import get_store
    return JSONResponse(get_store().summary())


@app.post("/wake_word/{state}")
async def wake_word_toggle(state: str):
    if not _wake_word:
        return JSONResponse({"status": "unavailable"}, status_code=409)
    _wake_word.enable(state == "on")
    return JSONResponse({"status": "ok", "enabled": _wake_word.enabled})


@app.post("/patrol/start")
async def patrol_start(payload: dict = Body(default={})):
    if not _patrol:
        return JSONResponse({"status": "unavailable"}, status_code=409)
    language = (payload or {}).get("language", "ko")
    _patrol.start(language=language)
    return JSONResponse({"status": "ok", "state": _patrol.state})


@app.post("/patrol/stop")
async def patrol_stop():
    if _patrol:
        _patrol.stop()
    return JSONResponse({"status": "ok"})


@app.get("/status")
async def robot_status():
    return JSONResponse({
        "camera": bool(_camera and _camera.connected),
        "robot_ip": ROBOT_IP,
        "robot_name": ROBOT_NAME,
        "telemetry": _telemetry.get() if _telemetry else {},
        "robot_state": {
            "obstacle_distance_m": _worker._robot.executor.robot_state.obstacle_distance_m,
            "battery_level_static": _worker._robot.executor.robot_state.battery_level,
        } if (_worker and _worker._robot) else None,
        "wake_word": {"available": _wake_word is not None, "enabled": bool(_wake_word and _wake_word.enabled)},
        "patrol": {
            "available": _patrol is not None,
            "state": _patrol.state if _patrol else "idle",
            "error": _patrol.last_error if _patrol else None,
        },
        "tracker": {
            "available": _tracker is not None,
            "running": _tracker._running if _tracker else False,
            "error": _tracker.last_error if _tracker else None,
        },
        "slam": {
            "pose": _pose_tracker.get() if _pose_tracker else None,
            "pose_fresh": bool(_pose_tracker and _pose_tracker.is_fresh()),
            "navigator_state": _navigator.state if _navigator else "unavailable",
            "navigator_error": _navigator.last_error if _navigator else None,
            "navigator_progress": _navigator.progress if _navigator else None,
            "waypoints": list((_waypoints.list() if _waypoints else {}).keys()),
        },
    })


@app.get("/api/debug/person_check")
async def get_debug_person_check():
    """One-shot diagnostic report from the person safety gate's YOLO
    detection -- every detected person's bounding-box area fraction and
    whether it would currently block a forward move, WITHOUT issuing any
    move command. Meant for calibrating person_safety_gate.py's
    AREA_TOO_CLOSE_FRACTION against real measured distances (place a
    mannequin/cardboard stand-in at a known distance, call this repeatedly
    while moving it, and read off the area_fraction at each distance) --
    check_forward_path() itself only reports blocked/not-blocked, with no
    diagnostic detail on a near-miss, which isn't enough to calibrate by.
    """
    if _person_gate is None:
        return JSONResponse({"status": "error", "message": "person gate not initialized"}, status_code=409)
    report = _person_gate.inspect()
    return JSONResponse({"status": "ok", **report})


@app.post("/api/debug/obstacle_distance")
async def set_debug_obstacle_distance(payload: dict = Body(default={})):
    """Manually sets RobotState.obstacle_distance_m on the live executor, for
    the context-aware gate's physically-staged case study. There is no live
    obstacle sensor wired into the voice-command path (see RobotState's
    obstacle_distance_m docstring in behavior_schema.py) -- this is the
    equivalent, for a real physical trial, of benchmark_corpus.json's
    robot_state_override used in dry-run. Pass {"meters": 0.3} right before
    issuing a voice/text command with a real object physically placed at that
    measured distance, or {"meters": null} to clear it back to unknown/
    not-gated. The value is sticky (applies to every command until changed),
    not reset automatically after one use.
    """
    if not (_worker and _worker._robot):
        return JSONResponse({"status": "error", "message": "robot not connected yet"}, status_code=409)
    meters = (payload or {}).get("meters")
    _worker._robot.executor.robot_state.obstacle_distance_m = (
        float(meters) if meters is not None else None
    )
    return JSONResponse({
        "status": "ok",
        "obstacle_distance_m": _worker._robot.executor.robot_state.obstacle_distance_m,
    })


@app.post("/tracking/start")
async def tracking_start():
    if not _tracker:
        return JSONResponse({"status": "unavailable"}, status_code=409)
    _tracker.start(enable_tracking=True)
    return JSONResponse({"status": "ok", "running": _tracker._running})


@app.post("/tracking/stop")
async def tracking_stop():
    if _tracker:
        _tracker.stop()
    return JSONResponse({"status": "ok"})


@app.post("/waypoint/save")
async def waypoint_save(payload: dict = Body(...)):
    name = (payload or {}).get("name", "")
    if _worker:
        _worker.save_waypoint(name)
    return JSONResponse({"status": "ok"})


@app.get("/waypoints")
async def waypoint_list():
    return JSONResponse(_waypoints.list() if _waypoints else {})


@app.post("/route/save")
async def route_save(payload: dict = Body(...)):
    name = (payload or {}).get("name", "")
    legs = (payload or {}).get("legs", [])
    if not _waypoints:
        return JSONResponse({"status": "error", "message": "저장소를 사용할 수 없습니다"}, status_code=503)
    try:
        _waypoints.save_route(name, legs)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/waypoint/delete")
async def waypoint_delete(payload: dict = Body(...)):
    name = (payload or {}).get("name", "")
    if _waypoints:
        _waypoints.delete(name)
    return JSONResponse({"status": "ok"})


@app.post("/navigate_to")
async def navigate_to_endpoint(payload: dict = Body(...)):
    location = (payload or {}).get("location", "")
    language = (payload or {}).get("language", "ko")
    if _worker:
        _worker.navigate_to(location, language=language, source="direct")
    return JSONResponse({"status": "ok"})


@app.get("/tracking_feed")
async def tracking_feed():
    async def gen():
        while True:
            if _tracker:
                frame = _tracker.get_display_frame()
                if frame is not None:
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    global ROBOT_IP, SERVER_PORT, OPENAI_API_KEY, LLM_MODEL, ROBOT_NAME, _camera, _telemetry, _wake_word, _person_gate

    import argparse
    p = argparse.ArgumentParser(description="Go2 Robot Dashboard")
    p.add_argument("--robot-ip", default="192.168.123.18")
    p.add_argument("--robot-name", default="GO2-01")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--openai-key", help="OpenAI API key (hoặc set OPENAI_API_KEY env)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model (mặc định: gpt-4o-mini)")
    p.add_argument("--no-wake-word", action="store_true",
                   help="Tắt tính năng luôn lắng nghe '윤재야' qua mic robot (mặc định: bật, chỉ ở chế độ WiFi)")
    args = p.parse_args()

    ROBOT_IP = args.robot_ip
    ROBOT_NAME = args.robot_name
    SERVER_PORT = args.port
    LLM_MODEL = args.model
    if args.openai_key:
        OPENAI_API_KEY = args.openai_key

    _camera = CameraStreamer(robot_ip=ROBOT_IP)
    _telemetry = TelemetryStreamer(robot_ip=ROBOT_IP, camera=_camera)
    from app.vision.person_safety_gate import PersonSafetyGate
    _person_gate = PersonSafetyGate(camera=_camera)

    wake_word_on = _camera._is_wifi() and not args.no_wake_word
    if wake_word_on:
        from app.voice.wake_word_listener import WakeWordListener
        _wake_word = WakeWordListener(
            camera=_camera,
            on_command=lambda audio: _worker.process_audio_array(audio) if _worker else None,
        )

    wake_word_label = "'윤재야' (robot mic, always listening)" if wake_word_on else "off (LAN mode has no mic path, or --no-wake-word set)"

    print(f"\n🤖 Go2 Robot Dashboard")
    print(f"   Robot IP : {ROBOT_IP}")
    print(f"   Server   : http://localhost:{SERVER_PORT}")
    print(f"   Camera   : {'WebRTC/WiFi' if _camera._is_wifi() else 'DDS/LAN'}")
    print(f"   Telemetry: {'WebRTC/WiFi' if _camera._is_wifi() else 'DDS/LAN'} (battery %, motor temp)")
    print(f"   Wake-word: {wake_word_label}")
    print(f"   Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")


if __name__ == "__main__":
    main()
