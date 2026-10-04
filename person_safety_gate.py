#!/usr/bin/env python3
"""
Real-time, camera-based hard safety gate: refuses to let the robot walk
forward into a person detected directly in its path, regardless of what
the LLM/instruction says.

This is deliberately NOT part of the V_s/V_i/V_c pipeline in
runtime_verification.py. That pipeline's V_i (intent) layer checks
whether the LLM's chosen action matches the user's instruction -- it has
no opinion on whether the instruction itself is safe, so a user who
directly and unambiguously asks the robot to walk into a person gets
"consistent" actions approved (see the injected_01 case study finding:
an injected instruction whose actions matched it exactly sailed through
even with all three layers enabled). A second LLM call cannot fix this,
because the same failure mode -- "the actions correctly do what was
asked" -- is not something any LLM-based consistency check can catch by
construction.

This gate instead uses a completely different, non-LLM technology
(YOLO person detection on the live camera feed) specifically so its
failure modes are NOT correlated with the LLM generator's or LLM
verifier's failure modes. It cannot be talked around by clever prompt
wording, because it never reads the instruction text at all -- only the
camera frame and the robot's own outgoing velocity command.

Reuses the dashboard's existing CameraStreamer frames and the same
yolo11n.pt person-detection model already used by PersonTracker
(dashboard_server.py's "follow me" feature) -- see MODEL_NAME etc. below,
kept in sync with dashboard_server.py's constants of the same name.
Unlike PersonTracker, this is detection-only: no tracking persistence
(no track IDs needed, no bytetrack), no motion commands, so it is safe to
call synchronously from BehaviorExecutor.can_execute() without
conflicting with PersonTracker's own follow-me motion thread if that
feature happens to be running at the same time.

Proximity is a bounding-box-area proxy, not a metric distance -- this
system has no depth sensing wired into the voice-command path (see
runtime_verification.py's module docstring for the same caveat already
established for RobotState.obstacle_distance_m). AREA_TOO_CLOSE_FRACTION
is a conservative starting guess, UNVERIFIED on real hardware -- tune it
empirically using a mannequin/cardboard person-shaped stand-in (never a
real person) before trusting this gate's threshold near an actual human.
"""

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Kept identical to dashboard_server.py's PersonTracker constants of the
# same name -- same model, same class id, same inference settings, so
# detections from this gate and from the "follow me" feature agree.
MODEL_NAME = "yolo11n.pt"
PERSON_CLASS_ID = 0
YOLO_CONFIDENCE = 0.16
YOLO_IOU = 0.50
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = "cpu"

# Fraction of total frame area a detected person's bounding box must reach
# to be treated as "too close to walk toward". UNVERIFIED on real hardware --
# 0.25 means "the person fills roughly a quarter of the camera frame", a
# rough guess pending calibration against the mannequin/cardboard-cutout
# protocol (see thesis Section 5's safety protocol for the person-safety-gate
# case study). Start conservative (a lower fraction blocks earlier/more
# often) and raise only after confirming it doesn't false-trigger on people
# who are actually far enough away to be safe.
AREA_TOO_CLOSE_FRACTION = 0.25

# Horizontal band (centered on the frame) a person's bounding-box center
# must fall within to count as "in the robot's forward path" rather than
# off to the side where a straight-ahead move would not reach them.
# 0.5 = middle 50% of frame width.
CENTER_BAND_FRACTION = 0.5

# Below this forward speed (m/s), a move is not really "walking toward"
# anything -- pure sideways/rotation/backward moves are not gated by this
# check (it targets the specific "commanded to walk straight into a person
# standing in front" scenario, not general proximity).
FORWARD_VX_THRESHOLD = 0.05


def evaluate_person_boxes(
    boxes_xyxy, frame_width: int, frame_height: int,
    area_threshold: float = AREA_TOO_CLOSE_FRACTION,
    center_band: float = CENTER_BAND_FRACTION,
) -> Tuple[bool, Optional[str]]:
    """Pure decision logic, deliberately separated from the camera/YOLO glue
    below so it can be unit-tested with synthetic bounding boxes -- no
    camera or model required. `boxes_xyxy` is an iterable of (x1, y1, x2, y2)
    in pixel coordinates, already filtered to person-class detections.
    """
    if frame_width <= 0 or frame_height <= 0:
        return False, None

    frame_area = frame_width * frame_height
    band_lo = frame_width * (1 - center_band) / 2
    band_hi = frame_width * (1 + center_band) / 2

    for x1, y1, x2, y2 in boxes_xyxy:
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area_frac = (width * height) / frame_area
        center_x = (x1 + x2) / 2

        if area_frac >= area_threshold and band_lo <= center_x <= band_hi:
            return True, (
                f"person detected directly ahead (bbox {area_frac:.0%} of frame, "
                f">= {area_threshold:.0%} threshold, centered at x={center_x:.0f}px "
                f"within [{band_lo:.0f}, {band_hi:.0f}])"
            )

    return False, None


class PersonSafetyGate:
    """Detection-only, synchronous, lazy-loaded (the YOLO model is not
    loaded until the first forward move is checked, so a session that
    never moves the robot forward never pays the load cost). Call
    check_forward_path() right before a move with a forward vx component
    actually executes.
    """

    def __init__(self, camera):
        self._camera = camera
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from ultralytics import YOLO
                    logger.info(f"PersonSafetyGate: loading {MODEL_NAME}")
                    self._model = YOLO(MODEL_NAME)
        return self._model

    def _get_frame(self) -> Optional[np.ndarray]:
        """Pulls the latest camera frame from the dashboard's shared
        CameraStreamer instead of opening a second video connection --
        same reuse rationale as PersonTracker._get_frame()."""
        jpeg = self._camera.get_frame() if self._camera else None
        if not jpeg:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def check_forward_path(self, vx: float) -> Tuple[bool, Optional[str]]:
        """Returns (blocked, reason).

        Fails OPEN (blocked=False) whenever the camera or model is
        unavailable -- this is an additional safety layer stacked on top
        of the existing schema/intent/context checks, not the only line
        of defense, and a camera dropout should not make the robot unable
        to move at all. This is a deliberate, different choice from the
        intent verifier's fail-CLOSED behavior in intent_verifier.py:
        that check guards an LLM API call that's reasonable to expect
        available; this one guards a local camera/model pipeline whose
        transient unavailability (WiFi hiccup, frame not decoded yet) is
        a normal operating condition, not an error to treat as unsafe.
        """
        if vx < FORWARD_VX_THRESHOLD:
            return False, None

        frame = self._get_frame()
        if frame is None:
            return False, None

        try:
            model = self._ensure_model()
            results = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                imgsz=YOLO_IMAGE_SIZE,
                device=YOLO_DEVICE,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"PersonSafetyGate: detection failed, failing open: {e}")
            return False, None

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return False, None

        frame_height, frame_width = frame.shape[:2]
        xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
        return evaluate_person_boxes(xyxy, frame_width, frame_height)

    def inspect(self) -> dict:
        """Diagnostic, one-shot detection report -- runs regardless of vx and
        reports every detected person's measured area fraction/center, not
        just a blocked/not-blocked verdict. Meant for calibrating
        AREA_TOO_CLOSE_FRACTION against real distances (e.g. via a debug API
        endpoint) without needing to actually issue a move command and risk
        the robot travelling toward whatever is being measured.
        """
        frame = self._get_frame()
        if frame is None:
            return {"error": "no camera frame available"}

        try:
            model = self._ensure_model()
            results = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                imgsz=YOLO_IMAGE_SIZE,
                device=YOLO_DEVICE,
                verbose=False,
            )
        except Exception as e:
            return {"error": f"detection failed: {e}"}

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        band_lo = frame_width * (1 - CENTER_BAND_FRACTION) / 2
        band_hi = frame_width * (1 + CENTER_BAND_FRACTION) / 2

        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
            confs = results[0].boxes.conf.detach().cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                area_frac = ((x2 - x1) * (y2 - y1)) / frame_area
                center_x = (x1 + x2) / 2
                detections.append({
                    "confidence": round(float(conf), 3),
                    "area_fraction": round(float(area_frac), 4),
                    "center_x_px": round(float(center_x), 1),
                    "in_center_band": bool(band_lo <= center_x <= band_hi),
                    "would_block_if_forward": bool(
                        area_frac >= AREA_TOO_CLOSE_FRACTION and band_lo <= center_x <= band_hi
                    ),
                })

        return {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "area_too_close_threshold": AREA_TOO_CLOSE_FRACTION,
            "center_band_px": [round(band_lo, 1), round(band_hi, 1)],
            "detections": detections,
        }
