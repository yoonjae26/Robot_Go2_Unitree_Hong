#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sajogi_commando_test24.py

Go2 카메라 기반 사람 추적 시험 프로그램

스레드 구조
1. Vision Thread
   - VideoClient
   - JPEG 디코딩
   - YOLO.track()
   - ByteTrack
   - 표적 선택
   - 수평 오차와 요청 yaw 계산
   - 오버레이 작성
   - 완성된 display_frame 공유

2. Motion Thread
   - SportClient.Move() 전담
   - IDLE / TURNING / SETTLING 상태기계
   - 연속 회전 시작 및 정지
   - 자세 안정화 대기
   - imshow 호출 없음

3. Main Thread
   - cv2.imshow()
   - cv2.waitKey()
   - 효과음 및 종료 입력
   - Move 호출 없음

핵심 제어
- 짧은 Move 펄스를 반복하지 않는다.
- 한 번 회전을 시작한 뒤 영상을 계속 확인한다.
- 목표가 정지 기준에 들어오면 Move(0, 0, 0)을 한 번 전송한다.
- 정지 후 일정 시간 동안 추가 회전을 금지해 자세 복원을 기다린다.
- 자세 안정 후 재시작 기준보다 크게 벗어난 경우에만 다시 회전한다.
"""

import sys
import time
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


WINDOW_NAME = "Go2 YOLO Person Tracking Test 24"
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

YAW_KP = 0.62
MIN_YAW_SPEED = 0.45
YAW_DIRECTION = -1.0

TURN_STOP_ERROR_RATIO = 0.045
TURN_RESTART_ERROR_RATIO = 0.075
TURN_SETTLING_TIME = 0.60
TURN_MAX_DURATION = 1.30

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


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def initialize_audio() -> pygame.mixer.Sound:
    if pygame.mixer.get_init():
        pygame.mixer.quit()

    pygame.mixer.init(
        frequency=44100,
        size=-16,
        channels=2,
        buffer=256,
    )

    sample_rate = 44100
    duration = 0.60
    sample_count = int(sample_rate * duration)
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    rng = np.random.default_rng()

    noise = rng.standard_normal(sample_count).astype(np.float32)
    blast = noise * np.exp(-17.0 * t)

    low = (
        1.00 * np.sin(2.0 * np.pi * 52.0 * t)
        + 0.55 * np.sin(2.0 * np.pi * 78.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 104.0 * t)
    ) * np.exp(-6.0 * t)

    crack = (
        np.sin(2.0 * np.pi * 1500.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 2300.0 * t)
    ) * np.exp(-55.0 * t)

    delayed = np.maximum(t - 0.045, 0.0)
    echo_noise = rng.standard_normal(sample_count).astype(np.float32)
    echo = echo_noise * np.exp(-28.0 * delayed)
    echo[t < 0.045] = 0.0

    mono = 0.90 * blast + 0.72 * low + 0.19 * crack + 0.25 * echo
    mono = np.tanh(2.0 * mono)

    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono /= peak

    pcm = (mono * 32700).astype(np.int16)
    stereo = np.column_stack((pcm, pcm))

    sound = pygame.sndarray.make_sound(stereo)
    sound.set_volume(1.0)
    return sound


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
    """밝고 어두운 배경 모두에서 보이도록 검은 외곽선 뒤에 본문을 그린다."""
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
    """표적 박스 위에 글자 크기만큼의 작은 검은 라벨 배경을 그린다."""
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
        "F/SPACE:sound  Q/ESC:quit",
        (18, frame.shape[0] - 18),
        0.49,
        COLOR_TEXT,
        1,
    )


def main() -> None:
    if len(sys.argv) > 1:
        network_interface = sys.argv[1]
        ChannelFactoryInitialize(0, network_interface)
        print(f"[네트워크] 인터페이스: {network_interface}")
    else:
        ChannelFactoryInitialize(0)
        print("[네트워크] 기본 인터페이스 사용")

    print(f"[1/4] YOLO 모델 준비: {MODEL_NAME}")
    yolo_model = YOLO(MODEL_NAME)

    print("[2/4] 수동 효과음 준비")
    demo_sound = initialize_audio()

    print("[3/4] Go2 SportClient 연결")
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    result = sport_client.StandUp()
    print(f"      StandUp 반환값: {result}")
    time.sleep(3.0)

    try:
        result = sport_client.FreeAvoid(False)
        print(f"      FreeAvoid(False) 반환값: {result}")
        time.sleep(0.5)
    except Exception as error:
        print(f"      FreeAvoid(False) 사용 불가: {error}")

    print("[4/4] Vision / Motion 스레드 시작")

    stop_event = threading.Event()

    vision_lock = threading.Lock()
    motion_lock = threading.Lock()
    display_lock = threading.Lock()

    vision_state = VisionState(
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

    motion_state = MotionState(
        mode=MotionMode.IDLE,
        current_yaw=0.0,
        turn_started_at=0.0,
        settling_until=0.0,
        last_result=None,
        updated_at=0.0,
        error=None,
    )

    display_state = DisplayState(
        frame=None,
        updated_at=0.0,
    )

    def get_motion_snapshot() -> MotionState:
        with motion_lock:
            return MotionState(
                mode=motion_state.mode,
                current_yaw=motion_state.current_yaw,
                turn_started_at=motion_state.turn_started_at,
                settling_until=motion_state.settling_until,
                last_result=motion_state.last_result,
                updated_at=motion_state.updated_at,
                error=motion_state.error,
            )

    def set_motion_state(
        *,
        mode: MotionMode,
        current_yaw: float,
        turn_started_at: float,
        settling_until: float,
        last_result: Optional[object],
        error: Optional[str],
    ) -> None:
        nonlocal motion_state
        with motion_lock:
            motion_state = MotionState(
                mode=mode,
                current_yaw=current_yaw,
                turn_started_at=turn_started_at,
                settling_until=settling_until,
                last_result=last_result,
                updated_at=time.monotonic(),
                error=error,
            )

    def vision_worker() -> None:
        nonlocal vision_state, display_state

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
            video_client = VideoClient()
            video_client.SetTimeout(3.0)
            video_client.Init()
            print("[VISION] VideoClient 연결 완료")

            while not stop_event.is_set():
                try:
                    return_code, image_bytes = video_client.GetImageSample()

                    if return_code != 0 or image_bytes is None:
                        stop_event.wait(0.01)
                        continue

                    encoded = np.frombuffer(
                        bytes(image_bytes),
                        dtype=np.uint8,
                    )

                    if encoded.size == 0:
                        continue

                    frame = cv2.imdecode(
                        encoded,
                        cv2.IMREAD_COLOR,
                    )

                    if frame is None:
                        continue

                    frame_height, frame_width = frame.shape[:2]
                    started = time.monotonic()

                    results = yolo_model.track(
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

                    inference_ms = (
                        time.monotonic() - started
                    ) * 1000.0

                    detections = (
                        parse_yolo_result(
                            results[0],
                            frame_width,
                            frame_height,
                        )
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
                                + (1.0 - POSITION_SMOOTHING_ALPHA)
                                * smoothed_center_x
                            )

                        (
                            requested_yaw,
                            aligned,
                            error_ratio,
                        ) = calculate_requested_yaw(
                            smoothed_center_x,
                            frame_width,
                        )

                        last_target = target
                        last_requested_yaw = requested_yaw
                        last_error_ratio = error_ratio
                        last_aligned = aligned

                    else:
                        lost_frames += 1

                        if (
                            last_target is not None
                            and lost_frames <= LOST_FRAME_LIMIT
                        ):
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

                    new_vision_state = VisionState(
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

                    motion_snapshot = get_motion_snapshot()
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

                    with vision_lock:
                        vision_state = new_vision_state

                    with display_lock:
                        display_state = DisplayState(
                            frame=display_frame,
                            updated_at=now,
                        )

                except Exception as error:
                    now = time.monotonic()

                    with vision_lock:
                        vision_state = VisionState(
                            detections=[],
                            target=None,
                            requested_yaw=0.0,
                            error_ratio=0.0,
                            aligned=False,
                            inference_ms=0.0,
                            vision_fps=vision_fps,
                            updated_at=now,
                            error=str(error),
                            lost_frames=LOST_FRAME_LIMIT + 1,
                            target_held=False,
                        )

                    stop_event.wait(0.05)

        except Exception as error:
            now = time.monotonic()
            with vision_lock:
                vision_state = VisionState(
                    detections=[],
                    target=None,
                    requested_yaw=0.0,
                    error_ratio=0.0,
                    aligned=False,
                    inference_ms=0.0,
                    vision_fps=0.0,
                    updated_at=now,
                    error=str(error),
                    lost_frames=LOST_FRAME_LIMIT + 1,
                    target_held=False,
                )

    def motion_worker() -> None:
        mode = MotionMode.IDLE
        current_yaw = 0.0
        turn_started_at = 0.0
        settling_until = 0.0
        last_result = None
        last_log_time = 0.0

        def send_stop() -> Optional[object]:
            return sport_client.Move(
                0.0,
                0.0,
                0.0,
            )

        try:
            set_motion_state(
                mode=mode,
                current_yaw=current_yaw,
                turn_started_at=turn_started_at,
                settling_until=settling_until,
                last_result=last_result,
                error=None,
            )

            while not stop_event.is_set():
                now = time.monotonic()

                with vision_lock:
                    command_target = vision_state.target
                    command_yaw = vision_state.requested_yaw
                    command_error = vision_state.error_ratio
                    command_aligned = vision_state.aligned
                    command_updated_at = vision_state.updated_at
                    command_error_text = vision_state.error

                state_age = (
                    now - command_updated_at
                    if command_updated_at > 0.0
                    else float("inf")
                )

                target_valid = (
                    command_target is not None
                    and command_error_text is None
                    and state_age <= VISION_STALE_TIMEOUT
                )

                if mode == MotionMode.IDLE:
                    current_yaw = 0.0

                    if (
                        target_valid
                        and not command_aligned
                        and command_error >= TURN_RESTART_ERROR_RATIO
                        and command_yaw != 0.0
                    ):
                        last_result = sport_client.Move(
                            0.0,
                            0.0,
                            float(command_yaw),
                        )

                        current_yaw = float(command_yaw)
                        turn_started_at = now
                        mode = MotionMode.TURNING

                        print(
                            f"[TURN START] "
                            f"yaw={current_yaw:+.3f} "
                            f"error={command_error:.3f} "
                            f"result={last_result}"
                        )

                elif mode == MotionMode.TURNING:
                    turn_elapsed = now - turn_started_at

                    direction_reversed = (
                        command_yaw != 0.0
                        and current_yaw != 0.0
                        and math.copysign(1.0, command_yaw)
                        != math.copysign(1.0, current_yaw)
                    )

                    should_stop = (
                        not target_valid
                        or command_aligned
                        or command_error <= TURN_STOP_ERROR_RATIO
                        or direction_reversed
                        or turn_elapsed >= TURN_MAX_DURATION
                    )

                    if should_stop:
                        last_result = send_stop()
                        current_yaw = 0.0
                        settling_until = now + TURN_SETTLING_TIME
                        mode = MotionMode.SETTLING

                        print(
                            f"[TURN STOP] "
                            f"error={command_error:.3f} "
                            f"reversed={int(direction_reversed)} "
                            f"elapsed={turn_elapsed:.2f}s "
                            f"result={last_result}"
                        )

                elif mode == MotionMode.SETTLING:
                    current_yaw = 0.0

                    if now >= settling_until:
                        mode = MotionMode.IDLE

                if now - last_log_time >= 1.0:
                    print(
                        f"[MOTION] "
                        f"mode={mode.value} "
                        f"yaw={current_yaw:+.3f} "
                        f"error={command_error:.3f} "
                        f"age={state_age:.3f}s "
                        f"target={int(target_valid)}"
                    )
                    last_log_time = now

                set_motion_state(
                    mode=mode,
                    current_yaw=current_yaw,
                    turn_started_at=turn_started_at,
                    settling_until=settling_until,
                    last_result=last_result,
                    error=None,
                )

                stop_event.wait(MOTION_LOOP_INTERVAL)

        except Exception as error:
            print(f"[MOTION 오류] {error}")

            try:
                last_result = send_stop()
            except Exception:
                pass

            set_motion_state(
                mode=MotionMode.IDLE,
                current_yaw=0.0,
                turn_started_at=0.0,
                settling_until=0.0,
                last_result=last_result,
                error=str(error),
            )

            stop_event.set()

        finally:
            for _ in range(10):
                try:
                    sport_client.Move(0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(0.05)

            try:
                result = sport_client.StopMove()
                print(f"[정지] StopMove 반환값: {result}")
            except Exception as error:
                print(f"[정지 오류] {error}")

    vision_thread = threading.Thread(
        target=vision_worker,
        daemon=True,
        name="vision-worker",
    )

    motion_thread = threading.Thread(
        target=motion_worker,
        daemon=True,
        name="motion-worker",
    )

    vision_thread.start()
    motion_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_vision_error = None
    last_motion_error = None

    try:
        while not stop_event.is_set():
            with display_lock:
                display_frame = (
                    display_state.frame.copy()
                    if display_state.frame is not None
                    else None
                )

            with vision_lock:
                vision_error = vision_state.error

            with motion_lock:
                motion_error = motion_state.error

            if vision_error and vision_error != last_vision_error:
                print(f"[VISION 오류] {vision_error}")
                last_vision_error = vision_error

            if motion_error and motion_error != last_motion_error:
                print(f"[MOTION 상태 오류] {motion_error}")
                last_motion_error = motion_error

            if display_frame is not None:
                cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("f"), ord("F"), 32):
                demo_sound.play()
                print("[SOUND] 수동 효과음 출력")

            elif key in (ord("q"), ord("Q"), 27):
                print("[종료] Motion Thread에 정지 요청")
                stop_event.set()
                break

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("[중단] Ctrl+C")
        stop_event.set()

    finally:
        stop_event.set()

        vision_thread.join(timeout=3.0)
        motion_thread.join(timeout=3.0)

        cv2.destroyAllWindows()

        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except BaseException:
            pass

        print("[완료] Go2 정지 및 프로그램 종료")


if __name__ == "__main__":
    main()
