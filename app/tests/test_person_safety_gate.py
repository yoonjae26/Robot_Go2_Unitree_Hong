#!/usr/bin/env python3
"""
Tests for person_safety_gate.py. Follows this repo's plain-assert,
`python test_x.py` style (see test_go2_behaviors.py) rather than
pytest/unittest.

evaluate_person_boxes() is pure geometry (no camera/YOLO needed) and is
tested directly with synthetic bounding boxes. check_forward_path()'s
vx-threshold short-circuit and fail-open behavior are tested with a fake
camera object -- no real camera connection or YOLO model load happens in
these tests.
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FRAME_W, FRAME_H = 640, 480


def test_close_centered_person_blocks():
    print("\n" + "=" * 70)
    print("TEST: large, centered bounding box blocks")
    print("=" * 70)

    from app.vision.person_safety_gate import evaluate_person_boxes

    # A box covering ~40% of the frame, centered -- well past the 25% default.
    w, h = 350, 350
    x1 = (FRAME_W - w) / 2
    y1 = (FRAME_H - h) / 2
    boxes = [(x1, y1, x1 + w, y1 + h)]

    blocked, reason = evaluate_person_boxes(boxes, FRAME_W, FRAME_H)
    assert blocked is True, "expected a large centered box to block"
    assert reason is not None
    print(f"✅ blocked: {reason}")


def test_small_far_person_does_not_block():
    print("\n" + "=" * 70)
    print("TEST: small (far away) bounding box does not block")
    print("=" * 70)

    from app.vision.person_safety_gate import evaluate_person_boxes

    # A small box (~1% of frame), centered -- far below the area threshold.
    w, h = 60, 80
    x1 = (FRAME_W - w) / 2
    y1 = (FRAME_H - h) / 2
    boxes = [(x1, y1, x1 + w, y1 + h)]

    blocked, reason = evaluate_person_boxes(boxes, FRAME_W, FRAME_H)
    assert blocked is False, f"expected a small box not to block, got: {reason}"
    print("✅ small/far box does not block")


def test_large_but_off_to_side_does_not_block():
    print("\n" + "=" * 70)
    print("TEST: large bounding box off to the side does not block")
    print("=" * 70)

    from app.vision.person_safety_gate import evaluate_person_boxes

    # Large box, but pinned to the far right edge -- outside the center band.
    w, h = 300, 300
    x1 = FRAME_W - w
    y1 = (FRAME_H - h) / 2
    boxes = [(x1, y1, x1 + w, y1 + h)]

    blocked, reason = evaluate_person_boxes(boxes, FRAME_W, FRAME_H)
    assert blocked is False, f"expected an off-center box not to block, got: {reason}"
    print("✅ large but off-center box does not block")


def test_no_detections_does_not_block():
    print("\n" + "=" * 70)
    print("TEST: empty detections list does not block")
    print("=" * 70)

    from app.vision.person_safety_gate import evaluate_person_boxes

    blocked, reason = evaluate_person_boxes([], FRAME_W, FRAME_H)
    assert blocked is False
    assert reason is None
    print("✅ no detections -> does not block")


def test_check_forward_path_skips_below_vx_threshold():
    print("\n" + "=" * 70)
    print("TEST: check_forward_path() short-circuits for non-forward moves")
    print("=" * 70)

    from app.vision.person_safety_gate import PersonSafetyGate

    class ExplodingCamera:
        def get_frame(self):
            raise AssertionError("camera should not be read for a non-forward move")

    gate = PersonSafetyGate(camera=ExplodingCamera())
    blocked, reason = gate.check_forward_path(vx=0.0)
    assert blocked is False
    assert reason is None
    print("✅ vx=0.0 never touches the camera, returns (False, None)")


def test_check_forward_path_fails_open_without_frame():
    print("\n" + "=" * 70)
    print("TEST: check_forward_path() fails open when no camera frame is available")
    print("=" * 70)

    from app.vision.person_safety_gate import PersonSafetyGate

    class NoFrameCamera:
        def get_frame(self):
            return None

    gate = PersonSafetyGate(camera=NoFrameCamera())
    blocked, reason = gate.check_forward_path(vx=0.3)
    assert blocked is False, "expected fail-open (not blocked) when camera has no frame"
    assert reason is None
    print("✅ missing frame fails open (does not block), as documented")


if __name__ == "__main__":
    test_close_centered_person_blocks()
    test_small_far_person_does_not_block()
    test_large_but_off_to_side_does_not_block()
    test_no_detections_does_not_block()
    test_check_forward_path_skips_below_vx_threshold()
    test_check_forward_path_fails_open_without_frame()
    print("\n" + "=" * 70)
    print("ALL PERSON SAFETY GATE TESTS PASSED")
    print("=" * 70)
