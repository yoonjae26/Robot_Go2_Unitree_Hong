#!/usr/bin/env python3
"""
Go2 Real-time Camera Stream
Phương pháp 1: DDS VideoClient (dùng LAN, 192.168.123.18)
Phương pháp 2: WebRTC (dùng WiFi STA mode, 192.168.12.1)
"""

import sys
import time
import argparse
import subprocess
import logging
import threading
import queue
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Phương pháp 1: DDS VideoClient (LAN)
# ──────────────────────────────────────────────────────────────────────────────

def detect_interface(robot_ip: str) -> str:
    try:
        subnet = ".".join(robot_ip.split(".")[:3])
        result = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.splitlines():
            if subnet in line:
                iface = line.split()[1]
                logger.info(f"Detected interface: {iface}")
                return iface
    except Exception as e:
        logger.warning(f"Interface detect failed: {e}")
    return ""


def run_dds_camera(robot_ip: str = "192.168.123.18", save_video: bool = False):
    """Real-time camera stream via DDS VideoClient (kết nối LAN)."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    interface = detect_interface(robot_ip)
    logger.info(f"DDS — robot: {robot_ip}, interface: '{interface}'")

    try:
        ChannelFactoryInitialize(0, interface)
    except Exception as e:
        logger.info(f"ChannelFactory already initialized: {e}")

    client = VideoClient()
    client.SetTimeout(5.0)
    client.Init()

    print("\n📷 Go2 Camera Stream (DDS/LAN)")
    print("  ESC = thoát  |  S = chụp ảnh  |  R = bắt đầu/dừng ghi video")
    print("=" * 55)

    code, data = client.GetImageSample()
    if code != 0:
        print(f"❌ Không kết nối được camera. Error code: {code}")
        print("   Kiểm tra: robot đang bật, dây LAN đã cắm, IP đúng")
        return

    print("✅ Camera connected!\n")

    video_writer = None
    recording = False
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0
    frame_count = 0

    while True:
        code, data = client.GetImageSample()
        if code != 0:
            time.sleep(0.01)
            continue

        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame_count += 1
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        display = frame.copy()
        cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if recording:
            cv2.circle(display, (display.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display, "REC", (display.shape[1] - 70, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Go2 Front Camera", display)

        if recording and video_writer:
            video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('s'), ord('S')):
            fname = f"go2_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸 Saved: {fname}")
        elif key in (ord('r'), ord('R')):
            if not recording:
                fname = f"go2_video_{int(time.time())}.avi"
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    fname, cv2.VideoWriter_fourcc(*'XVID'), 15, (w, h)
                )
                recording = True
                print(f"🔴 Recording: {fname}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("⏹️  Recording stopped")

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("👋 Camera stream ended")


# ──────────────────────────────────────────────────────────────────────────────
# Phương pháp 2: WebRTC (WiFi, không cần LAN)
# ──────────────────────────────────────────────────────────────────────────────

def run_webrtc_camera(robot_ip: str = "192.168.12.1"):
    """Real-time camera stream via WebRTC (kết nối WiFi Go2)."""
    import asyncio

    webrtc_path = str(Path(__file__).parent / "Test" / "unitree_webrtc_connect")
    if webrtc_path not in sys.path:
        sys.path.insert(0, webrtc_path)

    try:
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection, WebRTCConnectionMethod
        )
        from aiortc import MediaStreamTrack
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Chạy: pip install aiortc")
        return

    frame_queue: queue.Queue = queue.Queue(maxsize=5)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)

    async def recv_frames(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if not frame_queue.full():
                frame_queue.put_nowait(img)

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            await conn.connect()
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_frames)
        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    t.start()

    print("\n📷 Go2 Camera Stream (WebRTC/WiFi)")
    print(f"  Robot IP: {robot_ip}")
    print("  ESC = thoát  |  S = chụp ảnh")
    print("=" * 55)
    print("⏳ Đang kết nối WebRTC...")

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0
    last_frame = None

    try:
        while True:
            if not frame_queue.empty():
                last_frame = frame_queue.get()
                fps_counter += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    fps_time = time.time()

                display = last_frame.copy()
                cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, "WebRTC", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
                cv2.imshow("Go2 Front Camera (WebRTC)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('s'), ord('S')) and last_frame is not None:
                fname = f"go2_capture_{int(time.time())}.jpg"
                cv2.imwrite(fname, last_frame)
                print(f"📸 Saved: {fname}")
            else:
                time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        print("👋 WebRTC stream ended")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Go2 Real-time Camera Stream")
    parser.add_argument(
        "--method", choices=["dds", "webrtc"], default="dds",
        help="dds=LAN/기본, webrtc=WiFi"
    )
    parser.add_argument(
        "--robot-ip", default="192.168.123.18",
        help="Robot IP (dds: 192.168.123.18 / webrtc: 192.168.12.1)"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Tự động ghi video ngay khi mở"
    )
    args = parser.parse_args()

    if args.method == "dds":
        run_dds_camera(robot_ip=args.robot_ip, save_video=args.save_video)
    else:
        run_webrtc_camera(robot_ip=args.robot_ip)


if __name__ == "__main__":
    main()
