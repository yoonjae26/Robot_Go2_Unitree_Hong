#!/usr/bin/env python3
"""
Quick check: is the Go2's SLAM lidar (the L1 unit + onboard uslam service)
actually connected and streaming, over the same WebRTC path the dashboard
uses (rt/utlidar/... topics) -- see vendor/unitree_webrtc_connect/examples/go2/
data_channel/lidar/lidar_stream.py for the vendored reference this is based
on, and dashboard_server.py's TelemetryStreamer for the equivalent pattern
already used for battery/motor telemetry (subscribe to a topic, read
message["data"] as a plain dict with the same field names as the DDS struct
-- here unitree_sdk2py/idl/unitree_go/msg/dds_/_LidarState_.py).

Run from a laptop connected to the Go2's own WiFi hotspot (192.168.12.x):
  python3 test_lidar_slam.py

Checks two things, not just one -- they can fail independently:
  1. rt/utlidar/lidar_state -- is the raw lidar SENSOR itself connected and
     spinning (firmware_version present, error_state == 0, cloud_frequency > 0)?
  2. rt/utlidar/robot_pose  -- is SLAM LOCALIZATION actually running on top of
     that sensor data (a pose estimate is being published)? A lidar can be
     physically connected and spinning while the SLAM/mapping service on top
     of it is not yet initialized -- these are different failure modes and
     the fix for each is different, so both are reported separately.

Turns the lidar back OFF before exiting (step [4]) -- an earlier version of
this script left it on, which then kept flooding the dashboard's shared
WebRTC data channel with point-cloud data for as long as the sensor stayed
on, breaking telemetry/Move() there even though this script itself had
already exited (see slam_navigator.py's module docstring for the full
incident). If this script is ever killed with Ctrl+C mid-run instead of
exiting normally, that cleanup step doesn't run -- use lidar_off.py to force
it off afterward.
"""

import asyncio
import logging
import os
import sys
import time

logging.basicConfig(level=logging.WARNING)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")
CHECK_SECONDS = float(os.environ.get("LIDAR_CHECK_SECONDS", "8"))

# unitree_webrtc_connect is pip-installed editable (see
# vendor/unitree_webrtc_connect/), so no sys.path hack is needed.
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC


async def main():
    print(f"[1] Connecting to Go2 at {ROBOT_IP} over WebRTC...")
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await conn.connect()
    print("    Connected.\n")

    states = []
    poses = []

    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["ULIDAR_STATE"], lambda m: states.append(m.get("data") or {}))
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["ROBOTODOM"], lambda m: poses.append(m.get("data") or {}))

    print("[2] Turning lidar ON (rt/utlidar/switch)...")
    conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")

    print(f"[3] Listening for {CHECK_SECONDS:.0f}s...\n")
    await asyncio.sleep(CHECK_SECONDS)

    print("=" * 64)
    print("SENSOR (rt/utlidar/lidar_state)")
    print("-" * 64)
    if not states:
        print("❌ No lidar_state message received in time.")
        print("   -> L1 unit is not reporting over WebRTC at all. Check: is it")
        print("      physically attached and powered? Does the Unitree app see it?")
    else:
        s = states[-1]
        err = s.get("error_state")
        freq = s.get("cloud_frequency")
        loss = s.get("cloud_packet_loss_rate")
        print(f"✅ Received {len(states)} message(s) in {CHECK_SECONDS:.0f}s")
        print(f"   firmware_version      : {s.get('firmware_version')}")
        print(f"   error_state           : {err}  {'(OK)' if err == 0 else '(!! non-zero = hardware reporting an error)'}")
        print(f"   cloud_frequency (Hz)  : {freq}  {'(OK)' if freq and freq > 0 else '(!! 0 Hz = spinning but no point cloud data)'}")
        print(f"   cloud_packet_loss_rate: {loss}")

    print()
    print("LOCALIZATION (rt/utlidar/robot_pose)")
    print("-" * 64)
    if not poses:
        print("⚠️  No robot_pose message received -- SLAM localization doesn't")
        print("   seem to be running yet. This is separate from the sensor check")
        print("   above: the lidar can be fine while mapping/localization on top")
        print("   of it hasn't been started (see rt/uslam/client_command in")
        print("   vendor/unitree_webrtc_connect/unitree_webrtc_connect/constants.py")
        print("   for the topic that likely starts it -- not wired up yet here).")
    else:
        print(f"✅ Received {len(poses)} message(s) -- SLAM localization is live")
        print(f"   latest: {poses[-1]}")
    print("=" * 64)

    # Turn the lidar back off before disconnecting -- leaving it on (as an
    # earlier version of this script did) keeps the robot broadcasting
    # point-cloud data to any other WebRTC client that connects afterward
    # (e.g. the dashboard), flooding its shared data channel exactly like
    # the incident documented in slam_navigator.py's module docstring.
    print("\n[4] Turning lidar back OFF...")
    conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "off")
    await asyncio.sleep(0.5)
    print("    Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nConnection failed: {e}")
        sys.exit(1)
