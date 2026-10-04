#!/usr/bin/env python3
"""
Force the Go2's lidar sensor OFF (rt/utlidar/switch = "off"). Run this once
if the dashboard was left with the lidar stuck on (see slam_navigator.py's
module docstring for the incident this is a recovery tool for) -- the
switch state lives on the robot, not in the dashboard process, so simply
restarting dashboard_server.py does not by itself turn a stuck-on sensor
back off.

Run from a laptop connected to the Go2's own WiFi hotspot (192.168.12.x):
  python3 lidar_off.py
"""

import asyncio
import os
import sys

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")

# unitree_webrtc_connect is pip-installed editable (see
# vendor/unitree_webrtc_connect/), so no sys.path hack is needed.
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC


async def main():
    print(f"Connecting to Go2 at {ROBOT_IP}...")
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await conn.connect()
    print("Connected -- sending lidar OFF...")
    conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "off")
    await asyncio.sleep(1.0)  # give the send a moment before the connection tears down
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
