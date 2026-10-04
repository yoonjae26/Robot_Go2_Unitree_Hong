#!/usr/bin/env python3
"""
Quick DDS connection test over WiFi.
Run from laptop connected to Go2 WiFi (192.168.12.x):
  python3 test_wifi_dds.py
"""
import os
import time
import sys

ROBOT_IP = "192.168.12.1"

# Disable multicast — force unicast only to motion board WiFi IP.
# Without this, CycloneDDS may try to respond via 192.168.123.161 (internal LAN)
# which is unreachable from WiFi clients.
os.environ["CYCLONEDDS_URI"] = (
    "<CycloneDDS><Domain>"
    "<General><AllowMulticast>false</AllowMulticast></General>"
    f"<Discovery><Peers><Peer address='{ROBOT_IP}'/></Peers></Discovery>"
    "</Domain></CycloneDDS>"
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

print(f"[1] Initializing DDS on wlo1 → peer {ROBOT_IP}")
ChannelFactoryInitialize(0, "wlo1")

print("[2] Creating SportClient...")
sc = SportClient()
sc.SetTimeout(10.0)
sc.Init()

print("[3] Waiting 3s for DDS discovery...")
time.sleep(3)

print("[4] Calling Hello()...")
ret = sc.Hello()
print(f"    Result: {ret}  {'✅ OK!' if ret == 0 else '❌ Failed (code ' + str(ret) + ')'}")

if ret == 0:
    print("\n[5] Calling StandDown()...")
    ret2 = sc.StandDown()
    print(f"    Result: {ret2}  {'✅ OK!' if ret2 == 0 else '❌ Failed (code ' + str(ret2) + ')'}")
