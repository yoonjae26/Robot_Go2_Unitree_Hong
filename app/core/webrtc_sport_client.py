#!/usr/bin/env python3
"""
Drop-in stand-in for unitree_sdk2py's SportClient, backed by the WebRTC data
channel instead of DDS.

Over the Go2's own WiFi hotspot (192.168.12.x) the DDS RPC services (SportClient,
MotionSwitcherClient, ...) never get a response -- confirmed via test_wifi_dds.py,
which times out with code 3102 (RPC_ERR_CLIENT_SEND) even for a plain Hello().
The WebRTC data channel (same mechanism the official Unitree app uses over WiFi)
does reach the internal Sport service, via RTC_TOPIC["SPORT_MOD"] pub/sub calls.

This robot runs firmware >= 1.1.7, which defaults to "MCF" (Motion Control
Framework) mode -- confirmed live via MOTION_SWITCHER CheckMode returning
{"name": "mcf"}. No motion_switcher handshake/mode-switch is needed or possible:
attempting to SelectMode("ai") is silently ignored (mode stays "mcf"), which is
why the flip family used to fail with error 3203 -- not a missing mode switch,
but because the *older* api_id numbers (e.g. BackFlip=1044) are wrong for MCF.
MCF uses a different id for that subset of commands.

Every id/parameter-shape below is taken directly from the vendored, community-
maintained MCF reference example (not guessed):
  vendor/unitree_webrtc_connect/examples/go2/data_channel/sportmode_mcf/sportmode_mcf.py
which itself pulls from vendor/unitree_webrtc_connect/unitree_webrtc_connect/constants.py's
SPORT_CMD_MCF table. Commands not listed in SPORT_CMD_MCF (Wallow, MoonWalk,
RightFlip, ...) have no confirmed MCF id and are deliberately left unsupported --
calling one raises AttributeError, which BehaviorExecutor._execute_sportclient_method
already treats as "unsupported action" (via hasattr) rather than crashing.
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# name.lower() -> SPORT_MOD api_id, for zero-parameter commands.
_SUPPORTED = {
    "damp": 1001,
    "balancestand": 1002,
    "stopmove": 1003,
    "standup": 1004,
    "standdown": 1005,
    "recoverystand": 1006,
    "sit": 1009,
    "risesit": 1010,
    "hello": 1016,
    "stretch": 1017,
    "content": 1020,
    "dance1": 1022,
    "dance2": 1023,
    "scrape": 1029,
    "frontjump": 1031,
    "frontpounce": 1032,
    "heart": 1036,
    "fingerheart": 1036,  # alias -- same id, older WebRTC table used this name
    "getautorecovery": 2055,
}

# Single-shot trick commands that always fire with {"data": True} (not a
# True/False toggle) -- confirmed by sportmode_mcf.py's dispatch().
_SUPPORTED_TRIGGER = {
    "frontflip": 1030,
    "leftflip": 2041,
    "backflip": 2043,
}

# True/False toggle commands -- {"data": bool}. MCF ids (2041+ range) replace
# the older, wrong ones (e.g. HandStand was 1301, now 2044); FreeBound/FreeJump/
# FreeAvoid/WalkUpright/ClassicWalk/SetAutoRecovery/LeadFollow/SwitchAvoidMode
# were previously thought unsupported -- they just weren't in the older table.
_SUPPORTED_TOGGLE = {
    "switchjoystick": 1027,
    "staticwalk": 1061,
    "trotrun": 1062,
    "economicgait": 1063,
    "handstand": 2044,
    "freewalk": 2045,
    "freebound": 2046,
    "freejump": 2047,
    "freeavoid": 2048,
    "classicwalk": 2049,
    "walkupright": 2050,  # sportmode_mcf.py names this "BackStand"; same id as DDS SPORT_API_ID_WALKUPRIGHT
    "backstand": 2050,    # alias, matches sportmode_mcf.py's own name for it
    "crossstep": 2051,
    "setautorecovery": 2054,
    "leadfollow": 2056,
    "switchavoidmode": 2058,
}

# Pose (1028) is *not* a symmetric bool toggle: {"data": True} enters the pose,
# but the reference example's own hint says to exit via StopMove (1003), not
# {"data": False}. BehaviorExecutor's toggle action path calls the method with
# True then False -- handled as a special case in __getattr__ below so that
# generic call pattern still does the right thing for this one command.
POSE_API_ID = 1028
STOPMOVE_API_ID = 1003

MOVE_API_ID = 1008

# Confirmed present at the same id in both the legacy SPORT_CMD table and
# SPORT_CMD_MCF (unlike most tricks, which got renumbered into the 2041+
# range for MCF) -- see sportmode_mcf.py's own dispatcher, which sends the
# identical {"x": roll, "y": pitch, "z": yaw} shape. Like Move, this is a raw
# body-pose command with no schema/cooldown gating -- but UNLIKE Move it does
# NOT auto-decay: Move is fire-and-forget (_CallNoReply, drifts back to zero
# on its own if not refreshed), Euler is a held state ("tilt to X and stay")
# with no automatic return to neutral. Callers must explicitly send
# Euler(0, 0, 0) when done tilting, or the robot stays leaned over.
EULER_API_ID = 1007

# {"data": int}, no documented valid range in either the DDS SDK or the MCF
# reference example (sportmode_mcf.py just asks for a raw int) -- unverified
# on real hardware, start with small values.
SPEEDLEVEL_API_ID = 1015


class WebRTCSportClient:
    """Same call shape as unitree_sdk2py SportClient: SetTimeout/Init, then
    method calls returning 0 on success, matching BehaviorExecutor's expectations."""

    def __init__(
        self,
        robot_ip: str,
        shared_conn=None,
        shared_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Args:
            shared_conn/shared_loop: an already-connected UnitreeWebRTCConnection
                (+ the asyncio loop it's running on), reused instead of opening a
                second WebRTC peer connection. Go2's local WebRTC bridge handles a
                second simultaneous connection unreliably -- two independent
                connections (e.g. this client's own + the dashboard camera's) can
                both stall/timeout when they negotiate at the same time. Pass the
                camera's connection here when one is already open.
        """
        self.robot_ip = robot_ip
        self._timeout = 10.0
        self._loop = shared_loop
        self._conn = shared_conn
        self._owns_connection = shared_conn is None

    def SetTimeout(self, timeout: float):
        self._timeout = timeout

    def Init(self):
        if not self._owns_connection:
            if self._conn is None or self._loop is None:
                raise RuntimeError("Shared WebRTC connection was not ready")
            logger.info(f"WebRTC sport channel reusing shared connection: {self.robot_ip}")
            return

        # unitree_webrtc_connect is pip-installed editable (see
        # vendor/unitree_webrtc_connect/), so no sys.path hack is needed.
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection, WebRTCConnectionMethod,
        )

        self._conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.robot_ip)

        ready = threading.Event()
        error_box = {}

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._conn.connect())
            except Exception as e:
                error_box["error"] = e
            finally:
                ready.set()
            loop.run_forever()

        threading.Thread(target=run_loop, daemon=True, name="webrtc-sport").start()

        if not ready.wait(timeout=self._timeout):
            raise TimeoutError("WebRTC sport channel did not connect in time")
        if "error" in error_box:
            raise error_box["error"]

        logger.info(f"WebRTC sport channel connected: {self.robot_ip}")

    def Move(self, vx: float, vy: float, omega: float) -> int:
        return self._publish(MOVE_API_ID, {"x": vx, "y": vy, "z": omega})

    def Euler(self, roll: float, pitch: float, yaw: float) -> int:
        return self._publish(EULER_API_ID, {"x": roll, "y": pitch, "z": yaw})

    def SpeedLevel(self, level: int) -> int:
        return self._publish(SPEEDLEVEL_API_ID, {"data": int(level)})

    def __getattr__(self, name: str):
        key = name.lower()

        if key == "pose":
            def call_pose(flag=True, *_args, **_kwargs):
                if flag:
                    return self._publish(POSE_API_ID, {"data": True})
                return self._publish(STOPMOVE_API_ID)  # exit pose via StopMove, not {"data": False}
            return call_pose

        api_id = _SUPPORTED.get(key)
        if api_id is not None:
            def call(*_args, **_kwargs):
                return self._publish(api_id)
            return call

        trigger_api_id = _SUPPORTED_TRIGGER.get(key)
        if trigger_api_id is not None:
            def call_trigger(*_args, **_kwargs):
                return self._publish(trigger_api_id, {"data": True})
            return call_trigger

        toggle_api_id = _SUPPORTED_TOGGLE.get(key)
        if toggle_api_id is not None:
            def call_toggle(flag=True, *_args, **_kwargs):
                return self._publish(toggle_api_id, {"data": bool(flag)})
            return call_toggle

        raise AttributeError(name)

    def _publish(self, api_id: int, parameter: Optional[dict] = None) -> int:
        from unitree_webrtc_connect.constants import RTC_TOPIC

        payload = {"api_id": api_id}
        if parameter is not None:
            payload["parameter"] = parameter

        async def do_publish():
            return await self._conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], payload
            )

        future = asyncio.run_coroutine_threadsafe(do_publish(), self._loop)
        try:
            response = future.result(timeout=self._timeout)
        except Exception as e:
            logger.error(f"WebRTC sport command {api_id} failed: {e}")
            return 3102  # mirror DDS RPC_ERR_CLIENT_SEND so callers that check "!= 0" behave the same

        try:
            return response["data"]["header"]["status"]["code"]
        except Exception:
            return 0
