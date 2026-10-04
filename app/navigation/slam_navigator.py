#!/usr/bin/env python3
"""
SLAM waypoint navigation -- drives the Go2 to a named location using the
onboard L1 lidar's SLAM pose (rt/utlidar/robot_pose over WebRTC). Confirmed
live end-to-end via test_lidar_slam.py (error_state=0, ~15Hz point cloud,
~18Hz pose updates over the robot's own WiFi hotspot).

UNTESTED on real hardware beyond that connectivity check -- the control loop
below (turn-to-face, then drive-with-correction, stop within ARRIVE_DISTANCE_M)
is a reasonable starting point, not a calibrated one. Expect to retune the
speed/gain constants after watching the robot actually drive, the same way
voice_control_go2.py's turn-duration table took 3 rounds of real-robot
measurement before it was accurate.

IMPORTANT -- pose must be enabled ON-DEMAND, not left on, and via SUBSCRIBE/
UNSUBSCRIBE specifically, not the ULIDAR_SWITCH: two versions of this bug
happened back to back while building it, both confirmed live against the
real robot.

  v1: PoseTracker switched the lidar ON once at dashboard startup and left
      it on for the whole session -- broke telemetry (stuck on "연결 중...")
      and D-pad movement entirely, only the camera kept working (video rides
      a separate media track, not the WebRTC data channel everything else
      shares). Fixed by making the switch on-demand.

  v2: Still broken after v1's fix, because subscribe() -- not the switch --
      is what actually starts the robot pushing rt/utlidar/robot_pose to a
      given connection. v1's start() subscribed unconditionally at dashboard
      startup and never unsubscribed; turning the switch off (confirmed via
      lidar_off.py) did NOT stop the stream to that already-subscribed
      connection -- pose kept flowing 10+ minutes later. The other half of
      the damage: the vendored webrtc_datachannel.py logs literally every
      data-channel message via a bare logging.info() call (record.name ==
      "root", unlike this project's own logging.getLogger(__name__) loggers)
      -- ~18-20Hz of that synchronous logging, stacked on top of the
      existing audio-frame traffic, was enough to starve the shared asyncio
      event loop. Fixed by (a) moving the actual subscribe()/unsubscribe()
      calls into enable_lidar()/disable_lidar() instead of start(), and (b)
      dashboard_server.py filters out root-logger INFO records so this
      vendored noise doesn't get printed regardless of subscription state.

Net effect: PoseTracker.enable_lidar()/disable_lidar() must always be called
in pairs (enable right before reading a fresh pose, disable in a finally
block right after) -- never leave a subscription open when not actively
saving a waypoint or navigating.

Reuses rather than duplicates -- same rationale as AutoPatrol/PersonTracker
(see their docstrings in auto_patrol.py / dashboard_server.py): pose comes
from the dashboard's existing shared WebRTC connection (via CameraStreamer)
instead of opening a second one, and driving goes through PipelineWorker's
already-connected RobotController (._robot.executor.sport_client).

Coordinate frame: rt/utlidar/robot_pose reports position (x, y, z) and
orientation (quaternion) in the "odom" frame -- a fixed world frame anchored
wherever the SLAM service last initialized, NOT relative to the robot's
current heading. sport_client.Move(vx, vy, omega) is a BODY-frame velocity
command (vx = robot's own forward), so driving toward a world-frame target
requires knowing the robot's current heading (extracted from the quaternion)
to compute which way "forward" points in odom coordinates right now.

Because the odom frame is anchored at SLAM init time, saved waypoints are
only valid for the SLAM session they were recorded in -- if the lidar/SLAM
service restarts, (0, 0) moves to wherever the robot happens to be at that
moment, and old waypoints silently point at the wrong physical spot. Re-save
waypoints after any SLAM restart. Nothing in this module can detect that a
restart happened; there's no persistent map ID to check it against.

Not a real path planner: "go to X" drives an approximately-straight line
while continuously re-aiming at the target, and relies on FreeAvoid (see
webrtc_sport_client.py) to keep it from driving into anything in the way --
same as everywhere else in this project that uses FreeAvoid instead of an
actual planner. It will not route *around* an obstacle blocking the direct
line to the target -- confirmed live: a point saved in the middle of a room
was unreachable in a straight line from a hallway outside it, because that
line crosses a wall. rt/utlidar/voxel_map (what FreeAvoid itself likely
uses) was investigated as a way to build a real occupancy-grid planner
instead, but its decoder (vendor/unitree_webrtc_connect/.../lidar_decoder_native.py)
shows a fixed 128x128x38-voxel grid at ~0.05m resolution -- a ~6.4m local
window around the robot's current position, not a persistent map of
previously-visited-but-now-out-of-range areas, so it can't help plan a
route back through a doorway once that doorway is out of sensor range.
Unitree's own apparent global mapping/nav service (rt/uslam/client_command,
rt/uslam/navigation/global_path in constants.py) was also considered, but
its command protocol is undocumented anywhere found (searched public repos
and the RoboVerse wiki) -- every public Go2 navigation project instead
builds its own SLAM+Nav2 stack from raw sensor data rather than driving
that service, which is itself a signal nobody has reverse-engineered it
successfully. Given both, the pragmatic fix landed on instead: routes (see
WaypointStore) -- an ordered chain of already-saved points (e.g. "door" then
"room_center") driven leg by leg, so a human picks the waypoints that keep
each leg's straight line inside open floor space, same as how a person
would describe directions ("go to the door, then to the middle of the
room") rather than the robot figuring out the floor plan itself.
"""

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WAYPOINTS_FILE = Path(__file__).parent / "waypoints.json"

# ── Drive-to-point control loop tuning ──────────────────────────────────────
# UNVERIFIED on real hardware -- conservative starting values, adjust after
# watching actual behavior (see module docstring).
ARRIVE_DISTANCE_M = 0.35            # stop once within this many meters of the target
FACE_TARGET_HEADING_ERROR = 0.35    # rad (~20deg) -- must be roughly facing target before driving forward
HEADING_KP = 1.1                    # matches dashboard_server.py's YAW_KP used for person-tracking turns
MIN_TURN_SPEED = 0.5
MAX_TURN_SPEED = 1.2
DRIVE_SPEED = 0.25
MAX_DRIVE_HEADING_CORRECTION = 0.6  # omega cap while walking forward -- corrects course without spinning in place
CONTROL_REFRESH_S = 0.3             # Move() decays after ~1s if unrefreshed, same as everywhere else in this project
MAX_NAV_DURATION_S = 60.0           # safety cap -- abort rather than wander indefinitely if something's wrong
POSE_STALE_S = 1.0                  # if pose hasn't updated within this window, stop rather than drive blind


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rotation about Z / ground-plane heading) from a quaternion.
    Sign-invariant under global quaternion negation (q and -q are the same
    rotation) -- both the numerator and denominator here are bilinear in the
    components, so this works regardless of which sign convention a given
    pose message happens to use (observed both signs of `w` in live testing)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


class PoseTracker:
    """Subscribes to rt/utlidar/robot_pose over the shared WebRTC connection
    and keeps the latest (x, y, yaw, updated_at). Mirrors dashboard_server.py's
    TelemetryStreamer -- same shared-connection rationale (see
    CameraStreamer's docstring for why a second WebRTC connection is unsafe)."""

    def __init__(self, camera):
        self._camera = camera
        self._lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._updated_at = 0.0
        self._started = False
        self._conn = None
        self._loop = None
        self._topic = None
        self._callback = None

    def start(self):
        """Resolves the shared connection/topic/callback but does NOT
        subscribe yet -- see enable_lidar()/disable_lidar(). Safe to call
        once at dashboard startup."""
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._run, daemon=True, name="slam-pose").start()

    def _run(self):
        try:
            if not self._camera or not self._camera.webrtc_ready.wait(timeout=20):
                logger.warning("PoseTracker: camera WebRTC not ready -- no pose available")
                return
            conn, loop = self._camera.webrtc_conn, self._camera.webrtc_loop
            if conn is None or loop is None:
                logger.warning("PoseTracker: camera WebRTC connection missing -- no pose available")
                return
            self._conn, self._loop = conn, loop

            # unitree_webrtc_connect is pip-installed editable (see
            # vendor/unitree_webrtc_connect/), so no sys.path hack is needed.
            from unitree_webrtc_connect.constants import RTC_TOPIC
            self._topic = RTC_TOPIC["ROBOTODOM"]

            def on_pose(message):
                try:
                    data = message.get("data") or {}
                    pose = data.get("pose") or {}
                    pos = pose.get("position") or {}
                    ori = pose.get("orientation") or {}
                    yaw = quaternion_to_yaw(
                        ori.get("x", 0.0), ori.get("y", 0.0), ori.get("z", 0.0), ori.get("w", 1.0)
                    )
                    with self._lock:
                        self._x = pos.get("x", self._x)
                        self._y = pos.get("y", self._y)
                        self._yaw = yaw
                        self._updated_at = time.time()
                except Exception as e:
                    logger.warning(f"PoseTracker parse error: {e}")

            self._callback = on_pose
            # Deliberately does NOT subscribe here -- see enable_lidar()'s
            # docstring: subscribing (not the ULIDAR_SWITCH) is what actually
            # starts the robot pushing rt/utlidar/robot_pose to this
            # connection, confirmed empirically (switching the sensor off did
            # NOT stop the stream to an already-subscribed connection -- only
            # unsubscribe does). Subscribing here unconditionally at startup
            # was the real cause of the "everything breaks" incident, not the
            # switch -- see module docstring.
            logger.info("PoseTracker: ready (not subscribed yet -- lidar off until enable_lidar())")
        except Exception as e:
            logger.error(f"PoseTracker crashed: {e}", exc_info=True)

    def enable_lidar(self):
        """Subscribes to rt/utlidar/robot_pose (and switches the raw sensor
        on too) -- only call right before a navigation operation needs fresh
        pose, and always pair with disable_lidar() in a finally block.

        IMPORTANT, confirmed empirically: it's the SUBSCRIBE call that starts
        the robot pushing pose data to this connection, not the ULIDAR_SWITCH
        -- switching the sensor off did NOT stop rt/utlidar/robot_pose from
        continuing to flow to an already-subscribed connection. An earlier
        version of this method only toggled the switch and left the
        subscription active from start() -- that meant pose streamed
        continuously for the whole dashboard session regardless of the
        switch, flooding the shared WebRTC data channel (also used for
        telemetry + sport commands) and breaking both. See module docstring."""
        if self._conn is None or self._loop is None or self._topic is None or self._callback is None:
            return
        from unitree_webrtc_connect.constants import RTC_TOPIC
        self._loop.call_soon_threadsafe(
            self._conn.datachannel.pub_sub.subscribe, self._topic, self._callback
        )
        self._loop.call_soon_threadsafe(
            self._conn.datachannel.pub_sub.publish_without_callback, RTC_TOPIC["ULIDAR_SWITCH"], "on"
        )

    def disable_lidar(self):
        """Unsubscribes (this is what actually stops the stream -- see
        enable_lidar()'s docstring) and switches the sensor back off. Always
        call after enable_lidar(), even on failure (use try/finally)."""
        if self._conn is None or self._loop is None or self._topic is None:
            return
        from unitree_webrtc_connect.constants import RTC_TOPIC
        self._loop.call_soon_threadsafe(
            self._conn.datachannel.pub_sub.unsubscribe, self._topic
        )
        self._loop.call_soon_threadsafe(
            self._conn.datachannel.pub_sub.publish_without_callback, RTC_TOPIC["ULIDAR_SWITCH"], "off"
        )
        # Deliberately does NOT clear the last stored pose -- is_fresh() will
        # naturally go False once POSE_STALE_S elapses with no new updates,
        # but get() still returning the last-known value is useful (the
        # dashboard's UI shows it as "last checked position").

    def wait_for_fresh(self, timeout: float = 5.0) -> bool:
        """Blocks (polling) until a pose update has arrived recently, or
        timeout. Call after enable_lidar() -- there's a real delay between
        switching the sensor on and the first rt/utlidar/robot_pose message
        arriving (confirmed live via test_lidar_slam.py: point cloud/pose
        start flowing within the first second or two, not instantly)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_fresh():
                return True
            time.sleep(0.2)
        return self.is_fresh()

    def get(self) -> Optional[dict]:
        with self._lock:
            if self._updated_at == 0.0:
                return None
            return {"x": self._x, "y": self._y, "yaw": self._yaw, "updated_at": self._updated_at}

    def is_fresh(self) -> bool:
        with self._lock:
            return self._updated_at > 0 and (time.time() - self._updated_at) < POSE_STALE_S


class WaypointStore:
    """name -> {type: "point", x, y, yaw, saved_at} for a single saved pose,
    or name -> {type: "route", legs: [point_name, ...], saved_at} for an
    ordered chain of existing point waypoints -- persisted together to the
    same waypoints.json (plain JSON, not vision_people.xlsx's spreadsheet
    format -- nothing here benefits from being human-editable in a
    spreadsheet the way face registration's title/introduction fields do).

    Routes exist because navigate_to() drives an approximately-straight
    line to a single target and does not route around anything not
    immediately in front of the robot (see SlamNavigator's module
    docstring -- confirmed on real hardware: a saved point in the middle of
    a room was unreachable in a straight line from outside the room,
    through the doorway). A route breaks that one straight-line drive into
    several, one per already-saved point, driven in order -- e.g. saving
    "door" and "room_center" as points, then a route "room" = [door,
    room_center] lets navigate_to("room") walk through the doorway instead
    of into the wall next to it. Routes may only reference point waypoints,
    not other routes (no nesting -- avoids cycles and keeps the one-drive-
    per-leg model in SlamNavigator simple)."""

    def __init__(self, path: Path = WAYPOINTS_FILE):
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"WaypointStore: failed to read {self._path}: {e}")
            return {}

    def _save(self, data: dict):
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, name: str, x: float, y: float, yaw: float):
        name = (name or "").strip()
        if not name:
            raise ValueError("Waypoint name cannot be empty")
        with self._lock:
            data = self._load()
            data[name] = {"type": "point", "x": x, "y": y, "yaw": yaw, "saved_at": time.time()}
            self._save(data)

    def save_route(self, name: str, legs: list):
        """legs: ordered list of already-saved point-waypoint names. Each is
        resolved via find_by_name_fuzzy at save time (not just navigate
        time) so a typo is caught immediately instead of failing mid-drive
        later -- the resolved (exact) names are what's stored, so a leg's
        name being fuzzy-matched at save time doesn't silently change if a
        similarly-named point is added later."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Route name cannot be empty")
        legs = [str(leg).strip() for leg in (legs or []) if str(leg).strip()]
        if not legs:
            raise ValueError("Route must have at least one point")
        with self._lock:
            data = self._load()
            resolved_legs = []
            for leg in legs:
                resolved = leg if leg in data else None
                if resolved is None:
                    q = leg.lower()
                    for candidate in data:
                        if q in candidate.lower() or candidate.lower() in q:
                            resolved = candidate
                            break
                leg_entry = data.get(resolved) if resolved else None
                if leg_entry is None or leg_entry.get("type", "point") != "point":
                    raise ValueError(f"저장된 지점을 찾을 수 없습니다: {leg}")
                resolved_legs.append(resolved)
            data[name] = {"type": "route", "legs": resolved_legs, "saved_at": time.time()}
            self._save(data)

    def get(self, name: str) -> Optional[dict]:
        with self._lock:
            return self._load().get(name)

    def find_by_name_fuzzy(self, query: str) -> Optional[str]:
        """Exact match first, then substring either direction -- mirrors
        BehaviorLibrary.get_by_voice_command()'s fallback shape (behavior_library.py),
        since the LLM may pass back a slightly reworded location name instead
        of the exact saved key."""
        data = self._load()
        if query in data:
            return query
        q = (query or "").strip().lower()
        if not q:
            return None
        for name in data:
            if q in name.lower() or name.lower() in q:
                return name
        return None

    def list(self) -> dict:
        with self._lock:
            return self._load()

    def delete(self, name: str):
        with self._lock:
            data = self._load()
            if name in data:
                del data[name]
                self._save(data)


class SlamNavigator:
    """Drives the robot to a saved waypoint (or through an ordered chain of
    them -- a "route", see WaypointStore) using SLAM pose feedback (see
    module docstring for the single-leg control loop's shape and its limits)."""

    def __init__(self, camera, worker, pose_tracker: PoseTracker, waypoints: WaypointStore):
        self._camera = camera
        self._worker = worker
        self._pose = pose_tracker
        self._waypoints = waypoints
        self._stop_evt = threading.Event()
        self._last_error: Optional[str] = None
        self._state = "idle"  # idle | navigating | arrived | failed
        self._progress: Optional[str] = None  # "2/3 -> room_center" while driving a multi-leg route

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def progress(self) -> Optional[str]:
        return self._progress

    def stop(self):
        self._stop_evt.set()

    def _drive_to_target(self, sc, target: dict) -> bool:
        """One straight-line-with-heading-correction leg, to a single
        {x, y} target. Assumes lidar is already enabled/fresh and FreeAvoid
        is already on (navigate_to() manages those once for the whole
        route, not per leg -- toggling the lidar subscription between legs
        would re-pay wait_for_fresh()'s few-second delay for no benefit).
        Returns True on arrival, False on failure/abort/timeout (sets
        self._last_error)."""
        start_time = time.monotonic()
        while not self._stop_evt.is_set():
            if time.monotonic() - start_time > MAX_NAV_DURATION_S:
                self._last_error = f"이동 시간 초과 ({MAX_NAV_DURATION_S:.0f}초)"
                return False
            if not self._pose.is_fresh():
                self._last_error = "이동 중 SLAM 위치 정보 유실 (lidar 연결 끊김?)"
                return False

            pose = self._pose.get()
            dx = target["x"] - pose["x"]
            dy = target["y"] - pose["y"]
            distance = math.hypot(dx, dy)
            if distance < ARRIVE_DISTANCE_M:
                return True

            target_heading = math.atan2(dy, dx)
            heading_error = normalize_angle(target_heading - pose["yaw"])

            if abs(heading_error) > FACE_TARGET_HEADING_ERROR:
                # Turn-in-place phase -- not yet roughly facing the target.
                turn_speed = HEADING_KP * heading_error
                if 0 < abs(turn_speed) < MIN_TURN_SPEED:
                    turn_speed = MIN_TURN_SPEED if turn_speed > 0 else -MIN_TURN_SPEED
                turn_speed = max(-MAX_TURN_SPEED, min(MAX_TURN_SPEED, turn_speed))
                sc.Move(0.0, 0.0, turn_speed)
            else:
                # Driving phase, with a small proportional heading correction.
                correction = max(-MAX_DRIVE_HEADING_CORRECTION,
                                  min(MAX_DRIVE_HEADING_CORRECTION, HEADING_KP * heading_error))
                sc.Move(DRIVE_SPEED, 0.0, correction)

            self._stop_evt.wait(CONTROL_REFRESH_S)

        self._last_error = "이동이 취소되었습니다"
        return False

    def navigate_to(self, name: str) -> bool:
        """Blocking -- call on a background thread (see PipelineWorker.navigate_to).
        `name` may be a point waypoint or a route (drives each of the
        route's legs in order, see WaypointStore.save_route). Returns True
        on arrival at the final target, False on failure/abort (see
        .last_error) -- a route that fails partway does not roll back or
        retry, the robot simply stops wherever that leg left it."""
        self._last_error = None
        self._progress = None

        wp_name = self._waypoints.find_by_name_fuzzy(name)
        if wp_name is None:
            self._last_error = f"저장된 위치를 찾을 수 없습니다: {name}"
            self._state = "failed"
            return False
        wp = self._waypoints.get(wp_name)

        if wp.get("type", "point") == "route":
            legs = wp.get("legs") or []
            if not legs:
                self._last_error = f"경로 '{wp_name}'에 저장된 지점이 없습니다"
                self._state = "failed"
                return False
            targets = []
            for leg_name in legs:
                leg_wp = self._waypoints.get(leg_name)
                if leg_wp is None or leg_wp.get("type", "point") != "point":
                    self._last_error = f"경로 안의 지점을 찾을 수 없습니다: {leg_name}"
                    self._state = "failed"
                    return False
                targets.append((leg_name, leg_wp))
        else:
            targets = [(wp_name, wp)]

        try:
            self._worker._ensure_robot()
        except Exception as e:
            self._last_error = f"로봇 연결 실패: {e}"
            self._state = "failed"
            return False
        sc = self._worker._robot.executor.sport_client
        if sc is None or not self._worker._robot.executor.connected:
            self._last_error = "로봇이 연결되어 있지 않습니다"
            self._state = "failed"
            return False

        self._stop_evt.clear()
        self._state = "navigating"

        # Lidar and FreeAvoid are switched on once for the whole route (not
        # per leg) -- see PoseTracker.enable_lidar()'s docstring for why
        # leaving lidar on longer than necessary is bad, but re-subscribing
        # between every leg would also re-pay wait_for_fresh()'s delay for
        # no benefit since the connection never dropped.
        self._pose.enable_lidar()
        try:
            if not self._pose.wait_for_fresh(timeout=5.0):
                self._last_error = "SLAM 위치 정보가 없습니다 (lidar 연결을 확인하세요)"
                self._state = "failed"
                return False

            try:
                sc.FreeAvoid(True)
            except Exception as e:
                logger.warning(f"SlamNavigator: FreeAvoid(True) failed: {e}")

            arrived = False
            try:
                for i, (leg_name, leg_wp) in enumerate(targets):
                    if len(targets) > 1:
                        self._progress = f"{i + 1}/{len(targets)} → {leg_name}"
                    arrived = self._drive_to_target(sc, leg_wp)
                    if not arrived:
                        break
            finally:
                try:
                    sc.Move(0.0, 0.0, 0.0)
                except Exception:
                    pass
                try:
                    sc.FreeAvoid(False)
                except Exception as e:
                    logger.warning(f"SlamNavigator: FreeAvoid(False) failed: {e}")
        finally:
            self._pose.disable_lidar()
            self._progress = None

        self._state = "arrived" if arrived else "failed"
        if not arrived and not self._last_error:
            self._last_error = "이동이 중단되었습니다"
        return arrived
