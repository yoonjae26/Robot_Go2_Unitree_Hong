#!/usr/bin/env python3
"""
Behavior Executor - Executes Go2 robot behaviors with safety checks and monitoring
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import time
import logging
import subprocess
from typing import Optional, Dict, Any, Callable
from app.core.behavior_schema import RobotState, ActionSchema, RobotCommandResult, ActionStatus
from app.core.behavior_library import get_library, BehaviorLibrary
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from app.core.webrtc_sport_client import WebRTCSportClient
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How old live telemetry is allowed to be before can_execute() stops
# trusting it and falls back to the last known robot_state.battery_level.
# Matches runtime_verification.py's TELEMETRY_STALE_SECONDS -- duplicated
# rather than imported, since this per-action check is deliberately
# independent defense-in-depth alongside the whole-sequence check that
# module does before any action starts (see its module docstring).
TELEMETRY_STALE_SECONDS = 10.0


class BehaviorExecutor:
    """Executes robot behaviors with safety checks"""
    
    def __init__(
        self,
        robot_ip: str = "192.168.123.18",
        network_interface: str = "",
        webrtc_conn=None,
        webrtc_loop=None,
        telemetry=None,
        person_gate=None,
    ):
        """
        Initialize behavior executor

        Args:
            robot_ip: IP address of Go2 robot
            network_interface: Network interface connected to robot (e.g. "enp2s0").
                               If empty, auto-detected from robot_ip subnet.
            webrtc_conn/webrtc_loop: an already-connected WebRTC connection (e.g. the
                                     dashboard's camera connection) to reuse for Sport
                                     commands on the WiFi path, instead of opening a
                                     second one -- see WebRTCSportClient's docstring.
            telemetry: an optional TelemetryStreamer (dashboard_server.py) whose
                       live battery_pct should gate execution instead of
                       robot_state.battery_level's static default. Without it,
                       can_execute() falls back to that static value -- this is
                       what previously made the battery precondition a no-op
                       (RobotState.battery_level always defaulted to 100.0 and
                       nothing ever updated it).
            person_gate: an optional PersonSafetyGate (person_safety_gate.py)
                       that vetoes a forward "move" using real-time camera
                       person-detection, independent of and in addition to
                       min_obstacle_clearance -- see that module's docstring
                       for why this is not implemented as another LLM check.
        """
        self.robot_ip = robot_ip
        self.network_interface = network_interface or self._detect_interface(robot_ip)
        self.webrtc_conn = webrtc_conn
        self.webrtc_loop = webrtc_loop
        self.telemetry = telemetry
        self.person_gate = person_gate
        self.library = get_library()
        self.sport_client = None
        self.robot_state = RobotState()
        self.connected = False
        self.last_error = None
        
        # Action tracking
        self.current_action = None
        self.action_start_time = None
        self.execution_thread = None
        self.should_stop = False
        
        # Callbacks
        self.on_action_start: Optional[Callable] = None
        self.on_action_complete: Optional[Callable] = None
        self.on_action_error: Optional[Callable] = None
        
        logger.info(f"Initializing BehaviorExecutor — robot: {robot_ip}, interface: {self.network_interface}")
        self._connect()
    
    def _detect_interface(self, robot_ip: str) -> str:
        """Auto-detect network interface connected to the robot's subnet."""
        try:
            subnet = ".".join(robot_ip.split(".")[:3]) + "."  # trailing dot: "192.168.12." won't match "192.168.123."
            result = subprocess.run(
                ["ip", "-o", "-4", "addr", "show"],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                if subnet in line:
                    iface = line.split()[1]
                    logger.info(f"Auto-detected network interface: {iface}")
                    return iface
        except Exception as e:
            logger.warning(f"Interface auto-detect failed: {e}")
        logger.warning("Could not auto-detect interface — using empty string (DDS default)")
        return ""

    def _connect(self) -> bool:
        """Connect to robot via DDS SportClient (LAN) or WebRTC (Go2's own WiFi hotspot)."""
        try:
            logger.info(f"Connecting to robot at {self.robot_ip} via interface '{self.network_interface}'")

            if self.robot_ip.startswith("192.168.12."):
                # DDS RPC (SportClient/MotionSwitcherClient) never gets a response
                # over Go2's own WiFi hotspot -- confirmed via test_wifi_dds.py
                # (times out with RPC_ERR_CLIENT_SEND / 3102 even for Hello()).
                # Only the WebRTC data channel reaches the internal Sport service
                # from there, same as the official Unitree app over WiFi.
                self.sport_client = WebRTCSportClient(
                    self.robot_ip, shared_conn=self.webrtc_conn, shared_loop=self.webrtc_loop
                )
                self.sport_client.SetTimeout(10.0)
                self.sport_client.Init()
            else:
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

                # ChannelFactoryInitialize is not idempotent — ignore if already initialized
                try:
                    ChannelFactoryInitialize(0, self.network_interface)
                except Exception as e:
                    logger.info(f"ChannelFactory already initialized or minor error (continuing): {e}")

                self.sport_client = SportClient()
                self.sport_client.SetTimeout(10.0)
                self.sport_client.Init()

            time.sleep(0.5)

            self.connected = True
            self.robot_state.is_standing = True  # sport mode is active → robot is standing
            logger.info("✅ Connected to robot")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def is_state_safe(self) -> bool:
        """Check if robot is in safe state to execute actions"""
        if not self.connected:
            logger.error("Robot not connected — make sure robot is powered on at 192.168.123.18")
            return False
        if self.robot_state.error_state:
            logger.error(f"Robot in error state: {self.robot_state.error_message}")
            return False
        return True
    
    def _sync_battery_from_telemetry(self):
        """Pull live battery % from the shared TelemetryStreamer into
        robot_state.battery_level, when available and fresh. Without this,
        robot_state.battery_level never moves from RobotState's static 100.0
        default, and the min_battery precondition below can never actually
        trigger -- which is what happened before self.telemetry existed."""
        if self.telemetry is None:
            return
        try:
            data = self.telemetry.get()
        except Exception as e:
            logger.warning(f"Telemetry read failed, keeping last known battery_level: {e}")
            return
        pct, updated_at = data.get("battery_pct"), data.get("updated_at")
        if pct is None or updated_at is None:
            return
        if (time.time() - updated_at) > TELEMETRY_STALE_SECONDS:
            logger.warning(
                f"Telemetry stale ({time.time() - updated_at:.1f}s old) — "
                f"keeping last known battery_level={self.robot_state.battery_level}"
            )
            return
        self.robot_state.battery_level = pct

    def can_execute(
        self, action: ActionSchema, parameters: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if action can be executed

        Args:
            action: Action schema
            parameters: the actual runtime parameters this call will use
                        (e.g. vx for "move") -- needed by self.person_gate,
                        which cares about the specific velocity being sent,
                        not just which action schema this is.

        Returns:
            (can_execute, reason_if_not)
        """
        self._sync_battery_from_telemetry()

        if not self.is_state_safe():
            return False, "Robot not in safe state"

        if not self.robot_state.is_safe_for_action(action):
            if self.robot_state.battery_level < action.min_battery:
                return False, f"Battery too low ({self.robot_state.battery_level}% < {action.min_battery}%)"

            if action.requires_standing and not self.robot_state.is_standing:
                return False, "Robot must be standing for this action"

            if (
                action.min_obstacle_clearance is not None
                and self.robot_state.obstacle_distance_m is not None
                and self.robot_state.obstacle_distance_m < action.min_obstacle_clearance
            ):
                return False, (
                    f"Obstacle too close ({self.robot_state.obstacle_distance_m}m < "
                    f"{action.min_obstacle_clearance}m required)"
                )

            return False, "Robot state incompatible with action"

        # Real-time, non-LLM person-in-path check -- independent of
        # min_obstacle_clearance above (which is a manually-set test value,
        # not live sensing) and independent of V_i's intent-consistency
        # check (which cannot catch an instruction that is faithfully and
        # exactly carried out but is itself unsafe -- see
        # person_safety_gate.py's module docstring). Only applies to "move"
        # with a meaningful forward component.
        if self.person_gate is not None and action.name == "move":
            vx = (parameters or {}).get("vx", 0.0) or 0.0
            blocked, reason = self.person_gate.check_forward_path(vx)
            if blocked:
                return False, f"Person safety gate: {reason}"

        return True, None
    
    def _execute_sportclient_method(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> bool:
        """Execute a SportClient method. Returns True only if SDK returns code 0.
        Sets self.last_error with the real reason on failure, so callers can surface
        it instead of a generic message."""
        self.last_error = None
        try:
            if not hasattr(self.sport_client, method_name):
                self.last_error = f"'{method_name}' is not supported over the current connection (WiFi mode only exposes a subset of commands)"
                logger.warning(self.last_error)
                return False

            method = getattr(self.sport_client, method_name)

            if args or kwargs:
                result = method(*args, **kwargs)
            else:
                result = method()

            if result != 0:
                self.last_error = f"{method_name} returned error code {result}"
                logger.error(
                    f"{method_name} returned error code {result}. "
                    "Possible causes: precondition not met (battery/space/stance), "
                    "wrong api_id for the robot's current motion mode, or a DDS/WebRTC "
                    "communication failure."
                )
                return False

            logger.info(f"Executed {method_name}: OK")
            return True

        except Exception as e:
            self.last_error = f"{method_name} raised {e}"
            logger.error(f"Error executing {method_name}: {e}")
            return False
    
    def _ensure_standing(self) -> bool:
        """Ensure robot is standing before action"""
        if self.robot_state.is_standing:
            return True
        
        logger.info("Robot not standing, attempting to stand up...")
        try:
            self.sport_client.StandUp()
            time.sleep(2)
            self.robot_state.is_standing = True
            return True
        except Exception as e:
            logger.error(f"Failed to stand up: {e}")
            return False
    
    def _post_action_stand_if_needed(self, action: ActionSchema):
        """Automatically stand up after action if configured"""
        if action.auto_stand_after:
            logger.info("Auto-standing after action...")
            try:
                self.sport_client.StandUp()
                time.sleep(1)
                self.robot_state.is_standing = True
            except Exception as e:
                logger.warning(f"Auto-stand failed: {e}")
    
    def execute_action(
        self,
        action_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        wait_for_completion: bool = True
    ) -> RobotCommandResult:
        """
        Execute a behavior action
        
        Args:
            action_name: Name of action to execute
            parameters: Action parameters
            timeout: Execution timeout
            wait_for_completion: Wait for action to complete
            
        Returns:
            Execution result
        """
        # Get action schema
        action = self.library.get(action_name)
        if not action:
            return RobotCommandResult(
                success=False,
                action_name=action_name,
                error=f"Action not found: {action_name}"
            )
        
        # Check if can execute
        can_exec, reason = self.can_execute(action, parameters)
        if not can_exec:
            return RobotCommandResult(
                success=False,
                action_name=action_name,
                error=reason
            )
        
        # Ensure standing if required
        if action.requires_standing:
            if not self._ensure_standing():
                return RobotCommandResult(
                    success=False,
                    action_name=action_name,
                    error="Failed to stand up before action"
                )
        
        # Callback
        if self.on_action_start:
            self.on_action_start(action, parameters)
        
        start_time = time.time()
        result = RobotCommandResult(
            success=False,
            action_name=action_name,
            timestamp=start_time
        )
        
        try:
            logger.info(f"Executing action: {action_name}")
            
            # Merge parameters
            exec_params = parameters or {}
            
            # Build execution kwargs
            exec_kwargs = {}
            
            # Handle specific actions
            if action_name == "move":
                # Move command needs vx, vy, omega — use values directly from LLM
                vx = exec_params.get("vx", 0.3)
                vy = exec_params.get("vy", 0.0)
                omega = exec_params.get("omega", 0.0)

                duration = exec_params.get("duration", action.duration)

                # SportClient.Move() is a fire-and-forget velocity command
                # that decays after ~1s if not refreshed (see H1/G1
                # LocoClient.Move()'s default velocity duration of 1.0s) —
                # a single call followed by a long sleep only moves the
                # robot briefly, then it stalls for the rest of the sleep.
                # Resend it periodically to sustain motion for the full
                # requested duration.
                refresh_interval = 0.3
                code = self.sport_client.Move(vx, vy, omega)
                logger.info(f"Executed Move: {code}")
                elapsed = 0.0
                stopped_for_person = None
                while elapsed < duration:
                    step = min(refresh_interval, duration - elapsed)
                    time.sleep(step)
                    elapsed += step
                    if elapsed < duration:
                        # Re-check the person gate on every refresh tick, not
                        # just once before the loop started -- a person can
                        # walk closer (or the robot can close the remaining
                        # distance itself) during the up-to-several-second
                        # sustained motion below, and Move() is fire-and-forget
                        # with no closed-loop stopping distance of its own.
                        # Discovered via a live test: a person 80cm away did
                        # not trip the pre-check, then the robot's own 2s
                        # forward move closed most of that gap unchecked.
                        if self.person_gate is not None and vx >= 0.05:
                            blocked, reason = self.person_gate.check_forward_path(vx)
                            if blocked:
                                stopped_for_person = reason
                                break
                        code = self.sport_client.Move(vx, vy, omega)
                        if code != 0:
                            logger.warning(f"Move refresh returned error code {code}")
                self.sport_client.Move(0, 0, 0)

                if stopped_for_person:
                    result.success = False
                    result.error = f"Person safety gate (mid-motion): {stopped_for_person}"
                    logger.warning(f"Move stopped mid-motion: {result.error}")
                else:
                    result.success = True
                    result.message = f"Moved with vx={vx}, vy={vy}, omega={omega}"
            
            elif action_name == "euler":
                # Euler is a HELD tilt, unlike Move -- it does not auto-decay,
                # so it must be explicitly returned to neutral or the robot
                # stays leaned over after `duration` (see webrtc_sport_client.py).
                roll = exec_params.get("roll", 0.0)
                pitch = exec_params.get("pitch", 0.0)
                yaw = exec_params.get("yaw", 0.0)
                duration = exec_params.get("duration", action.duration)

                exec_success = self._execute_sportclient_method(action.method_name, roll, pitch, yaw)
                if exec_success:
                    time.sleep(duration)
                    self._execute_sportclient_method(action.method_name, 0.0, 0.0, 0.0)
                    result.success = True
                    result.message = f"Tilted with roll={roll}, pitch={pitch}, yaw={yaw}"
                else:
                    result.error = self.last_error or f"Failed to execute {action.display_name}"

            elif action_name == "speed_level":
                level = exec_params.get("level", 1)
                exec_success = self._execute_sportclient_method(action.method_name, int(level))
                if exec_success:
                    time.sleep(action.duration)
                    result.success = True
                    result.message = f"Set speed level to {level}"
                else:
                    result.error = self.last_error or f"Failed to execute {action.display_name}"

            elif action_name in ["hand_stand", "free_bound", "free_avoid", "walk_upright", "cross_step", "free_jump", "pose", "trot_run", "static_walk", "free_walk", "classic_walk"]:
                # Actions with on/off control
                exec_success = self._execute_sportclient_method(action.method_name, True)
                
                if exec_success:
                    duration = exec_params.get("duration", action.duration)
                    time.sleep(duration)
                    self._execute_sportclient_method(action.method_name, False)
                    result.success = True
                    result.message = f"Executed {action.display_name}"
                else:
                    result.error = self.last_error or f"Failed to execute {action.display_name}"

            else:
                # Simple actions — no extra args (Damp, StandUp, StandDown, etc.)
                exec_success = self._execute_sportclient_method(action.method_name)

                if exec_success:
                    result.success = True
                    result.message = f"Executed {action.display_name}"

                    # Wait for action duration
                    wait_time = action.duration
                    time.sleep(wait_time)
                else:
                    result.error = self.last_error or f"Failed to execute {action.display_name}"
            
            # Post-action operations
            self._post_action_stand_if_needed(action)
            
            # Update state based on action
            if action_name in ["stand_up", "rise_sit"]:
                self.robot_state.is_standing = True
            elif action_name in ["stand_down", "sit"]:
                self.robot_state.is_standing = False
            elif action_name == "stop_move":
                self.robot_state.is_moving = False
            
            result.duration = time.time() - start_time
            
            # Callback
            if self.on_action_complete:
                self.on_action_complete(action, result)
        
        except Exception as e:
            logger.error(f"Action execution error: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.duration = time.time() - start_time
            
            # Callback
            if self.on_action_error:
                self.on_action_error(action, e)
        
        finally:
            self.current_action = None
            self.action_start_time = None
        
        return result
    
    def execute_sequence(
        self,
        action_names: list[str],
        parameters_list: Optional[list[Dict[str, Any]]] = None,
        stop_on_error: bool = True
    ) -> list[RobotCommandResult]:
        """
        Execute a sequence of actions
        
        Args:
            action_names: List of action names
            parameters_list: Parameters for each action
            stop_on_error: Stop if any action fails
            
        Returns:
            List of execution results
        """
        results = []
        
        if parameters_list is None:
            parameters_list = [None] * len(action_names)
        
        logger.info(f"Starting action sequence: {action_names}")
        
        for i, (action_name, params) in enumerate(zip(action_names, parameters_list)):
            logger.info(f"Sequence step {i+1}/{len(action_names)}: {action_name}")
            
            result = self.execute_action(action_name, params)
            results.append(result)
            
            if not result.success and stop_on_error:
                logger.warning(f"Action failed, stopping sequence")
                break
            
            # Brief pause between actions
            time.sleep(0.5)
        
        logger.info(f"Sequence complete: {sum(1 for r in results if r.success)}/{len(results)} succeeded")
        return results
    
    def list_actions(self) -> list[str]:
        """List all available actions"""
        return self.library.list_names()
    
    def get_action_info(self, action_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about an action"""
        schema = self.library.get(action_name)
        if not schema:
            return None
        
        return {
            "name": schema.name,
            "display_name": schema.display_name,
            "description": schema.description,
            "category": schema.category.value,
            "difficulty": schema.difficulty.value,
            "duration": schema.duration,
            "requires_standing": schema.requires_standing,
            "voice_commands": schema.voice_commands,
            "min_battery": schema.min_battery,
        }
    
    def cleanup(self):
        """Clean up executor"""
        logger.info("Cleaning up BehaviorExecutor...")
        try:
            if self.sport_client:
                self.sport_client.StopMove()
            self.connected = False
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


# Test/Demo
if __name__ == "__main__":
    executor = BehaviorExecutor()
    
    print("\nAvailable Actions:")
    for name in executor.list_actions()[:5]:
        info = executor.get_action_info(name)
        print(f"  {info['display_name']}: {info['description']}")
    
    print("\nTesting action execution (demo mode)...")
    result = executor.execute_action("stand_up")
    print(f"Result: {result.success} - {result.message}")
    
    executor.cleanup()
