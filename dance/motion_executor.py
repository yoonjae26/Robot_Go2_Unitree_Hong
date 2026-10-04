#!/usr/bin/env python3
"""
Motion Executor
Dance Timeline -> real-time robot commands, timed against a monotonic clock.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # for robot_controller, behavior_executor, ...

from dance_timeline import DanceCue, DanceTimeline

logger = logging.getLogger(__name__)


class MotionExecutor:
    def __init__(self, robot_ip: str, webrtc_conn=None, webrtc_loop=None):
        from robot_controller import RobotController
        self.robot_ip = robot_ip
        self.robot = RobotController(robot_ip=robot_ip, webrtc_conn=webrtc_conn, webrtc_loop=webrtc_loop)
        # A gesture action (e.g. dance1, ~8s) can easily outlast the interval
        # between cues (a beat at 132 BPM is ~0.45s). Without this, every cue
        # that arrives mid-gesture would fire its own thread on top of the
        # still-running one -- overlapping calls into the same sport_client
        # pile up and the robot visibly gets stuck re-triggering the same move.
        # Only one gesture action runs at a time; cues that land while busy are
        # skipped (not queued) so the routine stays locked to the music instead
        # of drifting later and later.
        self._action_busy = threading.Event()

    def prepare(self):
        """Stand up and balance before dancing."""
        logger.info("Preparing: stand up + balance...")
        self.robot.execute_action("stand_up")
        time.sleep(1.5)

    def finish(self):
        """Settle back into a stable stance after the routine ends."""
        logger.info("Finishing: stop + recovery stand...")
        # Euler is a held pose, not a decaying velocity like Move -- if the
        # routine ends mid-rock the body stays tilted until told otherwise.
        try:
            self.robot.executor.sport_client.Euler(0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            self.robot.executor.sport_client.StopMove()
        except Exception:
            pass
        self.robot.execute_action("recovery")

    def execute(self, timeline: DanceTimeline, prepare: bool = True, finish: bool = True):
        """Run every cue at its scheduled time, relative to now."""
        if prepare:
            self.prepare()

        logger.info(
            f"Dancing '{timeline.song_name}' — {len(timeline.cues)} cues, "
            f"{timeline.total_duration():.1f}s, tempo={timeline.tempo:.1f} BPM"
        )
        start = time.monotonic()
        for cue in timeline.cues:
            target = start + cue.time
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._fire(cue)

        if finish:
            self.finish()

    def _fire(self, cue: DanceCue):
        if cue.kind == "action":
            if self._action_busy.is_set():
                logger.debug(f"Skipping '{cue.action}' — previous action still running")
                return
            # A schema trick (Dance1, Pose, BackFlip...) starting mid-rock
            # would launch from a tilted body -- untilt first so accents
            # always fire from a known-neutral pose.
            self._reset_euler()
            self._action_busy.set()
            threading.Thread(target=self._run_action_safe, args=(cue.action,), daemon=True).start()
        elif cue.kind == "sway":
            if not self._action_busy.is_set():
                self._sway(cue.params)
            # else: a gesture is mid-flight -- skip the sway so it doesn't step on it
        elif cue.kind == "rock":
            if not self._action_busy.is_set():
                self._rock(cue.params)
            # else: same as sway -- don't fight a gesture that's already running
        elif cue.kind == "stop":
            self._stop()

    def _run_action_safe(self, action_name: str):
        try:
            result = self.robot.execute_action(action_name)
            if not result.get("success"):
                logger.warning(f"Dance action '{action_name}' failed: {result.get('error')}")
        except Exception as e:
            logger.warning(f"Dance action '{action_name}' crashed: {e}")
        finally:
            self._action_busy.clear()

    def _sway(self, params: dict):
        try:
            self.robot.executor.sport_client.Move(
                params.get("vx", 0.0), params.get("vy", 0.0), params.get("omega", 0.0)
            )
        except Exception as e:
            logger.warning(f"Sway command failed: {e}")

    def _rock(self, params: dict):
        """Continuous body-tilt groove (Euler roll/pitch/yaw), refreshed every
        beat like _sway()'s Move() calls -- bypasses the action-schema/
        cooldown system entirely (see webrtc_sport_client.py's Euler docstring
        note: it holds a pose rather than decaying, so the timeline must keep
        calling this or explicitly reset to neutral, never just stop and
        assume the robot un-tilts on its own)."""
        try:
            self.robot.executor.sport_client.Euler(
                params.get("roll", 0.0), params.get("pitch", 0.0), params.get("yaw", 0.0)
            )
        except Exception as e:
            logger.warning(f"Rock (Euler) command failed: {e}")

    def _reset_euler(self):
        try:
            self.robot.executor.sport_client.Euler(0.0, 0.0, 0.0)
        except Exception as e:
            logger.warning(f"Euler reset failed: {e}")

    def _stop(self):
        try:
            self.robot.executor.sport_client.StopMove()
        except Exception as e:
            logger.warning(f"StopMove failed: {e}")
