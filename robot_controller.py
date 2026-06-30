#!/usr/bin/env python3
"""
Robot Controller for Go2
Pipeline bridge: OpenAI LLM JSON output → BehaviorExecutor → SportClient → Robot
Architecture: 🎤 Voice → STT (Whisper) → OpenAI LLM → JSON Parser → RobotController → BehaviorExecutor → 🤖 Robot → TTS 🔊
"""

import logging
from typing import Optional, Dict, Any, List
from behavior_executor import BehaviorExecutor
from behavior_library import get_library
from action_registry import get_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotController:
    """
    High-level controller — receives parsed JSON from OpenAI LLM and
    delegates execution to BehaviorExecutor.

    Expected LLM JSON format:
    {
        "understood": true,
        "action": "free_jump",    # canonical name or display name
        "speed": 0.7,             # optional (0.0–1.0)
        "duration": 2.0,          # optional (seconds)
        "vx": 0.3,                # optional, for move forward/backward
        "vy": 0.0,                # optional, for move lateral
        "omega": 0.0,             # optional, for move rotate
        "response": "I'll jump!"  # TTS response text
    }
    """

    def __init__(self, robot_ip: str = "192.168.12.1"):
        self.robot_ip = robot_ip
        self.executor = BehaviorExecutor(robot_ip)
        self.registry = get_registry()
        self.library = get_library()

    @property
    def connected(self) -> bool:
        return self.executor.connected

    # ------------------------------------------------------------------
    # Pipeline entry point — called after OpenAI LLM returns JSON
    # ------------------------------------------------------------------

    def process_llm_command(self, llm_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse OpenAI LLM JSON and execute the corresponding robot action.
        Returns a result dict that includes the TTS 'response' text.
        """
        if not llm_json.get("understood", False):
            return {
                "success": False,
                "error": "Command not understood",
                "response": llm_json.get("response", "I did not understand that command."),
            }

        raw_action = llm_json.get("action") or llm_json.get("command")
        if not raw_action:
            return {
                "success": False,
                "error": "No action in LLM response",
                "response": llm_json.get("response", "No action to perform."),
            }

        # Resolve to canonical action name via registry
        action_name = self.registry.get_action_name(str(raw_action).strip())
        if not action_name:
            return {
                "success": False,
                "error": f"Unknown action: {raw_action}",
                "response": llm_json.get("response", f"I don't know how to {raw_action}."),
            }

        # Extract motion parameters from LLM JSON
        parameters = {
            key: llm_json[key]
            for key in ("speed", "duration", "vx", "vy", "omega")
            if key in llm_json
        }

        result = self.executor.execute_action(action_name, parameters)

        return {
            "success": result.success,
            "action": action_name,
            "message": result.message,
            "error": result.error,
            "duration": result.duration,
            "response": llm_json.get("response", ""),
        }

    # ------------------------------------------------------------------
    # Direct execution — for testing or non-LLM callers
    # Supports all 16 Go2 actions:
    #   damp, stand_up, stand_down, move, stop_move,
    #   hand_stand, balanced_stand, recovery,
    #   left_flip, back_flip,
    #   free_walk, free_bound, free_avoid,
    #   walk_upright, cross_step, free_jump
    # ------------------------------------------------------------------

    def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute action by name with optional keyword parameters."""
        action_name = self.registry.get_action_name(action)
        if not action_name:
            return {"success": False, "error": f"Unknown action: {action}"}

        parameters = {k: v for k, v in kwargs.items() if v is not None}
        result = self.executor.execute_action(action_name, parameters)

        return {
            "success": result.success,
            "action": action_name,
            "message": result.message,
            "error": result.error,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers — mirrors go2_sport_client option_list IDs
    # ------------------------------------------------------------------

    def damp(self) -> Dict[str, Any]:
        return self.execute_action("damp")

    def stand_up(self) -> Dict[str, Any]:
        return self.execute_action("stand_up")

    def stand_down(self) -> Dict[str, Any]:
        return self.execute_action("stand_down")

    def move_forward(self, speed: float = 0.3, duration: float = 2.0) -> Dict[str, Any]:
        return self.execute_action("move", vx=speed, vy=0.0, omega=0.0, duration=duration)

    def move_lateral(self, speed: float = 0.3, duration: float = 2.0) -> Dict[str, Any]:
        return self.execute_action("move", vx=0.0, vy=speed, omega=0.0, duration=duration)

    def move_rotate(self, speed: float = 0.5, duration: float = 2.0) -> Dict[str, Any]:
        return self.execute_action("move", vx=0.0, vy=0.0, omega=speed, duration=duration)

    def stop_move(self) -> Dict[str, Any]:
        return self.execute_action("stop_move")

    def hand_stand(self, duration: float = 4.0) -> Dict[str, Any]:
        return self.execute_action("hand_stand", duration=duration)

    def balanced_stand(self) -> Dict[str, Any]:
        return self.execute_action("balanced_stand")

    def recovery(self) -> Dict[str, Any]:
        return self.execute_action("recovery")

    def left_flip(self) -> Dict[str, Any]:
        return self.execute_action("left_flip")

    def back_flip(self) -> Dict[str, Any]:
        return self.execute_action("back_flip")

    def free_walk(self) -> Dict[str, Any]:
        return self.execute_action("free_walk")

    def free_bound(self, duration: float = 2.0) -> Dict[str, Any]:
        return self.execute_action("free_bound", duration=duration)

    def free_avoid(self, duration: float = 2.0) -> Dict[str, Any]:
        return self.execute_action("free_avoid", duration=duration)

    def walk_upright(self, duration: float = 4.0) -> Dict[str, Any]:
        return self.execute_action("walk_upright", duration=duration)

    def cross_step(self, duration: float = 4.0) -> Dict[str, Any]:
        return self.execute_action("cross_step", duration=duration)

    def free_jump(self, duration: float = 3.0) -> Dict[str, Any]:
        return self.execute_action("free_jump", duration=duration)

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def list_actions(self) -> List[str]:
        """All available action names."""
        return self.library.list_names()

    def get_action_info(self, action: str) -> Optional[Dict[str, Any]]:
        """Get schema info for an action."""
        return self.executor.get_action_info(action)

    def cleanup(self):
        self.executor.cleanup()


# ------------------------------------------------------------------
# Quick smoke test (no real robot needed for schema/registry checks)
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Testing RobotController (schema + registry only)")

    from behavior_library import get_library as _lib
    library = _lib()
    registry = get_registry()

    print(f"\nTotal actions in library : {len(library.list_names())}")
    print("Actions:")
    for name in library.list_names():
        schema = library.get(name)
        print(f"  mode_id={schema.mode_id:2d}  {schema.name:<20} ({schema.difficulty.value})")

    print("\nLLM JSON parse test:")
    sample_llm_outputs = [
        {"understood": True, "action": "stand_up", "response": "Standing up now!"},
        {"understood": True, "action": "free_jump", "duration": 3.0, "response": "Jumping!"},
        {"understood": True, "action": "move forward", "speed": 0.3, "duration": 2.0, "response": "Moving forward."},
        {"understood": False, "response": "I don't understand."},
    ]

    for llm_json in sample_llm_outputs:
        action_raw = llm_json.get("action", "—")
        resolved = registry.get_action_name(str(action_raw)) if "action" in llm_json else None
        status = "✓" if resolved else ("✗" if "action" in llm_json else "—")
        print(f"  {status} '{action_raw}' → {resolved}")
