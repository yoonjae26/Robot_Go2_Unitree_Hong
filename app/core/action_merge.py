#!/usr/bin/env python3
"""
Merge consecutive "move" actions from an LLM action list into a single
call so the robot doesn't stop-and-restart its gait between steps that
go in the same direction (e.g. "walk forward 2 steps" split into two
move actions by the LLM).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_MOVE_DURATION = 2.0  # matches MOVE_SCHEMA.duration in behavior_schema.py


def normalize_actions(
    raw_actions: List[Any], command_dict: Dict[str, Any]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Convert the LLM's raw actions list (objects or plain strings) into
    a uniform list of (action_name, params) tuples."""
    normalized = []
    for item in raw_actions:
        if isinstance(item, dict):
            action_name = item.get("name") or item.get("action") or ""
            params = {k: v for k, v in item.items() if k not in ("name", "action")}
        else:
            action_name = str(item)
            params = {
                "speed": command_dict.get("speed", 0.5),
                "duration": command_dict.get("duration", 3.0),
                "vx": command_dict.get("vx"),
                "vy": command_dict.get("vy"),
                "omega": command_dict.get("omega"),
            }
        normalized.append((action_name, params))
    return normalized


def _move_duration(params: Dict[str, Any]) -> float:
    duration = params.get("duration")
    return float(duration) if duration is not None else DEFAULT_MOVE_DURATION


def _same_direction(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-6) -> bool:
    for key in ("vx", "vy", "omega"):
        av = a.get(key) or 0.0
        bv = b.get(key) or 0.0
        if abs(av - bv) > tol:
            return False
    return True


def merge_consecutive_moves(
    raw_actions: List[Any],
    command_dict: Dict[str, Any],
    get_canonical_name: Callable[[str], Optional[str]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Normalize the raw actions list and merge consecutive "move" actions
    that share the same direction (vx, vy, omega) into a single move with
    the summed duration. Non-move actions and direction changes break the
    merge chain — the robot only ever gets one continuous Move() call per
    contiguous same-direction stretch instead of stop/restart per step.
    """
    normalized = normalize_actions(raw_actions, command_dict)

    merged: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
    for action_name, params in normalized:
        canonical = get_canonical_name(action_name) if action_name else None

        if (
            canonical == "move"
            and merged
            and merged[-1][2] == "move"
            and _same_direction(merged[-1][1], params)
        ):
            prev_name, prev_params, prev_canonical = merged[-1]
            prev_params["duration"] = _move_duration(prev_params) + _move_duration(params)
            continue

        merged.append((action_name, dict(params), canonical))

    return [(name, params) for name, params, _ in merged]
