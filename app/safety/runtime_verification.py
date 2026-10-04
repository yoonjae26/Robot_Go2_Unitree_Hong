#!/usr/bin/env python3
"""
Sequential runtime verification pipeline: V_s (schema/parameter) -> V_i
(intent consistency) -> V_c (context-aware gating), matching Algorithm 1 in
the thesis's Runtime Command Verification section.

`verify_sequence()` is the single choke point both dashboard_server.py
execution paths (auto-execute voice path and the confirm-gated text-panel
path) call before any action in a generated sequence reaches the robot. It
evaluates the three layers in cost order -- schema is a local check, intent
costs an LLM call, context costs a telemetry read -- and stops at the first
rejecting layer (short-circuit), so a command already rejected by a cheap
layer never pays for an expensive one.

Each layer's decision is independent of the others: disabling a layer via
VerificationConfig makes it trivially pass (not "skipped" in a way that
would be indistinguishable from a very fast accept), which is what lets the
same code path serve all of the paper's C0-C3 configurations.

The pre-existing action-name whitelist (action_registry.get_action_name) is
NOT part of what VerificationConfig can disable -- it is baseline
infrastructure exercised by the schema pass regardless of config, and stays
active even in C0. See verification_config.py.
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.action_registry import get_registry
from app.core.behavior_library import get_library
from app.core.behavior_schema import RobotState
from app.safety.intent_verifier import verify_intent
from app.safety.verification_config import VerificationConfig, get_verification_config

logger = logging.getLogger(__name__)

# How old live telemetry is allowed to be before the context layer stops
# trusting it and falls back to the conservative static RobotState value.
TELEMETRY_STALE_SECONDS = 10.0

# Actions that never go through BehaviorExecutor/ActionSchema at all --
# navigate_to (slam_navigator.py) and vision_analyze (camera Q&A) are
# special-cased out of the normal execution loop by both dashboard_server.py
# call sites (see _llm_and_execute and _execute_pending). They have no
# ActionSchema to resolve against, so the schema/context passes below must
# not try to look one up -- doing so would reject every navigate_to command
# as "unknown action", which is what an earlier version of this file did.
# _llm_and_execute filters vision_analyze/navigate_to out before building
# `merged`, but _execute_pending's `pending["actions"]` (staged from the
# text-command panel) can still contain a bare "navigate_to" -- see its
# `if name == "navigate_to":` branch -- so both call sites route through the
# same verify_sequence() and this exemption has to hold for either.
PSEUDO_ACTIONS = {"navigate_to", "vision_analyze"}


@dataclass
class VerificationResult:
    accepted: bool
    reject_layer: Optional[str] = None  # "schema" | "intent" | "intent_infra_error" | "context" | None
    reject_reason: Optional[str] = None
    schema_ms: float = 0.0
    intent_ms: float = 0.0
    context_ms: float = 0.0
    # Canonical action names resolved during the schema pass (falls back to
    # the raw LLM-provided name for anything that didn't resolve), useful
    # for logging even when actions is empty or verification rejected early.
    resolved_actions: List[str] = field(default_factory=list)
    # The battery percentage the context layer actually used for its
    # decision (live telemetry, or the static RobotState fallback -- see
    # _live_battery()), populated whenever config.context_enabled ran,
    # regardless of accept/reject. Callers should log this instead of
    # robot_state.battery_level directly: that static field is only ever
    # refreshed by BehaviorExecutor.can_execute() as a side effect of
    # actually attempting execute_action(), so for a command rejected before
    # execution it still holds a stale/default value even though this
    # decision used the real live reading.
    live_battery_pct: Optional[float] = None
    # Latency of the always-on, non-LLM person-in-path pre-check (see
    # _check_person_gate). 0.0 when no person_gate was supplied to
    # verify_sequence() (e.g. the dry-run benchmark harness, which has no
    # camera) or no "move" action was present to check.
    person_gate_ms: float = 0.0


def _check_schema(
    actions: List[Tuple[str, Dict[str, Any]]], clamp: bool
) -> Tuple[bool, Optional[str], float]:
    t0 = time.monotonic()
    for raw_name, params in actions:
        if raw_name in PSEUDO_ACTIONS:
            continue

        canonical = get_registry().get_action_name(raw_name) if raw_name else None
        if not canonical:
            return False, f"unknown action: {raw_name!r}", (time.monotonic() - t0) * 1000

        schema = get_library().get(canonical)
        if schema is None:
            # Registry resolved a name the library has no schema for --
            # shouldn't happen since both are built from the same library,
            # but treated as a schema failure rather than a crash.
            return False, f"no schema for resolved action: {canonical!r}", (time.monotonic() - t0) * 1000

        for param in schema.parameters:
            if param.name not in params or params[param.name] is None:
                continue  # not supplied by the LLM -- execution falls back to its own default, unchanged
            value = params[param.name]
            if param.validate(value):
                continue
            if clamp and isinstance(value, (int, float)) and param.min_value is not None and param.max_value is not None:
                clamped = max(param.min_value, min(param.max_value, value))
                logger.info(f"Schema layer: clamping {canonical}.{param.name} {value!r} -> {clamped}")
                params[param.name] = clamped
                continue
            return False, (
                f"{canonical}.{param.name}={value!r} outside allowed range "
                f"[{param.min_value}, {param.max_value}]"
            ), (time.monotonic() - t0) * 1000

    return True, None, (time.monotonic() - t0) * 1000


def _live_battery(robot_state: RobotState, telemetry) -> float:
    """Best-effort live battery reading, falling back to the static
    RobotState value when telemetry is unavailable or stale. Mirrors
    BehaviorExecutor.can_execute()'s per-action check (see behavior_executor.py)
    -- duplicated here deliberately as defense-in-depth so the whole
    sequence is checked before any action starts, not just each action as
    it's about to run."""
    if telemetry is None:
        return robot_state.battery_level
    try:
        data = telemetry.get()
    except Exception as e:
        logger.warning(f"Context gate: telemetry.get() failed ({e}), using static battery_level")
        return robot_state.battery_level

    pct, updated_at = data.get("battery_pct"), data.get("updated_at")
    if pct is None or updated_at is None:
        return robot_state.battery_level
    if (time.time() - updated_at) > TELEMETRY_STALE_SECONDS:
        logger.warning(
            f"Context gate: telemetry is stale ({time.time() - updated_at:.1f}s old), "
            f"falling back to static battery_level={robot_state.battery_level}"
        )
        return robot_state.battery_level
    return pct


def _check_context(
    actions: List[Tuple[str, Dict[str, Any]]], robot_state: RobotState, telemetry,
) -> Tuple[bool, Optional[str], float, float]:
    t0 = time.monotonic()
    battery = _live_battery(robot_state, telemetry)

    for raw_name, _params in actions:
        if raw_name in PSEUDO_ACTIONS:
            continue
        canonical = get_registry().get_action_name(raw_name) if raw_name else None
        schema = get_library().get(canonical) if canonical else None
        if schema is None:
            continue  # unresolvable names are the schema layer's job to catch

        if battery < schema.min_battery:
            return False, (
                f"battery too low for {canonical}: {battery}% < {schema.min_battery}% required"
            ), (time.monotonic() - t0) * 1000, battery

        if (
            schema.min_obstacle_clearance is not None
            and robot_state.obstacle_distance_m is not None
            and robot_state.obstacle_distance_m < schema.min_obstacle_clearance
        ):
            return False, (
                f"obstacle too close for {canonical}: "
                f"{robot_state.obstacle_distance_m}m < {schema.min_obstacle_clearance}m required"
            ), (time.monotonic() - t0) * 1000, battery

    return True, None, (time.monotonic() - t0) * 1000, battery


def _check_person_gate(
    actions: List[Tuple[str, Dict[str, Any]]], person_gate,
) -> Tuple[bool, Optional[str], float]:
    """Sequence-level pre-check mirror of BehaviorExecutor.can_execute()'s
    person_gate call -- run here too so a person already visible and too
    close is caught BEFORE the TTS response is spoken, not just at
    execution time. The execution-time check stays as defense-in-depth for
    a person who steps into frame in the gap between this check and actual
    execution.

    Deliberately NOT gated by VerificationConfig.context_enabled (unlike
    _check_context above) -- like the action-name whitelist described in
    this module's docstring, this is always-on infrastructure independent
    of the C0-C3 ablation, precisely because it does not depend on any LLM
    judgment at all (see person_safety_gate.py's module docstring for why
    that independence is the point). Skips entirely (returns ok=True) when
    person_gate is None, so callers that don't have one (e.g. the
    benchmark harness's dry-run tier, which has no camera) are unaffected.
    """
    t0 = time.monotonic()
    if person_gate is None:
        return True, None, 0.0

    for raw_name, params in actions:
        if raw_name in PSEUDO_ACTIONS:
            continue
        canonical = get_registry().get_action_name(raw_name) if raw_name else None
        if canonical != "move":
            continue
        vx = (params or {}).get("vx", 0.0) or 0.0
        blocked, reason = person_gate.check_forward_path(vx)
        if blocked:
            return False, f"person safety gate: {reason}", (time.monotonic() - t0) * 1000

    return True, None, (time.monotonic() - t0) * 1000


def verify_sequence(
    instruction: str,
    actions: List[Tuple[str, Dict[str, Any]]],
    robot_state: RobotState,
    config: Optional[VerificationConfig] = None,
    telemetry=None,
    person_gate=None,
) -> VerificationResult:
    """Run the sequential V_s -> V_i -> V_c pipeline over a whole generated
    action sequence. `actions` is the (name, params) list already produced
    by action_merge.merge_consecutive_moves() + apply_tone_to_moves() --
    names are raw/LLM-provided, not necessarily canonical (resolved here).
    Mutates `actions`' param dicts in place when config.clamp_instead_of_reject
    is set (clamped values are what should then be passed to execution).
    """
    config = config or get_verification_config()
    resolved = [get_registry().get_action_name(n) or n for n, _p in actions]
    result = VerificationResult(accepted=True, resolved_actions=resolved)

    if config.schema_enabled:
        ok, reason, ms = _check_schema(actions, config.clamp_instead_of_reject)
        result.schema_ms = ms
        if not ok:
            result.accepted = False
            result.reject_layer = "schema"
            result.reject_reason = reason
            return result

    # Always-on, independent of config -- see _check_person_gate's docstring.
    # Placed before the (costly, LLM-based) intent check both on safety
    # grounds -- a hard, non-LLM veto should not depend on an LLM call
    # succeeding first -- and to avoid paying for that call when this
    # already vetoes the sequence.
    ok, reason, ms = _check_person_gate(actions, person_gate)
    result.person_gate_ms = ms
    if not ok:
        result.accepted = False
        result.reject_layer = "person_gate"
        result.reject_reason = reason
        return result

    if config.intent_enabled:
        verdict = verify_intent(instruction, actions)
        result.intent_ms = verdict.latency_ms
        if not verdict.accepted:
            result.accepted = False
            result.reject_layer = "intent_infra_error" if verdict.infra_error else "intent"
            result.reject_reason = verdict.reason
            return result

    if config.context_enabled:
        ok, reason, ms, battery = _check_context(actions, robot_state, telemetry)
        result.context_ms = ms
        result.live_battery_pct = battery
        if not ok:
            result.accepted = False
            result.reject_layer = "context"
            result.reject_reason = reason
            return result

    return result
