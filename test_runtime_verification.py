#!/usr/bin/env python3
"""
Tests for the three runtime verification layers (V_s, V_i, V_c) and their
orchestration in runtime_verification.py. Follows this repo's existing
plain-assert, `python test_x.py` style (see test_go2_behaviors.py) rather
than pytest/unittest, so it needs no extra dependency.

None of these tests touch the physical robot or make a real network call --
the intent layer is exercised by monkeypatching runtime_verification.verify_intent
with a stub, and the context layer's telemetry is a small fake object.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_schema_layer_rejects_out_of_range_param():
    print("\n" + "=" * 70)
    print("TEST: Schema layer (V_s) -- parameter range")
    print("=" * 70)

    from runtime_verification import verify_sequence
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=True, intent_enabled=False, context_enabled=False)
    state = RobotState(is_standing=True, battery_level=100.0)

    # vx=99 is far outside MOVE_SCHEMA's VX_PARAMETER range (-1.0, 1.0).
    bad = verify_sequence("move forward", [("move", {"vx": 99.0, "vy": 0.0, "omega": 0.0})], state, config)
    assert bad.accepted is False, "expected out-of-range vx to be rejected"
    assert bad.reject_layer == "schema"
    print(f"✅ vx=99 rejected: {bad.reject_reason}")

    good = verify_sequence("move forward", [("move", {"vx": 0.3, "vy": 0.0, "omega": 0.0})], state, config)
    assert good.accepted is True, f"expected valid move to pass, got: {good.reject_reason}"
    print("✅ vx=0.3 accepted")


def test_schema_layer_rejects_unknown_action():
    print("\n" + "=" * 70)
    print("TEST: Schema layer (V_s) -- unknown/hallucinated action")
    print("=" * 70)

    from runtime_verification import verify_sequence
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=True, intent_enabled=False, context_enabled=False)
    state = RobotState(is_standing=True, battery_level=100.0)

    result = verify_sequence("do a barrel roll", [("do_a_barrel_roll", {})], state, config)
    assert result.accepted is False
    assert result.reject_layer == "schema"
    print(f"✅ hallucinated action rejected: {result.reject_reason}")


def test_schema_layer_ignores_pseudo_actions():
    print("\n" + "=" * 70)
    print("TEST: Schema layer (V_s) -- navigate_to/vision_analyze are exempt")
    print("=" * 70)

    from runtime_verification import verify_sequence
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=True, intent_enabled=False, context_enabled=False)
    state = RobotState(is_standing=True, battery_level=100.0)

    result = verify_sequence("go to the kitchen", [("navigate_to", {"location": "kitchen"})], state, config)
    assert result.accepted is True, f"navigate_to should not be schema-checked, got: {result.reject_reason}"
    print("✅ navigate_to passes the schema layer untouched")


def test_schema_layer_can_be_disabled():
    print("\n" + "=" * 70)
    print("TEST: C0 baseline -- schema layer disabled lets bad params through")
    print("=" * 70)

    from runtime_verification import verify_sequence
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=False, intent_enabled=False, context_enabled=False)
    state = RobotState(is_standing=True, battery_level=100.0)

    result = verify_sequence("move forward", [("move", {"vx": 99.0})], state, config)
    assert result.accepted is True, "C0 should not run parameter checks at all"
    print("✅ C0 (all layers off) accepts an out-of-range command, as intended for the baseline")


def test_intent_layer_accept_and_reject():
    print("\n" + "=" * 70)
    print("TEST: Intent layer (V_i) -- consistency check via stub")
    print("=" * 70)

    import runtime_verification
    from verification_config import VerificationConfig
    from behavior_schema import RobotState
    from intent_verifier import IntentVerdict

    config = VerificationConfig(schema_enabled=False, intent_enabled=True, context_enabled=False)
    state = RobotState(is_standing=True, battery_level=100.0)
    calls = []

    def fake_accept(instruction, actions, model=None, client=None):
        calls.append(instruction)
        return IntentVerdict(accepted=True, reason="matches", latency_ms=1.0)

    def fake_reject(instruction, actions, model=None, client=None):
        calls.append(instruction)
        return IntentVerdict(accepted=False, reason="user asked to stay still", latency_ms=1.0)

    orig = runtime_verification.verify_intent
    try:
        runtime_verification.verify_intent = fake_accept
        ok = runtime_verification.verify_sequence("stand still", [("stand_up", {})], state, config)
        assert ok.accepted is True
        print("✅ intent accept path works")

        runtime_verification.verify_intent = fake_reject
        bad = runtime_verification.verify_sequence("stand still", [("move", {"vx": 0.3})], state, config)
        assert bad.accepted is False
        assert bad.reject_layer == "intent"
        print(f"✅ intent reject path works: {bad.reject_reason}")
    finally:
        runtime_verification.verify_intent = orig

    assert len(calls) == 2, "expected exactly 2 intent-verifier calls"


def test_intent_layer_skipped_when_disabled():
    print("\n" + "=" * 70)
    print("TEST: Intent layer (V_i) -- disabled config makes zero calls")
    print("=" * 70)

    import runtime_verification
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=False, intent_enabled=False, context_enabled=False)
    state = RobotState(is_standing=True, battery_level=100.0)
    calls = []

    def fake_verify_intent(instruction, actions, model=None, client=None):
        calls.append(instruction)
        raise AssertionError("verify_intent should not be called when intent_enabled=False")

    orig = runtime_verification.verify_intent
    try:
        runtime_verification.verify_intent = fake_verify_intent
        result = runtime_verification.verify_sequence("anything", [("stand_up", {})], state, config)
        assert result.accepted is True
    finally:
        runtime_verification.verify_intent = orig

    assert len(calls) == 0
    print("✅ intent_enabled=False makes zero verifier calls")


def test_context_layer_battery_and_obstacle():
    print("\n" + "=" * 70)
    print("TEST: Context layer (V_c) -- battery and obstacle gating")
    print("=" * 70)

    from runtime_verification import verify_sequence
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=False, intent_enabled=False, context_enabled=True)

    low_batt = RobotState(is_standing=True, battery_level=5.0)  # back_flip needs 20%
    result = verify_sequence("do a back flip", [("back_flip", {})], low_batt, config)
    assert result.accepted is False
    assert result.reject_layer == "context"
    print(f"✅ low battery rejected: {result.reject_reason}")

    ok_batt = RobotState(is_standing=True, battery_level=100.0)
    result = verify_sequence("do a back flip", [("back_flip", {})], ok_batt, config)
    assert result.accepted is True
    print("✅ sufficient battery accepted")

    close_obstacle = RobotState(is_standing=True, battery_level=100.0, obstacle_distance_m=0.1)
    result = verify_sequence("move forward", [("move", {"vx": 0.3})], close_obstacle, config)
    assert result.accepted is False
    assert result.reject_layer == "context"
    print(f"✅ obstacle within clearance rejected: {result.reject_reason}")

    clear_path = RobotState(is_standing=True, battery_level=100.0, obstacle_distance_m=2.0)
    result = verify_sequence("move forward", [("move", {"vx": 0.3})], clear_path, config)
    assert result.accepted is True
    print("✅ clear path accepted")


def test_context_layer_uses_live_telemetry():
    print("\n" + "=" * 70)
    print("TEST: Context layer (V_c) -- live telemetry overrides stale/static battery")
    print("=" * 70)

    from runtime_verification import verify_sequence
    from verification_config import VerificationConfig
    from behavior_schema import RobotState

    config = VerificationConfig(schema_enabled=False, intent_enabled=False, context_enabled=True)
    # Static default is 100% (would pass) -- live telemetry says 5%. This is
    # exactly the bug that made battery gating a no-op before
    # BehaviorExecutor.telemetry existed: robot_state.battery_level never
    # moved off its dataclass default.
    state = RobotState(is_standing=True, battery_level=100.0)

    class FakeTelemetry:
        def __init__(self, pct, age_s):
            self._pct, self._age = pct, age_s

        def get(self):
            return {"battery_pct": self._pct, "updated_at": time.time() - self._age}

    fresh_low = FakeTelemetry(pct=5.0, age_s=1.0)
    result = verify_sequence("do a back flip", [("back_flip", {})], state, config, telemetry=fresh_low)
    assert result.accepted is False, "fresh live telemetry should override the static 100% default"
    print(f"✅ fresh live telemetry (5%) overrides static default: {result.reject_reason}")

    stale_low = FakeTelemetry(pct=5.0, age_s=60.0)
    result = verify_sequence("do a back flip", [("back_flip", {})], state, config, telemetry=stale_low)
    assert result.accepted is True, "stale telemetry (>10s old) should fall back to the static default"
    print("✅ stale live telemetry (60s old) is ignored, falls back to static default")


def test_short_circuit_order():
    print("\n" + "=" * 70)
    print("TEST: Sequential short-circuit -- schema failure skips the intent layer")
    print("=" * 70)

    import runtime_verification
    from verification_config import VerificationConfig
    from behavior_schema import RobotState
    from intent_verifier import IntentVerdict

    config = VerificationConfig(schema_enabled=True, intent_enabled=True, context_enabled=True)
    state = RobotState(is_standing=True, battery_level=100.0)
    calls = []

    def fake_verify_intent(instruction, actions, model=None, client=None):
        calls.append(instruction)
        return IntentVerdict(accepted=True, reason="ok", latency_ms=1.0)

    orig = runtime_verification.verify_intent
    try:
        runtime_verification.verify_intent = fake_verify_intent
        result = runtime_verification.verify_sequence(
            "do a barrel roll", [("do_a_barrel_roll", {})], state, config
        )
        assert result.accepted is False
        assert result.reject_layer == "schema"
        assert len(calls) == 0, "intent layer should never be reached after a schema rejection"
    finally:
        runtime_verification.verify_intent = orig
    print("✅ schema rejection short-circuits before the intent layer runs")


def test_person_gate_blocks_before_intent_and_is_config_independent():
    print("\n" + "=" * 70)
    print("TEST: Person safety gate -- always-on pre-check, independent of config")
    print("=" * 70)

    import runtime_verification
    from verification_config import VerificationConfig
    from behavior_schema import RobotState
    from intent_verifier import IntentVerdict

    state = RobotState(is_standing=True, battery_level=100.0)
    intent_calls = []

    def fake_verify_intent(instruction, actions, model=None, client=None):
        intent_calls.append(instruction)
        return IntentVerdict(accepted=True, reason="ok", latency_ms=1.0)

    class BlockingGate:
        def check_forward_path(self, vx):
            return True, "mannequin 60% of frame, centered"

    class ClearGate:
        def check_forward_path(self, vx):
            return False, None

    orig = runtime_verification.verify_intent
    try:
        runtime_verification.verify_intent = fake_verify_intent

        # C0-equivalent (everything disabled) -- person gate still fires.
        c0 = VerificationConfig(schema_enabled=False, intent_enabled=False, context_enabled=False)
        result = runtime_verification.verify_sequence(
            "walk forward", [("move", {"vx": 0.3})], state, c0, person_gate=BlockingGate()
        )
        assert result.accepted is False
        assert result.reject_layer == "person_gate"
        assert len(intent_calls) == 0, "person gate should short-circuit before the (disabled) intent layer is even reached"
        print(f"✅ blocks even under C0 (all ablatable layers off): {result.reject_reason}")

        # Full config, but a clear gate -- must not block.
        c3 = VerificationConfig(schema_enabled=True, intent_enabled=True, context_enabled=True)
        result = runtime_verification.verify_sequence(
            "walk forward", [("move", {"vx": 0.3})], state, c3, person_gate=ClearGate()
        )
        assert result.accepted is True, f"expected a clear path to be accepted, got: {result.reject_reason}"
        print("✅ clear path is accepted")

        # No person_gate supplied at all (e.g. dry-run harness with no camera) -- must not block or crash.
        result = runtime_verification.verify_sequence(
            "walk forward", [("move", {"vx": 0.3})], state, c3, person_gate=None
        )
        assert result.accepted is True
        print("✅ person_gate=None (no camera) does not block or crash")
    finally:
        runtime_verification.verify_intent = orig


if __name__ == "__main__":
    test_schema_layer_rejects_out_of_range_param()
    test_schema_layer_rejects_unknown_action()
    test_schema_layer_ignores_pseudo_actions()
    test_schema_layer_can_be_disabled()
    test_intent_layer_accept_and_reject()
    test_intent_layer_skipped_when_disabled()
    test_context_layer_battery_and_obstacle()
    test_context_layer_uses_live_telemetry()
    test_short_circuit_order()
    test_person_gate_blocks_before_intent_and_is_config_independent()
    print("\n" + "=" * 70)
    print("ALL RUNTIME VERIFICATION TESTS PASSED")
    print("=" * 70)
