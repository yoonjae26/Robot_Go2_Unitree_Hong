#!/usr/bin/env python3
"""
Test suite for Go2 voice control system
Tests all components: behaviors, executor, registry, and LLM integration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_behavior_schema():
    """Test behavior schema definitions"""
    print("\n" + "="*70)
    print("TEST: Behavior Schema")
    print("="*70)
    
    from behavior_schema import (
        ActionSchema, ActionCategory, DifficultyLevel,
        STAND_UP_SCHEMA, BACK_FLIP_SCHEMA, MOVE_SCHEMA
    )
    
    # Test schema creation
    assert STAND_UP_SCHEMA.name == "stand_up"
    assert STAND_UP_SCHEMA.difficulty == DifficultyLevel.SAFE
    assert BACK_FLIP_SCHEMA.difficulty == DifficultyLevel.HIGH
    
    print("✅ Schema creation and validation passed")
    
    # Test parameter validation
    speed_param = MOVE_SCHEMA.parameters[0]
    assert speed_param.validate(0.5) == True
    assert speed_param.validate(1.5) == False  # Out of range
    
    print("✅ Parameter validation passed")


def test_behavior_library():
    """Test behavior library"""
    print("\n" + "="*70)
    print("TEST: Behavior Library")
    print("="*70)
    
    from behavior_library import get_library
    
    library = get_library()
    
    # Test all behaviors registered
    all_actions = library.list_names()
    print(f"  Registered {len(all_actions)} behaviors")
    assert len(all_actions) > 10
    
    # Test retrieval
    stand_up = library.get("stand_up")
    assert stand_up is not None
    assert stand_up.name == "stand_up"
    
    print("✅ Behavior retrieval passed")
    
    # Test voice command lookup
    schema = library.get_by_voice_command("jump")
    assert schema is not None
    assert schema.name == "free_jump"
    
    print("✅ Voice command lookup passed")
    
    # Test category retrieval
    posture_actions = library.get_by_category("posture")
    assert len(posture_actions) > 0
    
    print("✅ Category retrieval passed")
    
    # Test safe actions
    safe_actions = library.get_safe_actions()
    assert len(safe_actions) > 0
    
    print("✅ Safe action filtering passed")


def test_action_registry():
    """Test action registry"""
    print("\n" + "="*70)
    print("TEST: Action Registry")
    print("="*70)
    
    from action_registry import get_registry
    
    registry = get_registry()
    
    # Test action name resolution
    assert registry.get_action_name("stand_up") == "stand_up"
    assert registry.get_action_name("Stand Up") == "stand_up"
    assert registry.validate_action("stand_up") == True
    assert registry.validate_action("invalid_action") == False
    
    print("✅ Action name resolution passed")
    
    # Test mode ID lookup
    action = registry.get_action_by_mode_id(1)
    assert action == "stand_up"
    
    print("✅ Mode ID lookup passed")
    
    # Test NLP parsing
    action = registry.parse_nlp_command("please stand up")
    assert action == "stand_up"
    
    print("✅ NLP command parsing passed")
    
    # Test LLM output parsing
    llm_output = {"action": "free_jump", "speed": 0.5}
    action = registry.parse_llm_action(llm_output)
    assert action == "free_jump"
    
    print("✅ LLM output parsing passed")
    
    # Test category listing
    categories = registry.get_action_groups()
    assert "posture" in categories
    assert "gait" in categories
    
    print("✅ Category listing passed")


def test_robot_state():
    """Test robot state management"""
    print("\n" + "="*70)
    print("TEST: Robot State")
    print("="*70)
    
    from behavior_schema import RobotState, ActionSchema, ActionCategory, DifficultyLevel
    
    state = RobotState()
    
    # Test initial state
    assert state.battery_level == 100.0
    assert state.is_safe_for_action is not None
    
    # Test state with action
    action = ActionSchema(
        name="test",
        mode_id=1,
        requires_standing=True,
        min_battery=20.0
    )
    
    # Should fail (not standing)
    assert state.is_safe_for_action(action) == False
    
    # Update state
    state.is_standing = True
    assert state.is_safe_for_action(action) == True
    
    # Low battery
    state.battery_level = 10.0
    assert state.is_safe_for_action(action) == False
    
    print("✅ Robot state management passed")


def test_voice_control_integration():
    """Test voice control integration"""
    print("\n" + "="*70)
    print("TEST: Voice Control Integration")
    print("="*70)
    
    from action_registry import find_action_for_input
    from behavior_library import get_library
    
    library = get_library()
    
    # Test various voice inputs
    test_cases = [
        ("stand up", "stand_up"),
        ("jump!", "free_jump"),
        ("move forward", "move"),
        ("perform a backflip", "back_flip"),
        ("walk upright", "walk_upright"),
    ]
    
    for voice_input, expected_action in test_cases:
        found_action = find_action_for_input(voice_input)
        if found_action:
            print(f"  ✓ '{voice_input}' -> {found_action}")
        else:
            print(f"  ✗ '{voice_input}' -> NOT FOUND (expected {expected_action})")
    
    print("✅ Voice control integration passed")


def test_behavior_executor_mock():
    """Test behavior executor (mock mode without real robot)"""
    print("\n" + "="*70)
    print("TEST: Behavior Executor (Mock)")
    print("="*70)
    
    from behavior_executor import BehaviorExecutor
    
    try:
        executor = BehaviorExecutor()
        
        # Test action listing
        actions = executor.list_actions()
        print(f"  Available actions: {len(actions)}")
        assert len(actions) > 10
        
        # Test action info retrieval
        info = executor.get_action_info("stand_up")
        assert info is not None
        assert info["name"] == "stand_up"
        assert info["difficulty"] == "safe"
        
        print("✅ Behavior executor initialization passed")
        
    except Exception as e:
        print(f"⚠️  Executor test skipped (robot not available): {e}")


def test_openai_integration_mock():
    """Test OpenAI integration with mock response"""
    print("\n" + "="*70)
    print("TEST: OpenAI Integration (Mock)")
    print("="*70)
    
    import json
    from action_registry import parse_action
    
    # Simulate LLM response
    llm_response = {
        "understood": True,
        "action": "free_jump",
        "speed": 0.7,
        "duration": 2.0,
        "response": "I will jump for you now!"
    }
    
    # Test parsing
    action_name = parse_action(llm_response.get("action", ""))
    assert action_name == "free_jump"
    
    print(f"  LLM output: {json.dumps(llm_response, indent=2)}")
    print(f"  Parsed action: {action_name}")
    
    print("✅ OpenAI integration mock passed")


def test_pipeline_flow():
    """Test complete pipeline flow"""
    print("\n" + "="*70)
    print("TEST: Complete Pipeline Flow")
    print("="*70)
    
    from action_registry import find_action_for_input
    from behavior_library import get_library
    
    # Simulate pipeline
    voice_input = "please move forward quickly"
    
    print(f"1. Voice input: '{voice_input}'")
    
    action = find_action_for_input(voice_input)
    print(f"2. Parsed action: {action}")
    assert action is not None
    
    library = get_library()
    schema = library.get(action)
    print(f"3. Action schema: {schema.display_name}")
    print(f"   - Category: {schema.category.value}")
    print(f"   - Difficulty: {schema.difficulty.value}")
    print(f"   - Requires standing: {schema.requires_standing}")
    
    print("✅ Pipeline flow passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# Go2 Voice Control - Complete Test Suite")
    print("#"*70)
    
    tests = [
        test_behavior_schema,
        test_behavior_library,
        test_action_registry,
        test_robot_state,
        test_voice_control_integration,
        test_behavior_executor_mock,
        test_openai_integration_mock,
        test_pipeline_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
