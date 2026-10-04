#!/usr/bin/env python3
"""
Action Registry - Fast lookup registry for all Go2 robot actions
Used by LLM and voice command parsing for quick action identification
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.behavior_schema import ActionCategory, DifficultyLevel
from app.core.behavior_library import get_library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionRegistry:
    """Fast lookup registry for robot actions"""
    
    def __init__(self):
        """Initialize registry from library"""
        self.library = get_library()
        self._build_indexes()
    
    def _build_indexes(self):
        """Build fast lookup indexes"""
        self.by_name: Dict[str, str] = {}  # name -> canonical name
        self.by_alias: Dict[str, str] = {}  # alias -> canonical name
        self.by_mode_id: Dict[int, str] = {}  # mode_id -> name
        self.by_category: Dict[str, List[str]] = {}  # category -> [names]
        self.by_difficulty: Dict[str, List[str]] = {}  # difficulty -> [names]
        
        for schema in self.library.list_all():
            name = schema.name
            
            # Add to by_name
            self.by_name[name] = name
            
            # Add mode_id mapping (mode_id=0 is valid — damp)
            if schema.mode_id is not None:
                self.by_mode_id[schema.mode_id] = name
            
            # Add to by_category
            cat = schema.category.value
            if cat not in self.by_category:
                self.by_category[cat] = []
            self.by_category[cat].append(name)
            
            # Add to by_difficulty
            diff = schema.difficulty.value
            if diff not in self.by_difficulty:
                self.by_difficulty[diff] = []
            self.by_difficulty[diff].append(name)
            
            # Add alias mappings (display_name, method_name, etc.)
            aliases = [
                schema.display_name.lower(),
                schema.method_name.lower() if schema.method_name else None,
            ]
            
            for alias in aliases:
                if alias and alias not in self.by_alias:
                    self.by_alias[alias] = name
    
    def get_action_name(self, identifier: str) -> Optional[str]:
        """
        Get canonical action name from various identifiers
        
        Args:
            identifier: Action name, display name, alias, etc.
            
        Returns:
            Canonical action name or None
        """
        identifier_lower = identifier.lower().strip()
        
        # Try exact match
        if identifier_lower in self.by_name:
            return self.by_name[identifier_lower]
        
        if identifier_lower in self.by_alias:
            return self.by_alias[identifier_lower]
        
        # Try partial match
        for name in self.by_name.keys():
            if name in identifier_lower or identifier_lower in name:
                return name
        
        for alias in self.by_alias.keys():
            if alias in identifier_lower or identifier_lower in alias:
                return self.by_alias[alias]
        
        return None
    
    def get_action_by_mode_id(self, mode_id: int) -> Optional[str]:
        """Get action name by SportClient mode ID"""
        return self.by_mode_id.get(mode_id)
    
    def get_actions_for_category(self, category: str) -> List[str]:
        """Get all actions in a category"""
        return self.by_category.get(category.lower(), [])
    
    def get_actions_for_difficulty(self, difficulty: str) -> List[str]:
        """Get all actions of a difficulty level"""
        return self.by_difficulty.get(difficulty.lower(), [])
    
    def get_safe_action_names(self) -> List[str]:
        """Get all safe action names"""
        return self.by_difficulty.get("safe", [])
    
    def get_all_action_names(self) -> List[str]:
        """Get all action names"""
        return sorted(self.by_name.keys())
    
    def find_similar_actions(
        self,
        action_name: str,
        max_results: int = 5
    ) -> List[str]:
        """Find similar action names"""
        schema = self.library.get(action_name)
        if not schema:
            return []
        
        similar = self.library.get_similar_actions(action_name, max_results)
        return [s.name for s in similar]
    
    def validate_action(self, action_name: str) -> bool:
        """Check if action is valid and registered"""
        return action_name.lower() in self.by_name
    
    # LLM-specific methods for command parsing
    
    def parse_llm_action(self, llm_output: Dict) -> Optional[str]:
        """
        Parse LLM output and extract action name
        
        Args:
            llm_output: Dict with 'action' or 'command' key
            
        Returns:
            Validated action name or None
        """
        action = llm_output.get("action") or llm_output.get("command")
        
        if not action:
            return None
        
        # Get canonical name
        canonical = self.get_action_name(str(action).strip())
        
        if canonical and self.validate_action(canonical):
            return canonical
        
        return None
    
    def parse_nlp_command(self, text: str) -> Optional[str]:
        """
        Parse natural language command and find matching action
        
        Args:
            text: Natural language input
            
        Returns:
            Action name or None
        """
        text_lower = text.lower().strip()
        
        # Try to find action that matches
        for name, schema in zip(
            self.by_name.keys(),
            self.library.list_all()
        ):
            # Check voice commands
            for voice_cmd in schema.voice_commands:
                if voice_cmd in text_lower or text_lower in voice_cmd:
                    return name
        
        # Try fuzzy match on action name
        canonical = self.get_action_name(text)
        if canonical:
            return canonical
        
        return None
    
    def get_action_groups(self) -> Dict[str, List[str]]:
        """Get actions grouped by category"""
        return {k: v for k, v in self.by_category.items()}
    
    def print_registry(self):
        """Print registry summary"""
        print("\n" + "="*70)
        print("Action Registry Summary")
        print("="*70)
        
        print(f"\nTotal Actions: {len(self.by_name)}")
        print(f"Total Aliases: {len(self.by_alias)}")
        
        print("\n\nBy Category:")
        for cat, actions in sorted(self.by_category.items()):
            print(f"  {cat.upper():<15} ({len(actions)} actions)")
        
        print("\n\nBy Difficulty:")
        for diff, actions in sorted(self.by_difficulty.items()):
            print(f"  {diff.upper():<15} ({len(actions)} actions)")
        
        print("\n\nMode IDs:")
        for mode_id, action in sorted(self.by_mode_id.items()):
            print(f"  Mode {mode_id:2d}  -> {action}")
        
        print("\n" + "="*70)


# Global registry instance
_registry: Optional[ActionRegistry] = None


def get_registry() -> ActionRegistry:
    """Get or create global action registry"""
    global _registry
    if _registry is None:
        _registry = ActionRegistry()
    return _registry


def parse_action(identifier: str) -> Optional[str]:
    """Parse and validate action identifier"""
    return get_registry().get_action_name(identifier)


def is_valid_action(action_name: str) -> bool:
    """Check if action is valid"""
    return get_registry().validate_action(action_name)


def list_all_actions() -> List[str]:
    """Get all action names"""
    return get_registry().get_all_action_names()


def find_action_for_input(user_input: str) -> Optional[str]:
    """Find action matching user input (command or NLP)"""
    registry = get_registry()
    
    # Try parsing as command dict first
    if isinstance(user_input, dict):
        return registry.parse_llm_action(user_input)
    
    # Try parsing as natural language
    return registry.parse_nlp_command(str(user_input))


def get_similar_actions(action_name: str) -> List[str]:
    """Get similar actions for disambiguation"""
    return get_registry().find_similar_actions(action_name)


def get_safe_actions() -> List[str]:
    """Get all safe action names"""
    return get_registry().get_safe_action_names()


# Example/Testing
if __name__ == "__main__":
    registry = get_registry()
    
    print(f"\nTotal actions: {len(registry.get_all_action_names())}")
    
    print("\n\nTesting action lookup:")
    test_inputs = [
        "stand_up",
        "Stand Up",
        "StandUp",
        "jump",
        "free_jump",
        "Free Jump",
        "move forward",
        "back_flip",
        "unknown_action"
    ]
    
    for inp in test_inputs:
        result = registry.get_action_name(inp)
        status = "✓" if result else "✗"
        print(f"  {status} '{inp}' -> {result}")
    
    print("\n\nMode ID Lookup:")
    for mode_id in [0, 1, 2, 19]:
        action = registry.get_action_by_mode_id(mode_id)
        print(f"  Mode {mode_id} -> {action}")
    
    print("\n\nLLM Output Parsing:")
    llm_outputs = [
        {"action": "stand_up", "speed": 0.5},
        {"action": "free_jump"},
        {"command": "back_flip"},
    ]
    
    for output in llm_outputs:
        result = registry.parse_llm_action(output)
        print(f"  {output} -> {result}")
    
    print("\n\nNLP Command Parsing:")
    commands = [
        "please stand up",
        "move forward slowly",
        "do a flip",
        "jump high",
    ]
    
    for cmd in commands:
        result = registry.parse_nlp_command(cmd)
        print(f"  '{cmd}' -> {result}")
    
    print("\n")
    registry.print_registry()
