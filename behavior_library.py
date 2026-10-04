#!/usr/bin/env python3
"""
Behavior Library - Comprehensive library of Go2 robot behaviors/actions
Manages all available actions with metadata, descriptions, and execution info
"""

from typing import Dict, List, Optional, Set
import logging
from behavior_schema import (
    ActionSchema,
    ActionCategory,
    DifficultyLevel,
    # All pre-defined schemas
    STAND_UP_SCHEMA,
    STAND_DOWN_SCHEMA,
    MOVE_SCHEMA,
    HAND_STAND_SCHEMA,
    BACK_FLIP_SCHEMA,
    LEFT_FLIP_SCHEMA,
    WALK_UPRIGHT_SCHEMA,
    FREE_JUMP_SCHEMA,
    FREE_WALK_SCHEMA,
    FREE_BOUND_SCHEMA,
    CROSS_STEP_SCHEMA,
    RECOVERY_SCHEMA,
    DAMP_SCHEMA,
    STOP_MOVE_SCHEMA,
    BALANCED_STAND_SCHEMA,
    FREE_AVOID_SCHEMA,
    HELLO_SCHEMA,
    STRETCH_SCHEMA,
    DANCE1_SCHEMA,
    DANCE2_SCHEMA,
    FRONT_FLIP_SCHEMA,
    SCRAPE_SCHEMA,
    FRONT_JUMP_SCHEMA,
    FRONT_POUNCE_SCHEMA,
    POSE_SCHEMA,
    TROT_RUN_SCHEMA,
    STATIC_WALK_SCHEMA,
    SIT_SCHEMA,
    RISE_SIT_SCHEMA,
    CONTENT_SCHEMA,
    CLASSIC_WALK_SCHEMA,
    EULER_SCHEMA,
    SPEED_LEVEL_SCHEMA,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorLibrary:
    """Complete library of robot behaviors"""
    
    def __init__(self):
        """Initialize behavior library with all available actions"""
        self.behaviors: Dict[str, ActionSchema] = {}
        self.by_category: Dict[ActionCategory, List[ActionSchema]] = {}
        self.by_difficulty: Dict[DifficultyLevel, List[ActionSchema]] = {}
        self.voice_command_map: Dict[str, str] = {}  # voice -> action name
        
        # Register all behaviors
        self._register_all_behaviors()
    
    def _register_all_behaviors(self):
        """Register all available behaviors"""
        all_schemas = [
            STAND_UP_SCHEMA,
            STAND_DOWN_SCHEMA,
            MOVE_SCHEMA,
            HAND_STAND_SCHEMA,
            BACK_FLIP_SCHEMA,
            LEFT_FLIP_SCHEMA,
            WALK_UPRIGHT_SCHEMA,
            FREE_JUMP_SCHEMA,
            FREE_WALK_SCHEMA,
            FREE_BOUND_SCHEMA,
            CROSS_STEP_SCHEMA,
            RECOVERY_SCHEMA,
            DAMP_SCHEMA,
            STOP_MOVE_SCHEMA,
            BALANCED_STAND_SCHEMA,
            FREE_AVOID_SCHEMA,
            HELLO_SCHEMA,
            STRETCH_SCHEMA,
            DANCE1_SCHEMA,
            DANCE2_SCHEMA,
            FRONT_FLIP_SCHEMA,
            SCRAPE_SCHEMA,
            FRONT_JUMP_SCHEMA,
            FRONT_POUNCE_SCHEMA,
            POSE_SCHEMA,
            TROT_RUN_SCHEMA,
            STATIC_WALK_SCHEMA,
            SIT_SCHEMA,
            RISE_SIT_SCHEMA,
            CONTENT_SCHEMA,
            CLASSIC_WALK_SCHEMA,
            EULER_SCHEMA,
            SPEED_LEVEL_SCHEMA,
        ]
        
        for schema in all_schemas:
            self.register(schema)
    
    def register(self, schema: ActionSchema):
        """Register a behavior"""
        name = schema.name
        
        # Add to main registry
        self.behaviors[name] = schema
        
        # Add to category index
        category = schema.category
        if category not in self.by_category:
            self.by_category[category] = []
        self.by_category[category].append(schema)
        
        # Add to difficulty index
        difficulty = schema.difficulty
        if difficulty not in self.by_difficulty:
            self.by_difficulty[difficulty] = []
        self.by_difficulty[difficulty].append(schema)
        
        # Add voice command mappings
        for voice_cmd in schema.voice_commands:
            self.voice_command_map[voice_cmd.lower()] = name
        
        logger.info(f"Registered behavior: {name}")
    
    def get(self, name: str) -> Optional[ActionSchema]:
        """Get behavior by name"""
        return self.behaviors.get(name.lower())
    
    def get_by_voice_command(self, voice_input: str) -> Optional[ActionSchema]:
        """Find behavior by voice command"""
        voice_lower = voice_input.lower().strip()
        
        # Try exact match first
        if voice_lower in self.voice_command_map:
            action_name = self.voice_command_map[voice_lower]
            return self.behaviors[action_name]
        
        # Try partial match
        for cmd, action_name in self.voice_command_map.items():
            if cmd in voice_lower or voice_lower in cmd:
                return self.behaviors[action_name]
        
        return None
    
    def get_by_category(self, category) -> List[ActionSchema]:
        """Get all behaviors of a category (accepts ActionCategory enum or string)"""
        if isinstance(category, str):
            category = ActionCategory(category.lower())
        return self.by_category.get(category, [])
    
    def get_by_difficulty(self, difficulty: DifficultyLevel) -> List[ActionSchema]:
        """Get all behaviors of a difficulty level"""
        return self.by_difficulty.get(difficulty, [])
    
    def get_safe_actions(self) -> List[ActionSchema]:
        """Get all safe actions"""
        return self.by_difficulty.get(DifficultyLevel.SAFE, [])
    
    def list_all(self) -> List[ActionSchema]:
        """List all available behaviors"""
        return list(self.behaviors.values())
    
    def list_names(self) -> List[str]:
        """List all behavior names"""
        return sorted(self.behaviors.keys())
    
    def get_categories(self) -> Set[ActionCategory]:
        """Get all categories"""
        return set(self.by_category.keys())
    
    def get_difficulties(self) -> Set[DifficultyLevel]:
        """Get all difficulty levels"""
        return set(self.by_difficulty.keys())
    
    def get_similar_actions(self, name: str, max_results: int = 5) -> List[ActionSchema]:
        """Find similar actions (same category or difficulty)"""
        schema = self.get(name)
        if not schema:
            return []
        
        similar = []
        
        # Get actions in same category
        for s in self.by_category.get(schema.category, []):
            if s.name != name:
                similar.append(s)
        
        # Get actions with similar difficulty
        if len(similar) < max_results:
            for s in self.by_difficulty.get(schema.difficulty, []):
                if s.name != name and s not in similar:
                    similar.append(s)
        
        return similar[:max_results]
    
    def search(self, query: str) -> List[ActionSchema]:
        """Search behaviors by name or description"""
        query_lower = query.lower()
        results = []
        
        for behavior in self.behaviors.values():
            if (query_lower in behavior.name.lower() or
                query_lower in behavior.display_name.lower() or
                query_lower in behavior.description.lower()):
                results.append(behavior)
        
        return results
    
    def get_status_report(self) -> str:
        """Get library status report"""
        total = len(self.behaviors)
        safe = len(self.by_difficulty.get(DifficultyLevel.SAFE, []))
        low = len(self.by_difficulty.get(DifficultyLevel.LOW, []))
        medium = len(self.by_difficulty.get(DifficultyLevel.MEDIUM, []))
        high = len(self.by_difficulty.get(DifficultyLevel.HIGH, []))
        
        report = f"""
Behavior Library Status
========================
Total Behaviors: {total}

By Difficulty:
  Safe:   {safe}
  Low:    {low}
  Medium: {medium}
  High:   {high}

By Category:
"""
        for category in sorted(self.by_category.keys()):
            count = len(self.by_category[category])
            report += f"  {category.value.capitalize():<15} {count}\n"
        
        return report
    
    def print_help(self):
        """Print help with all available behaviors"""
        print("\n" + "="*70)
        print("Go2 Robot - Available Behaviors/Actions")
        print("="*70)
        
        for category in sorted(self.by_category.keys()):
            print(f"\n{category.value.upper()}")
            print("-" * 70)
            
            for schema in self.by_category[category]:
                print(f"  {schema.display_name:<20} ({schema.difficulty.value})")
                print(f"    Name:        {schema.name}")
                print(f"    Description: {schema.description}")
                if schema.voice_commands:
                    print(f"    Voice:       {', '.join(schema.voice_commands[:3])}")
                print()


# Global library instance
_library: Optional[BehaviorLibrary] = None


def get_library() -> BehaviorLibrary:
    """Get or create global behavior library"""
    global _library
    if _library is None:
        _library = BehaviorLibrary()
    return _library


def list_all_actions() -> List[str]:
    """Get list of all action names"""
    return get_library().list_names()


def get_action_schema(name: str) -> Optional[ActionSchema]:
    """Get action schema by name"""
    return get_library().get(name)


def find_action_by_voice(voice_input: str) -> Optional[ActionSchema]:
    """Find action by voice command"""
    return get_library().get_by_voice_command(voice_input)


def search_actions(query: str) -> List[ActionSchema]:
    """Search for actions"""
    return get_library().search(query)


def get_safe_actions() -> List[ActionSchema]:
    """Get all safe actions"""
    return get_library().get_safe_actions()


def print_library():
    """Print complete behavior library"""
    get_library().print_help()


# Example usage and testing
if __name__ == "__main__":
    library = get_library()
    
    print(library.get_status_report())
    
    print("\nAll Actions:")
    for name in library.list_names():
        schema = library.get(name)
        print(f"  {name:<20} - {schema.description}")
    
    print("\nVoice Command Examples:")
    examples = ["stand up", "move forward", "jump", "back flip", "walk upright"]
    for cmd in examples:
        schema = library.get_by_voice_command(cmd)
        if schema:
            print(f"  '{cmd}' -> {schema.name}")
        else:
            print(f"  '{cmd}' -> NOT FOUND")
    
    print("\nSafe Actions:")
    for schema in library.get_safe_actions():
        print(f"  {schema.display_name}")
    
    print("\n" + "="*70)
    library.print_help()
