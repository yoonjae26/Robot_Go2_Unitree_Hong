#!/usr/bin/env python3
"""
Behavior Schema - Data structures for Go2 robot behaviors/actions
Defines the structure of actions, commands, and their parameters
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum


class ActionCategory(Enum):
    """Categories of robot actions"""
    POSTURE = "posture"          # stand, sit, lie down
    MOVEMENT = "movement"        # move, rotate
    DANCE = "dance"              # dance moves, choreography
    FLIP_STUNT = "flip_stunt"    # flips, hand stands
    GAIT = "gait"                # different walking modes
    SAFETY = "safety"            # recovery, damp, stop
    SOCIAL = "social"            # friendly poses


class DifficultyLevel(Enum):
    """Difficulty level of actions"""
    SAFE = "safe"               # Always safe
    LOW = "low"                 # Low risk actions
    MEDIUM = "medium"           # Moderate risk
    HIGH = "high"               # Risky, needs space


class ActionStatus(Enum):
    """Status of action execution"""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class Parameter:
    """Definition of an action parameter"""
    name: str
    type: type
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Validate parameter value"""
        if not isinstance(value, self.type):
            return False
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        return True


@dataclass
class ActionSchema:
    """Schema for a robot action/behavior"""
    
    # Basic info
    name: str                                    # e.g., "stand_up"
    display_name: str = ""                      # e.g., "Stand Up"
    description: str = ""                       # What robot does
    category: ActionCategory = ActionCategory.POSTURE
    
    # Execution
    mode_id: int = 0                            # Sport mode ID in Go2 controller
    duration: float = 1.0                       # Default duration in seconds
    estimated_energy: float = 0.1               # Estimated battery usage
    
    # Constraints
    difficulty: DifficultyLevel = DifficultyLevel.SAFE
    requires_standing: bool = False             # Must be standing first
    requires_flat_ground: bool = True           # Needs flat surface
    min_battery: float = 10.0                   # Minimum battery percentage
    
    # Parameters
    parameters: List[Parameter] = field(default_factory=list)
    
    # Configuration
    timeout: float = 10.0                       # Execution timeout
    can_interrupt: bool = True                  # Can be interrupted
    
    # Implementation
    method_name: str = ""                       # SportClient method name
    alternate_methods: List[str] = field(default_factory=list)  # Fallback methods
    
    # Voice commands
    voice_commands: List[str] = field(default_factory=list)  # Natural language triggers
    
    # Post-action
    auto_stand_after: bool = False              # Automatically stand after action
    wait_time_after: float = 0.5                # Wait time after completion
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


@dataclass
class BehaviorSequence:
    """A sequence of actions to perform"""
    name: str
    description: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)  # List of action specs
    loop_count: int = 1
    wait_between: float = 0.5
    can_interrupt: bool = True
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM


@dataclass
class RobotCommandResult:
    """Result of executing a command"""
    success: bool
    action_name: str = ""
    duration: float = 0.0
    message: str = ""
    error: Optional[str] = None
    battery_after: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    """Current state of the robot"""
    is_standing: bool = False
    is_moving: bool = False
    is_executing_action: bool = False
    battery_level: float = 100.0
    last_action: Optional[str] = None
    current_position: str = "unknown"  # standing, sitting, lying, etc.
    error_state: bool = False
    error_message: Optional[str] = None
    gait_mode: Optional[str] = None
    
    def is_safe_for_action(self, action_schema: ActionSchema) -> bool:
        """Check if robot state allows this action"""
        if self.error_state:
            return False
        
        if self.battery_level < action_schema.min_battery:
            return False
        
        if action_schema.requires_standing and not self.is_standing:
            return False
        
        return True


# Pre-defined parameter sequences for common actions
SPEED_PARAMETER = Parameter(
    name="speed",
    type=float,
    default=0.5,
    min_value=0.0,
    max_value=1.0,
    description="Movement speed (0.0-1.0)"
)

DURATION_PARAMETER = Parameter(
    name="duration",
    type=float,
    default=2.0,
    min_value=0.1,
    max_value=30.0,
    description="Action duration in seconds"
)

INTENSITY_PARAMETER = Parameter(
    name="intensity",
    type=float,
    default=0.5,
    min_value=0.0,
    max_value=1.0,
    description="Action intensity/power (0.0-1.0)"
)

DIRECTION_PARAMETER = Parameter(
    name="direction",
    type=str,
    default="forward",
    description="Movement direction (forward, backward, left, right)"
)

ROTATION_PARAMETER = Parameter(
    name="rotation",
    type=float,
    default=0.5,
    min_value=-1.0,
    max_value=1.0,
    description="Rotation speed/amount"
)

# Common action schemas
STAND_UP_SCHEMA = ActionSchema(
    name="stand_up",
    display_name="Stand Up",
    description="Robot stands up from sitting or lying position",
    category=ActionCategory.POSTURE,
    mode_id=1,
    duration=2.0,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=False,
    auto_stand_after=True,
    method_name="StandUp",
    voice_commands=["stand up", "get up", "stand", "please stand",
                    "일어서", "일어나", "기립", "서", "일어서봐"],
)

STAND_DOWN_SCHEMA = ActionSchema(
    name="stand_down",
    display_name="Stand Down",
    description="Robot lowers body to ground",
    category=ActionCategory.POSTURE,
    mode_id=2,
    duration=2.0,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=True,
    method_name="StandDown",
    voice_commands=["stand down", "sit down", "lower", "crouch",
                    "앉아", "앉아봐", "엎드려", "내려가", "숙여"],
)

MOVE_SCHEMA = ActionSchema(
    name="move",
    display_name="Move",
    description="Robot moves in specified direction",
    category=ActionCategory.MOVEMENT,
    mode_id=3,
    duration=2.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    requires_flat_ground=True,
    parameters=[SPEED_PARAMETER, DURATION_PARAMETER, DIRECTION_PARAMETER],
    method_name="Move",
    voice_commands=["move", "go", "walk", "move forward", "move backward",
                    "앞으로", "전진", "이동", "가", "뒤로", "후진", "옆으로", "걸어"],
)

HAND_STAND_SCHEMA = ActionSchema(
    name="hand_stand",
    display_name="Hand Stand",
    description="Robot performs a hand stand",
    category=ActionCategory.FLIP_STUNT,
    mode_id=7,
    duration=4.0,
    estimated_energy=0.5,
    difficulty=DifficultyLevel.HIGH,
    requires_standing=True,
    requires_flat_ground=True,
    timeout=15.0,
    can_interrupt=True,
    method_name="HandStand",
    voice_commands=["hand stand", "handstand", "stand on hands",
                    "물구나무", "손으로 서기", "손서기"],
)

BACK_FLIP_SCHEMA = ActionSchema(
    name="back_flip",
    display_name="Back Flip",
    description="Robot performs a back flip",
    category=ActionCategory.FLIP_STUNT,
    mode_id=12,
    duration=3.0,
    estimated_energy=0.4,
    difficulty=DifficultyLevel.HIGH,
    requires_standing=True,
    requires_flat_ground=True,
    min_battery=20.0,
    timeout=10.0,
    method_name="BackFlip",
    voice_commands=["back flip", "backflip", "flip back",
                    "백플립", "뒤로 공중제비", "공중제비", "뒤돌기"],
)

LEFT_FLIP_SCHEMA = ActionSchema(
    name="left_flip",
    display_name="Left Flip",
    description="Robot performs a left flip",
    category=ActionCategory.FLIP_STUNT,
    mode_id=11,
    duration=3.0,
    estimated_energy=0.4,
    difficulty=DifficultyLevel.HIGH,
    requires_standing=True,
    requires_flat_ground=True,
    min_battery=20.0,
    timeout=10.0,
    method_name="LeftFlip",
    voice_commands=["left flip", "flip left",
                    "왼쪽 플립", "왼쪽 공중제비", "좌측 플립"],
)

WALK_UPRIGHT_SCHEMA = ActionSchema(
    name="walk_upright",
    display_name="Walk Upright",
    description="Robot walks in a more human-like upright posture",
    category=ActionCategory.GAIT,
    mode_id=17,
    duration=4.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    auto_stand_after=True,
    method_name="WalkUpright",
    voice_commands=["walk upright", "human walk", "upright walk",
                    "직립 보행", "직립", "사람처럼 걸어", "두발로 걸어"],
)

FREE_JUMP_SCHEMA = ActionSchema(
    name="free_jump",
    display_name="Free Jump",
    description="Robot performs a freestyle jump",
    category=ActionCategory.FLIP_STUNT,
    mode_id=19,
    duration=3.0,
    estimated_energy=0.3,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    timeout=10.0,
    method_name="FreeJump",
    voice_commands=["jump", "free jump", "bounce",
                    "점프", "뛰어", "뛰어올라", "점프해"],
)

FREE_WALK_SCHEMA = ActionSchema(
    name="free_walk",
    display_name="Free Walk",
    description="Robot walks with freestyle movement",
    category=ActionCategory.GAIT,
    mode_id=13,
    duration=5.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    requires_flat_ground=True,
    timeout=15.0,
    method_name="FreeWalk",
    voice_commands=["free walk", "walk freely",
                    "자유롭게 걸어", "걸어", "자유 보행"],
)

FREE_BOUND_SCHEMA = ActionSchema(
    name="free_bound",
    display_name="Free Bound",
    description="Robot performs bounding motion",
    category=ActionCategory.GAIT,
    mode_id=14,
    duration=5.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    timeout=15.0,
    method_name="FreeBound",
    voice_commands=["bound", "free bound", "jump around",
                    "바운딩", "달려", "뛰어다녀"],
)

CROSS_STEP_SCHEMA = ActionSchema(
    name="cross_step",
    display_name="Cross Step",
    description="Robot performs crossing step movement",
    category=ActionCategory.GAIT,
    mode_id=18,
    duration=4.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    timeout=15.0,
    method_name="CrossStep",
    voice_commands=["cross step", "cross walk", "sideways",
                    "크로스 스텝", "교차 걸음"],
)

RECOVERY_SCHEMA = ActionSchema(
    name="recovery",
    display_name="Recovery",
    description="Robot recovers to standing position if knocked down",
    category=ActionCategory.SAFETY,
    mode_id=10,
    duration=4.0,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=False,
    timeout=15.0,
    method_name="RecoveryStand",
    voice_commands=["recover", "get up", "help",
                    "회복", "회복해", "다시 일어나", "복구"],
)

DAMP_SCHEMA = ActionSchema(
    name="damp",
    display_name="Damp Mode",
    description="Robot enters damp mode (no stiffness, can be moved)",
    category=ActionCategory.SAFETY,
    mode_id=0,
    duration=0.5,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=False,
    can_interrupt=True,
    method_name="Damp",
    voice_commands=["damp", "relax", "no power",
                    "이완", "릴렉스", "힘 빼", "편안하게"],
)

STOP_MOVE_SCHEMA = ActionSchema(
    name="stop_move",
    display_name="Stop Move",
    description="Robot stops all movement",
    category=ActionCategory.SAFETY,
    mode_id=6,
    duration=0.5,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=False,
    can_interrupt=True,
    method_name="StopMove",
    voice_commands=["stop", "halt", "freeze", "stop moving",
                    "멈춰", "정지", "그만", "스톱"],
)

BALANCED_STAND_SCHEMA = ActionSchema(
    name="balanced_stand",
    display_name="Balanced Stand",
    description="Robot stands with balance control",
    category=ActionCategory.POSTURE,
    mode_id=9,
    duration=2.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=False,
    auto_stand_after=True,
    method_name="BalanceStand",
    voice_commands=["balance", "balanced stand",
                    "하트", "하트 포즈", "균형", "균형 자세"],
)

FREE_AVOID_SCHEMA = ActionSchema(
    name="free_avoid",
    display_name="Free Avoid",
    description="Robot performs obstacle avoidance movement",
    category=ActionCategory.GAIT,
    mode_id=15,
    duration=5.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=False,
    timeout=15.0,
    method_name="FreeAvoid",
    voice_commands=["avoid", "free avoid", "navigate",
                    "피해", "회피", "장애물 피해"],
)
