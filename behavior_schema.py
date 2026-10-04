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
        """Validate parameter value.

        Numeric types (int/float) are treated as interchangeable here: JSON
        numbers decoded from an LLM's JSON-mode output don't reliably
        preserve the int/float distinction (e.g. `0` vs `0.0`), so a strict
        `isinstance(value, self.type)` check would spuriously reject valid
        values on nothing more than that formatting accident. Range checks
        below still apply either way.
        """
        if self.type in (int, float):
            if not isinstance(value, (int, float)):
                return False
        elif not isinstance(value, self.type):
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
    # Minimum clear distance (meters) required in front of the robot for
    # this action to be allowed, used by the context-aware gate (V_c) for
    # movement-type actions. None = not gated on obstacle distance. This is
    # a controlled/test-harness-settable value (see RobotState.obstacle_distance_m),
    # not live LIDAR fusion -- see runtime_verification.py's module docstring.
    min_obstacle_clearance: Optional[float] = None
    
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
    # Distance (meters) to the nearest known obstacle in the robot's path,
    # for the context-aware gate's obstacle case study. None = unknown/not
    # tracked (fail-open: actions are not obstacle-gated unless this is
    # explicitly set), since this system has no continuous onboard
    # obstacle-distance sensing wired into the voice-command path -- it is
    # set deliberately by the test harness / experimenter for controlled
    # trials. See runtime_verification.py.
    obstacle_distance_m: Optional[float] = None

    def is_safe_for_action(self, action_schema: ActionSchema) -> bool:
        """Check if robot state allows this action"""
        if self.error_state:
            return False

        if self.battery_level < action_schema.min_battery:
            return False

        if action_schema.requires_standing and not self.is_standing:
            return False

        if (
            action_schema.min_obstacle_clearance is not None
            and self.obstacle_distance_m is not None
            and self.obstacle_distance_m < action_schema.min_obstacle_clearance
        ):
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

# The "move" action's real execution path (behavior_executor.py's
# execute_action, "move" branch) reads vx/vy/omega directly and sends them
# to SportClient.Move() -- it does NOT use SPEED_PARAMETER/DIRECTION_PARAMETER
# below (those describe fields the executor never reads for this action).
# Bounds are conservative estimates based on this codebase's own observed
# operating range rather than an authoritative Unitree spec (D-pad driving
# caps linear speed at 0.6 m/s -- dashboard_server.py's adjust_speed(); voice
# turn commands are calibrated around omega=1.0 rad/s per voice_control_go2.py's
# system prompt examples) -- like SPEED_LEVEL_SCHEMA elsewhere in this file,
# treat as unverified-on-hardware defaults to tune, not a ground truth.
VX_PARAMETER = Parameter(
    name="vx", type=float, default=0.0, min_value=-1.0, max_value=1.0,
    description="Forward(+)/backward(-) velocity in m/s",
)
VY_PARAMETER = Parameter(
    name="vy", type=float, default=0.0, min_value=-1.0, max_value=1.0,
    description="Left(+)/right(-) lateral velocity in m/s",
)
OMEGA_PARAMETER = Parameter(
    name="omega", type=float, default=0.0, min_value=-2.0, max_value=2.0,
    description="Yaw rotation speed in rad/s",
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
    method_name="StandUp",
    voice_commands=["stand up", "get up", "stand", "please stand",
                    "일어서", "일어나", "기립", "서", "일어서봐"],
)

STAND_DOWN_SCHEMA = ActionSchema(
    name="stand_down",
    display_name="Stand Down",
    description="Robot lowers its whole body flat to the ground, prone/lying down (distinct from sit, which stays up on its haunches like a dog)",
    category=ActionCategory.POSTURE,
    mode_id=2,
    duration=2.0,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=True,
    method_name="StandDown",
    # "앉아"/"앉기" (sit) deliberately NOT here -- that's genuinely "sit", now
    # correctly mapped to SIT_SCHEMA below. This action is lying/crouching down.
    voice_commands=["stand down", "lie down", "lower", "crouch",
                    "엎드려", "내려가", "숙여", "누워"],
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
    min_obstacle_clearance=0.5,
    parameters=[SPEED_PARAMETER, DURATION_PARAMETER, DIRECTION_PARAMETER,
                VX_PARAMETER, VY_PARAMETER, OMEGA_PARAMETER],
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
    display_name="Heart Pose (Love)",
    description="Robot performs heart/love pose — raises front legs to form a heart shape",
    category=ActionCategory.POSTURE,
    mode_id=9,
    duration=5.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=False,
    auto_stand_after=True,
    method_name="Heart",
    voice_commands=["heart", "love", "balanced stand",
                    "하트", "하트 포즈", "러브", "사랑"],
)

HELLO_SCHEMA = ActionSchema(
    name="hello",
    display_name="Hello / Wave",
    description="Robot waves a leg to greet",
    category=ActionCategory.POSTURE,
    mode_id=16,
    duration=3.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    method_name="Hello",
    voice_commands=["hello", "wave", "hi", "greet",
                    "안녕", "인사", "손 흔들어", "반가워"],
)

STRETCH_SCHEMA = ActionSchema(
    name="stretch",
    display_name="Stretch",
    description="Robot performs a stretching motion",
    category=ActionCategory.POSTURE,
    mode_id=17,
    duration=4.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=False,
    method_name="Stretch",
    voice_commands=["stretch", "스트레칭", "기지개"],
)

DANCE1_SCHEMA = ActionSchema(
    name="dance1",
    display_name="Dance 1",
    description="Robot performs dance routine 1",
    category=ActionCategory.POSTURE,
    mode_id=22,
    duration=8.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    method_name="Dance1",
    voice_commands=["dance", "dance1", "춤", "댄스", "춤춰", "춤 1", "댄스 1"],
)

DANCE2_SCHEMA = ActionSchema(
    name="dance2",
    display_name="Dance 2",
    description="Robot performs dance routine 2",
    category=ActionCategory.POSTURE,
    mode_id=23,
    duration=8.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    method_name="Dance2",
    voice_commands=["dance2", "춤 2", "댄스 2"],
)

FRONT_FLIP_SCHEMA = ActionSchema(
    name="front_flip",
    display_name="Front Flip",
    description="Robot performs a forward somersault",
    category=ActionCategory.FLIP_STUNT,
    mode_id=30,
    duration=5.0,
    difficulty=DifficultyLevel.HIGH,
    requires_standing=True,
    requires_flat_ground=True,
    method_name="FrontFlip",
    voice_commands=["front flip", "앞 공중제비", "앞으로 뒤집어", "앞 플립"],
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

SCRAPE_SCHEMA = ActionSchema(
    name="scrape",
    display_name="Scrape",
    description="Robot performs a scraping paw motion",
    category=ActionCategory.SOCIAL,
    mode_id=31,
    duration=3.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    method_name="Scrape",
    voice_commands=["scrape", "긁기", "발 긁어"],
)

FRONT_JUMP_SCHEMA = ActionSchema(
    name="front_jump",
    display_name="Front Jump",
    description="Robot performs a forward jump",
    category=ActionCategory.FLIP_STUNT,
    mode_id=32,
    duration=3.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    method_name="FrontJump",
    voice_commands=["front jump", "jump forward", "앞으로 점프", "앞점프"],
)

FRONT_POUNCE_SCHEMA = ActionSchema(
    name="front_pounce",
    display_name="Front Pounce",
    description="Robot performs a forward pouncing motion",
    category=ActionCategory.FLIP_STUNT,
    mode_id=33,
    duration=3.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    method_name="FrontPounce",
    voice_commands=["pounce", "front pounce", "덮치기", "앞으로 덮쳐"],
)

POSE_SCHEMA = ActionSchema(
    name="pose",
    display_name="Pose",
    description="Robot leans/poses its body without stepping",
    category=ActionCategory.SOCIAL,
    mode_id=34,
    duration=3.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    method_name="Pose",
    voice_commands=["pose", "포즈", "몸 기울이기"],
)

TROT_RUN_SCHEMA = ActionSchema(
    name="trot_run",
    display_name="Trot Run",
    description="Robot switches to a bouncy trotting gait (closest built-in match to a horse-like running step)",
    category=ActionCategory.GAIT,
    mode_id=35,
    duration=3.0,
    difficulty=DifficultyLevel.MEDIUM,
    requires_standing=True,
    requires_flat_ground=True,
    method_name="TrotRun",
    voice_commands=["trot", "trot run", "horse gait", "트롯 런", "말처럼 뛰기"],
)

STATIC_WALK_SCHEMA = ActionSchema(
    name="static_walk",
    display_name="Static Walk",
    description="Robot switches to a slow, statically-stable walking gait",
    category=ActionCategory.GAIT,
    mode_id=36,
    duration=3.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    requires_flat_ground=True,
    method_name="StaticWalk",
    voice_commands=["static walk", "slow walk", "정적 보행", "천천히 걸어"],
)

SIT_SCHEMA = ActionSchema(
    name="sit",
    display_name="Sit",
    description="Robot sits down on its haunches, dog-style (distinct from stand_down, which lowers the whole body to the ground)",
    category=ActionCategory.POSTURE,
    mode_id=37,
    duration=2.0,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=True,
    method_name="Sit",
    voice_commands=["sit", "sit down", "앉아", "앉기", "앉아봐",
                    "엉덩이 대고 앉아", "개다리 자세로 앉아"],
)

RISE_SIT_SCHEMA = ActionSchema(
    name="rise_sit",
    display_name="Rise From Sit",
    description="Robot stands back up from the sit posture (distinct from stand_up, which recovers from lying down)",
    category=ActionCategory.POSTURE,
    mode_id=38,
    duration=2.0,
    difficulty=DifficultyLevel.SAFE,
    requires_standing=False,
    method_name="RiseSit",
    # Deliberately not bare "일어나"/"일어서" -- stand_up already claims those.
    voice_commands=["rise", "stand up from sit", "앉은 데서 일어나"],
)

CONTENT_SCHEMA = ActionSchema(
    name="content",
    display_name="Content",
    description="Robot performs a happy/affectionate gesture",
    category=ActionCategory.SOCIAL,
    mode_id=39,
    duration=3.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    method_name="Content",
    voice_commands=["content", "happy", "애교", "기쁜 표정", "좋아하는 동작"],
)

CLASSIC_WALK_SCHEMA = ActionSchema(
    name="classic_walk",
    display_name="Classic Walk",
    description="Robot switches to the classic/standard walking gait",
    category=ActionCategory.GAIT,
    mode_id=40,
    duration=4.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    requires_flat_ground=True,
    method_name="ClassicWalk",
    voice_commands=["classic walk", "normal walk", "클래식 워크", "기본 걸음", "일반 걸음"],
)

ROLL_PARAMETER = Parameter(
    name="roll", type=float, default=0.0, min_value=-0.5, max_value=0.5,
    description="Body roll/lean, left(-)/right(+) in radians",
)
PITCH_PARAMETER = Parameter(
    name="pitch", type=float, default=0.0, min_value=-0.5, max_value=0.5,
    description="Body pitch, nose-down(-)/nose-up(+) in radians",
)
YAW_PARAMETER = Parameter(
    name="yaw", type=float, default=0.0, min_value=-0.5, max_value=0.5,
    description="Body yaw tilt in radians",
)

EULER_SCHEMA = ActionSchema(
    name="euler",
    display_name="Body Tilt",
    description="Robot tilts its body (roll/pitch/yaw) and holds the pose -- unlike Move, this does NOT auto-decay, so the executor must explicitly return it to (0, 0, 0) after `duration`",
    category=ActionCategory.SOCIAL,
    mode_id=41,
    duration=2.0,
    difficulty=DifficultyLevel.LOW,
    requires_standing=True,
    parameters=[ROLL_PARAMETER, PITCH_PARAMETER, YAW_PARAMETER, DURATION_PARAMETER],
    method_name="Euler",
    voice_commands=["tilt", "lean", "몸 기울여", "고개 숙여", "옆으로 기울여"],
)

LEVEL_PARAMETER = Parameter(
    name="level", type=int, default=1, min_value=1, max_value=5,
    description="Speed level (integer) -- exact valid range is NOT documented "
                "by Unitree; 1-5 is a conservative guess, unverified on real "
                "hardware",
)

SPEED_LEVEL_SCHEMA = ActionSchema(
    name="speed_level",
    display_name="Speed Level",
    description="Sets the robot's overall gait speed level (integer), as an alternative to changing vx/vy on individual move commands -- UNVERIFIED on real hardware, test with small values first",
    category=ActionCategory.MOVEMENT,
    mode_id=42,
    duration=0.5,
    difficulty=DifficultyLevel.MEDIUM,  # untested -- effect on gait/stability not yet confirmed live
    requires_standing=True,
    parameters=[LEVEL_PARAMETER],
    method_name="SpeedLevel",
    voice_commands=["speed level", "속도 단계", "속도 레벨"],
)
