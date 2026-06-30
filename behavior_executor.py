#!/usr/bin/env python3
"""
Behavior Executor - Executes Go2 robot behaviors with safety checks and monitoring
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from behavior_schema import RobotState, ActionSchema, RobotCommandResult, ActionStatus
from behavior_library import get_library, BehaviorLibrary
from unitree_sdk2py.go2.sport.sport_client import SportClient
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorExecutor:
    """Executes robot behaviors with safety checks"""
    
    def __init__(self, robot_ip: str = "192.168.12.1"):
        """
        Initialize behavior executor
        
        Args:
            robot_ip: IP address of Go2 robot
        """
        self.robot_ip = robot_ip
        self.library = get_library()
        self.sport_client = None
        self.robot_state = RobotState()
        self.connected = False
        
        # Action tracking
        self.current_action = None
        self.action_start_time = None
        self.execution_thread = None
        self.should_stop = False
        
        # Callbacks
        self.on_action_start: Optional[Callable] = None
        self.on_action_complete: Optional[Callable] = None
        self.on_action_error: Optional[Callable] = None
        
        logger.info(f"Initializing BehaviorExecutor for {robot_ip}")
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to robot"""
        try:
            logger.info(f"Connecting to robot at {self.robot_ip}")
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()

            time.sleep(0.5)

            self.connected = True
            logger.info("✅ Connected to robot")
            return True
        
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def is_state_safe(self) -> bool:
        """Check if robot is in safe state to execute actions"""
        if not self.connected:
            return False
        if self.robot_state.error_state:
            return False
        return True
    
    def can_execute(self, action: ActionSchema) -> tuple[bool, Optional[str]]:
        """
        Check if action can be executed
        
        Args:
            action: Action schema
            
        Returns:
            (can_execute, reason_if_not)
        """
        if not self.is_state_safe():
            return False, "Robot not in safe state"
        
        if not self.robot_state.is_safe_for_action(action):
            if self.robot_state.battery_level < action.min_battery:
                return False, f"Battery too low ({self.robot_state.battery_level}% < {action.min_battery}%)"
            
            if action.requires_standing and not self.robot_state.is_standing:
                return False, "Robot must be standing for this action"
            
            return False, "Robot state incompatible with action"
        
        return True, None
    
    def _execute_sportclient_method(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> bool:
        """Execute a SportClient method"""
        try:
            if not hasattr(self.sport_client, method_name):
                logger.warning(f"SportClient has no method: {method_name}")
                # Try alternate methods
                return False
            
            method = getattr(self.sport_client, method_name)
            
            if args or kwargs:
                result = method(*args, **kwargs)
            else:
                result = method()
            
            logger.info(f"Executed {method_name}: {result}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing {method_name}: {e}")
            return False
    
    def _ensure_standing(self) -> bool:
        """Ensure robot is standing before action"""
        if self.robot_state.is_standing:
            return True
        
        logger.info("Robot not standing, attempting to stand up...")
        try:
            self.sport_client.StandUp()
            time.sleep(2)
            self.robot_state.is_standing = True
            return True
        except Exception as e:
            logger.error(f"Failed to stand up: {e}")
            return False
    
    def _post_action_stand_if_needed(self, action: ActionSchema):
        """Automatically stand up after action if configured"""
        if action.auto_stand_after:
            logger.info("Auto-standing after action...")
            try:
                self.sport_client.StandUp()
                time.sleep(1)
                self.robot_state.is_standing = True
            except Exception as e:
                logger.warning(f"Auto-stand failed: {e}")
    
    def execute_action(
        self,
        action_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        wait_for_completion: bool = True
    ) -> RobotCommandResult:
        """
        Execute a behavior action
        
        Args:
            action_name: Name of action to execute
            parameters: Action parameters
            timeout: Execution timeout
            wait_for_completion: Wait for action to complete
            
        Returns:
            Execution result
        """
        # Get action schema
        action = self.library.get(action_name)
        if not action:
            return RobotCommandResult(
                success=False,
                action_name=action_name,
                error=f"Action not found: {action_name}"
            )
        
        # Check if can execute
        can_exec, reason = self.can_execute(action)
        if not can_exec:
            return RobotCommandResult(
                success=False,
                action_name=action_name,
                error=reason
            )
        
        # Ensure standing if required
        if action.requires_standing:
            if not self._ensure_standing():
                return RobotCommandResult(
                    success=False,
                    action_name=action_name,
                    error="Failed to stand up before action"
                )
        
        # Callback
        if self.on_action_start:
            self.on_action_start(action, parameters)
        
        start_time = time.time()
        result = RobotCommandResult(
            success=False,
            action_name=action_name,
            timestamp=start_time
        )
        
        try:
            logger.info(f"Executing action: {action_name}")
            
            # Merge parameters
            exec_params = parameters or {}
            
            # Build execution kwargs
            exec_kwargs = {}
            
            # Handle specific actions
            if action_name == "move":
                # Move command needs vx, vy, omega
                vx = exec_params.get("vx", 0.5) * exec_params.get("speed", 0.5)
                vy = exec_params.get("vy", 0) * exec_params.get("speed", 0.5)
                omega = exec_params.get("omega", 0) * exec_params.get("speed", 0.5)
                
                duration = exec_params.get("duration", action.duration)
                
                # Execute move
                self.sport_client.Move(vx, vy, omega)
                time.sleep(duration)
                self.sport_client.Move(0, 0, 0)
                
                result.success = True
                result.message = f"Moved with vx={vx}, vy={vy}, omega={omega}"
            
            elif action_name in ["hand_stand", "free_bound", "free_avoid", "walk_upright", "cross_step", "free_jump"]:
                # Actions with on/off control
                exec_success = self._execute_sportclient_method(action.method_name, True)
                
                if exec_success:
                    duration = exec_params.get("duration", action.duration)
                    time.sleep(duration)
                    self._execute_sportclient_method(action.method_name, False)
                    result.success = True
                    result.message = f"Executed {action.display_name}"
            
            else:
                # Simple actions — no extra args (Damp, StandUp, StandDown, etc.)
                exec_success = self._execute_sportclient_method(action.method_name)
                
                if exec_success:
                    result.success = True
                    result.message = f"Executed {action.display_name}"
                    
                    # Wait for action duration
                    wait_time = action.duration
                    time.sleep(wait_time)
            
            # Post-action operations
            self._post_action_stand_if_needed(action)
            
            # Update state based on action
            if action_name == "stand_up":
                self.robot_state.is_standing = True
            elif action_name in ["stand_down", "sit"]:
                self.robot_state.is_standing = False
            elif action_name == "stop_move":
                self.robot_state.is_moving = False
            
            result.duration = time.time() - start_time
            
            # Callback
            if self.on_action_complete:
                self.on_action_complete(action, result)
        
        except Exception as e:
            logger.error(f"Action execution error: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.duration = time.time() - start_time
            
            # Callback
            if self.on_action_error:
                self.on_action_error(action, e)
        
        finally:
            self.current_action = None
            self.action_start_time = None
        
        return result
    
    def execute_sequence(
        self,
        action_names: list[str],
        parameters_list: Optional[list[Dict[str, Any]]] = None,
        stop_on_error: bool = True
    ) -> list[RobotCommandResult]:
        """
        Execute a sequence of actions
        
        Args:
            action_names: List of action names
            parameters_list: Parameters for each action
            stop_on_error: Stop if any action fails
            
        Returns:
            List of execution results
        """
        results = []
        
        if parameters_list is None:
            parameters_list = [None] * len(action_names)
        
        logger.info(f"Starting action sequence: {action_names}")
        
        for i, (action_name, params) in enumerate(zip(action_names, parameters_list)):
            logger.info(f"Sequence step {i+1}/{len(action_names)}: {action_name}")
            
            result = self.execute_action(action_name, params)
            results.append(result)
            
            if not result.success and stop_on_error:
                logger.warning(f"Action failed, stopping sequence")
                break
            
            # Brief pause between actions
            time.sleep(0.5)
        
        logger.info(f"Sequence complete: {sum(1 for r in results if r.success)}/{len(results)} succeeded")
        return results
    
    def list_actions(self) -> list[str]:
        """List all available actions"""
        return self.library.list_names()
    
    def get_action_info(self, action_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about an action"""
        schema = self.library.get(action_name)
        if not schema:
            return None
        
        return {
            "name": schema.name,
            "display_name": schema.display_name,
            "description": schema.description,
            "category": schema.category.value,
            "difficulty": schema.difficulty.value,
            "duration": schema.duration,
            "requires_standing": schema.requires_standing,
            "voice_commands": schema.voice_commands,
            "min_battery": schema.min_battery,
        }
    
    def cleanup(self):
        """Clean up executor"""
        logger.info("Cleaning up BehaviorExecutor...")
        try:
            if self.sport_client:
                self.sport_client.StopMove()
            self.connected = False
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


# Test/Demo
if __name__ == "__main__":
    executor = BehaviorExecutor()
    
    print("\nAvailable Actions:")
    for name in executor.list_actions()[:5]:
        info = executor.get_action_info(name)
        print(f"  {info['display_name']}: {info['description']}")
    
    print("\nTesting action execution (demo mode)...")
    result = executor.execute_action("stand_up")
    print(f"Result: {result.success} - {result.message}")
    
    executor.cleanup()
