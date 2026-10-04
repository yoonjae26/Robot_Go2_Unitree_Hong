#!/usr/bin/env python3
"""
Main entry point for Go2 Voice Control with OpenAI
Supports multiple operation modes and configurations
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
import logging

from app.voice.voice_control_go2 import VoiceControlGo2
from app.voice.full_voice_pipeline import FullVoicePipeline
from app.core.robot_controller import RobotController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mode_interactive(args):
    """Interactive voice control mode"""
    logger.info("Starting interactive mode")
    
    controller = VoiceControlGo2(
        openai_api_key=args.openai_key,
        robot_ip=args.robot_ip,
        model=args.model,
        stt_model=args.stt_model
    )
    
    controller.interactive_mode()


def mode_single(args):
    """Single command mode"""
    logger.info("Starting single command mode")
    
    controller = VoiceControlGo2(
        openai_api_key=args.openai_key,
        robot_ip=args.robot_ip,
        model=args.model,
        stt_model=args.stt_model
    )
    
    result = controller.single_command_mode(timeout=args.timeout)
    
    if result:
        import json
        print("\nResult:")
        print(json.dumps(result, indent=2))


def mode_full_pipeline(args):
    """Full pipeline mode with TTS feedback"""
    logger.info("Starting full pipeline mode")
    
    pipeline = FullVoicePipeline(
        openai_api_key=args.openai_key,
        robot_ip=args.robot_ip,
        model=args.model,
        stt_model=args.stt_model,
        tts_enabled=not args.no_tts,
        ptt_mode=getattr(args, "ptt", False),
        ptt_key=getattr(args, "ptt_key", "q"),
    )
    
    if args.num_commands:
        pipeline.interactive_session(num_commands=args.num_commands)
    else:
        pipeline.interactive_session()


def mode_test_robot(args):
    """Test robot connection and commands"""
    logger.info("Starting robot test mode")

    import time

    controller = RobotController(robot_ip=args.robot_ip)

    try:
        print("\n🤖 Robot Test Mode")
        print("="*50)

        # Test stand up
        print("1. Testing stand_up...")
        result = controller.execute_action("stand_up")
        print(f"   Result: {result['success']}")
        time.sleep(2)

        # Test move forward
        print("2. Testing move forward (2s)...")
        result = controller.execute_action("move", vx=0.3, vy=0.0, omega=0.0, duration=2.0)
        print(f"   Result: {result['success']}")
        time.sleep(1)

        # Test rotate
        print("3. Testing move rotate (2s)...")
        result = controller.execute_action("move", vx=0.0, vy=0.0, omega=0.3, duration=2.0)
        print(f"   Result: {result['success']}")
        time.sleep(1)

        # Test stop
        print("4. Testing stop_move...")
        result = controller.execute_action("stop_move")
        print(f"   Result: {result['success']}")
        
        print("\n✅ Robot test completed")
    
    except Exception as e:
        logger.error(f"Robot test error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        controller.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Go2 Robot Voice Control with OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive voice control with continuous listening
  python main_go2.py --mode interactive
  
  # Single command mode
  python main_go2.py --mode single
  
  # Full pipeline with TTS feedback
  python main_go2.py --mode pipeline
  
  # Test robot commands
  python main_go2.py --mode test
  
  # Use specific OpenAI model
  python main_go2.py --model gpt-3.5-turbo
  
  # Use better STT model
  python main_go2.py --stt-model small
        """
    )
    
    # General arguments
    parser.add_argument(
        "--mode",
        choices=["interactive", "single", "pipeline", "test"],
        default="pipeline",
        help="Operation mode (default: pipeline)"
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.123.18",
        help="Go2 robot IP address (default: 192.168.123.18)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini). Options: gpt-4o-mini, gpt-4o, gpt-3.5-turbo"
    )
    parser.add_argument(
        "--stt-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper STT model size (default: base). Larger = slower but more accurate"
    )
    
    # Mode-specific arguments
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Listening timeout in seconds for single command mode"
    )
    parser.add_argument(
        "--num-commands",
        type=int,
        help="Number of commands to process in pipeline mode"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech feedback"
    )
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="Push-to-talk mode: press Q key to start/stop recording"
    )
    parser.add_argument(
        "--ptt-key",
        default="q",
        help="Key for push-to-talk (default: q)"
    )
    
    args = parser.parse_args()
    
    # Verify OpenAI API key
    if args.mode != "test":
        openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("❌ Error: OPENAI_API_KEY not provided")
            print("   Set it with: export OPENAI_API_KEY='your-key'")
            print("   Or use: --openai-key your-key")
            sys.exit(1)
    
    try:
        # Route to appropriate mode
        if args.mode == "interactive":
            mode_interactive(args)
        elif args.mode == "single":
            mode_single(args)
        elif args.mode == "pipeline":
            mode_full_pipeline(args)
        elif args.mode == "test":
            mode_test_robot(args)
    
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
        print("\n\n👋 Session ended")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
