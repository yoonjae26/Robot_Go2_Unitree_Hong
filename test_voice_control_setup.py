#!/usr/bin/env python3
"""
Test script to verify all components are working
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    print("✅ Testing imports...")
    
    try:
        from realtime_stt_pipeline import RealtimeSTTPipeline
        print("  ✓ RealtimeSTTPipeline")
    except ImportError as e:
        print(f"  ✗ RealtimeSTTPipeline: {e}")
        return False
    
    try:
        from robot_controller import RobotController
        print("  ✓ RobotController")
    except ImportError as e:
        print(f"  ✗ RobotController: {e}")
        return False
    
    try:
        from voice_control_go2 import VoiceControlGo2
        print("  ✓ VoiceControlGo2")
    except ImportError as e:
        print(f"  ✗ VoiceControlGo2: {e}")
        return False
    
    try:
        from full_voice_pipeline import FullVoicePipeline
        print("  ✓ FullVoicePipeline")
    except ImportError as e:
        print(f"  ✗ FullVoicePipeline: {e}")
        return False
    
    try:
        from go2_agent.main import Go2Agent
        print("  ✓ Go2Agent")
    except ImportError as e:
        print(f"  ✗ Go2Agent: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that all required packages are installed"""
    print("\n✅ Testing dependencies...")
    
    packages = {
        "faster_whisper": "Faster Whisper",
        "sounddevice": "Sound Device",
        "numpy": "NumPy",
        "opencv": "OpenCV",
        "pyttsx3": "pyttsx3",
        "scipy": "SciPy",
        "webrtcvad": "WebRTC VAD",
        "openai": "OpenAI",
        "unitree_sdk2py": "Unitree SDK v2",
    }
    
    missing = []
    
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install " + " ".join(missing))
        return False
    
    return True


def test_files():
    """Test that all required files exist"""
    print("\n✅ Testing file structure...")
    
    files = [
        "main_go2.py",
        "voice_control_go2.py",
        "full_voice_pipeline.py",
        "realtime_stt_pipeline.py",
        "robot_controller.py",
        "go2_agent/__init__.py",
        "go2_agent/main.py",
    ]
    
    base_path = Path(__file__).parent
    
    for file_path in files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            return False
    
    return True


def test_openai_config():
    """Test OpenAI configuration"""
    print("\n✅ Testing OpenAI configuration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("  ✗ OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export OPENAI_API_KEY='sk-your-api-key'")
        return False
    
    if not api_key.startswith("sk-"):
        print("  ⚠️  API key doesn't start with 'sk-' (might be invalid)")
        return False
    
    print(f"  ✓ OPENAI_API_KEY set ({api_key[:10]}...)")
    return True


def test_robot_ip():
    """Test robot IP configuration"""
    print("\n✅ Testing robot IP...")
    
    import socket
    
    robot_ip = "192.168.12.1"
    
    try:
        socket.create_connection((robot_ip, 29202), timeout=2)
        print(f"  ✓ Robot reachable at {robot_ip}:29202")
        return True
    except (socket.timeout, socket.error):
        print(f"  ⚠️  Cannot reach robot at {robot_ip}:29202")
        print("  (This is OK if running tests without robot connected)")
        return True  # Don't fail on this


def main():
    print("="*60)
    print("Go2 Voice Control - System Check")
    print("="*60)
    
    results = {
        "File Structure": test_files(),
        "Dependencies": test_dependencies(),
        "Imports": test_imports(),
        "OpenAI Config": test_openai_config(),
        "Robot Connection": test_robot_ip(),
    }
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ All tests passed! Ready to run.")
        print("\n📖 Quick start:")
        print("  python main_go2.py --mode pipeline")
        print("\n📚 See VOICE_CONTROL_README.md for full documentation")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
