# Go2 Voice Control with OpenAI API

A complete voice-controlled pipeline for Unitree Go2 robot using OpenAI's GPT models and Whisper STT.

**Pipeline Flow:**
```
🎤 Voice Input → STT (Whisper) → OpenAI LLM → Robot Commands → 🤖 Execution → 🔊 TTS Response
```

## Features

- **Voice Input**: Real-time speech recognition using Faster Whisper
- **LLM Processing**: OpenAI GPT-4 / GPT-3.5 for natural command understanding
- **Robot Control**: High-level Go2 robot commands (move, jump, flip, etc.)
- **Text-to-Speech**: Voice feedback using pyttsx3
- **Multiple Modes**: Interactive, single command, full pipeline, agent mode
- **State Management**: Real-time robot state tracking
- **Safety**: Built-in safety checks and conservative defaults

## Installation

### Prerequisites
- Python 3.8+
- Go2 robot on the network
- OpenAI API key
- Microphone and speakers

### Install Dependencies

```bash
# Install main dependencies
pip install -r setup.py install_requires

# Or manually:
pip install cyclonedds==0.10.2 numpy opencv-python pyttsx3
pip install faster-whisper sounddevice scipy webrtcvad openai

# For GPU acceleration (optional)
pip install torch torchaudio
```

### Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

Or pass it as a command-line argument:
```bash
python main_go2.py --openai-key 'sk-your-api-key-here'
```

## Usage

### 1. Main Entry Point - All-in-One

```bash
# Full pipeline mode (default) - continuous commands with TTS
python main_go2.py

# Single command
python main_go2.py --mode single

# Interactive mode (rapid commands)
python main_go2.py --mode interactive

# Test robot connection
python main_go2.py --mode test
```

### 2. Full Voice Pipeline (Recommended)

Complete pipeline with voice I/O and TTS feedback:

```bash
python full_voice_pipeline.py

# Options
python full_voice_pipeline.py --single                    # Single command
python full_voice_pipeline.py --num-commands 5            # 5 commands
python full_voice_pipeline.py --no-tts                    # No speech output
python full_voice_pipeline.py --model gpt-3.5-turbo       # Faster model
python full_voice_pipeline.py --stt-model small           # Better STT
```

### 3. Voice Control Go2

Direct voice control interface:

```bash
python voice_control_go2.py

# Options
python voice_control_go2.py --interactive                 # Continuous mode
python voice_control_go2.py --timeout 15                  # Custom timeout
python voice_control_go2.py --stt-model large             # Large Whisper model
```

### 4. Intelligent Agent

Advanced agent with reasoning and planning:

```bash
cd go2_agent
python main.py

# Options
python main.py --num-cycles 10                 # 10 cycles
python main.py --model gpt-3.5-turbo           # Cheaper model
```

### 5. Direct Robot Control Test

```bash
python robot_controller.py
```

## Command Examples

Here are some voice commands you can try:

- **Movement**: "Move forward", "Turn left", "Go backward"
- **Posture**: "Stand up", "Sit down", "Lie down"
- **Actions**: "Jump", "Flip", "Hand stand", "Walk upright"
- **Control**: "Stop", "Stay still", "Be careful"
- **Info**: "What's your battery?", "Are you okay?"

## File Structure

```
unitree_sdk2_python/
├── main_go2.py                    # Main entry point
├── voice_control_go2.py           # Voice control interface
├── full_voice_pipeline.py         # Complete pipeline with TTS
├── realtime_stt_pipeline.py       # STT module (Whisper)
├── robot_controller.py            # Robot control module
├── go2_agent/
│   └── main.py                    # Intelligent agent
└── example/go2/                   # SDK examples
```

## Architecture

### Components

1. **RealtimeSTTPipeline** (`realtime_stt_pipeline.py`)
   - Listens to microphone
   - Voice activity detection
   - Transcribes to text using Whisper

2. **RobotController** (`robot_controller.py`)
   - High-level Go2 commands
   - State management
   - Safety checks

3. **VoiceControlGo2** (`voice_control_go2.py`)
   - Coordinates STT, LLM, and Robot
   - JSON-based command parsing

4. **FullVoicePipeline** (`full_voice_pipeline.py`)
   - Complete end-to-end pipeline
   - Includes TTS feedback
   - Enhanced error handling

5. **Go2Agent** (`go2_agent/main.py`)
   - Intelligent agent with reasoning
   - Multi-step planning
   - State-aware decision making

## OpenAI Model Options

| Model | Speed | Cost | Quality |
|-------|-------|------|---------|
| gpt-4 | Fast✓ | Higher | Best |
| gpt-3.5-turbo | Faster | Lower | Good |
| gpt-4-turbo | Slow | Medium | Excellent |

Default: `gpt-4`

## STT Model Options

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| tiny | Very Fast | Fair | 39MB |
| base | Fast✓ | Good (default) | 140MB |
| small | Medium | Good+ | 466MB |
| medium | Slower | Excellent | 1.5GB |
| large | Very Slow | Best | 2.9GB |

## Troubleshooting

### "No connection to robot"
- Check robot IP: `ping 192.168.12.1`
- Verify network connectivity
- Ensure robot is powered on

### "OPENAI_API_KEY not set"
- Set environment: `export OPENAI_API_KEY='your-key'`
- Or pass as argument: `--openai-key 'your-key'`
- Check API key validity at openai.com

### "No speech detected"
- Check microphone: `arecord -l`
- Test with: `python test_microphone_input.py`
- Increase timeout: `--timeout 15`

### "Poor transcription quality"
- Use better STT model: `--stt-model small`
- Speak clearly and slower
- Reduce background noise

### "Low FPS / Slow response"
- Use faster model: `--model gpt-3.5-turbo`
- Use smaller STT: `--stt-model tiny`
- Reduce complexity

## Advanced Usage

### Batch Commands

```python
from full_voice_pipeline import FullVoicePipeline

pipeline = FullVoicePipeline()
pipeline.interactive_session(num_commands=5)
```

### Custom Action Handler

```python
from robot_controller import RobotController

controller = RobotController()
result = controller.execute_action("move_forward", duration=3, speed=0.7)
print(result)
```

### Agent with Custom Prompt

Modify the system prompt in `go2_agent/main.py` to add custom behaviors.

## Performance Tips

1. **Faster Response**: Use `gpt-3.5-turbo` instead of `gpt-4`
2. **Better Accuracy**: Use larger STT model (`small`, `medium`)
3. **Reduce Latency**: Run on machine close to robot
4. **Parallel Processing**: Pre-warm models on startup
5. **GPU Acceleration**: Install CUDA for faster Whisper

## Safety Considerations

- Default speed: 0.5 (conservative)
- Default duration: 2 seconds
- Always verify robot state before complex moves
- Keep emergency stop nearby
- Test in safe environment first

## API Reference

### VoiceControlGo2

```python
controller = VoiceControlGo2(openai_api_key="...", robot_ip="192.168.12.1")
result = controller.process_voice_command(timeout=10.0)
controller.interactive_mode()
```

### RobotController

```python
robot = RobotController(robot_ip="192.168.12.1")
result = robot.execute_action("move_forward", duration=2, speed=0.5)
robot.cleanup()
```

### FullVoicePipeline

```python
pipeline = FullVoicePipeline(tts_enabled=True)
result = pipeline.process_command_cycle(timeout=10.0)
pipeline.interactive_session(num_commands=5)
```

## Logging

Enable debug logging:

```bash
# Set log level
export LOG_LEVEL=DEBUG
python main_go2.py
```

Logs are printed to console and can be redirected:

```bash
python main_go2.py 2>&1 | tee robot_session.log
```

## Contributing

To extend or modify:

1. Add custom actions to `RobotController.ACTION_MAP`
2. Modify system prompts for different behaviors
3. Extend `Go2Agent` for multi-step planning
4. Add new STT/TTS backends

## License

See LICENSE file in repository

## Support

- Unitree Go2 Robot Documentation: https://unitreerobotics.com/
- OpenAI API Docs: https://platform.openai.com/docs/
- Faster Whisper: https://github.com/guillaumekln/faster-whisper

## Version History

- **v1.0.0** (2024): Initial release with voice control pipeline
