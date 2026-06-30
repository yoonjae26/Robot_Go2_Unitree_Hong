# unitree_sdk2_python
Python interface for unitree sdk2

## 🎤 Korean Voice Control for Go2 (한국어 음성 제어)

Control the Unitree Go2 robot with Korean voice commands via OpenAI Whisper STT and GPT-4o.

**Pipeline:**
```
🎤 한국어 음성 → Whisper STT → OpenAI GPT-4o → JSON → RobotController → 🤖 Go2 → 🔊 TTS
```

### Quick Start

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key"

# Full pipeline with TTS feedback (recommended)
python3 main_go2.py --mode pipeline

# Voice control only (no TTS)
python3 main_go2.py --mode interactive

# Test robot commands (no voice needed)
python3 main_go2.py --mode test --robot-ip 192.168.12.1
```

### Voice Command Examples (한국어)

| 말하기 | 행동 |
|--------|------|
| "하트 포즈하고 점프해" | balanced_stand → free_jump (sequence) |
| "앞으로 가" | move forward |
| "점프" | free_jump |
| "백플립" | back_flip |
| "물구나무" | hand_stand |
| "멈춰" | stop_move |
| "종료" | exit session |

### Dependencies

```bash
pip install openai>=1.0.0 faster-whisper sounddevice numpy
# Audio playback (one of):
apt install ffmpeg   # Linux
```

### Files

| File | Description |
|------|-------------|
| `main_go2.py` | Entry point — `--mode pipeline/interactive/test` |
| `full_voice_pipeline.py` | Full STT → LLM → Robot → TTS pipeline |
| `voice_control_go2.py` | Voice control core + Korean system prompt |
| `robot_controller.py` | High-level robot action interface |
| `behavior_executor.py` | SportClient execution layer |
| `behavior_schema.py` | Action definitions with Korean voice commands |
| `behavior_library.py` | Action registry by category |
| `action_registry.py` | Fast action lookup by name/alias/mode_id |
| `realtime_stt_pipeline.py` | Whisper STT pipeline (Korean) |
| `go2_agent/main.py` | Advanced agent with state tracking |

---

# Installation
## Dependencies
- Python >= 3.8
- cyclonedds == 0.10.2
- numpy
- opencv-python

### Installing from source
Execute the following commands in the terminal:
```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```
## FAQ
##### 1. Error when `pip3 install -e .`:
```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```
This error mentions that the cyclonedds path could not be found. First compile and install cyclonedds:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```
Enter the unitree_sdk2_python directory, set `CYCLONEDDS_HOME` to the path of the cyclonedds you just compiled, and then install unitree_sdk2_python.
```bash
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME=~/cyclonedds/install
pip3 install -e .
```
For details, see: https://pypi.org/project/cyclonedds/#installing-with-pre-built-binaries

# Usage
The Python sdk2 interface maintains consistency with the unitree_sdk2 interface, achieving robot status acquisition and control through request-response or topic subscription/publishing. Example programs are located in the `/example` directory. Before running the examples, configure the robot's network connection as per the instructions in the document at https://support.unitree.com/home/en/developer/Quick_start.
## DDS Communication
In the terminal, execute:
```bash
python3 ./example/helloworld/publisher.py
```
Open a new terminal and execute:
```bash
python3 ./example/helloworld/subscriber.py
```
You will see the data output in the terminal. The data structure transmitted between `publisher.py` and `subscriber.py` is defined in `user_data.py`, and users can define the required data structure as needed.
## High-Level Status and Control
The high-level interface maintains consistency with unitree_sdk2 in terms of data structure and control methods. For detailed information, refer to https://support.unitree.com/home/en/developer/sports_services.
### High-Level Status
Execute the following command in the terminal:
```bash
python3 ./example/high_level/read_highstate.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected,.
### High-Level Control
Execute the following command in the terminal:
```bash
python3 ./example/high_level/sportmode_test.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. This example program provides several test methods, and you can choose the required tests as follows:
```python
test.StandUpDown() # Stand up and lie down
# test.VelocityMove() # Velocity control
# test.BalanceAttitude() # Attitude control
# test.TrajectoryFollow() # Trajectory tracking
# test.SpecialMotions() # Special motions
```
## Low-Level Status and Control
The low-level interface maintains consistency with unitree_sdk2 in terms of data structure and control methods. For detailed information, refer to https://support.unitree.com/home/en/developer/Basic_services.
### Low-Level Status
Execute the following command in the terminal:
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The program will output the state of the right front leg hip joint, IMU, and battery voltage.
### Low-Level Motor Control
First, use the app to turn off the high-level motion service (sport_mode) to prevent conflicting instructions.
Execute the following command in the terminal:
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The left hind leg hip joint will maintain a 0-degree position (for safety, set kp=10, kd=1), and the left hind leg calf joint will continuously output 1Nm of torque.
## Wireless Controller Status
Execute the following command in the terminal:
```bash
python3 ./example/wireless_controller/wireless_controller.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The terminal will output the status of each key. For the definition and data structure of the remote control keys, refer to https://support.unitree.com/home/en/developer/Get_remote_control_status.
## Front Camera
Use OpenCV to obtain the front camera (ensure to run on a system with a graphical interface, and press ESC to exit the program):
```bash
python3 ./example/front_camera/camera_opencv.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected.

## Obstacle Avoidance Switch
```bash
python3 ./example/obstacles_avoid_switch/obstacles_avoid_switch.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected. The robot will cycle obstacle avoidance on and off. For details on the obstacle avoidance service, see https://support.unitree.com/home/en/developer/ObstaclesAvoidClient

## Light and volume control
```bash
python3 ./example/vui_client/vui_client_example.py enp2s0
```
Replace `enp2s0` with the name of the network interface to which the robot is connected.T he robot will cycle the volume and light brightness. The interface is detailed at https://support.unitree.com/home/en/developer/VuiClient
# Robot_Go2_Unitree_Hong
