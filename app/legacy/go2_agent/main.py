#!/usr/bin/env python3
"""
Go2 Agent Main Module — Korean Language
Advanced agent for Go2 robot with state management and behavior planning
Pipeline: 🎤 한국어 음성 → STT → OpenAI LLM → 로봇 실행
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.robot_controller import RobotController
from app.voice.realtime_stt_pipeline import RealtimeSTTPipeline
from app.voice.voice_control_go2 import KOREAN_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Current state of the robot"""
    mode: str = "idle"  # idle, moving, executing, error
    position: str = "standing"  # standing, sitting, lying
    last_command: Optional[str] = None
    battery_level: float = 100.0
    is_safe: bool = True
    error_message: Optional[str] = None


class Go2Agent:
    """Intelligent agent for Go2 robot"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        robot_ip: str = "192.168.12.1",
        model: str = "gpt-4",
        stt_model: str = "base",
    ):
        """
        Initialize Go2 Agent
        
        Args:
            openai_api_key: OpenAI API key
            robot_ip: Robot IP address
            model: OpenAI model to use
            stt_model: STT model size
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.model = model

        logger.info(f"Go2 Agent 초기화 — 모델: {model}, STT: {stt_model}, 로봇: {robot_ip}")

        self.robot_controller = RobotController(robot_ip=robot_ip)
        self.stt_pipeline = RealtimeSTTPipeline(model_name=stt_model, language="ko")

        self.state = RobotState()
        self.action_history = []
    
    def _get_state_prompt(self) -> str:
        """현재 로봇 상태를 시스템 프롬프트에 주입"""
        state_context = f"\n\n[현재 로봇 상태]\n위치: {self.state.position} | 모드: {self.state.mode} | 안전: {self.state.is_safe}"
        return KOREAN_SYSTEM_PROMPT + state_context
    
    def listen_command(self, timeout: float = 10.0) -> Optional[str]:
        """음성 명령 듣기"""
        logger.info("명령 대기 중")
        try:
            print("🎤 듣고 있습니다...")
            return self.stt_pipeline.listen_and_transcribe(timeout=timeout)
        except Exception as e:
            logger.error(f"듣기 오류: {e}")
            return None

    def think(self, user_input: str) -> Dict[str, Any]:
        """OpenAI LLM으로 명령 해석 및 계획"""
        import re

        logger.info(f"명령 처리 중: {user_input}")

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_state_prompt()},
                    {"role": "user", "content": user_input},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=512,
            )

            response_text = response.choices[0].message.content
            logger.info(f"LLM 응답: {response_text}")

            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return {"understood": False, "response": response_text}

        except Exception as e:
            logger.error(f"LLM 오류: {e}")
            return {"understood": False, "error": str(e)}
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """결정에 따라 로봇 행동 실행 (actions 배열 지원)"""
        if not decision.get("understood"):
            logger.info("명령을 이해하지 못함")
            return {"success": False, "reason": "not_understood"}

        actions = decision.get("actions") or []
        if not actions and decision.get("action"):
            actions = [decision["action"]]

        if not actions:
            logger.info("실행할 행동 없음")
            return {"success": False, "reason": "no_action"}

        logger.info(f"실행할 행동: {actions}")
        results = []
        try:
            for action_name in actions:
                result = self.robot_controller.execute_action(
                    action_name,
                    speed=decision.get("speed", 0.5),
                    duration=decision.get("duration", 2.0),
                    vx=decision.get("vx"),
                    vy=decision.get("vy"),
                    omega=decision.get("omega"),
                )
                self._update_state(action_name)
                self.action_history.append({
                    "action": action_name,
                    "result": result,
                })
                results.append(result)
                if not result.get("success"):
                    break

            return {
                "success": all(r.get("success") for r in results),
                "actions": actions,
                "results": results,
            }

        except Exception as e:
            logger.error(f"실행 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_state(self, action: str):
        """행동 후 로봇 상태 업데이트"""
        action = action.lower()

        if action == "stand_up":
            self.state.position = "standing"
        elif action == "stand_down":
            self.state.position = "sitting"
        elif action == "move":
            self.state.mode = "moving"
        elif action == "stop_move":
            self.state.mode = "idle"

        self.state.last_command = action
        logger.info(f"상태 업데이트: {self.state}")
    
    def run_cycle(self, timeout: float = 10.0) -> Dict[str, Any]:
        """한 사이클 실행: 듣기 → 판단 → 실행"""
        logger.info("Agent 사이클 시작")

        user_input = self.listen_command(timeout=timeout)
        if not user_input:
            return {"success": False, "reason": "no_input"}

        print(f"📝 인식: {user_input}")
        logger.info(f"음성 입력: {user_input}")

        decision = self.think(user_input)
        result = self.act(decision)

        return {
            "success": result.get("success", False),
            "user_input": user_input,
            "agent_decision": decision,
            "execution": result,
        }
    
    def interactive_session(self, num_cycles: Optional[int] = None):
        """연속 음성 명령 세션"""
        logger.info("Agent 세션 시작")

        print("\n" + "=" * 55)
        print("🤖 Go2 인텔리전트 에이전트 (한국어)")
        print("=" * 55)
        print(f"  모델: {self.model}  |  STT: 한국어 (Whisper)")
        print("  예시: '하트 포즈하고 점프해', '앞으로 가', '멈춰'")
        print("  '종료'라고 말하면 종료됩니다.")
        print("=" * 55 + "\n")

        cycle_count = 0
        try:
            while True:
                if num_cycles and cycle_count >= num_cycles:
                    break

                cycle_count += 1
                print(f"\n--- 사이클 {cycle_count} ---")
                result = self.run_cycle(timeout=15.0)

                if result.get("success"):
                    response = result.get("agent_decision", {}).get("response", "완료!")
                    print(f"🤖 {response}")
                else:
                    print(f"⚠️  실패: {result.get('reason', 'unknown')}")

                user_input = result.get("user_input", "")
                if any(kw in user_input for kw in ["종료", "나가기", "exit", "quit"]):
                    print("\n👋 종료합니다.")
                    break

        except KeyboardInterrupt:
            print("\n\n👋 사용자가 종료했습니다.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("Agent 종료")
        try:
            self.robot_controller.cleanup()
            self.stt_pipeline.cleanup()
        except Exception as e:
            logger.error(f"종료 오류: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Go2 인텔리전트 에이전트 (한국어)")
    parser.add_argument("--robot-ip", default="192.168.12.1")
    parser.add_argument("--openai-key")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--stt-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--num-cycles", type=int)
    args = parser.parse_args()

    try:
        agent = Go2Agent(
            openai_api_key=args.openai_key,
            robot_ip=args.robot_ip,
            model=args.model,
            stt_model=args.stt_model,
        )
        agent.interactive_session(num_cycles=args.num_cycles)

    except KeyboardInterrupt:
        logger.info("사용자 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
