#!/usr/bin/env python3
"""
Voice Control for Go2 Robot — Korean Language
Pipeline: 🎤 한국어 음성 → STT (Whisper/ko) → OpenAI LLM → JSON → RobotController → 🤖 Go2
"""

import argparse
import sys
import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from realtime_stt_pipeline import RealtimeSTTPipeline
from robot_controller import RobotController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Korean system prompt — instructs LLM to map Korean commands to robot actions
KOREAN_SYSTEM_PROMPT = """당신은 Unitree Go2 사족 보행 로봇을 제어하는 AI 어시스턴트입니다.
사용자의 한국어 음성 명령을 이해하여 로봇 행동 JSON으로 변환하세요.

[사용 가능한 행동 (action)]
- damp          : 관절 이완 (이완, 릴렉스, 힘 빼)
- stand_up      : 일어서기 (일어서, 기립, 서)
- stand_down    : 앉기/엎드리기 (앉아, 엎드려, 내려가)
- move          : 이동 — vx(앞+/뒤-), vy(왼쪽+/오른쪽-), omega(반시계+) (앞으로, 전진, 뒤로, 옆으로, 돌아)
- stop_move     : 정지 (멈춰, 그만, 스톱)
- hand_stand    : 물구나무 서기 (물구나무, 손으로 서기)
- balanced_stand: 균형 서기 / 하트 포즈 (하트, 균형, 하트 포즈)
- recovery      : 넘어짐 후 회복 (회복, 다시 일어나)
- left_flip     : 왼쪽 옆돌기 (왼쪽 플립, 왼쪽 공중제비)
- back_flip     : 뒤로 공중제비 (백플립, 공중제비)
- free_walk     : 자유 걷기 (걸어, 자유 보행)
- free_bound    : 바운딩 달리기 (바운딩, 달려)
- free_avoid    : 장애물 회피 (피해, 회피)
- walk_upright  : 직립 보행 (직립, 사람처럼 걸어, 두발로)
- cross_step    : 크로스 스텝 (크로스 스텝)
- free_jump     : 점프 (점프, 뛰어, 뛰어올라)

[규칙]
1. 반드시 JSON만 응답하세요 (설명 텍스트 없이).
2. 여러 행동을 말하면 "actions" 배열에 순서대로 넣으세요.
   예: "하트 포즈하고 점프해" → "actions": ["balanced_stand", "free_jump"]
3. move 명령은 vx/vy/omega 값을 설정하세요 (기본 속도 0.3).
4. response는 친근한 한국어로 작성하세요.

[출력 JSON 형식]
{
  "understood": true,
  "actions": ["action1", "action2"],
  "speed": 0.5,
  "duration": 2.0,
  "vx": 0.0,
  "vy": 0.0,
  "omega": 0.0,
  "response": "한국어 응답"
}"""


class VoiceControlGo2:
    """한국어 음성으로 Go2 로봇을 제어하는 클래스"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        robot_ip: str = "192.168.12.1",
        model: str = "gpt-4o",
        stt_model: str = "base",
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.model = model

        logger.info(f"OpenAI 모델: {model}, STT: {stt_model}, 로봇: {robot_ip}")

        # STT pipeline (Korean)
        self.stt_pipeline = RealtimeSTTPipeline(model_name=stt_model, language="ko")

        # Robot controller
        self.robot_controller = RobotController(robot_ip=robot_ip)

    def _call_llm(self, korean_text: str) -> Dict[str, Any]:
        """OpenAI API 호출 → JSON 반환"""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": KOREAN_SYSTEM_PROMPT},
                {"role": "user", "content": korean_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=512,
        )
        raw = response.choices[0].message.content

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract JSON block
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"understood": False, "actions": [], "response": "JSON 파싱 실패"}

    def _execute_command(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM JSON → 로봇 실행
        "actions" 배열이 있으면 순서대로 실행 (예: 하트 → 점프)
        """
        actions: List[str] = command_dict.get("actions") or []
        if not actions and command_dict.get("action"):
            actions = [command_dict["action"]]

        if not actions:
            return {"success": False, "reason": "실행할 행동이 없습니다."}

        results = []
        for action_name in actions:
            logger.info(f"실행 중: {action_name}")
            print(f"🤖 실행: {action_name}")
            result = self.robot_controller.execute_action(
                action_name,
                speed=command_dict.get("speed", 0.5),
                duration=command_dict.get("duration", 2.0),
                vx=command_dict.get("vx"),
                vy=command_dict.get("vy"),
                omega=command_dict.get("omega"),
            )
            results.append(result)
            if not result.get("success"):
                logger.warning(f"행동 실패: {action_name} — {result.get('error')}")
                break

        return {
            "success": all(r.get("success") for r in results),
            "actions": actions,
            "results": results,
        }

    def process_voice_command(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """음성 입력 한 사이클 처리: 듣기 → STT → LLM → 실행"""
        try:
            print("\n🎤 듣고 있습니다... (말씀하세요)")
            user_speech = self.stt_pipeline.listen_and_transcribe(timeout=timeout)

            if not user_speech:
                print("❌ 음성이 감지되지 않았습니다. 다시 시도해 주세요.")
                return None

            print(f"📝 인식된 텍스트: {user_speech}")
            logger.info(f"STT 결과: {user_speech}")

            print("🧠 OpenAI 처리 중...")
            command_dict = self._call_llm(user_speech)
            logger.info(f"LLM 응답: {command_dict}")

            tts_response = command_dict.get("response", "")
            print(f"🤖 응답: {tts_response}")

            if command_dict.get("understood", False):
                exec_result = self._execute_command(command_dict)
                command_dict["execution_result"] = exec_result
                if exec_result.get("success"):
                    print("✅ 완료!")
                else:
                    print(f"⚠️  실행 실패: {exec_result}")
            else:
                print(f"⚠️  {tts_response}")

            command_dict["user_input"] = user_speech
            return command_dict

        except Exception as e:
            logger.error(f"오류 발생: {e}", exc_info=True)
            print(f"❌ 오류: {e}")
            return None

    def interactive_mode(self):
        """연속 음성 명령 모드"""
        print("\n🤖 Go2 로봇 한국어 음성 제어")
        print("=" * 50)
        print(f"  모델: {self.model}  |  STT: 한국어 (Whisper)")
        print("  '종료' 또는 '나가기'라고 말하면 종료됩니다.")
        print("=" * 50)

        try:
            while True:
                result = self.process_voice_command(timeout=15.0)

                if result:
                    user_input = result.get("user_input", "")
                    if any(kw in user_input for kw in ["종료", "나가기", "exit", "quit"]):
                        print("\n👋 종료합니다.")
                        break

        except KeyboardInterrupt:
            print("\n\n👋 사용자가 종료했습니다.")
        finally:
            self.cleanup()

    def single_command_mode(self, timeout: float = 10.0):
        """단일 명령 처리"""
        return self.process_voice_command(timeout=timeout)

    def cleanup(self):
        self.robot_controller.cleanup()
        self.stt_pipeline.cleanup()
        logger.info("종료 완료")


def main():
    parser = argparse.ArgumentParser(description="Go2 한국어 음성 제어")
    parser.add_argument("--robot-ip", default="192.168.12.1")
    parser.add_argument("--openai-key")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI 모델 (기본: gpt-4o)")
    parser.add_argument("--stt-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--interactive", action="store_true",
                        help="연속 음성 명령 모드")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    try:
        controller = VoiceControlGo2(
            openai_api_key=args.openai_key,
            robot_ip=args.robot_ip,
            model=args.model,
            stt_model=args.stt_model,
        )
        if args.interactive:
            controller.interactive_mode()
        else:
            controller.single_command_mode(timeout=args.timeout)

    except KeyboardInterrupt:
        logger.info("사용자 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
