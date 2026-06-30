#!/usr/bin/env python3
"""
Full Voice Pipeline for Go2 Robot — Korean Language
Pipeline: 🎤 한국어 음성 → STT (Whisper/ko) → OpenAI LLM → JSON → RobotController → 🤖 Go2 → 🔊 TTS
"""

import argparse
import sys
import os
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from realtime_stt_pipeline import RealtimeSTTPipeline
from robot_controller import RobotController
from voice_control_go2 import KOREAN_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullVoicePipeline:
    """
    Full voice pipeline with TTS feedback — Korean language.
    🎤 음성 → STT → OpenAI LLM → 로봇 실행 → 🔊 TTS 응답
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        robot_ip: str = "192.168.12.1",
        model: str = "gpt-4o",
        stt_model: str = "base",
        tts_enabled: bool = True,
        tts_voice: str = "nova",
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.model = model
        self.tts_enabled = tts_enabled
        self.tts_voice = tts_voice

        logger.info(f"Pipeline 초기화 — 모델: {model}, STT: {stt_model}, TTS: {tts_enabled}")

        self.stt_pipeline = RealtimeSTTPipeline(model_name=stt_model, language="ko")
        self.robot_controller = RobotController(robot_ip=robot_ip)

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _call_llm(self, korean_text: str) -> Dict[str, Any]:
        """한국어 텍스트 → OpenAI API → JSON 명령"""
        import json
        import re

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
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"understood": False, "actions": [], "response": "JSON 파싱 실패"}

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """OpenAI TTS API로 한국어 음성 출력"""
        if not self.tts_enabled or not text.strip():
            return

        try:
            logger.info(f"TTS: {text}")
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=self.tts_voice,
                input=text,
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                response.stream_to_file(tmp_path)

            # Try available audio players
            for player in ["ffplay", "mpg123", "afplay"]:
                try:
                    if player == "ffplay":
                        subprocess.run(
                            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                            check=True, timeout=30
                        )
                    elif player == "mpg123":
                        subprocess.run(["mpg123", "-q", tmp_path], check=True, timeout=30)
                    elif player == "afplay":
                        subprocess.run(["afplay", tmp_path], check=True, timeout=30)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

        except Exception as e:
            logger.warning(f"TTS 오류 (계속 진행): {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_command(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """LLM JSON → 로봇 실행 (actions 배열 순서대로)"""
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

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def process_one_command(self, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
        """한 사이클: 듣기 → STT → LLM → 실행 → TTS"""
        try:
            print("\n🎤 듣고 있습니다... (말씀하세요)")
            user_speech = self.stt_pipeline.listen_and_transcribe(timeout=timeout)

            if not user_speech:
                print("❌ 음성이 감지되지 않았습니다.")
                return None

            print(f"📝 인식: {user_speech}")
            logger.info(f"STT: {user_speech}")

            print("🧠 OpenAI 처리 중...")
            command_dict = self._call_llm(user_speech)
            logger.info(f"LLM: {command_dict}")

            tts_text = command_dict.get("response", "")
            if tts_text:
                print(f"💬 {tts_text}")

            if command_dict.get("understood", False):
                exec_result = self._execute_command(command_dict)
                command_dict["execution_result"] = exec_result

                if exec_result.get("success"):
                    print("✅ 완료!")
                    self.speak(tts_text)
                else:
                    fail_msg = "행동을 실행할 수 없었어요."
                    print(f"⚠️  실행 실패")
                    self.speak(fail_msg)
            else:
                print(f"⚠️  명령을 이해하지 못했습니다.")
                self.speak(tts_text or "죄송해요, 다시 말씀해 주세요.")

            command_dict["user_input"] = user_speech
            return command_dict

        except Exception as e:
            logger.error(f"오류: {e}", exc_info=True)
            print(f"❌ 오류: {e}")
            return None

    # ------------------------------------------------------------------
    # Interactive session
    # ------------------------------------------------------------------

    def interactive_session(self, num_commands: Optional[int] = None):
        """연속 음성 명령 세션"""
        print("\n🤖 Go2 로봇 전체 음성 파이프라인 (한국어)")
        print("=" * 55)
        print(f"  모델: {self.model}  |  TTS 음성: {self.tts_voice}")
        print("  예시 명령: '하트 포즈하고 점프해', '앞으로 가', '점프'")
        print("  '종료'라고 말하면 종료됩니다.")
        print("=" * 55)

        count = 0
        try:
            while True:
                if num_commands and count >= num_commands:
                    break

                result = self.process_one_command(timeout=15.0)
                count += 1

                if result:
                    user_input = result.get("user_input", "")
                    if any(kw in user_input for kw in ["종료", "나가기", "exit", "quit"]):
                        self.speak("종료합니다. 안녕히 계세요!")
                        print("\n👋 종료합니다.")
                        break

        except KeyboardInterrupt:
            print("\n\n👋 사용자가 종료했습니다.")
        finally:
            self.cleanup()

    def cleanup(self):
        self.robot_controller.cleanup()
        self.stt_pipeline.cleanup()
        logger.info("Pipeline 종료")


def main():
    parser = argparse.ArgumentParser(description="Go2 전체 음성 파이프라인 (한국어)")
    parser.add_argument("--robot-ip", default="192.168.12.1")
    parser.add_argument("--openai-key")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--stt-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--no-tts", action="store_true", help="TTS 비활성화")
    parser.add_argument("--tts-voice", default="nova",
                        choices=["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
                        help="OpenAI TTS 음성 (기본: nova)")
    parser.add_argument("--num-commands", type=int, help="처리할 명령 수")
    args = parser.parse_args()

    try:
        pipeline = FullVoicePipeline(
            openai_api_key=args.openai_key,
            robot_ip=args.robot_ip,
            model=args.model,
            stt_model=args.stt_model,
            tts_enabled=not args.no_tts,
            tts_voice=args.tts_voice,
        )

        if args.num_commands:
            pipeline.interactive_session(num_commands=args.num_commands)
        else:
            pipeline.interactive_session()

    except KeyboardInterrupt:
        logger.info("사용자 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
