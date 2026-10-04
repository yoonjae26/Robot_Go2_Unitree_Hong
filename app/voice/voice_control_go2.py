#!/usr/bin/env python3
"""
Voice Control for Go2 Robot — Korean Language
Pipeline: 🎤 한국어 음성 → STT (Whisper/ko) → OpenAI LLM → JSON → RobotController → 🤖 Go2
"""


import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
import re
import json
import logging
from typing import Optional, Dict, Any, List

from openai import OpenAI
from app.voice.realtime_stt_pipeline import RealtimeSTTPipeline
from app.core.robot_controller import RobotController
from app.core.action_merge import merge_consecutive_moves

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
- stand_down    : 몸 전체를 바닥에 낮춰 엎드리기/눕기 (엎드려, 내려가, 누워, 숙여) — "앉아"는 쓰지
                  마세요, 그건 sit입니다.
- move          : 이동 — vx(앞+/뒤-), vy(왼쪽+/오른쪽-), omega(반시계+) (앞으로, 전진, 뒤로, 옆으로, 돌아)
- stop_move     : 정지 (멈춰, 그만, 스톱)
- hand_stand    : 물구나무 서기 (물구나무, 손으로 서기)
- balanced_stand: 균형 서기 / 하트 포즈 (하트, 균형, 하트 포즈)
- recovery      : 넘어짐 후 회복 (회복, 다시 일어나)
- left_flip     : 왼쪽 옆돌기 (왼쪽 플립, 레프트 플립, 왼쪽 공중제비)
- back_flip     : 뒤로 공중제비 (백플립, 팩폴립, 팩플립, 백 플립, back flip, 공중제비)
- free_walk     : 자유 걷기 (걸어, 자유 보행)
- free_bound    : 바운딩 달리기 (바운딩, 달려)
- free_avoid    : 장애물 회피 (피해, 회피)
- walk_upright  : 직립 보행 (직립, 사람처럼 걸어, 두발로)
- cross_step    : 크로스 스텝 (크로스 스텝)
- free_jump     : 점프 (점프, 뛰어, 뛰어올라)
- hello         : 인사/손 흔들기 (안녕, 인사해, 손 흔들어, 반가워)
- stretch       : 스트레칭 (스트레칭, 기지개)
- dance1        : 댄스 1 (춤춰, 댄스, 춤)
- dance2        : 댄스 2 (춤 2, 댄스 2)
- front_flip    : 앞 공중제비 (앞 플립, 프론트 플립, 앞으로 뒤집어)
- scrape        : 발로 긁기 (긁어, 발 긁기, 스크레이프)
- front_jump    : 앞으로 점프 (앞점프, 앞으로 뛰어, 프론트 점프)
- front_pounce  : 앞으로 덮치기 (덮쳐, 프론트 파운스)
- pose          : 몸 기울이기/포즈 (포즈, 몸 기울여)
- trot_run      : 트롯 걸음(말처럼 뛰는 걸음걸이) (트롯, 트롯 런, 말처럼 뛰기, 트롯날, 츄롯런)
- static_walk   : 천천히 정적 보행 (천천히 걸어, 스태틱 워크, 정적 보행)
- sit           : 개처럼 엉덩이 대고 앉기 (앉아, 앉기, 앉아봐, 엉덩이 대고 앉아). stand_down과
                  다름 — stand_down은 몸 전체를 바닥에 눕히는 것이고, sit은 앞다리를 세운 채
                  엉덩이만 바닥에 대는 것입니다.
- rise_sit      : 앉은(sit) 자세에서 일어서기. stand_up과 다름 — stand_up은 엎드린 자세에서 일어서는
                  것이고, rise_sit은 sit 자세 다음에만 쓰세요 (예: "앉은 데서 일어나").
- content       : 기쁨/애교 동작 (애교, 좋아하는 동작, 컨텐트)
- classic_walk  : 클래식(기본) 걸음걸이로 전환 (클래식 워크, 기본 걸음, 일반 걸음)
- euler         : 몸을 롤/피치/요 방향으로 기울이기 — roll(좌우 기울임), pitch(앞뒤 기울임),
                  yaw(좌우 회전 기울임)만 사용, vx/vy/omega는 넣지 마세요 (몸 기울여, 고개 숙여,
                  옆으로 기울여). Move와 달리 자동으로 원위치되지 않으므로 duration이 지나면
                  시스템이 자동으로 (0,0,0)으로 되돌립니다 — 별도로 원위치 명령을 만들지 마세요.
- speed_level   : 전체 걸음 속도 단계 설정, level(정수)만 사용 (속도 단계, 속도 레벨) — 실제
                  로봇에서 검증되지 않은 기능이므로 사용자가 명시적으로 요청했을 때만 사용하고,
                  큰 값을 임의로 넣지 말고 작은 정수(1~2)부터 사용하세요.
- vision_analyze: 로봇 동작이 아니라 카메라 화면 분석 — "지금 뭐가 보여?", "뭐가 보이나요?",
                  "이 사람 누구야?", "앞에 누구 있어?", "주변에 뭐가 있어?"처럼 카메라로 실제로
                  봐야만 답할 수 있는 질문에만 사용하세요 (일반 대화/잡담에는 사용하지 마세요).
- navigate_to   : 미리 저장된 위치로 SLAM 기반 자율 이동, location(위치 이름, 문자열)만 사용
                  (예: "문 앞으로 가줘", "OO로 이동해", "OO에 데려다줘"). location에는 사용자가
                  말한 위치 이름을 그대로 넣으세요 — 실제로 그 이름이 저장되어 있는지는 시스템이
                  확인합니다 (없으면 실행 시 오류로 안내됩니다). 이 위치가 저장되어 있는지 미리
                  알 수 없으므로 절대 거절하지 말고 일단 navigate_to로 시도하세요.

[규칙]
0. 영어 단어의 한국어 음차(STT 변환 결과)도 반드시 인식하세요.
   예: '팩폴립'/'팩플립' → back_flip, '프론트 플립' → front_flip, '레프트 플립' → left_flip
   STT가 영어 발음을 한국어 철자로 잘못 표기해도 가장 유사한 행동으로 매핑하세요.
1. 반드시 JSON만 응답하세요 (설명 텍스트 없이).
   actions 배열 순서는 반드시 사용자가 말한 순서와 동일해야 합니다. 절대 순서를 바꾸지 마세요.
   예: "오른쪽으로 돌고 앞으로 가" → [회전, 전진] 순서 (절대 [전진, 회전]으로 뒤집지 말 것)
2. "actions" 배열의 각 원소는 반드시 객체(object)로 작성하고, 사용자 발화 순서 그대로 나열하세요.
   - 단순 행동: {"name": "stand_up"}
   - move 행동: {"name": "move", "vx": ..., "vy": ..., "omega": ..., "duration": ...}
   - euler 행동: {"name": "euler", "roll": ..., "pitch": ..., "yaw": ..., "duration": ...}
     예: "왼쪽으로 몸 기울여" → {"name":"euler","roll":0.3,"pitch":0.0,"yaw":0.0,"duration":2.0}
3. move 파라미터 — 이동(vx/vy)과 회전(omega)을 반드시 구분하세요:
   [이동 — vx/vy 사용, omega=0]
   - 앞으로: vx=0.3, vy=0.0, omega=0.0
   - 뒤로: vx=-0.3, vy=0.0, omega=0.0
   - 왼쪽 옆으로(평행이동): vx=0.0, vy=0.3, omega=0.0
   - 오른쪽 옆으로(평행이동): vx=0.0, vy=-0.3, omega=0.0
   - 빠르게: vx/vy=0.5, 느리게: vx/vy=0.2
   - duration: 한 걸음=1.0초, 두 걸음=2.0초, 세 걸음=3.0초, 기본=2.0초
   [회전 — omega 사용, vx=0, vy=0]
   - 왼쪽(반시계)으로 돌기: vx=0.0, vy=0.0, omega=1.0
   - 오른쪽(시계)으로 돌기: vx=0.0, vy=0.0, omega=-1.0
   - duration: 30도=0.54초, 45도=0.86초, 90도=1.71초, 180도=3.32초, 360도(한바퀴)=6.64초, 각도 언급 없으면 기본=1.71초(90도)
     (실측 보정값 — 회전 시작/종료 시 가감속으로 명목 회전각보다 실제로 약 7% 덜 돌기 때문에
     이론값(각도/180*3.1초)보다 크게 잡은 값입니다. 다른 각도는 "각도/180*3.32초"로
     선형 계산하세요 — 절대 180도 값을 그대로 재사용하지 마세요. "왼쪽으로 돌아"처럼 각도를
     말하지 않았다면 절대 임의로 추측하지 말고 반드시 기본값 1.71초를 사용하세요 — 매번
     같은 발화에는 항상 같은 duration이 나와야 합니다.)
   예: "왼쪽으로 30도 돌기" → {"name":"move","vx":0.0,"vy":0.0,"omega":1.0,"duration":0.54}
   예: "오른쪽으로 90도 돌기" → {"name":"move","vx":0.0,"vy":0.0,"omega":-1.0,"duration":1.71}
   예: "제자리에서 한바퀴 돌기" → {"name":"move","vx":0.0,"vy":0.0,"omega":1.0,"duration":6.64}
4. 같은 방향 이동은 반드시 하나의 객체로 합치세요 (duration으로 거리 표현).
   틀린 예: [{"name":"move","vx":0.5,"duration":1.5}, {"name":"move","vx":0.5,"duration":1.5}]
   올바른 예: [{"name":"move","vx":0.5,"vy":0.0,"omega":0.0,"duration":3.0}]
5. 방향이 다를 때만 별도 객체로 분리하세요.
   예: "뒤로 두 걸음, 오른쪽으로 한 걸음" →
       [{"name":"move","vx":-0.5,"vy":0.0,"omega":0.0,"duration":3.0},
        {"name":"move","vx":0.0,"vy":-0.5,"omega":0.0,"duration":1.5}]
6. response는 친근한 한국어 미래형/의지형으로 작성하세요 (예: "~할게요!", "~갈게요!", "~춰볼게요!").
   로봇이 실행하기 전에 말하므로 과거형("~했어요", "~했습니다")은 절대 쓰지 마세요.
7. response는 반드시 실제로 실행되는 action과 일치해야 합니다 — 사용자가 말한 표현을 그대로
   따라 말하지 마세요. 예를 들어 front_jump는 항상 "앞으로" 점프하며 방향을 바꿀 수 없습니다.
   사용자가 "뒤로 점프해줘"라고 말해도 front_jump를 선택했다면 response는 "앞으로 점프할게요!"
   라고 해야지, "뒤로 점프할게요!"라고 하면 안 됩니다 (실제 동작과 다른 방향을 말하면 안전상
   위험 — 사용자가 로봇이 다르게 움직일 거라 착각하게 됩니다). 요청한 방향과 실제 action의
   동작이 다르면, response에서 실제로 어느 방향인지 정확히 밝히세요.
8. 발화의 문맥/어휘에서 감정·상황 톤을 추론하여 "tone" 필드에 아래 중 하나로 표시하세요
   (오디오 톤이 아니라 텍스트 내용만으로 추론하세요):
   - "urgent" : 급함, 서두름 (예: "빨리", "지금 당장", "급해")
   - "tired"  : 상대방이 지치거나 힘든 상황을 언급 (예: "더운데 참 수고가 많다", "피곤하다")
   - "calm"   : 차분함/여유를 요청 (예: "천천히", "편하게 해도 돼")
   - "happy"  : 밝고 기쁜 어조 (예: "신난다", "좋아!", 감탄사가 많은 발화)
   - "neutral": 위 어디에도 뚜렷이 해당하지 않는 일반적인 명령 (기본값)
   tone에 맞게 response의 어조도 자연스럽게 맞추세요 — "tired"면 다정하고 배려하는 말투로,
   "urgent"면 짧고 즉각적인 말투로 작성하세요. tone은 실행되는 action을 바꾸지 않습니다
   (예: move의 vx/vy/omega는 규칙 3~5를 그대로 따르세요) — 속도 조절은 시스템이 별도로 처리합니다.
9. vision_analyze를 선택했다면 actions 배열에는 반드시 {"name": "vision_analyze"} 하나만 넣고
   다른 행동(move, hello 등)과 절대 함께 넣지 마세요. response는 "확인해볼게요!"처럼 짧은
   대기 멘트로만 작성하세요 — 실제 화면 설명은 별도 시스템이 분석 후 다시 말합니다. response에서
   장면 내용을 미리 추측해서 말하지 마세요 (아직 보지 않았습니다).
10. navigate_to를 선택했다면 actions 배열에는 반드시 {"name": "navigate_to", "location": "..."}
    하나만 넣고 다른 행동과 절대 함께 넣지 마세요. response는 "이동할게요!"처럼 짧은 의지형으로만
    작성하세요 — 도착했는지 여부는 별도 시스템이 확인 후 다시 말합니다. response에서 도착을
    미리 단정하지 마세요 (아직 이동하지 않았습니다).

[출력 JSON 형식]
{
  "understood": true,
  "tone": "neutral",
  "actions": [
    {"name": "action1"},
    {"name": "move", "vx": 0.5, "vy": 0.0, "omega": 0.0, "duration": 3.0}
  ],
  "response": "한국어 응답"
}"""

# Language names used to parameterize Rule 6 (response language) below —
# keyed by the same code passed to Whisper's `language=` argument.
_LANGUAGE_NAMES = {"ko": "한국어", "vi": "베트남어", "en": "영어", "de": "독일어", "it": "이탈리아어"}

# Domain-hint sentences for Whisper's `initial_prompt` — short, in-domain
# phrases that bias decoding toward robot-command vocabulary instead of
# generic conversation, one per STT language the dashboard offers.
#
# "ko-jeju" is NOT a separate Whisper language -- Jeju is phonetically still
# Korean, so it's transcribed with language="ko" like standard Korean (see
# dashboard_server.py's _pipeline, which maps any "ko*" selection to "ko"
# before calling Whisper). What differs is the *vocabulary* Whisper is primed
# to expect: without a hint, Whisper tends to "correct" unfamiliar Jeju words
# to the nearest standard-Korean word it knows, i.e. it mishears rather than
# transcribes them. Priming with real Jeju words reduces that autocorrect bias.
WHISPER_INITIAL_PROMPTS = {
    "ko": "일어서, 앉아, 앞으로 가, 뒤로 가, 왼쪽으로 돌아, 오른쪽으로 돌아, 점프해, 정지, 인사해, 스트레칭",
    "vi": "đứng dậy, ngồi xuống, đi tới, đi lùi, rẽ trái, rẽ phải, nhảy lên, dừng lại, chào đi, duỗi người",
    "ko-jeju": "게마씸, 혼저옵서예, 허라게, 돌아사, 물러나곡, 해불라게, 핸",
}

# Jeju (제주어) -> standard Korean glossary, injected into the prompt only
# when "ko-jeju" is selected. This is the layer that actually matters for
# correctness: STT priming above only helps Whisper spell Jeju words right;
# without this, a *correctly transcribed* Jeju sentence can still fail to map
# to the right action because the LLM doesn't know what the dialect words
# mean. Few-shot vocabulary mapping in-prompt, not fine-tuning -- GPT-4o-class
# models generalize the pattern (ending/vowel shifts) to unseen Jeju words.
JEJU_VOCAB_HINT = """
[제주 방언(제주어) 참고]
사용자의 발화가 제주 방언일 수 있습니다. 아래는 표준어 대응 예시이며, 목록에 없는
제주어 표현도 같은 방식(어미·모음 변형)으로 뜻을 유추하세요. 의미 파악에만 참고하고,
response는 항상 표준 한국어로 작성하세요 (제주어로 답하지 마세요):
  "게마씸" → 정중한 설명/의문 어미 (표준어 "-거든요/-잖아요"에 해당)
  "핸" → "했어" (과거형 어미)
  "돌아사" → "돌아서"
  "물러나곡" → "물러나고"
  "해불라게" → "해버려"
  "허라게" → "해라"
  "혼저옵서예" → "어서 오세요"
"""


def get_system_prompt(language: str = "ko") -> str:
    """KOREAN_SYSTEM_PROMPT, adapted per selected language/dialect:
    - "ko-jeju": inject the Jeju vocabulary glossary above (Rule 6/response
      language stays standard Korean -- only comprehension of the input is
      being helped here, not dialect generation).
    - any other non-Korean language in _LANGUAGE_NAMES: swap Rule 6 so the
      spoken-back `response` matches the input language instead of always
      Korean.
    - "ko" (default): unchanged, byte-identical to KOREAN_SYSTEM_PROMPT."""
    if language == "ko-jeju":
        return KOREAN_SYSTEM_PROMPT.replace(
            "[출력 JSON 형식]", JEJU_VOCAB_HINT.strip() + "\n\n[출력 JSON 형식]"
        )
    if language == "ko" or language not in _LANGUAGE_NAMES:
        return KOREAN_SYSTEM_PROMPT
    lang_name = _LANGUAGE_NAMES[language]
    return KOREAN_SYSTEM_PROMPT.replace(
        "6. response는 친근한 한국어 미래형/의지형으로 작성하세요",
        f"6. response는 반드시 {lang_name}로 작성하세요 (한국어로 쓰지 마세요)."
        f" 친근한 미래형/의지형으로 작성하세요",
    )


# Emotion/tone -> movement speed multiplier. This is the only currently-
# controllable lever for making the robot's *physical* motion feel
# urgent/tired/calm: gesture tricks (Hello, Dance1, BackFlip...) run a fixed
# built-in animation with no speed parameter, only "move" has continuously
# variable vx/vy/omega we can scale. Deliberately small deviations from 1.0 --
# this is a felt-nuance feature, not a reason to move dangerously fast/slow.
TONE_SPEED_MULTIPLIER = {
    "urgent": 1.3,
    "tired": 0.7,
    "calm": 0.85,
    "happy": 1.1,
    "neutral": 1.0,
}


def apply_tone_to_moves(merged_actions: list, tone: str) -> list:
    """Scale vx/vy/omega on every 'move' entry in a merge_consecutive_moves()
    result by the tone's speed multiplier. Other action names are returned
    unchanged (their speed isn't controllable -- see TONE_SPEED_MULTIPLIER's
    docstring). No-ops for unknown/"neutral" tone (multiplier 1.0)."""
    mult = TONE_SPEED_MULTIPLIER.get(tone, 1.0)
    if mult == 1.0:
        return merged_actions
    scaled = []
    for name, params in merged_actions:
        if name == "move" and params:
            params = dict(params)
            for key in ("vx", "vy", "omega"):
                if key in params:
                    params[key] = round(params[key] * mult, 3)
        scaled.append((name, params))
    return scaled


class VoiceControlGo2:
    """한국어 음성으로 Go2 로봇을 제어하는 클래스"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        robot_ip: str = "192.168.123.18",
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
        "actions" 배열이 있으면 순서대로 실행 (객체/문자열 모두 지원)
        """
        raw_actions = command_dict.get("actions") or []
        if not raw_actions and command_dict.get("action"):
            raw_actions = [command_dict["action"]]

        if not raw_actions:
            return {"success": False, "reason": "실행할 행동이 없습니다."}

        # Merge consecutive same-direction "move" actions so the robot
        # doesn't stop and restart its gait between steps (see action_merge.py)
        merged_actions = merge_consecutive_moves(
            raw_actions, command_dict, self.robot_controller.registry.get_action_name
        )

        results = []
        action_names = []
        for action_name, params in merged_actions:
            action_names.append(action_name)
            logger.info(f"실행 중: {action_name} params={params}")
            print(f"🤖 실행: {action_name} {params}")
            result = self.robot_controller.execute_action(action_name, **params)
            results.append(result)
            if not result.get("success"):
                logger.warning(f"행동 실패: {action_name} — {result.get('error')}")
                break

        return {
            "success": all(r.get("success") for r in results),
            "actions": action_names,
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
