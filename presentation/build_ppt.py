#!/usr/bin/env python3
"""Generate the Go2 voice-control project presentation (.pptx)."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------- palette --
NAVY = RGBColor(0x11, 0x1C, 0x33)
BLUE = RGBColor(0x25, 0x63, 0xEB)
LIGHT_BLUE = RGBColor(0xDB, 0xEA, 0xFE)
ORANGE = RGBColor(0xF5, 0x9E, 0x0B)
DARK = RGBColor(0x1F, 0x29, 0x37)
GRAY = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY_BG = RGBColor(0xF8, 0xFA, 0xFC)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
RED = RGBColor(0xDC, 0x26, 0x26)

FONT = "Malgun Gothic"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def set_bg(slide, color):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, color, line=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line:
        shp.line.color.rgb = color
        shp.line.width = Pt(0.5)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, size=18, color=DARK, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT,
             line_spacing=1.0, italic=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = line_spacing
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = color
            r.font.name = font
    return box


def add_bullets(slide, x, y, w, h, items, size=18, color=DARK, font=FONT,
                 space_after=10, line_spacing=1.15, bullet_color=BLUE):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        prefix = "•  " if level == 0 else "-  "
        p.text = prefix + text
        p.level = 0
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        for r in p.runs:
            r.font.size = Pt(size - (2 if level else 0))
            r.font.color.rgb = color if level == 0 else GRAY
            r.font.name = font
    return box


def add_page_number(slide, n):
    add_text(slide, Inches(12.6), Inches(7.05), Inches(0.6), Inches(0.35),
              str(n), size=12, color=GRAY, align=PP_ALIGN.RIGHT)


def add_kicker_title(slide, kicker, title, n):
    set_bg(slide, WHITE)
    add_rect(slide, 0, 0, Inches(0.18), SLIDE_H, BLUE)
    add_text(slide, Inches(0.7), Inches(0.45), Inches(6), Inches(0.4),
              kicker, size=15, color=BLUE, bold=True)
    add_text(slide, Inches(0.7), Inches(0.85), Inches(11.8), Inches(0.9),
              title, size=32, color=NAVY, bold=True)
    add_rect(slide, Inches(0.7), Inches(1.65), Inches(1.1), Pt(3), ORANGE)
    add_page_number(slide, n)
    return slide


def set_notes(slide, text):
    notes = slide.notes_slide
    notes.notes_text_frame.text = text


def add_pill(slide, x, y, w, h, text, fill, text_color=WHITE, size=14, bold=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.5
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4)
    tf.margin_right = Pt(4)
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = text_color
        r.font.name = FONT
    return shp


def add_arrow(slide, x, y, w, h, color=GRAY):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


# ============================================================= SLIDE 1 ====
s = add_slide()
set_bg(s, NAVY)
add_rect(s, 0, Inches(6.6), SLIDE_W, Inches(0.9), BLUE)
add_text(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(0.6),
          "GO2 PROJECT · UNITREE", size=18, color=ORANGE, bold=True)
add_text(s, Inches(0.9), Inches(2.75), Inches(11.5), Inches(1.8),
          "말하는 대로 움직이는 로봇 강아지", size=44, color=WHITE, bold=True)
add_text(s, Inches(0.9), Inches(3.9), Inches(11.5), Inches(0.8),
          "Unitree Go2 음성 제어 프로젝트 소개", size=24, color=LIGHT_BLUE)
add_text(s, Inches(0.9), Inches(6.75), Inches(8), Inches(0.5),
          "발표자: ___________        발표 시간: 약 15분", size=14, color=WHITE)
set_notes(s, "(자기소개) 안녕하세요! 오늘은 제가 방학 동안 만든 프로젝트를 소개하려고 합니다. "
             "바로 '말로 명령하면 움직이는 로봇 강아지'입니다. Unitree Go2라는 사족보행 로봇에, "
             "음성인식과 AI를 연결해서 만든 프로젝트예요. 발표 시간은 15분 정도이고, "
             "실제 로봇이 어떻게 제 말을 알아듣고 움직이는지 하나씩 보여드릴게요.")

# ============================================================= SLIDE 2 ====
s = add_slide()
add_kicker_title(s, "INTRO", "오늘 발표 순서", 2)
items = [
    "1.  Go2 로봇이란?",
    "2.  프로젝트 목표 — 왜 '음성 제어'를 만들었나",
    "3.  전체 시스템 구조 (파이프라인)",
    "4.  로봇이 할 수 있는 20가지 행동",
    "5.  실제 동작 시연 시나리오",
    "6.  개발하며 만난 문제와 해결 과정",
    "7.  안전 설계",
    "8.  느낀 점과 앞으로의 계획",
]
add_bullets(s, Inches(0.9), Inches(2.1), Inches(10.5), Inches(4.8), items,
             size=22, space_after=16)
set_notes(s, "오늘 이야기할 순서는 이렇습니다. 먼저 Go2 로봇이 뭔지 간단히 소개하고, "
             "제가 왜 이 프로젝트를 시작했는지 말씀드릴게요. 그다음 전체 시스템이 어떻게 "
             "동작하는지 구조를 보여드리고, 로봇이 할 수 있는 동작들, 실제 시연 예시, "
             "그리고 개발하면서 겪었던 진짜 문제들과 어떻게 해결했는지까지 다 보여드리겠습니다.")

# ============================================================= SLIDE 3 ====
s = add_slide()
add_kicker_title(s, "01", "Go2 로봇이란?", 3)
add_bullets(s, Inches(0.7), Inches(2.1), Inches(6.6), Inches(4.6), [
    "중국 로봇 기업 Unitree가 만든 사족보행(4족) 로봇",
    "다리마다 3개씩, 총 12개의 관절 모터로 움직임",
    "카메라·라이다·IMU 센서로 주변 환경과 자세를 인식",
    "걷기·달리기뿐 아니라 백플립, 핸드스탠드 같은 고난도 동작도 가능",
    "연구, 교육, 순찰, 재난 현장 탐색 등 다양한 분야에서 활용",
], size=19, space_after=16)
box = add_rect(s, Inches(7.7), Inches(2.1), Inches(4.9), Inches(4.6), LIGHT_GRAY_BG)
add_text(s, Inches(7.7), Inches(2.3), Inches(4.5), Inches(0.5),
          "왜 하필 Go2였을까?", size=16, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(8.0), Inches(2.9), Inches(4.3), Inches(3.6),
          "제어용 SDK(개발도구)가 공개되어 있어서\n"
          "파이썬 코드로 직접 로봇에 명령을 보낼 수 있습니다.\n\n"
          "→ 그래서 '내가 만든 프로그램'으로\n     실제 로봇을 움직여볼 수 있었습니다.",
          size=16, color=DARK, align=PP_ALIGN.CENTER, line_spacing=1.3)
set_notes(s, "Go2는 Unitree라는 회사가 만든 강아지 로봇이에요. 다리가 4개, 관절이 12개나 "
             "있어서 사람처럼 균형을 잡으면서 걷습니다. 그냥 걷는 것만이 아니라 백플립이나 "
             "물구나무서기 같은 묘기도 부릴 수 있어요. 제가 이 로봇을 선택한 이유는 "
             "Unitree가 SDK, 그러니까 개발도구를 공개해놔서, 저처럼 학생도 파이썬 코드로 "
             "직접 로봇한테 명령을 내려볼 수 있기 때문입니다.")

# ============================================================= SLIDE 4 ====
s = add_slide()
add_kicker_title(s, "02", "프로젝트 목표", 4)
add_text(s, Inches(0.7), Inches(2.0), Inches(11.6), Inches(0.6),
          "기존 방식: 리모컨 버튼 또는 코드 한 줄 한 줄로 로봇을 조종", size=19, color=GRAY)
add_rect(s, Inches(0.7), Inches(2.75), Inches(11.6), Inches(1.1), RGBColor(0xFE,0xF2,0xF2))
add_text(s, Inches(1.0), Inches(2.9), Inches(11), Inches(0.8),
          "\"앞으로 가고 싶으면 move_forward() 함수를 실행해야 해...\"", size=18,
          color=RED, italic=True)
add_text(s, Inches(0.7), Inches(4.15), Inches(11.6), Inches(0.6),
          "제가 만들고 싶었던 것", size=19, color=GRAY)
add_rect(s, Inches(0.7), Inches(4.9), Inches(11.6), Inches(1.5), LIGHT_BLUE)
add_text(s, Inches(1.0), Inches(5.05), Inches(11), Inches(1.2),
          "\"일어나서 두 걸음 걸어가서 하트하고 다시 앉아줘\"\n"
          "     → 사람이 말하는 자연어 그대로 로봇이 알아듣고 행동!", size=19,
          color=NAVY, bold=True, line_spacing=1.3)
set_notes(s, "원래 이런 로봇을 움직이려면 리모컨을 쓰거나, move_forward() 같은 함수를 "
             "코드로 하나하나 실행해야 해요. 저는 이게 불편하다고 생각했습니다. "
             "그래서 목표를 이렇게 잡았어요 — 사람이 그냥 평소 말하듯이 '일어나서 두 걸음 "
             "걸어가서 하트하고 다시 앉아줘' 라고 말하면, 로봇이 그 말을 이해하고 순서대로 "
             "행동하게 만드는 것. 즉, 음성 명령 → AI 이해 → 로봇 실행까지 전부 자동화하는 "
             "것이 이번 프로젝트의 목표였습니다.")

# ============================================================= SLIDE 5 ====
s = add_slide()
add_kicker_title(s, "03", "전체 시스템 구조 (파이프라인)", 5)

stages = [
    ("음성 입력", "마이크로\n말하기"),
    ("STT", "Whisper로\n텍스트 변환"),
    ("LLM", "OpenAI GPT가\n명령을 이해"),
    ("실행", "로봇이\n동작 실행"),
    ("TTS", "음성으로\n결과 응답"),
]
box_w = Inches(1.9)
box_h = Inches(1.7)
gap = Inches(0.28)
arrow_w = Inches(0.4)
total_w = box_w * 5 + gap * 4 + arrow_w * 4
start_x = int((SLIDE_W - total_w) / 2)
y = Inches(2.7)
x = start_x
colors = [BLUE, BLUE, ORANGE, BLUE, BLUE]
for i, (label, desc) in enumerate(stages):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
    shp.adjustments[0] = 0.08
    shp.fill.solid()
    shp.fill.fore_color.rgb = colors[i]
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Pt(6)
    tf.margin_right = Pt(6)
    p1 = tf.paragraphs[0]
    p1.text = label
    p1.alignment = PP_ALIGN.CENTER
    for r in p1.runs:
        r.font.size = Pt(19)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = FONT
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.alignment = PP_ALIGN.CENTER
    for r in p2.runs:
        r.font.size = Pt(13)
        r.font.color.rgb = WHITE
        r.font.name = FONT
    x += box_w
    if i < len(stages) - 1:
        add_arrow(s, x, int(y + box_h/2 - arrow_w/2), arrow_w, arrow_w, GRAY)
        x += gap + arrow_w

add_text(s, Inches(0.7), Inches(5.2), Inches(11.9), Inches(1.6),
          "핵심 포인트: 사람의 '말'이 로봇의 '행동'으로 바뀌는 5단계 파이프라인입니다.\n"
          "이 중 STT · LLM · 실행 세 단계를 조금 더 자세히 살펴보겠습니다.",
          size=17, color=GRAY, align=PP_ALIGN.CENTER, line_spacing=1.4)
set_notes(s, "이게 이 프로젝트의 전체 구조입니다. 사람이 마이크에 말을 하면, 먼저 STT라는 "
             "기술이 그 말을 텍스트로 바꿔줍니다. 그 텍스트를 OpenAI의 GPT, 그러니까 AI한테 "
             "보내서 '이 사람이 로봇한테 뭘 시키고 싶은 건지' 해석하게 합니다. 그러면 로봇이 "
             "그 명령대로 실제로 움직이고, 마지막으로 음성으로 결과를 다시 알려줍니다. "
             "이 중에서 STT, LLM, 실행 세 단계를 조금 더 자세히 보여드릴게요.")

# ============================================================= SLIDE 6 ====
s = add_slide()
add_kicker_title(s, "03-1", "1단계 · 음성 인식 (STT)", 6)
add_bullets(s, Inches(0.7), Inches(2.1), Inches(6.6), Inches(4.4), [
    "Faster-Whisper 모델을 사용해 실시간으로 음성을 텍스트로 변환",
    "VAD(음성 구간 감지)로 사람이 말하기 시작하고 끝나는 순간을 자동으로 감지",
    "한국어 음성 인식 지원",
    "잡음이 있어도 어느 정도 정확하게 알아들음",
], size=19, space_after=18)
box = add_rect(s, Inches(7.7), Inches(2.1), Inches(4.9), Inches(4.4), LIGHT_GRAY_BG)
add_text(s, Inches(8.0), Inches(2.4), Inches(4.3), Inches(0.5),
          "실제 인식 예시", size=16, color=BLUE, bold=True)
add_text(s, Inches(8.0), Inches(3.0), Inches(4.3), Inches(2.8),
          "(사용자 발화)\n"
          "\"일어나서 두 걸음 앞으로\n"
          "걸어간 다음 하트를 하고\n"
          "엎드렸다가 다시 일어나\"\n\n"
          "(STT 변환 결과)\n"
          "위 문장 그대로 텍스트로 변환 성공",
          size=15, color=DARK, line_spacing=1.3)
set_notes(s, "먼저 STT, Speech-to-Text 단계입니다. 여기서는 Faster-Whisper라는 오픈소스 "
             "AI 모델을 써서 마이크로 들어온 소리를 실시간으로 텍스트로 바꿉니다. 사람이 "
             "말을 시작하고 끝내는 시점도 자동으로 감지해서, 버튼을 누르지 않아도 자연스럽게 "
             "대화하듯 말할 수 있어요. 한국어도 지원해서, 예를 들어 제가 '일어나서 두 걸음 "
             "앞으로 걸어간 다음 하트를 하고 엎드렸다가 다시 일어나' 라고 말하면 이 문장 "
             "그대로 텍스트로 바뀝니다.")

# ============================================================= SLIDE 7 ====
s = add_slide()
add_kicker_title(s, "03-2", "2단계 · LLM이 명령을 이해", 7)
add_bullets(s, Inches(0.7), Inches(2.1), Inches(5.7), Inches(4.6), [
    "OpenAI GPT 모델에게 로봇이 할 수 있는 행동 목록을 미리 알려줌",
    "사람의 자연어 문장을 로봇이 실행 가능한 명령(JSON)으로 변환",
    "여러 동작이 섞인 복합 문장도 순서대로 계획 가능",
    "→ AI가 이 프로젝트의 '두뇌' 역할",
], size=18, space_after=16)
box = add_rect(s, Inches(6.7), Inches(2.1), Inches(5.9), Inches(4.6), NAVY)
add_text(s, Inches(6.95), Inches(2.3), Inches(5.4), Inches(0.4),
          "LLM이 만들어낸 실제 명령 (일부)", size=14, color=ORANGE, bold=True)
code_txt = ('{\n'
            '  "actions": [\n'
            '    {"name": "stand_up"},\n'
            '    {"name": "move",\n'
            '     "vx": 0.5, "duration": 3.0},\n'
            '    {"name": "move",\n'
            '     "vx": 0.5, "duration": 3.0},\n'
            '    {"name": "balanced_stand"},\n'
            '    {"name": "stand_down"},\n'
            '    {"name": "stand_up"}\n'
            '  ]\n'
            '}')
add_text(s, Inches(6.95), Inches(2.75), Inches(5.4), Inches(3.8), code_txt,
          size=14, color=WHITE, font="Consolas", line_spacing=1.15)
set_notes(s, "두 번째 단계가 이 프로젝트의 핵심 두뇌라고 할 수 있는 LLM, 거대언어모델 "
             "단계입니다. 저는 OpenAI GPT한테 미리 '로봇이 할 수 있는 행동 목록'을 알려줍니다. "
             "그러면 GPT가 사람이 말한 자연어 문장을 분석해서, 로봇이 실행할 수 있는 명령어 "
             "형태, 그러니까 JSON이라는 데이터 형식으로 바꿔줍니다. 화면에 보이는 것처럼, "
             "'일어나서 두 걸음 걸어가서 하트하고 앉았다가 다시 일어나'라는 한 문장이 "
             "여섯 개의 순서 있는 명령으로 정확하게 쪼개진 걸 볼 수 있습니다.")

# ============================================================= SLIDE 8 ====
s = add_slide()
add_kicker_title(s, "03-3", "3단계 · 로봇이 실제로 움직이기까지", 8)
layers = [
    ("RobotController", "명령 이름을 받아서 실행 요청"),
    ("BehaviorExecutor", "안전 체크 후 실제 동작 실행 (대기시간·상태 관리)"),
    ("SportClient (Unitree SDK)", "로봇 하드웨어에 직접 신호 전달"),
    ("Go2 로봇 모터/관절", "실제로 다리가 움직임"),
]
y = Inches(2.15)
for i, (name, desc) in enumerate(layers):
    add_rect(s, Inches(0.9), y, Inches(11.5), Inches(0.85), LIGHT_BLUE if i % 2 == 0 else LIGHT_GRAY_BG)
    add_text(s, Inches(1.15), y + Inches(0.08), Inches(3.6), Inches(0.7),
              name, size=17, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(4.9), y + Inches(0.08), Inches(7.3), Inches(0.7),
              desc, size=15, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(layers) - 1:
        pass
    y += Inches(1.05)
set_notes(s, "명령이 로봇한테 전달되는 마지막 단계입니다. LLM이 만든 명령을 "
             "RobotController가 받아서, BehaviorExecutor한테 넘깁니다. 이 BehaviorExecutor가 "
             "'지금 배터리는 충분한지, 지금 서 있는 상태인지' 같은 안전 체크를 한 다음 "
             "실제로 Unitree에서 제공하는 SportClient라는 SDK를 통해 로봇 하드웨어에 신호를 "
             "보냅니다. 그러면 그 신호가 로봇의 12개 모터로 전달돼서 실제로 다리가 움직이게 "
             "되는 거예요.")

# ============================================================= SLIDE 9 ====
s = add_slide()
add_kicker_title(s, "04", "로봇이 할 수 있는 20가지 행동", 9)
cats = [
    ("자세", GREEN, "일어서기 · 앉기(엎드리기) · 회복 자세"),
    ("이동", BLUE, "전진 · 후진 · 좌우 이동 · 회전"),
    ("묘기", ORANGE, "백플립 · 프론트플립 · 핸드스탠드 · 옆으로 회전"),
    ("감성 표현", RGBColor(0xDB,0x27,0x77), "하트 포즈 · 인사(손 흔들기) · 춤 1·2 · 스트레칭"),
    ("안전", RED, "정지 · 댐핑(힘 빼기) · 장애물 회피 모드"),
]
y = Inches(2.15)
for name, color, desc in cats:
    add_pill(s, Inches(0.9), y, Inches(1.8), Inches(0.6), name, color, size=15)
    add_text(s, Inches(2.9), y + Inches(0.02), Inches(9.4), Inches(0.6),
              desc, size=17, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.9)
set_notes(s, "제가 만든 라이브러리에는 총 20가지 행동이 등록되어 있습니다. 크게 나누면 "
             "일어서고 앉는 '자세' 동작, 전후좌우로 움직이는 '이동' 동작, 백플립이나 "
             "핸드스탠드 같은 '묘기' 동작, 하트 포즈나 춤 같은 '감성 표현' 동작, 그리고 "
             "위험할 때 바로 멈추는 '안전' 동작이 있습니다. 이 모든 동작을 사람이 말 한마디로 "
             "선택할 수 있어요.")

# ============================================================= SLIDE 10 ===
s = add_slide()
add_kicker_title(s, "05", "실제 동작 시연 시나리오", 10)
add_rect(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.0), LIGHT_BLUE)
add_text(s, Inches(1.15), Inches(2.15), Inches(11), Inches(0.7),
          "\"일어나서 두 걸음 앞으로 걸어간 다음 하트를 하고 엎드렸다가 다시 일어나\"",
          size=18, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)

steps = ["① 일어서기", "② 전진 2걸음", "③ 하트 포즈", "④ 엎드리기", "⑤ 다시 일어서기"]
box_w = Inches(2.15)
gap = Inches(0.15)
total_w = box_w * 5 + gap * 4
start_x = int((SLIDE_W - total_w) / 2)
x = start_x
y2 = Inches(3.5)
for i, st in enumerate(steps):
    add_pill(s, x, y2, box_w, Inches(0.75), st, BLUE if i % 2 == 0 else NAVY, size=14)
    x += box_w + gap

add_text(s, Inches(0.9), Inches(4.8), Inches(11.5), Inches(0.5),
          "실행 결과", size=17, color=BLUE, bold=True)
add_bullets(s, Inches(0.9), Inches(5.3), Inches(11.3), Inches(1.9), [
    "5개의 동작이 순서대로, 하나도 빠짐없이 자동 실행됨",
    "동작 사이 안전 대기시간을 유지하면서도 전체 흐름은 자연스럽게 이어짐",
    "완료 후 TTS로 \"일어났어요! 두 걸음 걸어갔고, 하트를 하고 엎드렸다가 다시 일어났어요\" 음성 응답",
], size=16, space_after=8)
set_notes(s, "실제로 제가 이 문장을 말했을 때 로봇이 어떻게 움직였는지 순서대로 보여드릴게요. "
             "일어서고, 앞으로 두 걸음 걷고, 하트 포즈를 하고, 엎드렸다가, 다시 일어나는 "
             "다섯 단계가 순서대로 실행됩니다. 그리고 다 끝나면 로봇이 음성으로 '일어났어요! "
             "두 걸음 걸어갔고, 하트를 하고 엎드렸다가 다시 일어났어요' 라고 대답까지 해줍니다. "
             "(여기서 실제 데모 영상이나 실습을 보여주면 좋습니다)")

# ============================================================= SLIDE 11 ===
s = add_slide()
add_kicker_title(s, "06-1", "문제 1 · 명령이 많아질수록 너무 느려졌다", 11)
add_text(s, Inches(0.7), Inches(2.05), Inches(11.6), Inches(0.6),
          "동작 하나마다 안전을 위한 '대기시간'이 있는데, 명령을 이어붙일수록 그 시간이 계속 쌓임",
          size=18, color=GRAY)
add_rect(s, Inches(0.7), Inches(2.8), Inches(11.6), Inches(1.5), RGBColor(0xFE,0xF2,0xF2))
add_text(s, Inches(1.0), Inches(2.95), Inches(11), Inches(1.2),
          "\"일어서기 → 이동 → 하트 → 엎드리기 → 일어서기\" 명령 하나에\n"
          "     약 19초가 걸림 (그중 2초는 완전히 불필요한 중복 대기였음)",
          size=18, color=RED, line_spacing=1.4)
add_text(s, Inches(0.7), Inches(4.6), Inches(11.6), Inches(0.5),
          "해결 방법", size=19, color=GREEN, bold=True)
add_bullets(s, Inches(0.9), Inches(5.15), Inches(11.3), Inches(1.9), [
    "'일어서기' 동작이 끝난 뒤 불필요하게 한 번 더 반복 실행되던 버그 제거",
    "같은 방향으로 가는 이동 명령 여러 개를 하나로 합쳐서 실행 (멈췄다 다시 출발하는 낭비 제거)",
    "→ 안전 대기시간은 그대로 유지하면서, 불필요한 낭비 시간만 제거",
], size=17, space_after=10)
set_notes(s, "제가 처음 부딪힌 문제는 이거였어요. 명령을 여러 개 이어 붙일수록 로봇이 "
             "너무 오래 걸리는 거예요. 알고 보니, 동작 하나가 끝날 때마다 안전을 위해 "
             "잠깐씩 기다리는 시간이 있는데, 이게 명령 개수만큼 계속 쌓이는 구조였습니다. "
             "게다가 코드를 자세히 뜯어보니까, '일어서기' 동작이 끝나고 나서 똑같은 "
             "일어서기 동작을 한 번 더 실행하는 말도 안 되는 중복 버그도 발견했어요. "
             "그래서 그 중복을 없애고, 같은 방향으로 계속 걷는 이동 명령들은 하나로 합쳐서 "
             "실행하도록 바꿨습니다. 안전을 위한 대기시간은 그대로 지키면서, 낭비되는 "
             "시간만 줄인 거죠.")

# ============================================================= SLIDE 12 ===
s = add_slide()
add_kicker_title(s, "06-2", "문제 2 · 분명 '이동'을 시켰는데 로봇이 안 움직였다", 12)
add_bullets(s, Inches(0.7), Inches(2.05), Inches(11.6), Inches(1.4), [
    "이동 명령을 합쳤더니, 오히려 로봇이 제자리에 가만히 서있기만 하는 문제 발생",
], size=18, space_after=6)
add_rect(s, Inches(0.7), Inches(3.05), Inches(11.6), Inches(1.5), LIGHT_GRAY_BG)
add_text(s, Inches(1.0), Inches(3.2), Inches(11), Inches(1.2),
          "원인: 로봇에게 보내는 '속도 명령'에는 유통기한이 있다!\n"
          "     (약 1초 안에 다시 보내주지 않으면 자동으로 속도가 0으로 돌아감)",
          size=18, color=DARK, bold=True, line_spacing=1.4)
add_text(s, Inches(0.7), Inches(4.75), Inches(11.6), Inches(0.6),
          "비유: 자전거 페달을 계속 밟아야 앞으로 가지, 한 번 밟고 멈추면 곧 서버리는 것과 같음",
          size=17, color=GRAY, italic=True)
add_text(s, Inches(0.7), Inches(5.5), Inches(11.6), Inches(0.5),
          "해결 방법", size=19, color=GREEN, bold=True)
add_bullets(s, Inches(0.9), Inches(6.0), Inches(11.3), Inches(1.0), [
    "0.3초마다 '이동' 명령을 계속 다시 보내서, 정지시킬 때까지 끊김 없이 이동하도록 수정",
], size=17, space_after=8)
set_notes(s, "이걸 고치고 나니 더 이상한 문제가 생겼어요. 이동 명령을 하나로 합쳤더니, "
             "로봇이 그 시간 동안 그냥 가만히 서있기만 하는 거예요. 원인을 찾아보니까, "
             "로봇한테 보내는 '이 속도로 움직여라'라는 명령에는 유통기한이 있었습니다. "
             "대략 1초 안에 똑같은 명령을 다시 보내주지 않으면, 로봇이 자동으로 속도를 "
             "0으로 되돌려버려요. 이건 자전거 페달이랑 비슷해요. 페달을 한 번 밟고 놔버리면 "
             "자전거가 곧 멈추잖아요. 로봇도 계속 '가라, 가라' 하고 신호를 보내줘야 계속 "
             "가는 거였습니다. 그래서 이동하는 동안 0.3초마다 같은 명령을 계속 다시 "
             "보내주도록 코드를 고쳤고, 그제서야 로봇이 끊기지 않고 제대로 걸어갔습니다.")

# ============================================================= SLIDE 13 ===
s = add_slide()
add_kicker_title(s, "07", "안전 설계 (Safety First)", 13)
add_bullets(s, Inches(0.7), Inches(2.15), Inches(11.6), Inches(4.5), [
    "배터리가 부족하면 위험할 수 있는 동작은 아예 실행하지 않음",
    "서 있지 않은 상태에서 '걷기' 같은 명령이 오면, 먼저 자동으로 일어선 뒤 실행",
    "언제든 \"정지\" 한마디로 즉시 모든 동작을 멈출 수 있음",
    "이동 속도·지속시간의 기본값은 항상 보수적으로 설정 (급하게 움직이지 않도록)",
    "동작 사이 안전 대기시간은 성능 최적화 중에도 절대 줄이지 않음",
], size=20, space_after=20)
set_notes(s, "아무리 재미있는 동작을 많이 만들어도, 로봇 프로젝트에서 가장 중요한 건 "
             "안전이라고 생각합니다. 그래서 몇 가지 안전장치를 꼭 넣었어요. 배터리가 부족하면 "
             "위험한 동작은 실행하지 않고, 로봇이 서있지 않은 상태에서 걷기 명령이 오면 "
             "먼저 자동으로 일어서게 만들었습니다. 그리고 언제든 '정지'라고 말하면 바로 "
             "멈출 수 있고요. 아까 말씀드린 속도 최적화를 할 때도, 자세를 바꾸는 것처럼 "
             "안전과 직결된 대기시간은 절대 줄이지 않고, 진짜 불필요하게 낭비되는 부분만 "
             "골라서 줄였습니다.")

# ============================================================= SLIDE 14 ===
s = add_slide()
add_kicker_title(s, "08", "느낀 점 & 배운 점", 14)
add_bullets(s, Inches(0.7), Inches(2.15), Inches(11.6), Inches(4.4), [
    "로봇공학은 하드웨어뿐 아니라 음성인식·AI·안전설계까지 다양한 소프트웨어 기술이 만나는 분야",
    "코드 한 줄의 작은 버그(중복 명령, 속도 명령 유통기한)가 실제 로봇의 움직임을 완전히 바꿀 수 있다는 것을 직접 체험",
    "OpenAI의 LLM을 '로봇의 두뇌'로 활용해보며 최신 AI 기술을 실제 하드웨어에 적용하는 경험",
    "문제가 생겼을 때 로그를 하나씩 분석하며 원인을 찾아가는 디버깅 과정 자체가 큰 공부가 됨",
], size=19, space_after=18)
set_notes(s, "이 프로젝트를 하면서 느낀 점을 정리해봤습니다. 로봇공학이 단순히 모터나 "
             "기계 설계만의 영역이 아니라, 음성인식이나 AI, 안전설계 같은 소프트웨어 기술이 "
             "정말 많이 만나는 분야라는 걸 느꼈어요. 그리고 코드 한 줄의 작은 버그가 "
             "실제 로봇의 움직임을 완전히 바꿔놓을 수 있다는 것도 직접 경험했습니다. "
             "특히 최신 AI인 LLM을 로봇의 두뇌처럼 활용해본 게 정말 좋은 경험이었고, "
             "문제가 생겼을 때 로그를 하나하나 읽으면서 원인을 찾아가는 과정 자체가 "
             "저에게는 큰 공부가 되었습니다.")

# ============================================================= SLIDE 15 ===
s = add_slide()
set_bg(s, NAVY)
add_rect(s, 0, 0, SLIDE_W, Inches(0.15), ORANGE)
add_text(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(0.6),
          "앞으로의 계획", size=18, color=ORANGE, bold=True)
add_bullets(s, Inches(0.9), Inches(2.2), Inches(11), Inches(2.2), [
    "카메라 영상으로 장애물을 스스로 인식하고 피하는 기능 추가",
    "고정된 대기시간 대신, 로봇의 실제 센서 값을 보고 동작 완료를 판단하는 방식으로 개선",
    "더 복잡한 여러 단계의 명령도 안정적으로 처리할 수 있도록 발전",
], size=19, color=WHITE, space_after=14)
add_text(s, Inches(0.9), Inches(5.1), Inches(11.5), Inches(1.2),
          "들어주셔서 감사합니다!", size=36, color=WHITE, bold=True)
add_text(s, Inches(0.9), Inches(6.0), Inches(11.5), Inches(0.6),
          "질문 있으신가요?  Q & A", size=20, color=LIGHT_BLUE)
set_notes(s, "마지막으로 앞으로의 계획을 말씀드리면, 지금은 카메라를 활용해서 스스로 "
             "장애물을 피하는 기능을 추가하고 싶고요, 지금처럼 정해진 시간만큼 무조건 "
             "기다리는 방식 대신 로봇의 실제 센서 값을 보고 '아, 지금 동작이 끝났구나' "
             "하고 판단하는 더 똑똑한 방식으로 발전시키고 싶습니다. 여기까지 들어주셔서 "
             "정말 감사합니다. 질문 있으시면 편하게 해주세요!")

prs.save("/media/hong/data/unitree_sdk2_python/presentation/Go2_음성제어_프로젝트_발표.pptx")
print("Saved.", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
