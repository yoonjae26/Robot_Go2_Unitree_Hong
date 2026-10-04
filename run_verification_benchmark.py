#!/usr/bin/env python3
"""
Runs benchmark_corpus.json through the real LLM-generation -> verify_sequence
pipeline (the same call sequence dashboard_server.py's _llm_and_execute uses,
from get_system_prompt() onward) across the paper's C0-C3 configurations, and
writes one CSV row per (config, case). This is what turns the three verification
layers from "code that exists" into the actual numbers the thesis's Results
section needs (accept/reject-by-layer, per-layer latency, false-rejection rate
on the benign cases).

Two modes:
  --dry-run (default): computes the verification decision only. Robot
      execution is never called -- safe to run repeatedly at a desk without
      the robot powered on, and without risking a wrongly-accepted command
      actually moving the robot. exec_ms is always 0 in this mode.
  --live: additionally calls RobotController.execute_action() for accepted
      commands, for the final physical-latency numbers once ready to run on
      real hardware. Requires --robot-ip and the robot to be reachable.

Usage:
    python run_verification_benchmark.py --dry-run
    python run_verification_benchmark.py --dry-run --configs C1 C3
    python run_verification_benchmark.py --live --robot-ip 192.168.123.18
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import find_dotenv, load_dotenv

# .env may live above this project's own root (e.g. /media/hong/data/.env) --
# same lookup dashboard_server.py uses, search upward from cwd.
load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(level=logging.WARNING)  # keep benchmark stdout readable; per-row detail goes to the CSV
logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path(__file__).parent / "benchmark_corpus.json"
DEFAULT_OUTPUT = Path(__file__).parent / "benchmark_results.csv"

CSV_FIELDS = [
    "config", "case_id", "repeat", "category", "instruction", "expect_reject",
    "understood", "raw_actions", "llm_ms",
    "accepted", "reject_layer", "reject_reason",
    "schema_ms", "intent_ms", "context_ms", "exec_ms",
    "resolved_actions", "error",
]


def _load_corpus(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def _generate_actions(client, model: str, instruction: str) -> Dict[str, Any]:
    """Same call shape as dashboard_server.py's _llm_and_execute: get_system_prompt()
    + JSON-mode completion, parsed with the same regex fallback."""
    from voice_control_go2 import get_system_prompt

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_prompt("ko")},
            {"role": "user", "content": instruction},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=512,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw), raw
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        return (json.loads(m.group()) if m else {}), raw


def _build_robot_state(case: dict):
    from behavior_schema import RobotState
    overrides = case.get("robot_state_override") or {}
    state = RobotState(is_standing=True, battery_level=100.0)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def run_one(client, model: str, config_name: str, case: dict, live: bool, robot=None, repeat: int = 0,
            confirm_each_action: bool = False) -> Dict[str, Any]:
    from action_merge import merge_consecutive_moves
    from action_registry import get_registry
    from runtime_verification import verify_sequence
    from verification_config import get_verification_config
    from voice_control_go2 import apply_tone_to_moves

    row = {
        "config": config_name, "case_id": case["id"], "repeat": repeat, "category": case["category"],
        "instruction": case["instruction"], "expect_reject": case.get("expect_reject"),
        "understood": None, "raw_actions": "", "llm_ms": 0.0,
        "accepted": None, "reject_layer": "", "reject_reason": "",
        "schema_ms": 0.0, "intent_ms": 0.0, "context_ms": 0.0, "exec_ms": 0.0,
        "resolved_actions": "", "error": "",
    }

    t_llm = time.monotonic()
    try:
        cmd, raw = _generate_actions(client, model, case["instruction"])
    except Exception as e:
        row["error"] = f"LLM generation failed: {e}"
        return row
    row["llm_ms"] = (time.monotonic() - t_llm) * 1000

    understood = bool(cmd.get("understood", False))
    row["understood"] = understood
    if not understood:
        row["accepted"] = False
        row["reject_layer"] = "not_understood"
        row["reject_reason"] = cmd.get("response", "")
        return row

    raw_action_list = cmd.get("actions") or []
    row["raw_actions"] = json.dumps(raw_action_list, ensure_ascii=False)

    merged = merge_consecutive_moves(raw_action_list, cmd, get_registry().get_action_name)
    merged = apply_tone_to_moves(merged, cmd.get("tone", "neutral"))

    robot_state = _build_robot_state(case)
    result = verify_sequence(case["instruction"], merged, robot_state, get_verification_config())

    row["accepted"] = result.accepted
    row["reject_layer"] = result.reject_layer or ""
    row["reject_reason"] = result.reject_reason or ""
    row["schema_ms"] = round(result.schema_ms, 2)
    row["intent_ms"] = round(result.intent_ms, 2)
    row["context_ms"] = round(result.context_ms, 2)
    row["resolved_actions"] = ",".join(result.resolved_actions)

    if result.accepted and live and robot is not None:
        t_exec = time.monotonic()
        for name, params in merged:
            if name in ("navigate_to", "vision_analyze"):
                continue  # out of scope for this benchmark
            if confirm_each_action:
                ans = input(f"    >> about to execute on the REAL robot: {name}({params})  -- run it? [y/N/q=quit entire run] ").strip().lower()
                if ans == "q":
                    print("Aborted by operator.")
                    sys.exit(1)
                if ans != "y":
                    print(f"    >> skipped {name}")
                    continue
            robot.execute_action(name, **params)
        row["exec_ms"] = round((time.monotonic() - t_exec) * 1000, 2)

    return row


def main():
    p = argparse.ArgumentParser(description="C0-C3 runtime verification benchmark sweep")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--configs", nargs="+", default=["C0", "C1", "C2", "C3"])
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    p.add_argument("--live", action="store_true", help="Actually execute accepted commands on the robot (default: dry-run)")
    # 192.168.12.x = Go2's own WiFi hotspot (WebRTC path); this matches the
    # default used everywhere else in this codebase (voice_control_go2.py,
    # full_voice_pipeline.py, camera_stream.py, ...), not the DDS/Ethernet-LAN
    # address. See behavior_executor.py's _connect() for the branch logic.
    p.add_argument("--robot-ip", default="192.168.12.1")
    p.add_argument("--repeats", type=int, default=1,
                    help="Run each case N times per config (e.g. 3 for the dry-run tier), to capture "
                         "LLM sampling variance (temperature=0.3 in _generate_actions). exec_ms in "
                         "--live mode means each repeat also physically re-executes accepted commands, "
                         "so this is normally left at 1 for --live runs.")
    p.add_argument("--live-subset-only", action="store_true",
                    help="Filter the corpus to cases marked \"live_candidate\": true -- the 40-case "
                         "subset selected for physical trials (see benchmark_corpus.json's "
                         "_live_subset_note). Intended for use with --live --configs C0 C3.")
    p.add_argument("--confirm-each-action", action="store_true",
                    help="With --live: pause and ask y/N/q before each individual action is actually "
                         "sent to the robot, printing the action name and params first. Strongly "
                         "recommended for a first C0 (no-protection baseline) run, since C0 has no "
                         "software safety net beyond the pre-existing always-on battery/posture check.")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set (this benchmark needs the real LLM call, not the robot)")
        sys.exit(1)

    from openai import OpenAI
    from verification_config import set_preset
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    robot = None
    if args.live:
        from robot_controller import RobotController
        print(f"🤖 --live: connecting to robot at {args.robot_ip} ...")
        robot = RobotController(robot_ip=args.robot_ip)

    cases = _load_corpus(args.corpus)
    if args.live_subset_only:
        cases = [c for c in cases if c.get("live_candidate")]
    print(f"Loaded {len(cases)} cases from {args.corpus}"
          + (" (live_candidate subset)" if args.live_subset_only else ""))
    print(f"Configs: {args.configs}  |  mode: {'LIVE (robot will move)' if args.live else 'dry-run (no physical execution)'}"
          + (f"  |  repeats: {args.repeats}" if args.repeats > 1 else ""))

    rows = []
    for config_name in args.configs:
        set_preset(config_name)
        print(f"\n=== {config_name} ===")
        for case in cases:
            for repeat in range(args.repeats):
                row = run_one(client, args.model, config_name, case, args.live, robot, repeat=repeat,
                               confirm_each_action=args.confirm_each_action)
                status = "ACCEPT" if row["accepted"] else f"REJECT[{row['reject_layer']}]"
                rep_suffix = f" (rep {repeat + 1}/{args.repeats})" if args.repeats > 1 else ""
                print(f"  {case['id']:<22} {status:<22} {row['reject_reason'][:60]}{rep_suffix}")
                rows.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Wrote {len(rows)} rows to {args.output}")

    if robot is not None:
        robot.cleanup()


if __name__ == "__main__":
    main()
