#!/usr/bin/env python3
"""
Layer 2 (V_i) -- intent consistency verification.

This is a second, separately-scoped LLM call, independent of the call that
generated the action sequence in the first place: it receives only the
user's raw instruction and the already-resolved action list (canonical
action names + parameters), never the generator's own `response` text or
reasoning trace. The point is to reduce -- not eliminate -- the risk that
generator and verifier share the same blind spot because they were shown
the same framing of the problem (see the thesis's discussion of verifier
independence in Section 4).

Before this module existed, there was no automated intent check at all on
the voice-command path (dashboard_server.py's _llm_and_execute auto-executes
straight from the generator); only the text-command panel had a *manual*
UI confirm step, which is not a substitute for an automated check on the
path this thesis actually evaluates.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

INTENT_VERIFIER_MODEL = os.getenv("INTENT_VERIFIER_MODEL", "gpt-4o-mini")

_VERIFIER_SYSTEM_PROMPT = """You are a safety checker for a quadruped robot.
You will be shown a user's instruction and a list of actions a separate
system has already chosen to execute in response. Your only job is to judge
whether that action list is a plausible, reasonable way to carry out the
instruction -- not whether it's the only possible way, and not whether
individual parameter values are within safe limits (a different check
already handles that). Reject only if the action list clearly contradicts,
ignores, or does something materially different from what was asked (for
example: the user asked the robot to stay still but the list contains
movement; the user asked for one thing and the list does something
unrelated; the list is empty despite a clear action request).

Respond with a JSON object exactly like this, and nothing else:
{"consistent": true, "reason": "one short sentence"}
"""


@dataclass
class IntentVerdict:
    accepted: bool
    reason: str
    latency_ms: float
    # True when `accepted` is a fail-closed default caused by the verifier
    # itself failing (network/API error, unparseable response) rather than
    # a genuine "this doesn't match the instruction" judgment. Kept distinct
    # so later analysis doesn't conflate "the verifier caught something" with
    # "the verifier was unavailable" when computing rejection rates.
    infra_error: bool = False


def _format_actions(actions: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines = []
    for i, (name, params) in enumerate(actions, 1):
        if params:
            arg_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"{i}. {name}({arg_str})")
        else:
            lines.append(f"{i}. {name}")
    return "\n".join(lines) if lines else "(no actions)"


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _client


def verify_intent(
    instruction: str,
    actions: List[Tuple[str, Dict[str, Any]]],
    model: str = INTENT_VERIFIER_MODEL,
    client: Optional[Any] = None,
) -> IntentVerdict:
    """Ask a dedicated LLM call whether `actions` plausibly carries out
    `instruction`.

    `client` is injectable (anything exposing the same
    `.chat.completions.create(...)` shape as the OpenAI SDK client) so tests
    can supply a stub without hitting the network.
    """
    if not actions:
        return IntentVerdict(accepted=True, reason="no actions to verify", latency_ms=0.0)

    t0 = time.monotonic()
    try:
        c = client or _get_client()
        resp = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Instruction: {instruction}\n\n"
                    f"Chosen actions:\n{_format_actions(actions)}"
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        raw = resp.choices[0].message.content
        try:
            verdict = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            verdict = json.loads(m.group()) if m else {}

        accepted = bool(verdict.get("consistent", False))
        reason = str(verdict.get("reason", "")) or ("consistent" if accepted else "intent mismatch")
        return IntentVerdict(accepted=accepted, reason=reason, latency_ms=latency_ms)

    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error(f"Intent verifier call failed: {e}")
        # Fail CLOSED: if the safety check itself cannot be evaluated
        # (network error, API outage, malformed response), the command is
        # rejected rather than silently let through -- otherwise "the
        # verifier is unavailable" becomes an unmeasured way to bypass
        # verification entirely, which would be the wrong default for a
        # safety layer. infra_error=True lets analysis separate this from a
        # genuine negative verdict when computing false-rejection rates.
        return IntentVerdict(
            accepted=False,
            reason=f"intent verifier unavailable: {e}",
            latency_ms=latency_ms,
            infra_error=True,
        )
