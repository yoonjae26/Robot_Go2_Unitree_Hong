#!/usr/bin/env python3
"""
LLM Choreographer
Auto-detected segments (numeric features only -- no audio, no lyrics) -> a
fixed-schema JSON choreography, one entry per segment. Same call pattern as
dashboard_server.py's voice pipeline: OpenAI client, JSON-only response,
validated against a fixed schema before use.

Results are cached to songs/_cache/ keyed by the audio file + its detected
segment energies, so a given track only calls the LLM once ever -- every
later run (and the choreography actually used on the robot) is a plain local
JSON read, deterministic and free.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

# .env lives at /media/hong/data/.env — one level above this project's own
# root — so search upward from cwd rather than assuming a fixed relative path.
load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "songs" / "_cache"

# Moves allowed for unsupervised/automatic choreography (no per-song human
# review). Flips/HandStand deliberately excluded -- those are opt-in only via
# a hand-curated song file with a reviewed, one-off placement (see
# songs/gangnam_style.py's CLIMAX_ACTION), not something an LLM should sprinkle
# into an arbitrary, unreviewed track. All other confirmed-safe LOW/MEDIUM
# gait switches are included (trot_run, walk_upright, free_bound, free_avoid,
# static_walk) -- these actually move the robot across the floor rather than
# just gesturing in place, so mixing them in with the static gestures gives
# real variety instead of the same handful of stationary tricks on repeat.
ALLOWED_MOVES = [
    "stretch", "scrape", "front_pounce", "front_jump", "pose", "cross_step",
    "free_walk", "balanced_stand", "trot_run", "walk_upright", "free_bound",
    "free_avoid", "static_walk",
]

_SCHEMA_EXAMPLE = {
    "segments": [
        {"index": 0, "style": "calm", "beats_per_move": 8, "sway_amplitude": 0.15, "moves": ["hello"]},
        {"index": 1, "style": "energetic", "beats_per_move": 2, "sway_amplitude": 0.35, "moves": ["dance1", "front_pounce"]},
    ]
}


def _cache_key(audio_path: str, segments) -> str:
    sig = f"{Path(audio_path).name}:{len(segments)}:" + ",".join(f"{s.energy:.2f}" for s in segments)
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def _min_moves_for_duration(duration: float) -> int:
    """Longer segments cycling only 2 moves start to feel repetitive well
    before the segment ends (e.g. a 57s segment alternating just 2 gestures
    repeats ~10+ times) -- scale the variety floor with how long the segment
    actually runs."""
    if duration > 40:
        return 5
    if duration > 20:
        return 4
    return 2


def _ensure_variety(segments_styles: List[dict], segments=None) -> List[dict]:
    """Guarantee a minimum number of distinct moves per segment (more for
    longer segments -- see _min_moves_for_duration). The prompt asks the
    model for "2-4 different moves per segment for variety" but nothing
    enforced it -- a segment left with a single move gets that one gesture
    fired back-to-back for the segment's entire duration (looks like the
    routine is "stuck" repeating). Pads from ALLOWED_MOVES, offset by segment
    index so consecutive segments don't all pad with the same filler move.
    Applied on every read (fresh AND cached) so it also self-heals choreography
    files cached before this existed."""
    for i, seg in enumerate(segments_styles):
        moves = list(dict.fromkeys(seg.get("moves", [])))  # de-dup, keep order
        min_moves = _min_moves_for_duration(segments[i].duration) if segments else 2
        offset = 0
        while len(moves) < min_moves and offset < len(ALLOWED_MOVES):
            candidate = ALLOWED_MOVES[(i + offset) % len(ALLOWED_MOVES)]
            if candidate not in moves:
                moves.append(candidate)
            offset += 1
        seg["moves"] = moves
    return segments_styles


def choreograph(
    audio_path: str,
    tempo: float,
    segments,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_cache: bool = True,
) -> List[dict]:
    """Returns one validated style dict per segment (same order/length as `segments`)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(audio_path, segments)}.json"

    if use_cache and cache_file.exists():
        logger.info(f"Using cached choreography: {cache_file.name}")
        return _ensure_variety(json.loads(cache_file.read_text())["segments"], segments)

    from openai import OpenAI
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)

    seg_desc = "\n".join(
        f"  {i}: duration={s.duration:.1f}s, energy={s.energy:.2f} "
        f"(0=calmest, 1=most energetic *in this track*), beat_count={s.beat_count}"
        for i, s in enumerate(segments)
    )

    prompt = f"""You are choreographing a quadruped robot dance for a track at {tempo:.0f} BPM.

Below are the track's structural segments, detected purely from audio signal
features (timbre + harmony + energy) -- no lyrics or song identity involved:
{seg_desc}

For each segment index, choose a dance style:
- Only use moves from this exact list: {ALLOWED_MOVES}
- Higher energy -> shorter beats_per_move (2-3, more frequent gestures) and higher sway_amplitude (0.25-0.4)
- Lower energy -> longer beats_per_move (6-8) and lower sway_amplitude (0.1-0.2)
- Pick 2-4 different moves per segment for variety
- style is a short label of your choice (e.g. "calm", "buildup", "energetic", "climax")

Respond with ONLY valid JSON, one entry per segment index (0 to {len(segments)-1}),
matching exactly this schema:
{json.dumps(_SCHEMA_EXAMPLE, indent=2)}
"""

    logger.info(f"Requesting choreography from {model} for {len(segments)} segments...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = json.loads(response.choices[0].message.content)
    validated = _ensure_variety(_validate_and_fix(raw.get("segments", []), len(segments)), segments)

    cache_file.write_text(json.dumps({"segments": validated}, indent=2, ensure_ascii=False))
    logger.info(f"Cached choreography: {cache_file.name}")
    return validated


def _validate_and_fix(llm_segments: list, n_expected: int) -> List[dict]:
    """Never trust the model's output blindly: fill gaps, drop unknown moves,
    clamp numeric ranges. A malformed/missing entry falls back to a safe,
    low-energy default rather than failing the whole routine."""
    fixed = []
    by_index = {e.get("index"): e for e in llm_segments if isinstance(e, dict)}
    for i in range(n_expected):
        entry = by_index.get(i, {})
        moves = [m for m in entry.get("moves", []) if m in ALLOWED_MOVES]
        if not moves:
            logger.warning(f"Segment {i}: no valid moves in LLM output, defaulting to ['hello']")
            moves = ["hello"]
        fixed.append({
            "index": i,
            "style": str(entry.get("style", "calm")),
            "beats_per_move": max(1, min(16, int(entry.get("beats_per_move", 4) or 4))),
            "sway_amplitude": max(0.05, min(0.4, float(entry.get("sway_amplitude", 0.2) or 0.2))),
            "moves": moves,
        })
    return fixed
