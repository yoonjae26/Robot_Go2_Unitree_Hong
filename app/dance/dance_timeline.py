#!/usr/bin/env python3
"""
Dance Timeline
Beat times -> an ordered list of timed robot cues (DanceTimeline).

Two ways to build one:
  - build_timeline_from_pattern(): a curated, hand-authored routine (see
    songs/) resolved against this specific audio file's detected beats --
    deterministic, same choreography every time for a known song.
  - build_timeline_auto(): a generic fallback for any track with no curated
    routine -- cycles through a safe move pool on a fixed beat interval.
"""

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from beat_detection import BeatInfo

logger = logging.getLogger(__name__)


@dataclass
class DanceCue:
    """One scheduled event in a dance."""
    time: float                              # seconds from start of track
    kind: str                                # "action" | "sway" | "stop"
    action: str = ""                         # action name (kind="action"), e.g. "hello"
    params: Dict[str, Any] = field(default_factory=dict)
    duration: float = 1.0                    # how long this cue occupies, seconds


@dataclass
class DanceTimeline:
    song_name: str
    audio_path: str
    tempo: float
    cues: List[DanceCue]

    def total_duration(self) -> float:
        if not self.cues:
            return 0.0
        last = self.cues[-1]
        return last.time + last.duration


# Moves confirmed working over WiFi (see webrtc_sport_client.py) that are safe
# to chain repeatedly for a dance routine -- no flips/jumps by default, those
# are jarring/riskier for continuous rhythmic use. A curated song pattern can
# still reference any registered action explicitly if you want bigger moves.
SAFE_MOVE_POOL = ["stretch", "scrape", "front_pounce", "pose", "cross_step"]


def build_timeline_auto(
    beat_info: BeatInfo,
    move_pool: Optional[List[str]] = None,
    beats_per_move: int = 4,
    sway_amplitude: float = 0.25,
) -> DanceTimeline:
    """Generic beat-synced routine for a track with no curated pattern.

    Sways in place (Move) on every beat, and fires a gesture from move_pool
    every `beats_per_move` beats.
    """
    move_pool = move_pool or SAFE_MOVE_POOL
    beats = beat_info.beat_times
    period = beat_info.beat_period
    cues: List[DanceCue] = []

    for i, t in enumerate(beats):
        if i % beats_per_move == 0:
            action = move_pool[(i // beats_per_move) % len(move_pool)]
            cues.append(DanceCue(time=t, kind="action", action=action, duration=period * beats_per_move))
        else:
            direction = 1 if (i % 2 == 0) else -1
            cues.append(DanceCue(
                time=t, kind="sway",
                params={"vx": 0.0, "vy": sway_amplitude * direction, "omega": 0.0},
                duration=period,
            ))

    return DanceTimeline(
        song_name="auto",
        audio_path=beat_info.audio_path,
        tempo=beat_info.tempo,
        cues=cues,
    )


def build_timeline_from_pattern(beat_info: BeatInfo, pattern: List[dict]) -> DanceTimeline:
    """Resolve a curated song pattern against this file's actual detected beats.

    Each pattern entry: {"beat": <index into beat_info.beat_times>, "kind": ...,
    "action"/"params"/"duration_beats": ...}. Using beat *indices* (not raw
    seconds) means the same curated routine still lines up with the music even
    if this particular audio file has a slightly different intro length/edit
    than whichever copy the pattern was authored against.
    """
    beats = beat_info.beat_times
    period = beat_info.beat_period
    cues: List[DanceCue] = []

    for entry in pattern:
        idx = entry["beat"]
        if idx >= len(beats):
            logger.warning(f"Pattern beat index {idx} beyond detected beats ({len(beats)}) — skipping")
            continue
        duration = entry.get("duration_beats", 1) * period
        cues.append(DanceCue(
            time=beats[idx],
            kind=entry["kind"],
            action=entry.get("action", ""),
            params=entry.get("params", {}),
            duration=duration,
        ))

    return DanceTimeline(
        song_name=pattern[0].get("song_name", "curated") if pattern else "curated",
        audio_path=beat_info.audio_path,
        tempo=beat_info.tempo,
        cues=cues,
    )


def trim_timeline(timeline: DanceTimeline, end_at: float) -> DanceTimeline:
    """Cut a timeline down to only the cues before `end_at` seconds -- e.g. to
    perform just the first verse/chorus cycle of a song that repeats itself,
    instead of the full track."""
    kept = [c for c in timeline.cues if c.time < end_at]
    return replace(timeline, cues=kept)


def _snap_to_downbeat(beat_times: List[float], t: float, seg_end: float) -> Optional[float]:
    """Find the next beat >= t, then round forward to the next bar-1 (every
    4th beat, assuming 4/4 -- true for nearly all pop/dance tracks). Landing
    gesture changes on the downbeat instead of an arbitrary nearby beat is
    what makes a routine feel like it's actually dancing *to* the music
    instead of just ticking through a schedule. Falls back to the plain next
    beat if the next downbeat would overshoot the segment (short segment /
    near a boundary), so we never wait past where we're allowed to move."""
    idx = next((i for i, bt in enumerate(beat_times) if bt >= t), None)
    if idx is None:
        return None
    downbeat_idx = idx if idx % 4 == 0 else idx + (4 - idx % 4)
    if downbeat_idx < len(beat_times) and beat_times[downbeat_idx] < seg_end:
        return beat_times[downbeat_idx]
    return beat_times[idx] if beat_times[idx] < seg_end else None


def _real_action_duration(action_name: str, default: float = 2.0) -> float:
    """Look up an action's actual configured duration (seconds) from the
    behavior library, so scheduling matches how long the robot really takes
    -- not a nominal beat-count guess. Falls back to `default` if unknown."""
    try:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from app.core.behavior_library import get_library
        schema = get_library().get(action_name)
        return schema.duration if schema else default
    except Exception:
        return default


def build_timeline_from_segments(
    beat_info: BeatInfo,
    segments,
    styles: List[dict],
    climax_index: Optional[int] = None,
    climax_action: str = "back_flip",
) -> DanceTimeline:
    """Resolve auto-detected segments + their LLM-chosen styles into a timeline.

    `segments` (section_detection.Segment) and `styles` (llm_choreographer's
    validated output) must be the same length and in the same order --
    styles[i] describes how to move during segments[i].

    Gesture actions are chained back-to-back using each move's *real*
    configured duration (not the LLM's beats_per_move, which has no idea how
    long a move actually takes on the robot) -- this is what MotionExecutor's
    single-action-at-a-time lock expects. Scheduling a move every 2 beats when
    it actually takes 8s to run meant most cues silently got dropped while
    busy, leaving the robot standing still with nothing scheduled in between
    -- the routine's "disjointed" feel. Chaining by real duration means the
    next move starts right as the previous one finishes: no dead air, no
    dropped cues. A short beat-synced sway pulse fills any gap between one
    move ending and the next detected beat, to keep the routine feeling
    beat-locked rather than just a flat sequence of gestures.

    climax_index: if given, that segment gets a single one-off `climax_action`
    (default BackFlip) fired at its very first beat, before its normal move
    cycle resumes -- mirrors the curated songs/gangnam_style.py pattern, which
    deliberately keeps flips out of the regular cycling pool (repeating one
    every couple of seconds is a motor-wear/stability risk) but still wants a
    highlight moment. Callers should reserve this for the single
    highest-energy segment, not sprinkle it across many.
    """
    period = beat_info.beat_period
    cues: List[DanceCue] = []

    for seg_i, (seg, style) in enumerate(zip(segments, styles)):
        sway = style["sway_amplitude"]
        moves = style["moves"]
        move_i = 0
        direction = 1

        beat_idx = next((i for i, bt in enumerate(beat_info.beat_times) if bt >= seg.start), None)
        if beat_idx is None:
            continue
        t = beat_info.beat_times[beat_idx]

        if seg_i == climax_index:
            climax_duration = _real_action_duration(climax_action)
            cues.append(DanceCue(time=t, kind="action", action=climax_action, duration=climax_duration))
            t += climax_duration
            next_beat = _snap_to_downbeat(beat_info.beat_times, t, seg.end)
            if next_beat is None:
                continue
            t = next_beat

        while t < seg.end:
            action = moves[move_i % len(moves)]
            move_i += 1
            duration = _real_action_duration(action)
            cues.append(DanceCue(time=t, kind="action", action=action, duration=duration))
            t += duration

            next_beat = _snap_to_downbeat(beat_info.beat_times, t, seg.end)
            if next_beat is None:
                break
            direction *= -1
            # A plain side-step sway reads as "waiting", not dancing -- mix in
            # a bit of forward/back rock and a slight counter-twist so the gap
            # between gestures still looks like movement, not a pause.
            cues.append(DanceCue(
                time=next_beat, kind="sway",
                params={
                    "vx": sway * 0.4 * direction,
                    "vy": sway * direction,
                    "omega": sway * 0.3 * -direction,
                },
                duration=period,
            ))
            t = next_beat + period

    cues.sort(key=lambda c: c.time)
    return DanceTimeline(
        song_name="auto-llm",
        audio_path=beat_info.audio_path,
        tempo=beat_info.tempo,
        cues=cues,
    )
