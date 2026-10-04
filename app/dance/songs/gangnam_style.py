#!/usr/bin/env python3
"""
Curated choreography for PSY - "Gangnam Style" (~132 BPM, 252s M/V-length cut).

Section boundaries below are grounded in a measured RMS energy profile of the
actual local audio file (4s windows), not guesswork: energy is near-zero for
the first ~4s (matches the first detected beat at 4.06s), stays in a fairly
consistent moderate/high band from ~4s to ~216s (RMS alone doesn't cleanly
separate verse/chorus in a densely-mixed track like this), drops sharply to
~0.05 at 220s and jumps back up at 224s (an unambiguous breakdown-then-drop),
holds high through ~244s, then fades out to the end. Because RMS can't
distinguish verse from chorus within that long moderate-energy middle
stretch, the boundaries there still alternate on a fixed ~32s phrase length
as a reasonable approximation -- watch/listen to a run and nudge them if a
transition feels a beat or two early/late.

generate_pattern() maps each boundary to the nearest beat actually detected
in the audio file, so cues stay beat-locked regardless.

Section types drive which moves get used. Move choices draw from the full set
confirmed working over WiFi after the MCF api_id fix (see webrtc_sport_client.py)
-- no repeated flips (motor wear / stability risk if triggered every couple
seconds), but a single one-off BackFlip is inserted as a climax flourish right
on the breakdown-into-drop moment (see CLIMAX_BEAT_OFFSET below), not as part
of the regular cycling pool.

trot_run (a bouncy trotting gait -- the closest built-in match to Gangnam
Style's signature "invisible horse" move) leads the buildup/chorus move
pools, confirmed working standalone on hardware before being added here.

MotionExecutor only runs one gesture action at a time (see its _action_busy
guard) and skips a cue if the previous action is still mid-flight, so a
gesture's *real* duration (its ActionSchema.duration, not duration_beats here)
is what actually paces the routine -- duration_beats below is informational/
for --dry-run printing, not a hard timing guarantee.
"""

from typing import List

# (start_sec, end_sec, section_type) — adjust after watching a first run.
SECTIONS = [
    (0.0,   4.0,   "intro"),      # near-silence before the beat starts
    (4.0,   64.0,  "verse"),
    (64.0,  88.0,  "chorus"),     # first local energy peak
    (88.0,  148.0, "verse"),
    (148.0, 168.0, "chorus"),     # second local energy peak
    (168.0, 220.0, "verse"),
    (220.0, 224.0, "breakdown"),  # measured: sharp energy dip then jump back
    (224.0, 248.0, "chorus"),     # high energy after the drop
    (248.0, 252.0, "outro"),      # measured fade-out
]

# Per-section-type choreography: which action fires every Nth beat, and the
# sway amplitude used on the other beats.
SECTION_STYLE = {
    "intro":     {"beats_per_move": 8, "moves": ["balanced_stand", "stretch"],                   "sway": 0.15},
    "verse":     {"beats_per_move": 4, "moves": ["stretch", "pose", "cross_step", "free_walk"],  "sway": 0.20},
    "buildup":   {"beats_per_move": 2, "moves": ["trot_run", "scrape", "front_pounce", "front_jump", "walk_upright"], "sway": 0.30},
    "chorus":    {"beats_per_move": 2, "moves": ["trot_run", "front_pounce", "cross_step", "trot_run", "hand_stand"], "sway": 0.35},
    "breakdown": {"beats_per_move": 4, "moves": ["pose", "stretch", "balanced_stand"],           "sway": 0.15},
    "outro":     {"beats_per_move": 8, "moves": ["stretch", "balanced_stand"],                   "sway": 0.10},
}

# One-off highlight: fires once, on the beat where "breakdown" hands off to
# "chorus" (the measured drop). Not part of the cycling pool above -- a flip
# repeated every couple of beats would be excessive; a single one on the drop
# is a deliberate flourish. Ensure clear, flat space before running.
CLIMAX_SECTION_BOUNDARY = "breakdown", "chorus"
CLIMAX_ACTION = "back_flip"


def _section_for(t: float) -> str:
    for start, end, kind in SECTIONS:
        if start <= t < end:
            return kind
    return "outro"


def generate_pattern(beat_info) -> List[dict]:
    """Build the curated cue pattern for this specific audio file's beat grid."""
    pattern: List[dict] = []
    move_counters = {}
    prev_kind = None
    climax_from, climax_to = CLIMAX_SECTION_BOUNDARY

    for i, t in enumerate(beat_info.beat_times):
        kind = _section_for(t)

        if prev_kind == climax_from and kind == climax_to:
            pattern.append({
                "song_name": "gangnam_style",
                "beat": i,
                "kind": "action",
                "action": CLIMAX_ACTION,
                "duration_beats": 4,
            })
            move_counters[kind] = 0
            prev_kind = kind
            continue
        prev_kind = kind

        style = SECTION_STYLE[kind]
        n = move_counters.get(kind, 0)
        move_counters[kind] = n + 1

        if n % style["beats_per_move"] == 0:
            moves = style["moves"]
            action = moves[(n // style["beats_per_move"]) % len(moves)]
            pattern.append({
                "song_name": "gangnam_style",
                "beat": i,
                "kind": "action",
                "action": action,
                "duration_beats": style["beats_per_move"],
            })
        else:
            direction = 1 if (i % 2 == 0) else -1
            # Continuous Euler body-tilt groove between gesture accents --
            # Euler (like Move) bypasses the action-schema/cooldown system
            # entirely (see webrtc_sport_client.py / motion_executor.py's
            # _rock()), so it can fire every single beat without ever being
            # rate-limited the way discrete tricks are. This is what actually
            # fills the gaps between accents with motion instead of the robot
            # standing still "waiting" for the next trick -- a rocking/
            # bobbing body reads as dancing far more than a flat side-shuffle.
            # Amplitude is deliberately conservative (max ~0.12 rad / 7° at
            # the highest chorus sway=0.35) for safety on a first live run --
            # raise SECTION_STYLE's "sway" values once that's confirmed stable.
            amp = style["sway"]
            pattern.append({
                "song_name": "gangnam_style",
                "beat": i,
                "kind": "rock",
                "params": {
                    "roll": amp * 0.35 * direction,
                    "pitch": amp * 0.25 * -direction,
                    "yaw": amp * 0.20 * direction,
                },
                "duration_beats": 1,
            })

    return pattern
