#!/usr/bin/env python3
"""
Run Dance — full pipeline entry point.

Music -> Beat Detection -> Dance Timeline -> Motion Executor -> Go2

Usage:
  # Preview only — prints the timeline, touches neither robot nor speakers.
  python3 run_dance.py --audio ~/music/gangnam_style.mp3 --song gangnam_style --dry-run

  # Full run.
  python3 run_dance.py --audio ~/music/gangnam_style.mp3 --song gangnam_style --robot-ip 192.168.12.1
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

# .env may live above this project's own root (e.g. /media/hong/data/.env) —
# search upward from cwd rather than assuming a fixed relative path. Must run
# before build_timeline()'s OPENAI_API_KEY check below.
load_dotenv(find_dotenv(usecwd=True))

from beat_detection import detect_beats
from dance_timeline import build_timeline_auto, build_timeline_from_pattern, build_timeline_from_segments, trim_timeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_timeline(audio_path: str, song: str, openai_api_key: str = None, no_cache: bool = False):
    beat_info = detect_beats(audio_path)

    if song == "auto-simple":
        return build_timeline_auto(beat_info)

    if song == "auto":
        # Smart auto: detect structural segments, let an LLM choreograph each
        # one based on its measured energy -- works for any track, no
        # per-song script needed. Falls back to plain beat-cycling if no
        # OpenAI key is configured.
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No OPENAI_API_KEY set — falling back to simple beat-cycling (--song auto-simple).")
            return build_timeline_auto(beat_info)

        from section_detection import detect_segments
        from llm_choreographer import choreograph

        segments = detect_segments(audio_path, beat_info, n_segments=16)
        styles = choreograph(audio_path, beat_info.tempo, segments, openai_api_key=api_key, use_cache=not no_cache)

        # One-off highlight flip at the single highest-energy segment (mirrors
        # songs/gangnam_style.py's curated climax) -- only if that peak is
        # genuinely energetic, not just the loudest of an otherwise-mellow track.
        climax_index = None
        if segments:
            peak = max(range(len(segments)), key=lambda i: segments[i].energy)
            if segments[peak].energy >= 0.85:
                climax_index = peak
                logger.info(f"Climax flip scheduled at segment {peak} (energy={segments[peak].energy:.2f})")

        return build_timeline_from_segments(beat_info, segments, styles, climax_index=climax_index)

    songs_dir = Path(__file__).parent / "songs"
    module_path = songs_dir / f"{song}.py"
    if not module_path.exists():
        available = [p.stem for p in songs_dir.glob("*.py") if p.stem != "__init__"]
        raise SystemExit(f"No curated pattern for '{song}'. Available: {available or '(none)'} — or use --song auto")

    sys.path.insert(0, str(songs_dir))
    song_module = __import__(song)
    pattern = song_module.generate_pattern(beat_info)
    timeline = build_timeline_from_pattern(beat_info, pattern)
    timeline.song_name = song
    return timeline


def main():
    p = argparse.ArgumentParser(description="Go2 music-synced dance")
    p.add_argument("--audio", required=True, help="Path to audio file (mp3/wav)")
    p.add_argument("--song", default="auto",
                   help="'auto' (LLM-choreographed, any track, needs OPENAI_API_KEY), "
                        "'auto-simple' (no LLM, plain beat-cycling), or a curated pattern name (e.g. gangnam_style)")
    p.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env)")
    p.add_argument("--no-cache", action="store_true", help="Force a fresh LLM choreography call, ignore cached result")
    p.add_argument("--robot-ip", default="192.168.12.1")
    p.add_argument("--end-at", type=float, default=None,
                    help="Stop the routine (and the music) at this many seconds in — "
                         "e.g. to perform only the first verse/chorus cycle of a repeating song")
    p.add_argument("--dry-run", action="store_true", help="Print the timeline only — no robot, no audio")
    p.add_argument("--no-audio", action="store_true", help="Run the robot routine without playing sound")
    args = p.parse_args()

    timeline = build_timeline(args.audio, args.song, openai_api_key=args.openai_key, no_cache=args.no_cache)
    if args.end_at is not None:
        timeline = trim_timeline(timeline, args.end_at)
    print(f"\nTimeline: '{timeline.song_name}' — {len(timeline.cues)} cues, "
          f"{timeline.total_duration():.1f}s, tempo={timeline.tempo:.1f} BPM\n")

    if args.dry_run:
        for cue in timeline.cues:
            detail = cue.action if cue.kind == "action" else cue.params
            print(f"  {cue.time:7.2f}s  {cue.kind:6s} {detail}")
        return

    from motion_executor import MotionExecutor

    audio_thread = None
    if not args.no_audio:
        from audio_player import play_audio_async
        audio_thread = play_audio_async(args.audio, end_at=args.end_at)

    executor = MotionExecutor(args.robot_ip)
    executor.execute(timeline)

    if audio_thread:
        audio_thread.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(0)
