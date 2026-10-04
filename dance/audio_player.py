#!/usr/bin/env python3
"""
Audio Player
Plays the music locally (speakers) in parallel with the motion executor, so a
human watching hears the same track the robot is dancing to.
"""

import logging
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _convert_to_wav(audio_path: str) -> str:
    """soundfile can't decode mp3 on all builds — convert via ffmpeg first."""
    out_path = str(Path(tempfile.gettempdir()) / (Path(audio_path).stem + "_dance.wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, out_path],
        capture_output=True, timeout=60, check=True,
    )
    return out_path


def play_audio_async(audio_path: str, end_at: float = None) -> threading.Thread:
    """Start playback in a background thread; returns immediately.

    Args:
        end_at: if set, stop playback at this many seconds in — keeps the
            music in sync with a trimmed dance timeline (see
            dance_timeline.trim_timeline) instead of playing the full track
            after the robot's routine has already ended.
    """
    def _play():
        import soundfile as sf
        import sounddevice as sd

        path = audio_path
        if not path.lower().endswith(".wav"):
            logger.info("Converting to WAV for playback...")
            path = _convert_to_wav(audio_path)

        data, sr = sf.read(path, dtype="float32")
        if end_at is not None:
            data = data[: int(end_at * sr)]
        logger.info("Starting audio playback")
        sd.play(data, sr)
        sd.wait()
        logger.info("Audio playback finished")

    t = threading.Thread(target=_play, daemon=True, name="audio-player")
    t.start()
    return t
