#!/usr/bin/env python3
"""
Beat Detection
Music (file) -> tempo + beat timestamps, via librosa.
"""

import logging
from dataclasses import dataclass, replace
from typing import List

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BeatInfo:
    audio_path: str
    tempo: float                 # estimated BPM
    beat_times: List[float]      # seconds, one per detected beat
    duration: float              # total track length, seconds

    @property
    def beat_period(self) -> float:
        """Average seconds between beats (60 / tempo)."""
        return 60.0 / self.tempo if self.tempo > 0 else 0.5


def detect_beats(audio_path: str, regularize: bool = True) -> BeatInfo:
    """Load an audio file and detect its tempo and beat timestamps.

    Args:
        regularize: snap beats onto a perfectly steady grid at the estimated
            tempo, anchored on the first detected beat (see regularize_beats).
            Good default for club/dance-pop tracks with a locked, constant
            BPM; disable for anything with tempo changes (live band, rubato).
    """
    logger.info(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo = float(np.atleast_1d(tempo)[0])

    logger.info(f"Detected tempo={tempo:.1f} BPM, {len(beat_times)} beats, duration={duration:.1f}s")
    info = BeatInfo(audio_path=audio_path, tempo=tempo, beat_times=beat_times, duration=duration)

    if regularize:
        info = regularize_beats(info)
        logger.info(f"Regularized to a steady {info.tempo:.1f} BPM grid — {len(info.beat_times)} beats")

    return info


def regularize_beats(beat_info: BeatInfo) -> BeatInfo:
    """Snap detected beats onto a perfectly steady grid at the estimated tempo,
    anchored on the first detected beat. librosa's adaptive tracker can drift
    or skip a beat in busy/complex passages; a locked-tempo track doesn't need
    that adaptivity and a fixed grid keeps every downstream cue exactly on time.
    """
    if not beat_info.beat_times:
        return beat_info
    period = beat_info.beat_period
    start = beat_info.beat_times[0]
    n_beats = int((beat_info.duration - start) / period) + 1
    grid = [start + i * period for i in range(n_beats) if start + i * period < beat_info.duration]
    return replace(beat_info, beat_times=grid)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python3 beat_detection.py <audio_file>")
        sys.exit(1)
    info = detect_beats(sys.argv[1])
    print(f"\nTempo: {info.tempo:.1f} BPM")
    print(f"Duration: {info.duration:.1f}s")
    print(f"Beats: {len(info.beat_times)}")
    print(f"First 10 beat times: {[round(t, 2) for t in info.beat_times[:10]]}")
