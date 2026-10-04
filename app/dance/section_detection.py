#!/usr/bin/env python3
"""
Auto Section Detection
Any audio file -> structural segments (intro/verse/chorus-like blocks), purely
from signal features (MFCC timbre + chroma harmony + RMS energy). No manual
per-song boundaries, no lyrics/content analysis -- same technique used to find
Gangnam Style's breakdown by hand, generalized and automated.
"""

import logging
from dataclasses import dataclass
from typing import List

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    energy: float       # 0..1, normalized within this track (1 = this track's most energetic segment)
    beat_count: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_segments(audio_path: str, beat_info, n_segments: int = 8) -> List[Segment]:
    """Cluster the track into n_segments structural blocks and score each
    one's relative energy, so downstream choreography can tell calm sections
    from climactic ones without any manual timing."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = 512

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    features = np.vstack([mfcc, chroma, rms])

    bound_frames = librosa.segment.agglomerative(features, n_segments)
    bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    boundaries = sorted(set([0.0] + [float(t) for t in bound_times] + [duration]))

    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    segments: List[Segment] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask = (rms_times >= start) & (rms_times < end)
        seg_energy = float(np.mean(rms[mask])) if mask.any() else 0.0
        beat_count = sum(1 for t in beat_info.beat_times if start <= t < end)
        segments.append(Segment(start=start, end=end, energy=seg_energy, beat_count=beat_count))

    energies = [s.energy for s in segments]
    lo, hi = min(energies), max(energies)
    for s in segments:
        s.energy = (s.energy - lo) / (hi - lo) if hi > lo else 0.5

    logger.info(f"Detected {len(segments)} structural segments")
    for i, s in enumerate(segments):
        logger.info(f"  [{i}] {s.start:6.1f}s - {s.end:6.1f}s  energy={s.energy:.2f}  beats={s.beat_count}")

    return segments


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from beat_detection import detect_beats

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python3 section_detection.py <audio_file> [n_segments]")
        sys.exit(1)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    info = detect_beats(sys.argv[1])
    detect_segments(sys.argv[1], info, n_segments=n)
