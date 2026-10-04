#!/usr/bin/env python3
"""
Real-time STT Pipeline using Faster Whisper + webrtcvad
Stops recording immediately after speech ends — minimizes latency.
"""

import os
import sys
import tty
import termios
import select
import threading
import numpy as np
import sounddevice as sd
import logging
import collections
import time
from typing import Optional

import webrtcvad
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSTTPipeline:
    """
    Low-latency STT pipeline:
    - webrtcvad stops recording as soon as speech ends (~0.8s silence)
    - beam_size=1 for fast Whisper inference
    - int16 recording (required by webrtcvad), converted to float32 for Whisper
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30                          # webrtcvad supports 10/20/30ms
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
    TRAIL_FRAMES = 20                      # 600ms sliding window for silence detection

    def __init__(
        self,
        model_name: str = "base",
        language: str = "ko",
        vad_aggressiveness: int = 3,       # 0=lenient … 3=strict (3 avoids bg-noise false positives)
    ):
        self.language = language

        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 ship their .so files
        # under site-packages instead of a system CUDA install. Setting
        # LD_LIBRARY_PATH alone isn't enough -- ctranslate2 only dlopens cuBLAS
        # lazily on the first real inference call (not at model construction),
        # and that internal dlopen doesn't reliably pick up a path change made
        # from Python. Preloading the .so files directly via ctypes registers
        # them in the process so ctranslate2's own dlopen-by-soname finds the
        # already-loaded library instead of searching paths.
        import ctypes
        import nvidia.cublas, nvidia.cudnn
        cublas_dir = os.path.join(list(nvidia.cublas.__path__)[0], "lib")
        cudnn_dir = os.path.join(list(nvidia.cudnn.__path__)[0], "lib")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [cublas_dir, cudnn_dir, os.environ.get("LD_LIBRARY_PATH", "")]
        )
        for fname in ("libcublasLt.so.12", "libcublas.so.12"):
            ctypes.CDLL(os.path.join(cublas_dir, fname), mode=ctypes.RTLD_GLOBAL)
        for fname in ("libcudnn.so.9",):
            ctypes.CDLL(os.path.join(cudnn_dir, fname), mode=ctypes.RTLD_GLOBAL)

        logger.info(f"Loading Whisper model: {model_name}")
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device="cuda",
            compute_type="float16",
        )
        logger.info("STT pipeline ready")

    def listen_and_transcribe(self, timeout: float = 10.0) -> Optional[str]:
        """
        Record until speech ends, then transcribe.
        Returns transcribed text or None if no speech detected.
        """
        logger.info(f"Starting to listen (timeout: {timeout}s)")

        recorded_frames = []
        # Pre-trigger: ring buffer holds recent frames for pre-roll
        PRE_ROLL = 10
        ring_buffer = collections.deque(maxlen=PRE_ROLL)
        triggered = False           # True = we are in a speech segment
        # Adaptive energy tracking: stop when energy drops far below peak speech level
        peak_energy = 0.0
        silence_score = 0           # Weighted counter: +2 on silence, -1 on speech
        SILENCE_SCORE_THRESHOLD = 60  # ~1.5s net silence to stop (at 30ms frames)
        start_time = time.time()

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.FRAME_SIZE,
            ) as stream:
                logger.info("Recording audio...")

                while (time.time() - start_time) < timeout:
                    frame_int16, _ = stream.read(self.FRAME_SIZE)
                    frame_bytes = frame_int16.tobytes()

                    try:
                        is_speech = self.vad.is_speech(frame_bytes, self.SAMPLE_RATE)
                    except Exception:
                        is_speech = False

                    # RMS energy of this frame
                    rms = float(np.sqrt(np.mean(frame_int16.astype(np.float32) ** 2)))

                    if not triggered:
                        ring_buffer.append((frame_int16.copy(), is_speech))
                        voiced = sum(1 for _, s in ring_buffer if s)
                        # Start capturing when >50% of pre-roll buffer is voiced
                        if voiced > 0.5 * len(ring_buffer):
                            triggered = True
                            silence_score = 0
                            peak_energy = 0.0
                            # Include pre-roll frames so we don't cut the beginning
                            recorded_frames.extend(f for f, _ in ring_buffer)
                            ring_buffer.clear()
                    else:
                        recorded_frames.append(frame_int16.copy())

                        # Track peak energy during speech
                        if rms > peak_energy:
                            peak_energy = rms

                        # Adaptive silence threshold: 8% of peak speech energy
                        silence_threshold = peak_energy * 0.08

                        # Weighted counter: silence frames accumulate, speech frames decay
                        if peak_energy > 0 and rms < silence_threshold:
                            silence_score += 2   # rapid accumulation in silence
                        elif is_speech:
                            silence_score = max(0, silence_score - 1)  # slow decay on speech
                        # else: neither speech nor silence → hold current score

                        if silence_score >= SILENCE_SCORE_THRESHOLD:
                            logger.info("Speech ended — stopping recording")
                            break

        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
            return None

        if not recorded_frames:
            logger.warning("No speech detected")
            return None

        audio_data = (
            np.concatenate(recorded_frames)
            .flatten()
            .astype(np.float32) / 32768.0
        )
        logger.info(f"Captured {len(audio_data)} samples ({len(audio_data)/self.SAMPLE_RATE:.1f}s)")

        logger.info("Transcribing audio...")
        segments, _ = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=1,
            best_of=1,
            vad_filter=True,               # Whisper내부 VAD로 무음 제거
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"Transcription: {text}")
        return text if text else None

    def listen_push_to_talk(self, key: str = 'q') -> Optional[str]:
        """
        Push-to-talk mode: press `key` to start, press again to stop.
        No VAD needed — user controls the exact recording window.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        def wait_key():
            """Block until `key` is pressed (raw terminal)."""
            try:
                tty.setcbreak(fd)   # per-char input, Ctrl+C still works
                while True:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        if ch.lower() == key.lower():
                            return
            finally:
                pass  # caller restores settings

        try:
            print(f"\n⌨  [{key.upper()}] 키를 눌러 녹음 시작...", flush=True)
            tty.setcbreak(fd)
            # Wait for first keypress
            while True:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == key.lower():
                        break

            print(f"🎤 녹음 중... [{key.upper()}] 키를 눌러 중지", flush=True)

            recorded_frames = []
            stop_flag = threading.Event()

            def key_watcher():
                while not stop_flag.is_set():
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        if ch.lower() == key.lower():
                            stop_flag.set()
                            return

            watcher = threading.Thread(target=key_watcher, daemon=True)
            watcher.start()

            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.FRAME_SIZE,
            ) as stream:
                while not stop_flag.is_set():
                    frame_int16, _ = stream.read(self.FRAME_SIZE)
                    recorded_frames.append(frame_int16.copy())

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if not recorded_frames:
            logger.warning("No audio captured")
            return None

        audio_data = (
            np.concatenate(recorded_frames)
            .flatten()
            .astype(np.float32) / 32768.0
        )
        logger.info(f"PTT captured {len(audio_data)/self.SAMPLE_RATE:.1f}s")
        logger.info("Transcribing audio...")

        segments, _ = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=1,
            best_of=1,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"Transcription: {text}")
        return text if text else None

    def cleanup(self):
        pass


def test_stt_pipeline():
    pipeline = RealtimeSTTPipeline(model_name="base", language="ko")
    print("🎤 말씀하세요...")
    result = pipeline.listen_and_transcribe(timeout=10.0)
    if result:
        print(f"✅ 인식: {result}")
    else:
        print("❌ 음성 없음")


if __name__ == "__main__":
    test_stt_pipeline()
