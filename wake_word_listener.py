#!/usr/bin/env python3
"""
Wake-word listener — Go2's own onboard mic, always on.

Pipeline: robot mic (WebRTC audio, sendrecv track — see
Test/unitree_webrtc_connect/unitree_webrtc_connect/webrtc_audio.py) -> resample
to 16kHz mono (PyAV, exact rate/layout the robot sends is irrelevant, the
resampler reads it off the frame itself) -> webrtcvad endpointing (same
approach as realtime_stt_pipeline.py's push-to-talk-free mode) -> a small
Whisper pass just to check for the wake word -> once matched, the *next*
utterance (ended by silence, same endpointing) is treated as the actual
command and run through the dashboard's normal Whisper "medium" + LLM +
auto-execute pipeline (PipelineWorker._llm_and_execute).

Kept as its own module (not folded into dashboard_server.py) since it's a
self-contained state machine over a raw audio stream, not FastAPI plumbing.
"""

import collections
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)


class WakeWordListener:
    SAMPLE_RATE = 16000
    FRAME_MS = 30                                     # webrtcvad: 10/20/30ms only
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)   # 480 samples
    PRE_ROLL = 10                                     # ring buffer frames before trigger
    SILENCE_SCORE_THRESHOLD = 60                      # ~1.5s net silence -> utterance end
    MAX_UTTERANCE_S = 12.0                            # safety cap on a runaway trigger

    # STT-mishearing-tolerant wake phrase variants -- same spirit as the
    # phonetic action-name variants already handled in voice_control_go2.py's
    # KOREAN_SYSTEM_PROMPT (e.g. "트롯날" accepted for trot_run). Matched as a
    # substring against the tiny model's (space-stripped) transcript.
    WAKE_PHRASES = ["윤재야", "윤재", "윤재아", "윤제야", "운재야"]

    def __init__(self, camera, on_command: Callable[[np.ndarray], None]):
        """
        camera: the dashboard's CameraStreamer -- reused for its already-open
            WebRTC connection (same pattern as TelemetryStreamer) instead of
            opening a second one.
        on_command: called with a 16kHz mono float32 numpy array -- the audio
            captured *after* the wake word fired, up to the following
            silence. Runs on a background thread; the callee is responsible
            for its own threading (PipelineWorker.process_audio_array already
            hands off to a worker thread).
        """
        self._camera = camera
        self._on_command = on_command
        # 0=lenient..3=strict. Started at 2, but real hardware showed webrtcvad
        # classifying the robot's own servo/fan noise as speech continuously
        # (utterances ran to the MAX_UTTERANCE_S cap with an empty Whisper
        # transcript -- i.e. captured audio had no real speech in it). Level 3
        # is specifically the most aggressive about rejecting non-speech noise.
        self._vad = webrtcvad.Vad(3)
        self._resampler = None
        self._wake_whisper = None

        self._state = "wake"  # "wake" | "command"
        self._carry = np.zeros(0, dtype=np.int16)
        self._ring = collections.deque(maxlen=self.PRE_ROLL)
        self._recorded = []
        self._triggered = False
        self._silence_score = 0
        self._utterance_start = 0.0

        self._enabled = False
        self._got_first_frame = False

    def enable(self, enabled: bool):
        self._enabled = enabled
        logger.info(f"Wake-word: {'enabled' if enabled else 'disabled'}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self):
        threading.Thread(target=self._setup, daemon=True, name="wake-word-setup").start()

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup(self):
        try:
            if not self._camera.webrtc_ready.wait(timeout=20):
                logger.warning("Wake-word: camera WebRTC not ready — mic listener disabled")
                return
            conn, loop = self._camera.webrtc_conn, self._camera.webrtc_loop
            if conn is None or loop is None:
                logger.warning("Wake-word: no WebRTC connection — mic listener disabled")
                return

            import av
            self._resampler = av.AudioResampler(format="s16", layout="mono", rate=self.SAMPLE_RATE)

            self._load_wake_model()

            # switchAudioChannel/add_track_callback touch the shared
            # connection's datachannel/pc -- must run on its own event-loop
            # thread, not this one (see TelemetryStreamer for the same rule).
            loop.call_soon_threadsafe(conn.audio.switchAudioChannel, True)
            loop.call_soon_threadsafe(conn.audio.add_track_callback, self._on_frame)
            self._enabled = True
            logger.info("Wake-word: listening on robot mic for '윤재야'")
        except Exception as e:
            logger.error(f"Wake-word setup failed: {e}", exc_info=True)

    def _load_wake_model(self):
        # A second, tiny model alongside the dashboard's medium one is cheap
        # (~75MB) and keeps the continuous wake-check pass fast -- running
        # the medium model on every utterance regardless of whether it's
        # actually the wake word would waste GPU time for no benefit, since
        # wake-word matching only needs "close enough", not full accuracy.
        import os, ctypes
        import nvidia.cublas, nvidia.cudnn
        os.environ.setdefault("HF_HOME", "/media/hong/data/hf_cache")
        cublas_dir = os.path.join(list(nvidia.cublas.__path__)[0], "lib")
        cudnn_dir = os.path.join(list(nvidia.cudnn.__path__)[0], "lib")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [cublas_dir, cudnn_dir, os.environ.get("LD_LIBRARY_PATH", "")]
        )
        for fname in ("libcublasLt.so.12", "libcublas.so.12"):
            ctypes.CDLL(os.path.join(cublas_dir, fname), mode=ctypes.RTLD_GLOBAL)
        for fname in ("libcudnn.so.9",):
            ctypes.CDLL(os.path.join(cudnn_dir, fname), mode=ctypes.RTLD_GLOBAL)

        from faster_whisper import WhisperModel
        logger.info("Wake-word: loading Whisper tiny (GPU)...")
        self._wake_whisper = WhisperModel("tiny", device="cuda", compute_type="float16")

    # ── Audio ingestion ──────────────────────────────────────────────────

    async def _on_frame(self, frame):
        if not self._enabled or self._resampler is None:
            return
        try:
            for resampled in self._resampler.resample(frame):
                mono = resampled.to_ndarray().reshape(-1)
                self._ingest(mono)
        except Exception as e:
            logger.warning(f"Wake-word: frame error: {e}")

    def _ingest(self, mono_i16: np.ndarray):
        """Buffer arbitrary-length resampled chunks into fixed VAD-frame blocks."""
        if not self._got_first_frame:
            self._got_first_frame = True
            logger.info("Wake-word: first audio frame received from robot mic — stream is live")
        buf = np.concatenate([self._carry, mono_i16])
        n = len(buf) // self.FRAME_SIZE
        for i in range(n):
            self._process_frame(buf[i * self.FRAME_SIZE:(i + 1) * self.FRAME_SIZE])
        self._carry = buf[n * self.FRAME_SIZE:]

    def _process_frame(self, frame_i16: np.ndarray):
        """VAD-driven endpointing. realtime_stt_pipeline.py's original approach
        (still used for the laptop-mic path) gates silence on RMS relative to
        the utterance's peak energy, tuned for a quiet room. Next to the
        robot's own body that assumption breaks: constant servo/fan noise
        sits close enough to peak speech energy that the adaptive threshold
        is never crossed, so an utterance never ends on silence and instead
        always runs out the clock at MAX_UTTERANCE_S (confirmed: a real
        capture ran exactly 12.2s, i.e. hit the cap, not a detected pause).
        webrtcvad's is_speech() classifies speech-vs-not per frame using
        spectral shape, not raw amplitude, so it holds up much better against
        a steady mechanical noise floor -- use it as the primary signal."""
        frame_bytes = frame_i16.astype(np.int16).tobytes()
        try:
            is_speech = self._vad.is_speech(frame_bytes, self.SAMPLE_RATE)
        except Exception:
            is_speech = False

        if not self._triggered:
            self._ring.append((frame_i16.copy(), is_speech))
            voiced = sum(1 for _, s in self._ring if s)
            if voiced > 0.5 * len(self._ring):
                self._triggered = True
                self._silence_score = 0
                self._utterance_start = time.monotonic()
                self._recorded = [f for f, _ in self._ring]
                self._ring.clear()
        else:
            self._recorded.append(frame_i16.copy())
            if is_speech:
                self._silence_score = max(0, self._silence_score - 1)
            else:
                self._silence_score += 2

            timed_out = (time.monotonic() - self._utterance_start) > self.MAX_UTTERANCE_S
            if self._silence_score >= self.SILENCE_SCORE_THRESHOLD or timed_out:
                self._finish_utterance()

    def _finish_utterance(self):
        frames, self._recorded = self._recorded, []
        self._triggered = False
        self._silence_score = 0
        if not frames:
            return
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        logger.info(f"Wake-word: speech segment captured ({len(audio)/self.SAMPLE_RATE:.1f}s, state={self._state}) — transcribing...")
        threading.Thread(target=self._handle_utterance, args=(audio,), daemon=True).start()

    # ── State machine ────────────────────────────────────────────────────

    def _handle_utterance(self, audio: np.ndarray):
        try:
            if self._state == "wake":
                segs, _ = self._wake_whisper.transcribe(audio, language="ko", beam_size=1, best_of=1)
                text = "".join(s.text for s in segs).strip().replace(" ", "")
                matched = next((p for p in self.WAKE_PHRASES if p.replace(" ", "") in text), None)
                if matched:
                    remainder = text.replace(matched.replace(" ", ""), "", 1)
                    if len(remainder) >= 2:
                        # Wake word + command said in one breath (no pause in
                        # between) -- the common, natural way to call out to
                        # something ("hey X, do Y"), not two separate
                        # utterances. Run the *whole* clip through the real
                        # pipeline right away instead of discarding it as a
                        # wake-only check; the medium model + LLM will
                        # re-transcribe more accurately and simply ignore a
                        # leading "윤재야" as an address, not a command.
                        logger.info(f"Wake-word matched with command in same breath ('{text}') — running full pipeline")
                        self._state = "wake"
                        self._on_command(audio)
                    else:
                        logger.info(f"Wake-word matched ('{text}') — listening for command...")
                        self._state = "command"
                else:
                    # INFO, not DEBUG: this is the only signal that audio is
                    # actually arriving and being processed at all -- without
                    # it, "no audio reaching the robot's mic path" and "audio
                    # arriving but tiny-model transcription not matching" are
                    # indistinguishable from the logs.
                    logger.info(f"Wake-word check, no match: '{text}' (heard {len(audio)/self.SAMPLE_RATE:.1f}s)")
            else:
                self._state = "wake"
                self._on_command(audio)
        except Exception as e:
            logger.error(f"Wake-word utterance handling failed: {e}", exc_info=True)
            self._state = "wake"
