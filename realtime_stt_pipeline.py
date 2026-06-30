#!/usr/bin/env python3
"""
Real-time STT (Speech-to-Text) Pipeline using Faster Whisper
Handles audio input, voice activity detection, and transcription
"""

import numpy as np
import sounddevice as sd
import logging
from faster_whisper import WhisperModel
from typing import Optional, Tuple
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSTTPipeline:
    """Real-time Speech-to-Text pipeline using Faster Whisper"""
    
    def __init__(
        self,
        model_name: str = "base",
        language: str = "ko",
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        energy_threshold: float = 0.02,
        pause_threshold: float = 0.5,
    ):
        """
        Initialize STT pipeline
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'vi', 'ja')
            sample_rate: Audio sample rate in Hz
            chunk_size: Audio chunk size in samples
            energy_threshold: Energy threshold for voice detection
            pause_threshold: Pause threshold in seconds to finalize audio
        """
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device="auto",  # Use GPU if available
            compute_type="default"
        )
        
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.stream = None
    
    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Queue audio data
        audio_data = indata[:, 0].copy()  # Convert to mono
        self.audio_queue.put(audio_data)
    
    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains voice
        
        Args:
            audio_chunk: Audio data array
            
        Returns:
            True if voice detected, False otherwise
        """
        # Simple energy-based voice activity detection
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > self.energy_threshold
    
    def listen_and_transcribe(self, timeout: float = 10.0) -> Optional[str]:
        """
        Listen to microphone and transcribe speech
        
        Args:
            timeout: Maximum listening duration in seconds
            
        Returns:
            Transcribed text or None if no speech detected
        """
        logger.info(f"Starting to listen (timeout: {timeout}s)")
        
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            
            self.stream.start()
            self.is_listening = True
            
            audio_frames = []
            silence_duration = 0
            has_voice = False
            start_time = time.time()
            
            logger.info("Recording audio...")
            
            while (time.time() - start_time) < timeout:
                try:
                    # Get audio chunk from queue with timeout
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check for voice activity
                if self.detect_voice_activity(audio_chunk):
                    audio_frames.append(audio_chunk)
                    silence_duration = 0
                    has_voice = True
                    logger.debug("Voice detected")
                else:
                    # If we've detected voice, continue recording through pauses
                    if has_voice:
                        audio_frames.append(audio_chunk)
                        silence_duration += (len(audio_chunk) / self.sample_rate)
                        
                        # If pause is long enough, we can stop
                        if silence_duration > self.pause_threshold:
                            logger.info("Pause detected, finalizing audio")
                            break
            
            self.stream.stop()
            self.stream.close()
            self.is_listening = False
            
            if not audio_frames:
                logger.warning("No audio frames captured")
                return None
            
            # Concatenate audio frames
            audio_data = np.concatenate(audio_frames)
            logger.info(f"Captured {len(audio_data)} audio samples")
            
            # Transcribe using Whisper
            logger.info("Transcribing audio...")
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                best_of=5,
            )
            
            # Combine segments
            transcribed_text = " ".join([segment.text for segment in segments])
            logger.info(f"Transcription: {transcribed_text}")
            
            return transcribed_text if transcribed_text else None
        
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return None
        
        finally:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
            self.is_listening = False
    
    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            logger.info("Audio stream closed")


def test_stt_pipeline():
    """Test the STT pipeline"""
    logger.info("Testing STT Pipeline")
    
    pipeline = RealtimeSTTPipeline(model_name="base", language="ko")
    
    print("🎤 Listening for speech (10 seconds)...")
    result = pipeline.listen_and_transcribe(timeout=10.0)
    
    if result:
        print(f"✅ Transcribed: {result}")
    else:
        print("❌ No speech detected")
    
    pipeline.cleanup()


if __name__ == "__main__":
    test_stt_pipeline()
