"""
voice to text input module
using faster-whisper for offline speech recognition, it will record audio from microphone for a 
fixed duration or until silence is detected, then transcribe the audio and return the transcript as string.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE     = 16_000   # Whisper expects 16 kHz mono
MAX_DURATION    = 15       # seconds — hard cap on recording
SILENCE_THRESH  = 0.012    # RMS below this = silence
SILENCE_SECS    = 2      # seconds of silence before auto-stop
MIN_SPEECH_SECS = 0.4      # ignore recording shorter than this
CHUNK_SIZE      = 1024     # samples per chunk

_model_lock = threading.Lock()
_whisper_model = None


def _get_model(model_size: str = "base"):
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            logger.info("Loading Whisper model '%s' (first-time download may take a moment)…", model_size)
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8", 
            )
            logger.info("Whisper model ready.")
    return _whisper_model



def transcribe_from_mic(
    model_size: str = "base",
    max_duration: float = MAX_DURATION,
) -> Optional[str]:
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install sounddevice"
        )

    try:
        _get_model(model_size)  
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    logger.info(" Recording started (max %ss)…", max_duration)

    chunks: list[np.ndarray] = []
    silence_samples = 0
    speech_samples  = 0
    silence_limit   = int(SILENCE_SECS  * SAMPLE_RATE / CHUNK_SIZE)
    speech_required = int(MIN_SPEECH_SECS * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks      = int(max_duration  * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
    ) as stream:
        for _ in range(max_chunks):
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            chunk = audio_chunk.flatten()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            chunks.append(chunk)

            if rms > SILENCE_THRESH:
                speech_samples  += 1
                silence_samples  = 0
            else:
                silence_samples += 1
                if speech_samples >= speech_required and silence_samples >= silence_limit:
                    logger.info("Silence detected — stopping recording.")
                    break

    if not chunks:
        return None

    audio = np.concatenate(chunks)
    duration = len(audio) / SAMPLE_RATE
    logger.info("Recorded %.2fs of audio — transcribing…", duration)

    if duration < MIN_SPEECH_SECS:
        logger.warning("Recording too short (%.2fs) — ignoring.", duration)
        return None

    model = _get_model(model_size)
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,      # built-in filter for clearer segments
        vad_parameters={"min_silence_duration_ms": 500},
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        logger.warning("Transcription returned empty string.")
        return None

    logger.info("Transcribed: %r", text)
    return text


def check_dependencies() -> dict[str, bool]:
    results: dict[str, bool] = {}
    for module, key in [
        ("sounddevice",    "sounddevice"),
        ("faster_whisper", "faster_whisper"),
        ("numpy",          "numpy"),
    ]:
        try:
            __import__(module)
            results[key] = True
        except ImportError:
            results[key] = False
    return results