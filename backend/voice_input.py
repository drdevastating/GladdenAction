"""
voice_input.py

Offline speech recognition using faster-whisper.

Changes from the original
--------------------------
* Recording is now controlled by a threading.Event flag (`_stop_recording`)
  instead of blocking on terminal input().  This lets the FastAPI server
  call `request_stop()` when the frontend hits the /voice/stop endpoint,
  so the user can click the mic button (or press Enter in the Electron
  window) to end recording without needing a terminal.

* `transcribe_from_mic()` polls the stop-event every CHUNK_SIZE samples
  instead of spawning an input()-thread that is meaningless in a
  headless server context.

Public API
----------
    transcribe_from_mic(model_size, max_duration) -> str | None
    request_stop()   — signal the running recording to stop
    check_dependencies() -> dict[str, bool]
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE     = 16_000   # Whisper expects 16 kHz mono
MAX_DURATION    = 60       # hard cap in seconds
MIN_SPEECH_SECS = 0.4      # ignore recordings shorter than this
CHUNK_SIZE      = 1024     # samples per chunk (~64 ms at 16 kHz)

_model_lock  = threading.Lock()
_whisper_model = None

# ── Global stop-event ────────────────────────────────────────────────────
# Set by request_stop(); cleared at the start of each new recording.
_stop_event = threading.Event()


def request_stop() -> None:
    """
    Signal the currently-running transcribe_from_mic() call to stop
    recording and proceed to transcription.  Safe to call even when no
    recording is in progress (no-op).
    """
    _stop_event.set()


def _get_model(model_size: str = "base"):
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            logger.info(
                "Loading Whisper model '%s' (first-time download may take a moment)…",
                model_size,
            )
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Whisper model ready.")
    return _whisper_model


def transcribe_from_mic(
    model_size: str = "base",
    max_duration: float = MAX_DURATION,
) -> Optional[str]:
    """
    Record from the default microphone until either:
      • request_stop() is called  (frontend clicked the mic button / pressed Enter)
      • max_duration seconds elapse  (hard safety cap)

    Returns the transcribed string, or None if nothing was recorded /
    transcription produced no output.
    """
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

    # Clear any leftover stop signal from a previous call
    _stop_event.clear()

    logger.info("Recording started (max %.0fs) — waiting for stop signal…", max_duration)

    chunks: list[np.ndarray] = []
    max_chunks = int(max_duration * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
    ) as stream:
        for _ in range(max_chunks):
            if _stop_event.is_set():
                logger.info("Stop signal received — ending recording.")
                break

            audio_chunk, _ = stream.read(CHUNK_SIZE)
            chunks.append(audio_chunk.flatten())

    if not chunks:
        return None

    audio    = np.concatenate(chunks)
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
        vad_filter=True,
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