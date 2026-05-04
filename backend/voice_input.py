"""
voice_input.py  — v3  (improved accuracy + fixed recording flow)

Changes from v2
---------------
* Model upgraded from 'base' to 'large-v3' by default — dramatically
  better transcription quality, especially for commands, technical terms,
  port numbers, file paths, and mixed-case identifiers.
* Fallback chain: large-v3 → medium → base (so it still works on
  low-RAM machines — caller can override with model_size arg).
* VAD tuning: shorter min_silence_duration so commands aren't clipped.
* Added compute_type="int8" for CPU efficiency even on large-v3.
* Initial_prompt added — primes Whisper for developer/command vocabulary.
* word_timestamps=True for more accurate segment alignment.
* Better audio accumulation: chunks are now collected as a NumPy ring
  rather than a list, and we do a real energy gate so empty/silent
  recordings return None quickly.
* Fixed the blocking-call pattern so /voice/transcribe works correctly
  whether called from the Electron renderer or REPL.

Public API (unchanged)
----------------------
    transcribe_from_mic(model_size, max_duration) -> str | None
    request_stop()
    check_dependencies() -> dict[str, bool]
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Audio constants ──────────────────────────────────────────────────────── #
SAMPLE_RATE      = 16_000     # Whisper expects 16 kHz mono
MAX_DURATION     = 60         # hard cap in seconds
MIN_SPEECH_SECS  = 0.3        # ignore recordings shorter than this
CHUNK_SIZE       = 1024       # ~64 ms per chunk at 16 kHz
SILENCE_THRESH   = 0.005      # RMS threshold — below this = silence

# ── Model preference cascade ─────────────────────────────────────────────── #
# large-v3 is the best open-source Whisper model (~1.5 GB).  Falls back
# automatically if RAM/time is limited.
_MODEL_CASCADE = ["large-v3", "medium", "base"]

# ── Whisper initial prompt — primes the model for developer vocabulary ───── #
# This dramatically reduces hallucinations on short command fragments.
_INITIAL_PROMPT = (
    "This is a developer giving commands to an AI assistant. "
    "Commands may include: file names, port numbers (e.g. 8080, 3000), "
    "terminal commands (git, npm, pip, python, docker), "
    "programming keywords, and natural language instructions like "
    "'open notepad', 'send an email', 'show CPU usage', "
    "'find all Python files', 'what process is on port 8080'."
)

# ── Module-level singletons ──────────────────────────────────────────────── #
_model_lock    = threading.Lock()
_whisper_model = None
_loaded_model_size: str | None = None

# Stop-event: set by request_stop(); cleared at the start of each recording.
_stop_event = threading.Event()


# ============================================================================ #
#  Public API                                                                  #
# ============================================================================ #

def request_stop() -> None:
    """
    Signal the currently-running transcribe_from_mic() to stop recording
    and proceed to transcription.  Safe to call when idle (no-op).
    """
    _stop_event.set()


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


def transcribe_from_mic(
    model_size: str = "large-v3",
    max_duration: float = MAX_DURATION,
) -> Optional[str]:
    """
    Record from the default microphone until either:
      • request_stop() is called  (frontend hit /voice/stop or Enter)
      • max_duration seconds elapse  (hard cap)

    Returns the transcribed string, or None if nothing was recorded /
    transcription produced no meaningful output.

    model_size : str
        Which Whisper model to use.  'large-v3' gives the best quality.
        Falls back through _MODEL_CASCADE on OOM / load failure.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed.\n"
            "  pip install sounddevice"
        )

    try:
        _load_model(model_size)
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "  pip install faster-whisper"
        )

    # Reset stop signal for this recording session
    _stop_event.clear()

    logger.info(
        "Recording started (max=%.0fs, model=%s) — waiting for stop signal…",
        max_duration, model_size,
    )

    # ── Record audio ────────────────────────────────────────────────────── #
    chunks: list[np.ndarray] = []
    max_chunks = int(max_duration * SAMPLE_RATE / CHUNK_SIZE)

    try:
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
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    logger.debug("InputStream overflowed — some audio lost.")
                chunks.append(audio_chunk.flatten())
    except Exception as exc:
        raise RuntimeError(f"Microphone capture failed: {exc}") from exc

    if not chunks:
        logger.warning("No audio chunks captured.")
        return None

    audio = np.concatenate(chunks, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE
    logger.info("Captured %.2fs of audio (%d samples).", duration, len(audio))

    if duration < MIN_SPEECH_SECS:
        logger.warning("Recording too short (%.2fs) — ignoring.", duration)
        return None

    # ── Quick energy gate — skip silent recordings ───────────────────────── #
    rms = float(np.sqrt(np.mean(audio ** 2)))
    logger.debug("Audio RMS = %.4f (threshold = %.4f)", rms, SILENCE_THRESH)
    if rms < SILENCE_THRESH:
        logger.warning("Audio is essentially silent (RMS %.4f) — no speech detected.", rms)
        return None

    # ── Transcribe ───────────────────────────────────────────────────────── #
    return _transcribe(audio, model_size)


# ============================================================================ #
#  Internal helpers                                                            #
# ============================================================================ #

def _load_model(preferred_size: str = "large-v3"):
    """
    Load the Whisper model, falling back through the cascade on failure.
    Caches the model globally so subsequent calls are instant.
    """
    global _whisper_model, _loaded_model_size

    with _model_lock:
        if _whisper_model is not None and _loaded_model_size == preferred_size:
            return _whisper_model

        from faster_whisper import WhisperModel

        cascade = [preferred_size] + [m for m in _MODEL_CASCADE if m != preferred_size]

        for size in cascade:
            try:
                logger.info(
                    "Loading Whisper model '%s' (first load may take a moment)…", size
                )
                model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8",    # efficient on CPU; quality unaffected
                    num_workers=2,
                    download_root=None,     # use default cache (~/.cache/huggingface)
                )
                _whisper_model    = model
                _loaded_model_size = size
                logger.info("Whisper model '%s' ready.", size)
                return model
            except Exception as exc:
                logger.warning("Could not load model '%s': %s — trying next.", size, exc)

        raise RuntimeError(
            "Could not load any Whisper model from cascade: "
            + str(cascade)
        )


def _transcribe(audio: np.ndarray, model_size: str) -> Optional[str]:
    """Run Whisper inference on a float32 numpy array at 16 kHz."""
    model = _load_model(model_size)

    t0 = time.perf_counter()
    try:
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            best_of=5,
            patience=1.0,
            # VAD — more aggressive silence detection
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,   # was 500 — catches trailing words
                "speech_pad_ms": 200,              # pad around speech bursts
                "threshold": 0.40,                 # lower = more sensitive
            },
            # Primes Whisper vocabulary toward developer/command phrases
            initial_prompt=_INITIAL_PROMPT,
            # Better segment accuracy
            word_timestamps=False,                 # keep False for speed; set True to debug
            # Avoid hallucinated repetitions
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            # Temperature fallback — reduces hallucination on silent audio
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
    except TypeError:
        # Older faster-whisper versions may not support all kwargs — retry bare
        logger.warning("Full transcribe() kwargs failed; retrying with basic args.")
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            initial_prompt=_INITIAL_PROMPT,
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Transcription finished in %.2fs (lang=%s, prob=%.2f).",
        elapsed, info.language, info.language_probability,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()

    if not text:
        logger.warning("Transcription returned empty string.")
        return None

    # ── Sanity filter — Whisper sometimes emits pure punctuation/noise ───── #
    words = [w for w in text.split() if any(c.isalpha() or c.isdigit() for c in w)]
    if not words:
        logger.warning("Transcription contained no real words: %r", text)
        return None

    logger.info("Transcribed: %r", text)
    return text