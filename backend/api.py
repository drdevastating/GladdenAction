"""
api.py  —  GladdenAction FastAPI backend  (updated: Planner integration)

Changes from previous version
------------------------------
- Imports and constructs a Planner instance alongside the Agent.
- Passes it to Agent constructor.
- _run_instruction() now also registers a callback on the Agent so that
  planning/step lifecycle events are collected into the same events list
  that the frontend already streams.
- All existing routes, schemas, and voice endpoints are unchanged.

Voice flow (two-step design)
---------------------------------
1. Frontend POSTs /voice/start  → server begins recording in a background
   thread and returns immediately with {"recording": true}.
2. User speaks.
3. Frontend POSTs /voice/stop   → server signals the recording thread to
   stop, waits for transcription to finish, returns the transcript.
4. Frontend populates the input box with the transcript.
5. User reviews / edits, then presses Enter.
6. Frontend POSTs /execute      → agent runs as normal.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.agent import Agent
from agent.planner import Planner
from core.tools import FileCreationTool
from core.tools.registry import ToolRegistry
from core.tools.system_control_tool import SystemControlTool
from core.tools.ui_automation_tool import UIAutomationTool
from execution.executor import ToolExecutor


# ── Build agent once at startup ─────────────────────────────────────────── #

def _build_agent() -> Agent:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY not set — add it to backend/.env")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(UIAutomationTool())
    registry.register(FileCreationTool())
    registry.register(SystemControlTool())

    executor = ToolExecutor(registry)
    planner  = Planner(api_key=api_key)
    agent    = Agent(registry=registry, executor=executor, api_key=api_key, planner=planner)
    logger.info("Agent ready — tools: %s  planner: enabled", registry.list_names())
    return agent


agent = _build_agent()


# ── Voice recording state ────────────────────────────────────────────────── #

_recording_thread: threading.Thread | None = None
_recording_result: dict = {}
_recording_done   = threading.Event()
_recording_lock   = threading.Lock()


# ── FastAPI app ──────────────────────────────────────────────────────────── #

app = FastAPI(title="Gladden API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────── #

class InstructionRequest(BaseModel):
    instruction: str

class ExecuteResponse(BaseModel):
    success: bool
    output:  Any        = None
    error:   str | None = None
    events:  list[dict] = []

class VoiceStartRequest(BaseModel):
    model_size:   str   = "base"
    max_duration: float = 30.0

class VoiceStartResponse(BaseModel):
    recording: bool
    error:     str | None = None

class VoiceStopResponse(BaseModel):
    success:    bool
    transcript: str | None = None
    error:      str | None = None

class VoiceCheckResponse(BaseModel):
    sounddevice:    bool
    faster_whisper: bool
    numpy:          bool
    ready:          bool

class VoiceListenRequest(BaseModel):
    model_size:   str   = "base"
    max_duration: float = 10.0

class VoiceTranscribeResponse(BaseModel):
    success:    bool
    transcript: str | None = None
    error:      str | None = None


# ── Shared execution helper ──────────────────────────────────────────────── #

def _run_instruction(instruction: str) -> tuple[Any, str | None, list[dict], bool]:
    collected_events: list[dict] = []

    def _collect(event: dict) -> None:
        collected_events.append(event)

    # Patch executor so tool-execution events are collected
    original_execute = agent._executor.execute

    def _patched_execute(tool_name: str, **kwargs: Any):
        kwargs.pop("event_callback", None)
        return original_execute(tool_name, event_callback=_collect, **kwargs)

    agent._executor.execute = _patched_execute

    # Register callback on Agent so planning/step events are also collected
    agent.register_event_callback(_collect)

    try:
        result = agent.run(instruction)
    finally:
        agent._executor.execute = original_execute
        agent.clear_event_callback()

    return result.output, result.error, collected_events, result.success


# ── Core routes ──────────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "tools":   agent._registry.list_names(),
        "planner": agent._planner is not None,
    }


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: InstructionRequest):
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must not be empty.")

    output, error, events, success = _run_instruction(req.instruction)

    return ExecuteResponse(success=success, output=output, error=error, events=events)


# ── Voice routes ─────────────────────────────────────────────────────────── #

@app.get("/voice/check", response_model=VoiceCheckResponse)
def voice_check():
    """Return which voice dependencies are installed."""
    try:
        from voice_input import check_dependencies
        deps = check_dependencies()
    except Exception as exc:
        logger.warning("voice_input module not available: %s", exc)
        deps = {"sounddevice": False, "faster_whisper": False, "numpy": False}

    return VoiceCheckResponse(
        sounddevice=deps.get("sounddevice", False),
        faster_whisper=deps.get("faster_whisper", False),
        numpy=deps.get("numpy", False),
        ready=all(deps.values()),
    )


@app.post("/voice/start", response_model=VoiceStartResponse)
def voice_start(req: VoiceStartRequest = VoiceStartRequest()):
    global _recording_thread, _recording_result, _recording_done

    try:
        from voice_input import transcribe_from_mic, request_stop  # noqa: F401
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Voice module unavailable: {exc}. Install: pip install faster-whisper sounddevice",
        )

    with _recording_lock:
        if _recording_thread is not None and _recording_thread.is_alive():
            return VoiceStartResponse(recording=False, error="Already recording — call /voice/stop first.")

        _recording_result = {}
        _recording_done.clear()

        def _run():
            try:
                from voice_input import transcribe_from_mic
                transcript = transcribe_from_mic(
                    model_size=req.model_size,
                    max_duration=req.max_duration,
                )
                _recording_result["transcript"] = transcript
                _recording_result["success"]    = bool(transcript)
                if not transcript:
                    _recording_result["error"] = "No speech detected. Please speak clearly and try again."
            except Exception as exc:
                logger.exception("Recording thread failed")
                _recording_result["success"]    = False
                _recording_result["transcript"] = None
                _recording_result["error"]      = str(exc)
            finally:
                _recording_done.set()

        _recording_thread = threading.Thread(target=_run, daemon=True)
        _recording_thread.start()

    logger.info("Voice recording started (max %.0fs).", req.max_duration)
    return VoiceStartResponse(recording=True)


@app.post("/voice/stop", response_model=VoiceStopResponse)
def voice_stop():
    global _recording_thread

    try:
        from voice_input import request_stop
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    with _recording_lock:
        if _recording_thread is None or not _recording_thread.is_alive():
            if not _recording_done.is_set():
                return VoiceStopResponse(success=False, error="No active recording to stop.")

    request_stop()

    finished = _recording_done.wait(timeout=60)
    if not finished:
        return VoiceStopResponse(success=False, error="Transcription timed out.")

    result = _recording_result
    return VoiceStopResponse(
        success=result.get("success", False),
        transcript=result.get("transcript"),
        error=result.get("error"),
    )


@app.post("/voice/transcribe", response_model=VoiceTranscribeResponse)
def voice_transcribe(req: VoiceListenRequest = VoiceListenRequest()):
    """
    Legacy single-shot endpoint — kept for backward compat.
    Use /voice/start + /voice/stop instead.
    """
    try:
        from voice_input import transcribe_from_mic
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Voice module unavailable: {exc}. Install: pip install faster-whisper sounddevice",
        )

    try:
        transcript = transcribe_from_mic(
            model_size=req.model_size,
            max_duration=req.max_duration,
        )
    except RuntimeError as exc:
        return VoiceTranscribeResponse(success=False, error=str(exc))
    except Exception as exc:
        logger.exception("voice_transcribe failed")
        return VoiceTranscribeResponse(success=False, error=f"Transcription error: {exc}")

    if not transcript:
        return VoiceTranscribeResponse(
            success=False,
            error="No speech detected. Please speak clearly and try again.",
        )

    return VoiceTranscribeResponse(success=True, transcript=transcript)