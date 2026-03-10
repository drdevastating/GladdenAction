
from __future__ import annotations

import logging
import os
import sys
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
    agent    = Agent(registry=registry, executor=executor, api_key=api_key)
    logger.info("Agent ready — tools: %s", registry.list_names())
    return agent


agent = _build_agent()


# ── FastAPI app ──────────────────────────────────────────────────────────── #

app = FastAPI(title="Gladden API", version="2.0.0")

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

class VoiceListenRequest(BaseModel):
    model_size:   str   = "base"   # tiny | base | small
    max_duration: float = 10.0

class VoiceListenResponse(BaseModel):
    success:    bool
    transcript: str | None = None
    output:     Any        = None
    error:      str | None = None
    events:     list[dict] = []

class VoiceTranscribeResponse(BaseModel):
    success:    bool
    transcript: str | None = None
    error:      str | None = None

class VoiceCheckResponse(BaseModel):
    sounddevice:    bool
    faster_whisper: bool
    numpy:          bool
    ready:          bool


# ── Shared execution helper ──────────────────────────────────────────────── #

def _run_instruction(instruction: str) -> tuple[Any, str | None, list[dict]]:
    """
    Run instruction through the agent and return (output, error, events).
    Extracted so both /execute and /voice/listen can share it.
    """
    collected_events: list[dict] = []

    def _collect(event: dict) -> None:
        collected_events.append(event)

    original = agent._executor.execute

    def _patched(tool_name: str, **kwargs):
        return original(tool_name, event_callback=_collect, **kwargs)

    agent._executor.execute = _patched

    try:
        result = agent.run(instruction)
    finally:
        agent._executor.execute = original

    return result.output, result.error, collected_events, result.success


# ── Original routes ──────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {"status": "ok", "tools": agent._registry.list_names()}


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: InstructionRequest):
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must not be empty.")

    output, error, events, success = _run_instruction(req.instruction)

    return ExecuteResponse(
        success=success,
        output=output,
        error=error,
        events=events,
    )


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


@app.post("/voice/transcribe", response_model=VoiceTranscribeResponse)
def voice_transcribe(req: VoiceListenRequest = VoiceListenRequest()):
    """
    Record from microphone and return the transcript only (no agent execution).
    Use this to preview what was heard before deciding to execute.
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


@app.post("/voice/listen", response_model=VoiceListenResponse)
def voice_listen(req: VoiceListenRequest = VoiceListenRequest()):
    """
    Full pipeline: Record → Transcribe → Execute agent.

    Returns both the transcript AND the agent execution result so the
    frontend can display what was heard and what was done.
    """
    try:
        from voice_input import transcribe_from_mic
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Voice module unavailable: {exc}. Install: pip install faster-whisper sounddevice",
        )

    # Step 1: Transcribe
    try:
        transcript = transcribe_from_mic(
            model_size=req.model_size,
            max_duration=req.max_duration,
        )
    except RuntimeError as exc:
        return VoiceListenResponse(success=False, error=str(exc))
    except Exception as exc:
        logger.exception("voice_listen transcription failed")
        return VoiceListenResponse(success=False, error=f"Transcription error: {exc}")

    if not transcript:
        return VoiceListenResponse(
            success=False,
            error="No speech detected. Please speak clearly and try again.",
        )

    logger.info("Voice transcript: %r — executing agent…", transcript)

    # Step 2: Execute
    try:
        output, error, events, success = _run_instruction(transcript)
    except Exception as exc:
        logger.exception("voice_listen agent execution failed")
        return VoiceListenResponse(
            success=False,
            transcript=transcript,
            error=f"Agent execution error: {exc}",
        )

    return VoiceListenResponse(
        success=success,
        transcript=transcript,
        output=output,
        error=error,
        events=events,
    )