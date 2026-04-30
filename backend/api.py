"""
api.py  —  GladdenAction FastAPI backend  (v6 — NL→Command feature)

New in v6
---------
- NLCommandTool : Convert plain English to CMD/PowerShell via Groq LLM,
                  execute through ShellTool safety whitelist.
- GET /nl_command/preview : Translate a request and return the command
                            without executing (for Electron UI preview).
- All existing routes unchanged.
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
from agent.approval_gate import ApprovalGate
from agent.planner import Planner
from core.tools import FileCreationTool
from core.tools.code_edit_tool import CodeEditTool
from core.tools.context_tool import ContextTool
from core.tools.nl_command_tool import NLCommandTool          # ← NEW
from core.tools.registry import ToolRegistry
from core.tools.shell_tool import ShellTool
from core.tools.system_control_tool import SystemControlTool
from core.tools.ui_automation_tool import UIAutomationTool
from execution.executor import ToolExecutor


# ── Build agent once at startup ─────────────────────────────────────────── #

_approval_gate = ApprovalGate(auto_approve_delay=8.0, destructive_delay=15.0)


def _build_agent() -> Agent:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY not set — add it to backend/.env")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(UIAutomationTool())
    registry.register(FileCreationTool())
    registry.register(SystemControlTool())
    registry.register(CodeEditTool())
    registry.register(ContextTool())
    registry.register(ShellTool())
    registry.register(NLCommandTool())      # ← NEW

    executor = ToolExecutor(registry)
    planner  = Planner(api_key=api_key)
    agent    = Agent(
        registry=registry,
        executor=executor,
        api_key=api_key,
        planner=planner,
    )
    logger.info("Agent v6 ready — tools: %s  planner: enabled", registry.list_names())
    return agent


agent = _build_agent()

# ── Voice recording state ────────────────────────────────────────────────── #

_recording_thread: threading.Thread | None = None
_recording_result: dict = {}
_recording_done   = threading.Event()
_recording_lock   = threading.Lock()


# ── FastAPI app ──────────────────────────────────────────────────────────── #

app = FastAPI(title="Gladden API", version="6.0.0")

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

class ApprovalRequest(BaseModel):
    reason: str = ""

class GateStatusResponse(BaseModel):
    pending:  bool
    approved: bool | None = None

# ── NEW: NL Command schemas ──────────────────────────────────────────────── #

class NLCommandPreviewRequest(BaseModel):
    request: str
    working_dir: str = ""

class NLCommandPreviewResponse(BaseModel):
    success: bool
    command: str | None = None
    error:   str | None = None

class NLCommandRunRequest(BaseModel):
    request:     str
    working_dir: str = ""

# ── Voice schemas ────────────────────────────────────────────────────────── #

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

    result = agent.run(instruction, event_callback=_collect)
    return result.output, result.error, collected_events, result.success


# ── Core routes ──────────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "tools":   agent._registry.list_names(),
        "planner": agent._planner is not None,
        "version": "6.0.0",
    }


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: InstructionRequest):
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must not be empty.")
    output, error, events, success = _run_instruction(req.instruction)
    return ExecuteResponse(success=success, output=output, error=error, events=events)


# ── NL Command routes ────────────────────────────────────────────────────── #

@app.post("/nl_command/preview", response_model=NLCommandPreviewResponse)
def nl_command_preview(req: NLCommandPreviewRequest):
    """
    Translate a natural-language request to a shell command WITHOUT executing it.
    Returns the translated command for the UI to display before running.
    """
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="request must not be empty.")

    collected: list[dict] = []
    nl_tool = agent._registry.get_or_none("nl_command")
    if nl_tool is None:
        return NLCommandPreviewResponse(success=False, error="nl_command tool not registered.")

    result = nl_tool.execute(
        request=req.request.strip(),
        preview_only=True,
        working_dir=req.working_dir or "",
        event_callback=collected.append,
    )

    if result.success:
        # Extract just the command string from output like "Command (not executed): dir /s *.py"
        cmd = str(result.output or "").replace("Command (not executed): ", "").strip()
        return NLCommandPreviewResponse(success=True, command=cmd)

    return NLCommandPreviewResponse(success=False, error=result.error)


@app.post("/nl_command/run", response_model=ExecuteResponse)
def nl_command_run(req: NLCommandRunRequest):
    """
    Translate a natural-language request and execute the resulting command.
    Returns full event stream + output.
    """
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="request must not be empty.")

    instruction = req.request.strip()
    # Route through the normal agent so events are collected properly
    output, error, events, success = _run_instruction(
        f"Run a command that: {instruction}"
    )
    return ExecuteResponse(success=success, output=output, error=error, events=events)


# ── Approval gate routes ─────────────────────────────────────────────────── #

@app.post("/approve")
def approve(req: ApprovalRequest = ApprovalRequest()):
    _approval_gate.approve(req.reason or "User approved via UI")
    return {"status": "approved"}


@app.post("/reject")
def reject(req: ApprovalRequest = ApprovalRequest()):
    _approval_gate.reject(req.reason or "User rejected via UI")
    return {"status": "rejected"}


@app.get("/gate", response_model=GateStatusResponse)
def gate_status():
    return GateStatusResponse(pending=_approval_gate.is_pending)


# ── Voice routes ─────────────────────────────────────────────────────────── #

@app.get("/voice/check", response_model=VoiceCheckResponse)
def voice_check():
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
                    _recording_result["error"] = "No speech detected."
            except Exception as exc:
                logger.exception("Recording thread failed")
                _recording_result["success"]    = False
                _recording_result["transcript"] = None
                _recording_result["error"]      = str(exc)
            finally:
                _recording_done.set()

        _recording_thread = threading.Thread(target=_run, daemon=True)
        _recording_thread.start()

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
    try:
        from voice_input import transcribe_from_mic
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Voice module unavailable: {exc}.")
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
        return VoiceTranscribeResponse(success=False, error="No speech detected.")
    return VoiceTranscribeResponse(success=True, transcript=transcript)