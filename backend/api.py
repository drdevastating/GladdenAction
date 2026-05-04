"""
api.py  —  GladdenAction FastAPI backend  (v7 — voice recording fixed)

Voice recording changes (v7)
-----------------------------
The previous /voice/transcribe route blocked the HTTP connection for the
entire recording duration, which caused the Electron renderer's fetch()
to either time out or hang.  The fix splits recording into three stages:

  POST /voice/start        — begins background recording; returns immediately
  POST /voice/stop         — signals recording to stop, waits for transcript
  POST /voice/transcribe   — DEPRECATED compat alias: start + auto-stop after
                             max_duration; kept for backward compat but now
                             runs in a way that doesn't block indefinitely

All other routes are unchanged from v6.
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
from core.tools.nl_command_tool import NLCommandTool
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
    registry.register(NLCommandTool())

    executor = ToolExecutor(registry)
    planner  = Planner(api_key=api_key)
    agent    = Agent(
        registry=registry,
        executor=executor,
        api_key=api_key,
        planner=planner,
    )
    logger.info("Agent ready — tools: %s", registry.list_names())
    return agent


agent = _build_agent()

# ── Voice recording state ────────────────────────────────────────────────── #

# Shared recording session state
_recording_lock   = threading.Lock()
_recording_thread: threading.Thread | None = None
_recording_result: dict = {}
_recording_done   = threading.Event()
_recording_active = threading.Event()   # set while a recording is in progress


# ── FastAPI app ──────────────────────────────────────────────────────────── #

app = FastAPI(title="Gladden API", version="7.0.0")

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

class VoiceStartRequest(BaseModel):
    model_size:   str   = "large-v3"
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
    model_size:   str   = "large-v3"
    max_duration: float = 30.0

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
        "version": "7.0.0",
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
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="request must not be empty.")

    nl_tool = agent._registry.get_or_none("nl_command")
    if nl_tool is None:
        return NLCommandPreviewResponse(success=False, error="nl_command tool not registered.")

    result = nl_tool.execute(
        request=req.request.strip(),
        preview_only=True,
        working_dir=req.working_dir or "",
    )

    if result.success:
        cmd = str(result.output or "").replace("Command (not executed): ", "").strip()
        return NLCommandPreviewResponse(success=True, command=cmd)

    return NLCommandPreviewResponse(success=False, error=result.error)


@app.post("/nl_command/run", response_model=ExecuteResponse)
def nl_command_run(req: NLCommandRunRequest):
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="request must not be empty.")

    output, error, events, success = _run_instruction(
        f"Run a command that: {req.request.strip()}"
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


# ═══════════════════════════════════════════════════════════════════════════ #
#  Voice routes (fixed in v7)                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def _voice_deps_ok() -> bool:
    try:
        from voice_input import check_dependencies
        return all(check_dependencies().values())
    except Exception:
        return False


def _assert_voice_deps():
    if not _voice_deps_ok():
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice dependencies not installed. "
                "Run: pip install faster-whisper sounddevice numpy"
            ),
        )


@app.get("/voice/check", response_model=VoiceCheckResponse)
def voice_check():
    try:
        from voice_input import check_dependencies
        deps = check_dependencies()
    except Exception as exc:
        logger.warning("voice_input module unavailable: %s", exc)
        deps = {"sounddevice": False, "faster_whisper": False, "numpy": False}
    return VoiceCheckResponse(
        sounddevice=deps.get("sounddevice", False),
        faster_whisper=deps.get("faster_whisper", False),
        numpy=deps.get("numpy", False),
        ready=all(deps.values()),
    )


# ── POST /voice/start  ───────────────────────────────────────────────────── #

@app.post("/voice/start", response_model=VoiceStartResponse)
def voice_start(req: VoiceStartRequest = VoiceStartRequest()):
    """
    Begin a background recording session.  Returns immediately with
    recording=True.  The Electron frontend should then call /voice/stop
    when the user clicks the mic button again (or presses Enter).
    """
    _assert_voice_deps()

    global _recording_thread, _recording_result, _recording_done, _recording_active

    with _recording_lock:
        if _recording_active.is_set():
            return VoiceStartResponse(
                recording=False,
                error="Already recording — call /voice/stop first.",
            )

        _recording_result = {}
        _recording_done.clear()
        _recording_active.set()

        model_size   = req.model_size   or "large-v3"
        max_duration = req.max_duration or 30.0

        def _run():
            try:
                from voice_input import transcribe_from_mic, request_stop  # noqa: F401
                transcript = transcribe_from_mic(
                    model_size=model_size,
                    max_duration=max_duration,
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
                _recording_active.clear()
                _recording_done.set()

        _recording_thread = threading.Thread(target=_run, daemon=True)
        _recording_thread.start()

    return VoiceStartResponse(recording=True)


# ── POST /voice/stop  ────────────────────────────────────────────────────── #

@app.post("/voice/stop", response_model=VoiceStopResponse)
def voice_stop():
    """
    Signal the background recording to stop, then block until transcription
    completes and return the result.  Typically takes 1–10 seconds depending
    on model size and audio length.
    """
    _assert_voice_deps()

    from voice_input import request_stop

    if not _recording_active.is_set() and not _recording_done.is_set():
        return VoiceStopResponse(success=False, error="No active recording to stop.")

    # Signal the recording thread to stop capturing
    request_stop()

    # Wait for transcription to finish (up to 120 s for large-v3 on CPU)
    finished = _recording_done.wait(timeout=120)
    if not finished:
        return VoiceStopResponse(success=False, error="Transcription timed out.")

    result = _recording_result
    return VoiceStopResponse(
        success=result.get("success", False),
        transcript=result.get("transcript"),
        error=result.get("error"),
    )


# ── POST /voice/transcribe  ──────────────────────────────────────────────── #
#
# Legacy "one-shot" endpoint kept for backward compatibility with the
# original Electron renderer.  It now works correctly by:
#   1. Starting a background recording thread (just like /voice/start)
#   2. Waiting for max_duration OR a stop signal (whichever comes first)
#   3. Returning the transcript
#
# The key fix: we use a threading.Event to let the OS record in a thread
# while this HTTP handler waits on that event, so FastAPI's worker is not
# blocking the sounddevice stream.
# ──────────────────────────────────────────────────────────────────────────── #

@app.post("/voice/transcribe", response_model=VoiceTranscribeResponse)
def voice_transcribe(req: VoiceListenRequest = VoiceListenRequest()):
    """
    One-shot record + transcribe.

    The Electron renderer calls this endpoint when the user clicks the mic
    button.  It blocks until either the user clicks Stop (via request_stop())
    or max_duration elapses.

    The renderer should show a "recording…" state while this call is pending,
    then display the transcript when it returns.
    """
    _assert_voice_deps()

    global _recording_result, _recording_done, _recording_active

    with _recording_lock:
        if _recording_active.is_set():
            return VoiceTranscribeResponse(
                success=False,
                error="Already recording — wait for the current session to finish.",
            )

        _recording_result = {}
        _recording_done.clear()
        _recording_active.set()

        model_size   = req.model_size   or "large-v3"
        max_duration = req.max_duration or 30.0

        done_event = threading.Event()

        def _run():
            try:
                from voice_input import transcribe_from_mic
                transcript = transcribe_from_mic(
                    model_size=model_size,
                    max_duration=max_duration,
                )
                _recording_result["transcript"] = transcript
                _recording_result["success"]    = bool(transcript)
                if not transcript:
                    _recording_result["error"] = "No speech detected."
            except Exception as exc:
                logger.exception("voice_transcribe thread failed")
                _recording_result["success"]    = False
                _recording_result["transcript"] = None
                _recording_result["error"]      = str(exc)
            finally:
                _recording_active.clear()
                _recording_done.set()
                done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # Wait outside the lock — allow other requests (like /voice/stop) to proceed
    # Timeout = max_duration + 120 s transcription buffer
    timeout = float(max_duration) + 120.0
    finished = done_event.wait(timeout=timeout)

    if not finished:
        return VoiceTranscribeResponse(success=False, error="Transcription timed out.")

    result = _recording_result
    return VoiceTranscribeResponse(
        success=result.get("success", False),
        transcript=result.get("transcript"),
        error=result.get("error"),
    )