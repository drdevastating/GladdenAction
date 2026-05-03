"""
core/tools/recorder_tool.py  — v2 (fixed)

Root cause of previous failures
---------------------------------
1. ms-voicerecorder: URI fails on some Windows builds (app not registered).
2. Record button coordinate (int(sh * 0.72)) does not match actual button
   position after window reflow — the app resizes differently per system.

Fix: Same pattern as WhatsApp/Gmail.
  • Win search "Voice Recorder" — the most reliable way to open UWP apps
  • After launch, use keyboard shortcut Ctrl+R to toggle recording
    (Windows Voice Recorder supports Ctrl+R = Start/Stop recording)
  • No pixel coordinates at all

Actions
-------
  open              Open Voice Recorder
  start_recording   Open app and start recording via Ctrl+R
  stop_recording    Stop recording via Ctrl+R (same toggle shortcut)
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pyautogui

from core.tools.base import BaseTool, ToolResult
from core.tools.ui_automation_tool import (
    focus_app, hotkey, press_key, slow_type, wait,
)

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, action: str = "recorder") -> dict:
    return {"type": type_, "stage": stage, "message": message,
            "tool": f"recorder/{action}", "timestamp": _utc_now()}


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _launch_recorder(cb: EventCallback, action: str) -> bool:
    """
    Launch Windows Voice Recorder using 3-fallback strategy.
    Mirrors the WhatsApp/Gmail approach exactly.
    """
    _emit(cb, _event("info", "launching", "Launching Windows Voice Recorder…", action))

    # 1. ms-voicerecorder: URI
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "ms-voicerecorder:"],
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        wait(3.5, "Voice Recorder initialising via URI")
        return True
    except Exception as exc:
        logger.debug("ms-voicerecorder URI failed: %s", exc)

    # 2. UWP shell appsfolder
    try:
        subprocess.Popen(
            ["explorer.exe",
             "shell:appsfolder\\Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe!App"],
            shell=False,
        )
        wait(3.5, "Voice Recorder initialising via shell")
        return True
    except Exception as exc:
        logger.debug("Voice Recorder shell appsfolder failed: %s", exc)

    # 3. Win search — guaranteed fallback (same pattern as WhatsApp)
    _emit(cb, _event("info", "launch_search",
                      "Searching Voice Recorder via Start menu…", action))
    try:
        pyautogui.hotkey("win")
        wait(0.9)
        slow_type("Voice Recorder", interval=0.07)
        wait(0.8)
        press_key("enter")
        wait(3.5, "Voice Recorder initialising via Start search")
        return True
    except Exception as exc:
        logger.debug("Voice Recorder Start search failed: %s", exc)

    return False


def _focus_recorder() -> None:
    """Bring Voice Recorder window to foreground."""
    for title in ["Voice Recorder", "Sound Recorder", "Recorder"]:
        try:
            import pygetwindow as gw  # type: ignore
            wins = [w for w in gw.getAllWindows()
                    if title.lower() in w.title.lower()]
            if wins:
                wins[0].restore()
                wins[0].activate()
                wait(0.5)
                return
        except Exception:
            pass
    # Fallback: click centre
    sw, sh = pyautogui.size()
    pyautogui.click(sw // 2, sh // 2)
    wait(0.3)


class RecorderTool(BaseTool):
    """
    Automates Windows Voice Recorder using keyboard shortcuts only.
    Ctrl+R = Start / Stop recording (no pixel-coordinate clicking).
    """

    name = "recorder"
    description = (
        "Automates Windows Voice Recorder. "
        "Actions: 'open' — open the app; "
        "'start_recording' — open and start recording (Ctrl+R); "
        "'stop_recording' — stop the current recording (Ctrl+R toggle)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "start_recording", "stop_recording"],
            },
        },
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)
        action = kwargs.get("action", "").strip().lower()
        dispatch = {
            "open":            self._open,
            "start_recording": self._start_recording,
            "stop_recording":  self._stop_recording,
        }
        if action not in dispatch:
            return ToolResult(success=False,
                              error=f"Unknown action '{action}'. Options: {list(dispatch)}")
        try:
            return dispatch[action](cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"RecorderTool/{action} crashed: {exc}"
            logger.exception(msg)
            return ToolResult(success=False, error=msg)

    # ── open ──────────────────────────────────────────────────────────── #

    def _open(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        if not _launch_recorder(cb, "open"):
            return ToolResult(success=False, error="Failed to launch Voice Recorder.")
        _focus_recorder()
        _emit(cb, _event("status", "recorder_open",
                          "Voice Recorder is open. Use Ctrl+R to start recording.", "open"))
        return ToolResult(success=True, output="Voice Recorder opened.")

    # ── start_recording ───────────────────────────────────────────────── #

    def _start_recording(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "start_recording"
        if not _launch_recorder(cb, action):
            return ToolResult(success=False, error="Failed to launch Voice Recorder.")

        _focus_recorder()
        wait(0.5)

        # Ctrl+R = Start recording (official Voice Recorder keyboard shortcut)
        _emit(cb, _event("info", "starting", "Pressing Ctrl+R to start recording…", action))
        hotkey("ctrl", "r")
        wait(0.8)

        _emit(cb, _event("status", "recording_started",
                          "Recording started. Call stop_recording or press Ctrl+R to stop.", action))
        return ToolResult(success=True, output="Voice recording started (Ctrl+R).",
                          metadata={"shortcut": "Ctrl+R"})

    # ── stop_recording ────────────────────────────────────────────────── #

    def _stop_recording(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "stop_recording"
        _emit(cb, _event("info", "stopping", "Stopping recording (Ctrl+R)…", action))
        _focus_recorder()
        wait(0.4)

        # Ctrl+R toggles stop as well
        hotkey("ctrl", "r")
        wait(1.5, "Recording saving")

        _emit(cb, _event("status", "recording_stopped",
                          "Recording stopped and saved in Voice Recorder.", action))
        return ToolResult(success=True,
                          output="Recording stopped and saved in Voice Recorder.")