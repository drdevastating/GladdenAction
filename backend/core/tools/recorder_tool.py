"""
core/tools/recorder_tool.py

RecorderTool — Windows Voice Recorder (Sound Recorder) automation.

Actions
-------
  open              Open Voice Recorder app
  start_recording   Open app and click the record button
  stop_recording    Stop an in-progress recording and save

Uses PyAutoGUI primitives from ui_automation_tool.
No clipboard (Pyperclip) needed — this tool is purely action-based.
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
    click, focus_app, press_key, slow_type, wait,
)

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, action: str = "recorder") -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"recorder/{action}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _launch_recorder() -> bool:
    for cmd in [
        ["cmd", "/c", "start", "", "ms-voicerecorder:"],
        [
            "explorer.exe",
            "shell:appsfolder\\Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe!App",
        ],
        ["cmd", "/c", "start", "voice recorder"],
    ]:
        try:
            subprocess.Popen(
                cmd, shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recorder launch failed (%s): %s", cmd[0], exc)
    # Start-menu fallback
    try:
        pyautogui.hotkey("win")
        time.sleep(0.7)
        slow_type("voice recorder", interval=0.07)
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Recorder Start-menu launch failed: %s", exc)
    return False


class RecorderTool(BaseTool):
    """
    Automates Windows Voice Recorder to open, start, and stop recordings.
    All UI interaction uses PyAutoGUI with smooth mouse movement and delays.
    """

    name = "recorder"

    description = (
        "Automates Windows Voice Recorder. "
        "Actions: "
        "'open' — open the Voice Recorder app; "
        "'start_recording' — open the app and click the Record button; "
        "'stop_recording' — stop the current recording and save it."
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
            "open":            self._action_open,
            "start_recording": self._action_start_recording,
            "stop_recording":  self._action_stop_recording,
        }

        if action not in dispatch:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Must be: {list(dispatch.keys())}",
            )

        try:
            return dispatch[action](cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"RecorderTool/{action} crashed: {exc}"
            logger.exception(msg)
            _emit(cb, _event("error", "tool_crashed", msg, action))
            return ToolResult(success=False, error=msg)

    # ================================================================== #

    def _action_open(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "open"
        _emit(cb, _event("info", "launching", "Launching Windows Voice Recorder…", action))
        if not _launch_recorder():
            return ToolResult(success=False, error="Failed to launch Voice Recorder.")
        wait(3.0, "Voice Recorder initialising")
        focus_app("Voice Recorder")
        wait(0.6)
        _emit(cb, _event("status", "recorder_open",
                          "Voice Recorder is open. Click the mic button to record.", action))
        return ToolResult(success=True, output="Voice Recorder opened.",
                          metadata={"action": action})

    def _action_start_recording(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "start_recording"
        _emit(cb, _event("info", "launching",
                          "Launching Voice Recorder and starting recording…", action))
        if not _launch_recorder():
            return ToolResult(success=False, error="Failed to launch Voice Recorder.")
        wait(3.0, "Voice Recorder initialising")
        focus_app("Voice Recorder")
        wait(0.8)

        sw, sh = pyautogui.size()
        # The Record button is the large circular button in the centre-bottom area
        rec_x = int(sw * 0.50)
        rec_y = int(sh * 0.72)

        _emit(cb, _event("info", "clicking_record",
                          "Clicking the Record button…", action))
        click(rec_x, rec_y, duration=0.45)
        wait(0.8)

        _emit(cb, _event("status", "recording_started",
                          "Recording started. Use stop_recording to finish.", action))
        return ToolResult(
            success=True,
            output="Voice recording started.",
            metadata={"action": action, "record_button": (rec_x, rec_y)},
        )

    def _action_stop_recording(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "stop_recording"
        _emit(cb, _event("info", "stopping", "Stopping the current recording…", action))
        focus_app("Voice Recorder")
        wait(0.6)

        sw, sh = pyautogui.size()
        # Stop button is in the same position as the Record button (it toggles)
        stop_x = int(sw * 0.50)
        stop_y = int(sh * 0.72)

        _emit(cb, _event("info", "clicking_stop",
                          "Clicking Stop button to save recording…", action))
        click(stop_x, stop_y, duration=0.45)
        wait(1.2, "Recording saving")

        _emit(cb, _event("status", "recording_stopped",
                          "Recording stopped and saved.", action))
        return ToolResult(
            success=True,
            output="Recording stopped and saved in Voice Recorder.",
            metadata={"action": action},
        )