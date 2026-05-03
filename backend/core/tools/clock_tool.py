"""
core/tools/clock_tool.py  — v2 (fixed)

Root cause of previous failures
---------------------------------
Old version used absolute pixel coordinates (int(sw * 0.065), int(sh * 0.30))
to click nav items. These are highly screen-resolution- and window-size-dependent.

Fix: Keyboard-only navigation, mirroring the WhatsApp / Gmail approach.
  • ms-clock: URI → shell appsfolder → Win search (3-fallback launch)
  • Ctrl+1 = Alarm  |  Ctrl+4 = Timer  (Windows Alarms & Clock shortcuts)
  • Tab + slow_type for all time/label entry — zero coordinate clicking

Actions
-------
  open          Open Windows Clock
  start_timer   Navigate to Timer tab, add timer, start it
  set_alarm     Navigate to Alarm tab, fill dialog, save
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


def _event(type_: str, stage: str, message: str, action: str = "clock") -> dict:
    return {"type": type_, "stage": stage, "message": message,
            "tool": f"clock/{action}", "timestamp": _utc_now()}


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _launch_clock(cb: EventCallback, action: str) -> bool:
    """3-fallback launch, same pattern as WhatsApp/Gmail."""
    _emit(cb, _event("info", "launching", "Launching Windows Clock…", action))

    # 1. ms-clock: URI
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "ms-clock:"],
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        wait(3.5, "Clock initialising via URI")
        return True
    except Exception as exc:
        logger.debug("ms-clock URI failed: %s", exc)

    # 2. UWP shell appsfolder
    try:
        subprocess.Popen(
            ["explorer.exe",
             "shell:appsfolder\\Microsoft.WindowsAlarms_8wekyb3d8bbwe!App"],
            shell=False,
        )
        wait(3.5, "Clock initialising via shell")
        return True
    except Exception as exc:
        logger.debug("Clock shell appsfolder failed: %s", exc)

    # 3. Win search — guaranteed fallback (same as WhatsApp)
    _emit(cb, _event("info", "launch_search", "Searching Clock via Start menu…", action))
    try:
        pyautogui.hotkey("win")
        wait(0.9)
        slow_type("Alarms Clock", interval=0.07)
        wait(0.8)
        press_key("enter")
        wait(3.5, "Clock initialising via Start search")
        return True
    except Exception as exc:
        logger.debug("Clock Start search failed: %s", exc)

    return False


def _focus_clock() -> None:
    for title in ["Alarms & Clock", "Clock", "Alarm"]:
        try:
            import pygetwindow as gw  # type: ignore
            wins = [w for w in gw.getAllWindows()
                    if title.lower() in w.title.lower()]
            if wins:
                wins[0].restore()
                wins[0].activate()
                wait(0.4)
                return
        except Exception:
            pass
    # bare fallback
    sw, sh = pyautogui.size()
    pyautogui.click(sw // 2, sh // 2)
    wait(0.3)


class ClockTool(BaseTool):
    name = "clock"
    description = (
        "Automates Windows Clock (Alarms & Clock) app via keyboard-only navigation. "
        "Actions: 'open', 'start_timer' (minutes/seconds), 'set_alarm' (alarm_time HH:MM, optional alarm_label)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action":      {"type": "string", "enum": ["open", "start_timer", "set_alarm"]},
            "minutes":     {"type": "integer", "description": "Timer minutes."},
            "seconds":     {"type": "integer", "description": "Timer seconds (default 0)."},
            "alarm_time":  {"type": "string",  "description": "Alarm time HH:MM (24h)."},
            "alarm_label": {"type": "string",  "description": "Alarm name/label (optional)."},
        },
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)
        action = kwargs.get("action", "").strip().lower()
        dispatch = {"open": self._open, "start_timer": self._start_timer,
                    "set_alarm": self._set_alarm}
        if action not in dispatch:
            return ToolResult(success=False,
                              error=f"Unknown action '{action}'. Options: {list(dispatch)}")
        try:
            return dispatch[action](cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"ClockTool/{action} crashed: {exc}"
            logger.exception(msg)
            return ToolResult(success=False, error=msg)

    # ── open ──────────────────────────────────────────────────────────── #

    def _open(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        if not _launch_clock(cb, "open"):
            return ToolResult(success=False, error="Failed to launch Windows Clock.")
        _focus_clock()
        _emit(cb, _event("status", "clock_open", "Windows Clock is open.", "open"))
        return ToolResult(success=True, output="Windows Clock opened.")

    # ── start_timer ───────────────────────────────────────────────────── #

    def _start_timer(self, *, cb: EventCallback,
                     minutes: int = 0, seconds: int = 0, **_: Any) -> ToolResult:
        action = "start_timer"
        minutes = max(0, int(minutes or 0))
        seconds = max(0, int(seconds or 0))
        if not minutes and not seconds:
            return ToolResult(success=False,
                              error="Provide 'minutes' and/or 'seconds'.")

        if not _launch_clock(cb, action):
            return ToolResult(success=False, error="Failed to launch Windows Clock.")

        # Navigate to Timer section (Ctrl+4 in Alarms & Clock)
        _emit(cb, _event("info", "nav_timer", "Navigating to Timer tab (Ctrl+4)…", action))
        _focus_clock()
        wait(0.5)
        hotkey("ctrl", "4")
        wait(1.2)

        # Open the "Add new timer" dialog
        _emit(cb, _event("info", "add_timer", "Opening Add Timer dialog…", action))
        _focus_clock()
        wait(0.3)
        press_key("tab")    # Tab to the + button
        wait(0.3)
        press_key("space")  # Activate it
        wait(1.2, "Timer dialog opening")

        # Fill HH / MM / SS fields (Tab-navigated, no coordinate clicking)
        _emit(cb, _event("info", "entering_time",
                          f"Entering {minutes:02d}m {seconds:02d}s…", action))

        hotkey("ctrl", "a"); wait(0.1)
        slow_type("00", interval=0.18)   # hours
        wait(0.2)
        press_key("tab"); wait(0.2)

        hotkey("ctrl", "a"); wait(0.1)
        slow_type(f"{minutes:02d}", interval=0.18)   # minutes
        wait(0.2)
        press_key("tab"); wait(0.2)

        hotkey("ctrl", "a"); wait(0.1)
        slow_type(f"{seconds:02d}", interval=0.18)   # seconds
        wait(0.3)

        # Tab to Start / Save button and press Enter
        press_key("tab"); wait(0.2)
        press_key("enter")
        wait(0.8)

        total = minutes * 60 + seconds
        msg = f"Timer started: {minutes}m {seconds}s ({total}s total)."
        _emit(cb, _event("status", "timer_started", msg, action))
        return ToolResult(success=True, output=msg,
                          metadata={"minutes": minutes, "seconds": seconds,
                                    "total_seconds": total})

    # ── set_alarm ─────────────────────────────────────────────────────── #

    def _set_alarm(self, *, cb: EventCallback,
                   alarm_time: str = "", alarm_label: str = "", **_: Any) -> ToolResult:
        action = "set_alarm"
        if not alarm_time:
            return ToolResult(success=False, error="'alarm_time' (HH:MM) is required.")

        try:
            parts  = alarm_time.strip().replace(".", ":").split(":")
            hour   = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return ToolResult(success=False,
                              error=f"Cannot parse alarm_time '{alarm_time}'. Use HH:MM.")

        if not _launch_clock(cb, action):
            return ToolResult(success=False, error="Failed to launch Windows Clock.")

        # Navigate to Alarm section (Ctrl+1)
        _emit(cb, _event("info", "nav_alarm", "Navigating to Alarm tab (Ctrl+1)…", action))
        _focus_clock()
        wait(0.5)
        hotkey("ctrl", "1")
        wait(1.2)

        # Open Add Alarm dialog
        _emit(cb, _event("info", "add_alarm", "Opening Add Alarm dialog…", action))
        _focus_clock()
        wait(0.3)
        press_key("tab")    # Tab to the + button
        wait(0.3)
        press_key("space")  # Activate it
        wait(1.5, "Add alarm dialog loading")

        # Fill time fields
        _emit(cb, _event("info", "entering_time",
                          f"Setting alarm to {hour:02d}:{minute:02d}…", action))

        hotkey("ctrl", "a"); wait(0.1)
        slow_type(f"{hour:02d}", interval=0.20)   # hour
        wait(0.3)
        press_key("tab"); wait(0.2)

        hotkey("ctrl", "a"); wait(0.1)
        slow_type(f"{minute:02d}", interval=0.20)  # minute
        wait(0.3)
        press_key("tab"); wait(0.2)   # past AM/PM
        press_key("tab"); wait(0.2)   # to Name field

        if alarm_label:
            _emit(cb, _event("info", "entering_label",
                              f"Setting label: '{alarm_label}'…", action))
            hotkey("ctrl", "a"); wait(0.1)
            slow_type(alarm_label, interval=0.06)
            wait(0.3)

        # Save
        _emit(cb, _event("info", "saving", "Saving alarm…", action))
        press_key("tab"); wait(0.2)   # Tab to Save button
        press_key("enter")
        wait(1.0)

        result = (f"Alarm set for {hour:02d}:{minute:02d}"
                  + (f" '{alarm_label}'" if alarm_label else "") + ".")
        _emit(cb, _event("status", "alarm_set", result, action))
        return ToolResult(success=True, output=result,
                          metadata={"hour": hour, "minute": minute,
                                    "alarm_label": alarm_label})