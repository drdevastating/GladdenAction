"""
core/tools/clock_tool.py

ClockTool — Windows Clock (Alarms & Clock UWP app) automation.

Actions
-------
  open            Open the Clock app
  start_timer     Navigate to Timer tab and start a countdown
  set_alarm       Navigate to Alarm tab and create an alarm

All interaction uses PyAutoGUI primitives from ui_automation_tool.
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
    click, focus_app, hotkey, press_key, slow_type, wait,
)

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, action: str = "clock") -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"clock/{action}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _launch_clock() -> bool:
    for cmd in [
        ["cmd", "/c", "start", "", "ms-clock:"],
        [
            "explorer.exe",
            "shell:appsfolder\\Microsoft.WindowsAlarms_8wekyb3d8bbwe!App",
        ],
        ["cmd", "/c", "start", "clock"],
    ]:
        try:
            subprocess.Popen(
                cmd, shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Clock launch failed (%s): %s", cmd[0], exc)
    # Start-menu fallback
    try:
        pyautogui.hotkey("win")
        time.sleep(0.7)
        slow_type("clock", interval=0.08)
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Clock Start-menu launch failed: %s", exc)
    return False


class ClockTool(BaseTool):
    """
    Automates Windows Clock (Alarms & Clock) to open the app,
    start countdown timers, and set alarms.
    """

    name = "clock"

    description = (
        "Automates Windows Clock app. "
        "Actions: "
        "'open' — open the Clock app; "
        "'start_timer' — open Clock and start a countdown timer (specify minutes); "
        "'set_alarm' — open Clock and set an alarm (specify alarm_time as HH:MM)."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "start_timer", "set_alarm"],
            },
            "minutes": {
                "type": "integer",
                "description": "Timer duration in minutes (required for start_timer).",
            },
            "seconds": {
                "type": "integer",
                "description": "Additional seconds for the timer (optional, default 0).",
            },
            "alarm_time": {
                "type": "string",
                "description": "Alarm time in HH:MM format, 24h or 12h (required for set_alarm).",
            },
            "alarm_label": {
                "type": "string",
                "description": "Label/name for the alarm (optional).",
            },
        },
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)
        action = kwargs.get("action", "").strip().lower()

        dispatch = {
            "open":        self._action_open,
            "start_timer": self._action_start_timer,
            "set_alarm":   self._action_set_alarm,
        }

        if action not in dispatch:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Must be: {list(dispatch.keys())}",
            )

        try:
            return dispatch[action](cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"ClockTool/{action} crashed: {exc}"
            logger.exception(msg)
            _emit(cb, _event("error", "tool_crashed", msg, action))
            return ToolResult(success=False, error=msg)

    # ================================================================== #

    def _action_open(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "open"
        _emit(cb, _event("info", "launching", "Launching Windows Clock…", action))
        if not _launch_clock():
            return ToolResult(success=False, error="Failed to launch Windows Clock.")
        wait(3.0, "Clock initialising")
        focus_app("Clock")
        wait(0.6)
        _emit(cb, _event("status", "clock_open", "Windows Clock is open.", action))
        return ToolResult(success=True, output="Windows Clock opened.",
                          metadata={"action": action})

    def _action_start_timer(
        self,
        *,
        cb: EventCallback,
        minutes: int = 0,
        seconds: int = 0,
        **_: Any,
    ) -> ToolResult:
        action = "start_timer"

        if not minutes and not seconds:
            return ToolResult(success=False,
                              error="Provide 'minutes' and/or 'seconds' for start_timer.")

        _emit(cb, _event("info", "launching", "Launching Clock for timer…", action))
        if not _launch_clock():
            return ToolResult(success=False, error="Failed to launch Windows Clock.")
        wait(3.0, "Clock initialising")
        focus_app("Clock")
        wait(0.8)

        sw, sh = pyautogui.size()

        # Navigate to Timer tab
        # Windows Clock nav (left sidebar, vertical): Focus→Clock, Alarm, Timer, Stopwatch
        # Timer is typically ~3rd item ≈ 30% of screen height
        _emit(cb, _event("info", "nav_timer",
                          "Clicking Timer nav item…", action))
        timer_nav_x = int(sw * 0.065)
        timer_nav_y = int(sh * 0.30)
        click(timer_nav_x, timer_nav_y, duration=0.4)
        wait(1.0, "Timer tab loading")

        # Click the time input area (hours field) — centre-top of content area
        input_x = int(sw * 0.50)
        input_y = int(sh * 0.38)
        click(input_x, input_y, duration=0.35)
        wait(0.4)

        # Hours field: type 00 (we only use minutes/seconds in most cases)
        slow_type("00", interval=0.15)
        wait(0.20)
        press_key("tab")
        wait(0.20)

        # Minutes field
        mm_str = f"{int(minutes):02d}"
        _emit(cb, _event("info", "entering_time",
                          f"Entering {mm_str} minutes…", action))
        slow_type(mm_str, interval=0.15)
        wait(0.20)
        press_key("tab")
        wait(0.20)

        # Seconds field
        ss_str = f"{int(seconds):02d}"
        slow_type(ss_str, interval=0.15)
        wait(0.25)

        # Click "Start" button — roughly at bottom-centre of content
        start_x = int(sw * 0.50)
        start_y = int(sh * 0.62)
        _emit(cb, _event("info", "starting_timer",
                          f"Clicking Start for {minutes}m {seconds}s timer…", action))
        click(start_x, start_y, duration=0.35)
        wait(0.6)

        total_secs = int(minutes) * 60 + int(seconds)
        result_msg = f"Timer started: {minutes}m {seconds}s ({total_secs}s total)."
        _emit(cb, _event("status", "timer_started", result_msg, action))
        return ToolResult(
            success=True,
            output=result_msg,
            metadata={"action": action, "minutes": minutes, "seconds": seconds,
                      "total_seconds": total_secs},
        )

    def _action_set_alarm(
        self,
        *,
        cb: EventCallback,
        alarm_time: str = "",
        alarm_label: str = "",
        **_: Any,
    ) -> ToolResult:
        action = "set_alarm"

        if not alarm_time:
            return ToolResult(success=False,
                              error="'alarm_time' (HH:MM) is required for set_alarm.")

        _emit(cb, _event("info", "launching", "Launching Clock for alarm…", action))
        if not _launch_clock():
            return ToolResult(success=False, error="Failed to launch Windows Clock.")
        wait(3.0, "Clock initialising")
        focus_app("Clock")
        wait(0.8)

        sw, sh = pyautogui.size()

        # Navigate to Alarm tab (~18% down nav bar)
        _emit(cb, _event("info", "nav_alarm", "Clicking Alarm nav item…", action))
        alarm_nav_x = int(sw * 0.065)
        alarm_nav_y = int(sh * 0.18)
        click(alarm_nav_x, alarm_nav_y, duration=0.4)
        wait(1.0, "Alarm tab loading")

        # Click the "+" (Add alarm) button — typically bottom-right area
        _emit(cb, _event("info", "add_alarm", "Clicking Add Alarm button…", action))
        add_x = int(sw * 0.88)
        add_y = int(sh * 0.90)
        click(add_x, add_y, duration=0.4)
        wait(1.5, "Add alarm dialog loading")

        # Enter time digits without colon (Windows Clock accepts HHMM)
        time_digits = alarm_time.replace(":", "").replace(" ", "")
        _emit(cb, _event("info", "entering_alarm_time",
                          f"Entering alarm time: {alarm_time}…", action))
        slow_type(time_digits, interval=0.20)
        wait(0.4)

        # If there's a label field, Tab to it and enter the label
        if alarm_label:
            press_key("tab")
            wait(0.25)
            slow_type(alarm_label, interval=0.06)
            wait(0.3)

        # Confirm / Save alarm — Enter key or click Save button
        press_key("enter")
        wait(0.8)

        result_msg = f"Alarm set for {alarm_time}{f' ({alarm_label})' if alarm_label else ''}."
        _emit(cb, _event("status", "alarm_set", result_msg, action))
        return ToolResult(
            success=True,
            output=result_msg,
            metadata={"action": action, "alarm_time": alarm_time, "alarm_label": alarm_label},
        )