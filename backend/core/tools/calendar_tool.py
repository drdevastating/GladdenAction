"""
core/tools/calendar_tool.py

CalendarTool — Windows Calendar (Outlook / UWP) automation.

Uses PyAutoGUI for all UI interaction and Pyperclip for content
that must be accurate (descriptions, locations).

Typing strategy
---------------
  title       → slow_type()   (short, user-visible feedback)
  description → paste_text()  (may be long; accuracy critical)
  date/time   → slow_type()   (field-by-field input)

Supported actions
-----------------
  open_calendar()
  create_event(title, event_datetime, description, location)
  view_today()

Event contract
--------------
  type: info | status | error
  stage: <snake_case>
  message: <human-readable>
  tool: calendar/<action>
  timestamp: ISO-8601 UTC
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
    click,
    double_click,
    focus_app,
    hotkey,
    move_mouse,
    paste_text,
    press_key,
    slow_type,
    wait,
)

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]


# ── Event helpers ─────────────────────────────────────────────────────────── #

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, action: str = "calendar") -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"calendar/{action}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


# ── Launch helpers ────────────────────────────────────────────────────────── #

_CALENDAR_CMDS = [
    # Windows 10/11 Mail & Calendar app URI
    ["cmd", "/c", "start", "", "outlookcal:"],
    # UWP shell: launch directly
    [
        "explorer.exe",
        "shell:appsfolder\\microsoft.windowscommunicationsapps_8wekyb3d8bbwe!"
        "microsoft.windowslive.calendar",
    ],
    # Generic shell open
    ["cmd", "/c", "start", "calendar"],
]


def _launch_calendar() -> bool:
    """Try each launch strategy; return True if at least one Popen succeeded."""
    for cmd in _CALENDAR_CMDS:
        try:
            subprocess.Popen(
                cmd,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Calendar launch attempt failed (%s): %s", cmd[0], exc)

    # Last resort: Windows Start-menu search
    try:
        pyautogui.hotkey("win")
        time.sleep(0.7)
        slow_type("calendar", interval=0.08)
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Calendar Start-menu launch failed: %s", exc)
    return False


def _focus_calendar() -> bool:
    """Bring the Calendar window to the foreground."""
    return focus_app("Calendar")


# ============================================================================ #
#  CalendarTool                                                                 #
# ============================================================================ #

class CalendarTool(BaseTool):
    """
    Automates Windows Calendar (UWP Mail & Calendar app) to open the app
    and create calendar events with visible, natural-looking interaction.

    Design principles
    -----------------
    • All mouse movements use duration > 0 (smooth, visible).
    • Short fields (title, time) are typed character-by-character via slow_type.
    • Long fields (description, location) use paste_text for accuracy.
    • Delays between steps mimic a real user's pace.
    """

    name = "calendar"

    description = (
        "Automates Windows Calendar to open the app and create events. "
        "Actions: "
        "'open' — open Windows Calendar app; "
        "'create_event' — create a new calendar event with title, date/time, "
        "description and location; "
        "'view_today' — navigate to today's view."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "create_event", "view_today"],
                "description": "Action to perform.",
            },
            "title": {
                "type": "string",
                "description": "Event title (used with create_event).",
            },
            "event_datetime": {
                "type": "string",
                "description": (
                    "Event start date and time. "
                    "Formats accepted: 'YYYY-MM-DD HH:MM', 'MM/DD/YYYY HH:MM', "
                    "or natural strings like 'tomorrow 3pm'."
                ),
            },
            "description": {
                "type": "string",
                "description": "Event description / notes (pasted, not typed).",
            },
            "location": {
                "type": "string",
                "description": "Event location (pasted for accuracy).",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Event duration in minutes (default: 60).",
            },
        },
    }

    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)
        action = kwargs.get("action", "").strip().lower()

        dispatch = {
            "open":         self._action_open,
            "create_event": self._action_create_event,
            "view_today":   self._action_view_today,
        }

        if action not in dispatch:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Must be one of: {list(dispatch.keys())}",
            )

        try:
            return dispatch[action](cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"CalendarTool/{action} crashed: {exc}"
            logger.exception(msg)
            _emit(cb, _event("error", "tool_crashed", msg, action))
            return ToolResult(success=False, error=msg)

    # ================================================================== #
    #  open                                                                #
    # ================================================================== #

    def _action_open(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "open"
        _emit(cb, _event("info", "launching_calendar",
                          "Launching Windows Calendar app…", action))

        if not _launch_calendar():
            msg = "Failed to launch Windows Calendar via all strategies."
            _emit(cb, _event("error", "launch_failed", msg, action))
            return ToolResult(success=False, error=msg)

        wait(3.5, "Calendar initialising")
        _focus_calendar()
        # Maximize window to ensure click coordinates align
        pyautogui.hotkey('win', 'up')
        wait(0.8, "Window maximizing")

        _emit(cb, _event("status", "calendar_open",
                          "Windows Calendar is open.", action))
        return ToolResult(
            success=True,
            output="Windows Calendar opened successfully.",
            metadata={"action": action},
        )

    # ================================================================== #
    #  view_today                                                          #
    # ================================================================== #

    def _action_view_today(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "view_today"
        _emit(cb, _event("info", "opening_calendar",
                          "Opening Calendar and navigating to today…", action))

        if not _launch_calendar():
            return ToolResult(success=False,
                              error="Failed to launch Windows Calendar.")

        wait(3.0, "Calendar loading")
        _focus_calendar()
        wait(0.6)

        # Windows Calendar: Alt+Home or Ctrl+T = Today
        hotkey("alt", "Home")
        wait(0.5)

        _emit(cb, _event("status", "today_view",
                          "Navigated to today's calendar view.", action))
        return ToolResult(success=True, output="Calendar showing today.",
                          metadata={"action": action})

    # ================================================================== #
    #  create_event                                                        #
    # ================================================================== #

    def _action_create_event(
        self,
        *,
        cb: EventCallback,
        title: str = "New Event",
        event_datetime: str = "",
        description: str = "",
        location: str = "",
        duration_minutes: int = 60,
        **_: Any,
    ) -> ToolResult:
        action = "create_event"

        if not title.strip():
            return ToolResult(success=False,
                              error="'title' is required for create_event.")

        _emit(cb, _event("info", "opening_calendar",
                          "Opening Calendar to create a new event…", action))

        if not _launch_calendar():
            return ToolResult(success=False,
                              error="Failed to launch Windows Calendar.")

        wait(3.5, "Calendar initialising")
        _focus_calendar()
        wait(0.8)

        # ── Step 1: Open new event ─────────────────────────────────── #
        _emit(cb, _event("info", "new_event",
                          "Opening new event form (Ctrl+N)…", action))
        hotkey("ctrl", "n")
        wait(1.8, "New event dialog loading")

        sw, sh = pyautogui.size()

        # ── Step 2: Enter title (visible slow typing) ──────────────── #
        _emit(cb, _event("info", "typing_title",
                          f"Typing event title: '{title}'…", action))

        # The title field is usually at the top of the new event form.
        # Click it to be sure it has focus.
        title_x = int(sw * 0.50)
        title_y = int(sh * 0.17)
        click(title_x, title_y, duration=0.4)
        wait(0.4)
        hotkey("ctrl", "a")
        wait(0.15)
        # SLOW TYPE — short, visible, title field reacts per keystroke
        slow_type(title, interval=0.07)
        wait(0.4)

        # ── Step 3: Press Tab to move to date/start-time field ─────── #
        _emit(cb, _event("info", "setting_datetime",
                          f"Setting date/time: '{event_datetime or 'today'}'…", action))
        press_key("tab")
        wait(0.35)

        if event_datetime:
            # Clear the existing date and enter the new one
            hotkey("ctrl", "a")
            wait(0.15)
            # Date fields are numeric — slow_type is fine for short strings
            slow_type(self._format_date_for_input(event_datetime), interval=0.10)
            wait(0.25)
            press_key("tab")   # move to time field
            wait(0.25)
            slow_type(self._format_time_for_input(event_datetime), interval=0.10)
            wait(0.25)

        # Tab past end-time and duration fields
        for _ in range(3):
            press_key("tab")
            wait(0.20)

        # ── Step 4: Location (paste — may be long/complex address) ─── #
        if location:
            _emit(cb, _event("info", "entering_location",
                              f"Entering location: '{location[:60]}'…", action))
            # Tab to location field
            press_key("tab")
            wait(0.25)
            hotkey("ctrl", "a")
            wait(0.15)
            # PASTE — location could be a full address
            paste_text(location)
            wait(0.35)

        # ── Step 5: Description (paste — always potentially long) ───── #
        if description:
            _emit(cb, _event("info", "entering_description",
                              f"Adding description ({len(description)} chars)…", action))
            # Description / notes field — Tab to it or click a known region
            desc_x = int(sw * 0.50)
            desc_y = int(sh * 0.60)
            click(desc_x, desc_y, duration=0.4)
            wait(0.4)
            hotkey("ctrl", "a")
            wait(0.15)
            # PASTE — description can be multiple paragraphs
            paste_text(description)
            wait(0.4)

        # ── Step 6: Save / send ────────────────────────────────────── #
        _emit(cb, _event("info", "saving_event",
                          "Saving the event (Ctrl+S)…", action))
        hotkey("ctrl", "s")
        wait(1.2, "Event saving")

        # Some Calendar dialogs ask for confirmation — press Enter to accept
        press_key("enter")
        wait(0.6)

        _emit(cb, _event("status", "event_created",
                          f"Event '{title}' created successfully.", action))
        return ToolResult(
            success=True,
            output=f"Calendar event '{title}' created.",
            metadata={
                "action":           action,
                "title":            title,
                "event_datetime":   event_datetime,
                "location":         location,
                "description_len":  len(description),
                "duration_minutes": duration_minutes,
            },
        )

    # ================================================================== #
    #  Private helpers                                                     #
    # ================================================================== #

    @staticmethod
    def _format_date_for_input(dt_str: str) -> str:
        """
        Extract the date portion as MM/DD/YYYY.
        Accepts ISO format (YYYY-MM-DD …) or common human strings.
        Falls back to returning the raw string so the user can correct.
        """
        dt_str = dt_str.strip()
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
                    "%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(dt_str.split(" ")[0], fmt.split(" ")[0])
                return dt.strftime("%m/%d/%Y")
            except ValueError:
                continue
        # Return the first word (date part) verbatim
        return dt_str.split(" ")[0]

    @staticmethod
    def _format_time_for_input(dt_str: str) -> str:
        """
        Extract the time portion as H:MM AM/PM.
        Falls back to "12:00 PM" if no time found.
        """
        parts = dt_str.strip().split()
        if len(parts) >= 2:
            time_part = parts[1]
            for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
                try:
                    t = datetime.strptime(time_part, fmt)
                    return t.strftime("%-I:%M %p")
                except ValueError:
                    continue
            return time_part
        return "12:00 PM"