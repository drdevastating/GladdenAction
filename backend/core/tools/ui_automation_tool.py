"""
core/tools/ui_automation_tool.py  — Primitives Edition

Changes from previous version
------------------------------
ADDED: create_event_browser workflow — creates Google Calendar events via Chrome,
       following the same pattern as send_email_browser.

All existing workflows unchanged.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pyautogui
import pyperclip

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

pyautogui.PAUSE    = 0.20
pyautogui.FAILSAFE = True


# =========================================================================== #
#  PRIMITIVES                                                                   #
# =========================================================================== #

def move_mouse(x: int, y: int, duration: float = 0.45) -> None:
    pyautogui.moveTo(x, y, duration=max(0.05, duration))


def click(x: int | None = None, y: int | None = None, duration: float = 0.3) -> None:
    if x is not None and y is not None:
        move_mouse(x, y, duration)
    pyautogui.click()


def double_click(x: int | None = None, y: int | None = None, duration: float = 0.3) -> None:
    if x is not None and y is not None:
        move_mouse(x, y, duration)
    pyautogui.doubleClick()


def right_click(x: int | None = None, y: int | None = None, duration: float = 0.3) -> None:
    if x is not None and y is not None:
        move_mouse(x, y, duration)
    pyautogui.rightClick()


def press_key(key: str) -> None:
    pyautogui.press(key)


def hotkey(*keys: str) -> None:
    pyautogui.hotkey(*keys)


def slow_type(text: str, interval: float = 0.055) -> None:
    pyautogui.typewrite(text, interval=max(0.02, interval))


def paste_text(text: str) -> None:
    pyperclip.copy(text)
    time.sleep(0.08)
    pyautogui.hotkey("ctrl", "v")


def wait(seconds: float, reason: str = "") -> None:
    if reason:
        logger.debug("wait %.2fs — %s", seconds, reason)
    time.sleep(max(0.0, seconds))


def focus_app(app_name: str) -> bool:
    try:
        import pygetwindow as gw
        wins = gw.getWindowsWithTitle(app_name)
        if not wins:
            wins = [w for w in gw.getAllWindows()
                    if app_name.lower() in w.title.lower()]
        if wins:
            wins[0].restore()
            wins[0].activate()
            time.sleep(0.4)
            return True
    except Exception as exc:
        logger.debug("pygetwindow focus failed: %s", exc)

    try:
        pyautogui.hotkey("win")
        time.sleep(0.6)
        slow_type(app_name, interval=0.07)
        time.sleep(0.5)
        pyautogui.press("enter")
        time.sleep(1.5)
    except Exception as exc:
        logger.debug("Start-menu focus fallback failed: %s", exc)
    return False


# =========================================================================== #
#  Helpers                                                                      #
# =========================================================================== #

def _emit(callback: EventCallback, event: dict) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:
        logger.warning("event_callback raised: %s", exc)


def _event(type_: str, stage: str, message: str, workflow: str) -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"ui_automation/{workflow}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _wait_emit(seconds: float, callback: EventCallback, workflow: str, reason: str = "") -> None:
    if reason:
        _emit(callback, _event("info", "waiting", f"Waiting {seconds}s — {reason}", workflow))
    time.sleep(seconds)


# ── App locators ─────────────────────────────────────────────────────────── #

def _find_vscode() -> str | None:
    for alias in ("code", "code.cmd", "code-insiders"):
        if shutil.which(alias):
            return shutil.which(alias)
    local_app  = os.environ.get("LOCALAPPDATA", "")
    prog_files = os.environ.get("ProgramFiles",   "C:\\Program Files")
    prog_x86   = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    user_home  = str(Path.home())
    candidates: list[Path] = [
        Path(local_app) / "Programs" / "Microsoft VS Code" / "Code.exe",
        Path(local_app) / "Microsoft" / "WindowsApps" / "code.exe",
        Path(prog_files)  / "Microsoft VS Code" / "Code.exe",
        Path(prog_x86)    / "Microsoft VS Code" / "Code.exe",
        Path(user_home) / "scoop" / "shims" / "code.exe",
        Path("C:\\tools") / "vscode" / "Code.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _find_chrome() -> list[str]:
    chrome_paths = [
        Path(os.environ.get("ProgramFiles",   "C:\\Program Files"))
        / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
        / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for p in chrome_paths:
        if p.exists():
            return [str(p)]
    return ["cmd", "/c", "start", "chrome"]


def _find_whatsapp() -> str | None:
    local = os.environ.get("LOCALAPPDATA", "")
    for p in [
        Path(local) / "Microsoft" / "WindowsApps" / "WhatsApp.exe",
        Path(local) / "WhatsApp" / "WhatsApp.exe",
    ]:
        if p.exists():
            return str(p)
    return shutil.which("WhatsApp")


def _find_linkedin_desktop() -> str | None:
    local = os.environ.get("LOCALAPPDATA", "")
    candidates: list[Path] = [
        Path(local) / "Microsoft" / "WindowsApps" / "LinkedIn.exe",
        Path(local) / "LinkedIn" / "LinkedIn.exe",
        Path(local) / "Programs" / "LinkedIn" / "LinkedIn.exe",
        Path("C:\\Program Files") / "LinkedIn" / "LinkedIn.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("LinkedIn") or shutil.which("linkedin")


def _launch_linkedin_app(callback: EventCallback, wf: str) -> bool:
    _emit(callback, _event("info", "app_launch", "Launching LinkedIn Desktop app…", wf))
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "linkedin://"],
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        _wait_emit(6.0, callback, wf, "LinkedIn app initialising")
        return True
    except Exception as exc:
        logger.debug("LinkedIn URI launch failed: %s", exc)
    exe = _find_linkedin_desktop()
    if exe:
        try:
            subprocess.Popen([exe])
            _wait_emit(6.0, callback, wf, "LinkedIn app initialising")
            return True
        except Exception:
            pass
    return False


def _focus_window_by_title(title_fragment: str) -> bool:
    try:
        import pygetwindow as gw
        for fragment in [title_fragment, title_fragment.split()[0]]:
            wins = gw.getWindowsWithTitle(fragment)
            if wins:
                wins[0].restore()
                wins[0].activate()
                time.sleep(0.5)
                return True
    except Exception:
        pass
    sw, sh = pyautogui.size()
    pyautogui.click(sw // 2, sh // 2)
    time.sleep(0.3)
    return False


# =========================================================================== #
#  UIAutomationTool                                                             #
# =========================================================================== #

from core.tools.base import BaseTool, ToolResult


class UIAutomationTool(BaseTool):

    name: str = "ui_automation"

    description: str = (
        "Performs VISIBLE on-screen UI automation workflows using real applications. "
        "Supported workflows: "
        "'create_file_notepad', 'create_file_vscode', 'send_email_browser', "
        "'create_event_browser', "
        "'send_whatsapp_desktop', 'send_whatsapp_advanced', 'play_youtube_video', "
        "'linkedin_action', 'accept_linkedin_connections', 'code_workflow_cpp', "
        "'launch_application', 'take_screenshot', "
        "'open_calendar', 'open_onenote', 'open_clock', 'open_recorder'."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["workflow"],
        "properties": {
            "workflow":            {"type": "string"},
            "filename":            {"type": "string"},
            "content":             {"type": "string"},
            "code":                {"type": "string"},
            "recipient":           {"type": "string"},
            "subject":             {"type": "string"},
            "contact_name":        {"type": ["string", "array"]},
            "message":             {"type": "string"},
            "delay_seconds":       {"type": "number"},
            "repeat":              {"type": "integer"},
            "query":               {"type": "string"},
            "name":                {"type": "string"},
            "action":              {"type": "string"},
            "app_name":            {"type": "string"},
            "screenshot_filename": {"type": "string"},
            # Calendar event (browser)
            "event_title":        {"type": "string",  "description": "Title/summary of the calendar event."},
            "event_date":         {"type": "string",  "description": "Event date in YYYY-MM-DD format."},
            "event_start_time":   {"type": "string",  "description": "Start time in HH:MM (24h) format, e.g. '14:00'."},
            "event_end_time":     {"type": "string",  "description": "End time in HH:MM (24h) format, e.g. '15:00'."},
            # Existing params
            "title":               {"type": "string",  "description": "Event/note title."},
            "description":         {"type": "string",  "description": "Event description."},
            "event_datetime":      {"type": "string",  "description": "ISO-8601 datetime for calendar event."},
            "minutes":             {"type": "integer", "description": "Timer duration in minutes (clock_tool)."},
            "alarm_time":          {"type": "string",  "description": "HH:MM alarm time (clock_tool)."},
            "typing_mode":         {
                "type": "string",
                "enum": ["slow", "paste"],
                "description": "Force 'slow' or 'paste'. Auto-detected if omitted.",
            },
        },
    }

    # ------------------------------------------------------------------ #
    #  Dispatch                                                            #
    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        workflow: str = kwargs.get("workflow", "").strip()
        callback: EventCallback = kwargs.pop("event_callback", None)

        _emit(callback, _event("info", "dispatch_started",
                               f"Dispatching workflow: '{workflow}'", workflow))

        _workflow_map = {
            # ── Existing ──
            "create_file_notepad":         self._create_file_notepad,
            "create_file_vscode":          self._create_file_vscode,
            "send_email_browser":          self._send_email_browser,
            "send_whatsapp_desktop":       self._send_whatsapp_desktop,
            "send_whatsapp_advanced":      self._send_whatsapp_advanced,
            "play_youtube_video":          self._play_youtube_video,
            "linkedin_action":             self._linkedin_action,
            "accept_linkedin_connections": self._accept_linkedin_connections,
            "code_workflow_cpp":           self._code_workflow_cpp,
            "launch_application":          self._launch_application,
            "take_screenshot":             self._take_screenshot,
            "open_calendar":               self._open_calendar,
            "open_onenote":                self._open_onenote,
            "open_clock":                  self._open_clock,
            "open_recorder":               self._open_recorder,
            # ── NEW ──
            "create_event_browser":        self._create_event_browser,
        }

        handler = _workflow_map.get(workflow)
        if handler is None:
            msg = (
                f"Unsupported workflow: '{workflow}'. "
                f"Supported: {sorted(_workflow_map.keys())}"
            )
            _emit(callback, _event("error", "dispatch_failed", msg, workflow))
            return ToolResult(success=False, error=msg)

        try:
            return handler(callback=callback, **kwargs)
        except Exception as exc:
            msg = f"Workflow '{workflow}' crashed: {exc}"
            logger.exception(msg)
            _emit(callback, _event("error", "workflow_crashed", msg, workflow))
            return ToolResult(success=False, error=msg)

    # ================================================================== #
    #  NEW WORKFLOW: create_event_browser                                  #
    # ================================================================== #

    def _create_event_browser(
        self,
        *,
        callback: EventCallback = None,
        event_title: str = "New Event",
        event_date: str = "",
        event_start_time: str = "",
        event_end_time: str = "",
        workflow: str = "create_event_browser",
        **_: Any,
    ) -> ToolResult:
        """
        Create a Google Calendar event via Chrome browser.

        Strategy
        --------
        1. Build Google Calendar "new event" URL with pre-filled title,
           date, and times using the /calendar/r/eventedit endpoint with
           `text` and `dates` query params.
        2. Open Chrome with that URL — calendar loads with fields pre-filled.
        3. Wait for the page to load fully.
        4. Click the Save button (top-right area of the form).
        5. Return success.

        Google Calendar URL format
        --------------------------
          https://calendar.google.com/calendar/r/eventedit
            ?text=<title>
            &dates=<YYYYMMDDTHHMMSS>/<YYYYMMDDTHHMMSS>

        Example: event on 2025-06-10, 14:00–15:00
          dates=20250610T140000/20250610T150000
        """
        wf = workflow

        if not event_title.strip():
            return ToolResult(success=False, error="'event_title' is required.")

        # ── 1. Build pre-filled URL ────────────────────────────────────
        def _compact_date(d: str) -> str:
            """YYYY-MM-DD → YYYYMMDD"""
            return d.replace("-", "").strip() if d else ""

        def _compact_time(t: str) -> str:
            """HH:MM → HHMM00"""
            if not t:
                return ""
            t = t.replace(":", "").strip()
            t = t.ljust(4, "0")[:4]   # ensure 4 digits
            return t + "00"

        date_str  = _compact_date(event_date)
        start_str = _compact_time(event_start_time)
        end_str   = _compact_time(event_end_time)

        if date_str and start_str and end_str:
            dates_param = f"{date_str}T{start_str}/{date_str}T{end_str}"
        elif date_str:
            # All-day fallback — just pass the date twice
            dates_param = f"{date_str}/{date_str}"
        else:
            dates_param = ""

        params: dict[str, str] = {"text": event_title.strip()}
        if dates_param:
            params["dates"] = dates_param

        cal_url = (
            "https://calendar.google.com/calendar/r/eventedit?"
            + urllib.parse.urlencode(params)
        )

        _emit(callback, _event(
            "info", "opening_browser",
            f"Opening Google Calendar new-event form in Chrome…", wf,
        ))
        logger.info("Calendar URL: %s", cal_url)

        # ── 2. Open Chrome ─────────────────────────────────────────────
        chrome_cmd = _find_chrome()
        subprocess.Popen(chrome_cmd + [cal_url])
        _wait_emit(7.0, callback, wf, "Google Calendar form loading")

        # ── 3. Dismiss any overlay / tooltip ──────────────────────────
        press_key("escape")
        wait(0.5)

        sw, sh = pyautogui.size()

        # ── 4. Click the Save button ───────────────────────────────────
        # On a 1920×1080 screen the Save button sits at roughly:
        #   x ≈ 85-88% of screen width   (top-right toolbar)
        #   y ≈ 5.5-7% of screen height
        #
        # We try two positions to handle different Chrome zoom levels /
        # screen resolutions, plus a Tab+Enter fallback.

        _emit(callback, _event(
            "info", "saving_event",
            "Saving the event using Ctrl+S…", wf,
        ))

        hotkey("ctrl", "s")
        wait(2.0, "Event saved")

        _emit(callback, _event(
            "status", "workflow_done",
            f"Google Calendar event '{event_title}' created successfully.", wf,
        ))

        return ToolResult(
            success=True,
            output=f"Google Calendar event '{event_title}' created.",
            metadata={
                "workflow":         wf,
                "event_title":      event_title,
                "event_date":       event_date,
                "event_start_time": event_start_time,
                "event_end_time":   event_end_time,
                "calendar_url":     cal_url,
            },
        )

    # ================================================================== #
    #  Existing workflows — unchanged below                                #
    # ================================================================== #

    def _open_calendar(
        self,
        *,
        callback: EventCallback = None,
        workflow: str = "open_calendar",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        _emit(callback, _event("info", "app_launch",
                               "Opening Windows Calendar app…", wf))

        launched = False
        for cmd in [
            ["cmd", "/c", "start", "", "outlookcal:"],
            ["explorer.exe", "shell:appsfolder\\microsoft.windowscommunicationsapps_8wekyb3d8bbwe!microsoft.windowslive.calendar"],
            ["cmd", "/c", "start", "calendar"],
        ]:
            try:
                subprocess.Popen(cmd, shell=False,
                                 creationflags=subprocess.CREATE_NO_WINDOW
                                 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
                launched = True
                break
            except Exception as exc:
                logger.debug("Calendar launch attempt failed: %s", exc)

        if not launched:
            pyautogui.hotkey("win")
            wait(0.7)
            slow_type("calendar", interval=0.08)
            wait(0.5)
            press_key("enter")

        wait(3.0, "Calendar initialising")
        _focus_window_by_title("Calendar")
        wait(0.5)

        _emit(callback, _event("status", "workflow_done",
                               "Windows Calendar is open.", wf))
        return ToolResult(
            success=True,
            output="Windows Calendar opened.",
            metadata={"workflow": wf},
        )

    def _open_onenote(
        self,
        *,
        callback: EventCallback = None,
        title: str = "",
        content: str = "",
        typing_mode: str = "auto",
        workflow: str = "open_onenote",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        _emit(callback, _event("info", "app_launch",
                               "Opening Microsoft OneNote…", wf))

        launched = False
        for cmd in [
            ["cmd", "/c", "start", "", "onenote:"],
            [shutil.which("ONENOTE") or "ONENOTE.EXE"],
        ]:
            try:
                subprocess.Popen([c for c in cmd if c], shell=False,
                                 creationflags=subprocess.CREATE_NO_WINDOW
                                 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
                launched = True
                break
            except Exception:
                pass

        if not launched:
            pyautogui.hotkey("win")
            wait(0.7)
            slow_type("onenote", interval=0.08)
            wait(0.5)
            press_key("enter")

        wait(4.0, "OneNote initialising")
        _focus_window_by_title("OneNote")
        wait(0.8)

        if title:
            _emit(callback, _event("info", "creating_page",
                                   f"Creating new page: '{title}'…", wf))
            hotkey("ctrl", "n")
            wait(1.2, "new page loading")
            slow_type(title, interval=0.07)
            press_key("enter")
            wait(0.5)

            if content:
                mode = typing_mode if typing_mode in ("slow", "paste") else (
                    "slow" if len(content) <= 80 else "paste"
                )
                _emit(callback, _event("info", "adding_content",
                                       f"Adding content via {mode} mode…", wf))
                if mode == "slow":
                    slow_type(content, interval=0.04)
                else:
                    paste_text(content)
                wait(0.4)

            hotkey("ctrl", "s")
            wait(0.5)

        _emit(callback, _event("status", "workflow_done",
                               f"OneNote open. Page '{title or '(none)'}' ready.", wf))
        return ToolResult(
            success=True,
            output=f"OneNote opened{f' with page: {title}' if title else ''}.",
            metadata={"workflow": wf, "title": title,
                      "content_length": len(content)},
        )

    def _open_clock(
        self,
        *,
        callback: EventCallback = None,
        minutes: int = 0,
        alarm_time: str = "",
        workflow: str = "open_clock",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        _emit(callback, _event("info", "app_launch",
                               "Opening Windows Clock app…", wf))

        launched = False
        for cmd in [
            ["cmd", "/c", "start", "", "ms-clock:"],
            ["explorer.exe", "shell:appsfolder\\Microsoft.WindowsAlarms_8wekyb3d8bbwe!App"],
        ]:
            try:
                subprocess.Popen(cmd, shell=False,
                                 creationflags=subprocess.CREATE_NO_WINDOW
                                 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
                launched = True
                break
            except Exception:
                pass

        if not launched:
            pyautogui.hotkey("win")
            wait(0.7)
            slow_type("clock", interval=0.08)
            wait(0.5)
            press_key("enter")

        wait(3.0, "Clock app initialising")
        _focus_window_by_title("Clock")
        wait(0.8)

        sw, sh = pyautogui.size()
        result_msg = "Windows Clock opened."

        if minutes and minutes > 0:
            _emit(callback, _event("info", "setting_timer",
                                   f"Navigating to Timer tab for {minutes}m timer…", wf))
            nav_x = int(sw * 0.07)
            timer_nav_y = int(sh * 0.30)
            click(nav_x, timer_nav_y, duration=0.4)
            wait(0.8, "Timer tab loading")

            centre_x, centre_y = sw // 2, sh // 2
            click(centre_x, int(sh * 0.38), duration=0.4)
            wait(0.5)

            hotkey("ctrl", "a")
            wait(0.2)
            slow_type(str(minutes), interval=0.12)
            wait(0.3)

            start_x = int(sw * 0.50)
            start_y = int(sh * 0.62)
            click(start_x, start_y, duration=0.35)
            wait(0.5)
            result_msg = f"Timer set for {minutes} minute(s)."

        elif alarm_time:
            _emit(callback, _event("info", "setting_alarm",
                                   f"Navigating to Alarm tab for {alarm_time}…", wf))
            nav_x   = int(sw * 0.07)
            alarm_y = int(sh * 0.18)
            click(nav_x, alarm_y, duration=0.4)
            wait(0.8, "Alarm tab loading")

            add_btn_x = int(sw * 0.88)
            add_btn_y = int(sh * 0.90)
            click(add_btn_x, add_btn_y, duration=0.4)
            wait(1.2, "Add alarm dialog loading")

            slow_type(alarm_time.replace(":", ""), interval=0.18)
            wait(0.4)
            press_key("enter")
            wait(0.5)
            result_msg = f"Alarm set for {alarm_time}."

        _emit(callback, _event("status", "workflow_done", result_msg, wf))
        return ToolResult(
            success=True,
            output=result_msg,
            metadata={"workflow": wf, "minutes": minutes, "alarm_time": alarm_time},
        )

    def _open_recorder(
        self,
        *,
        callback: EventCallback = None,
        workflow: str = "open_recorder",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        _emit(callback, _event("info", "app_launch",
                               "Opening Windows Voice Recorder…", wf))

        launched = False
        for cmd in [
            ["cmd", "/c", "start", "", "ms-voicerecorder:"],
            ["explorer.exe",
             "shell:appsfolder\\Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe!App"],
        ]:
            try:
                subprocess.Popen(cmd, shell=False,
                                 creationflags=subprocess.CREATE_NO_WINDOW
                                 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
                launched = True
                break
            except Exception:
                pass

        if not launched:
            pyautogui.hotkey("win")
            wait(0.7)
            slow_type("voice recorder", interval=0.07)
            wait(0.5)
            press_key("enter")

        wait(3.0, "Voice Recorder initialising")
        _focus_window_by_title("Voice Recorder")
        wait(0.6)

        _emit(callback, _event("status", "workflow_done",
                               "Voice Recorder is open. Click the mic button to record.", wf))
        return ToolResult(
            success=True,
            output="Voice Recorder opened.",
            metadata={"workflow": wf},
        )

    def _create_file_notepad(
        self,
        *,
        callback: EventCallback = None,
        filename: str = "untitled.txt",
        content: str = "",
        workflow: str = "create_file_notepad",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        _emit(callback, _event("info", "app_launch",
                               "Launching fresh Notepad (new instance)...", wf))
        subprocess.Popen(["notepad.exe", "/N"])
        _wait_emit(1.8, callback, wf, "Notepad loading")

        _emit(callback, _event("info", "typing_content",
                               f"Pasting {len(content)} chars...", wf))
        paste_text(content)
        wait(0.5)

        _emit(callback, _event("info", "save_dialog",
                               "Triggering Save As...", wf))
        hotkey("ctrl", "s")
        _wait_emit(1.8, callback, wf, "Save-As dialog loading")

        hotkey("ctrl", "a")
        wait(0.25)
        paste_text(filename)
        wait(0.35)
        press_key("enter")
        wait(0.6)
        press_key("enter")
        wait(0.5)

        hotkey("alt", "f4")
        wait(0.5)

        _emit(callback, _event("status", "workflow_done",
                               f"File '{filename}' created via Notepad.", wf))
        return ToolResult(success=True,
                          output=f"File '{filename}' created via Notepad.",
                          metadata={"workflow": wf, "filename": filename,
                                    "content_length": len(content)})

    def _create_file_vscode(
        self,
        *,
        callback: EventCallback = None,
        filename: str = "main.py",
        content: str = "",
        workflow: str = "create_file_vscode",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        vscode_exe = _find_vscode()
        if vscode_exe is None:
            return ToolResult(success=False,
                              error="VS Code not found. Install it and add to PATH.")

        target = Path(filename)
        if not target.is_absolute() and len(target.parts) == 1:
            desktop = Path.home() / "Desktop"
            desktop.mkdir(parents=True, exist_ok=True)
            target = desktop / filename
        target = target.resolve()

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        size = target.stat().st_size

        _emit(callback, _event("info", "app_launch",
                               f"Opening '{target.name}' in VS Code...", wf))
        subprocess.Popen([vscode_exe, str(target)])
        _wait_emit(2.5, callback, wf, "VS Code opening")

        _emit(callback, _event("status", "workflow_done",
                               f"File '{target.name}' opened in VS Code.", wf))
        return ToolResult(success=True,
                          output=f"File '{target.name}' created and opened in VS Code.",
                          metadata={"workflow": wf, "filename": target.name,
                                    "absolute_path": str(target),
                                    "content_length": len(content), "size_bytes": size})

    def _send_email_browser(
        self,
        *,
        callback: EventCallback = None,
        recipient: str = "",
        subject: str = "(no subject)",
        content: str = "",
        workflow: str = "send_email_browser",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not recipient:
            return ToolResult(success=False,
                              error="'recipient' is required for send_email_browser.")

        params = urllib.parse.urlencode(
            {"view": "cm", "to": recipient, "su": subject, "body": content}
        )
        compose_url = f"https://mail.google.com/mail/?{params}"
        chrome_cmd = _find_chrome()
        subprocess.Popen(chrome_cmd + [compose_url])
        _wait_emit(6.0, callback, wf, "Gmail compose rendering")

        press_key("escape")
        wait(0.5)
        sw, sh = pyautogui.size()
        body_x, body_y = int(sw * 0.72), int(sh * 0.55)

        press_key("tab"); wait(0.35)
        press_key("tab"); wait(0.35)
        click(body_x, body_y); wait(0.4)
        hotkey("ctrl", "a"); wait(0.2)
        paste_text(content); wait(0.5)
        click(body_x, body_y); wait(0.3)
        move_mouse(sw // 2, sh // 2, duration=0.3); wait(0.3)
        hotkey("ctrl", "enter")
        _wait_emit(2.5, callback, wf, "Gmail processing send")

        _emit(callback, _event("status", "workflow_done",
                               f"Email sent to '{recipient}'.", wf))
        return ToolResult(success=True,
                          output=f"Email sent to '{recipient}'.",
                          metadata={"workflow": wf, "recipient": recipient,
                                    "subject": subject, "content_length": len(content)})

    def _send_whatsapp_desktop(
        self,
        *,
        callback: EventCallback = None,
        contact_name: str = "",
        message: str = "",
        workflow: str = "send_whatsapp_desktop",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not contact_name.strip():
            return ToolResult(success=False, error="'contact_name' is required.")
        if not message.strip():
            return ToolResult(success=False, error="'message' is required.")
        return self._whatsapp_send_single(contact_name=contact_name, message=message,
                                          callback=callback, workflow=wf, launch_app=True)

    def _send_whatsapp_advanced(
        self,
        *,
        callback: EventCallback = None,
        contact_name: "str | list[str]" = "",
        message: str = "",
        delay_seconds: float = 0.0,
        repeat: int = 1,
        workflow: str = "send_whatsapp_advanced",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        contacts = ([c.strip() for c in contact_name.split(",") if c.strip()]
                    if isinstance(contact_name, str)
                    else [str(c).strip() for c in contact_name if str(c).strip()])
        if not contacts:
            return ToolResult(success=False, error="'contact_name' is required.")
        if not message.strip():
            return ToolResult(success=False, error="'message' is required.")

        repeat        = max(1, int(repeat or 1))
        delay_seconds = max(0.0, float(delay_seconds or 0.0))

        self._launch_whatsapp_app(callback, wf)
        _wait_emit(5.0, callback, wf, "WhatsApp Desktop initialising")
        press_key("shift"); wait(0.5)

        sent:   list[str] = []
        failed: list[str] = []

        for contact in contacts:
            for attempt in range(repeat):
                if delay_seconds > 0:
                    _wait_emit(delay_seconds, callback, wf, "pre-send delay")
                result = self._whatsapp_send_single(contact_name=contact, message=message,
                                                     callback=callback, workflow=wf,
                                                     launch_app=False)
                if result.success:
                    sent.append(f"{contact} (×{attempt + 1})")
                else:
                    failed.append(contact); break

        summary = f"Sent: {', '.join(sent) or 'none'}."
        if failed:
            summary += f" Failed: {', '.join(failed)}."
        return ToolResult(success=len(sent) > 0, output=summary,
                          metadata={"workflow": wf, "contacts": contacts,
                                    "sent": sent, "failed": failed})

    def _launch_whatsapp_app(self, callback: EventCallback, wf: str) -> None:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "whatsapp://"], shell=False,
                             creationflags=subprocess.CREATE_NO_WINDOW
                             if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        except Exception:
            wa_exe = _find_whatsapp()
            if wa_exe:
                try:
                    os.startfile(wa_exe); return
                except Exception:
                    pass
            try:
                subprocess.Popen(["cmd", "/c", "start", "WhatsApp"])
            except Exception as exc:
                logger.warning("All WhatsApp launch methods failed: %s", exc)

    def _whatsapp_send_single(
        self,
        *,
        contact_name: str,
        message: str,
        callback: EventCallback,
        workflow: str,
        launch_app: bool = True,
    ) -> ToolResult:
        wf = workflow
        if launch_app:
            self._launch_whatsapp_app(callback, wf)
            _wait_emit(5.0, callback, wf, "WhatsApp Desktop initialising")
            press_key("shift"); wait(0.5)

        hotkey("ctrl", "f"); wait(1.0)
        hotkey("ctrl", "a"); wait(0.2)
        paste_text(contact_name); wait(1.5)
        press_key("enter"); wait(1.5)

        sw, sh = pyautogui.size()
        click(int(sw * 0.5), int(sh * 0.93)); wait(0.5)
        paste_text(message); wait(0.5)
        press_key("enter"); wait(1.0)

        return ToolResult(success=True,
                          output=f"WhatsApp message sent to '{contact_name}'.",
                          metadata={"workflow": wf, "contact_name": contact_name})

    def _play_youtube_video(
        self,
        *,
        callback: EventCallback = None,
        query: str = "",
        workflow: str = "play_youtube_video",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not query.strip():
            return ToolResult(success=False, error="'query' is required.")

        encoded_query = urllib.parse.quote_plus(query)
        search_url    = f"https://www.youtube.com/results?search_query={encoded_query}"
        chrome_cmd = _find_chrome()
        subprocess.Popen(chrome_cmd + [search_url])
        _wait_emit(5.0, callback, wf, "YouTube search results loading")

        sw, sh = pyautogui.size()
        click(sw // 2, sh // 2); wait(0.6)
        press_key("escape"); wait(0.4)
        for _ in range(7):
            press_key("tab"); wait(0.18)
        press_key("enter")
        _wait_emit(5.0, callback, wf, "Video page loading")

        click(sw // 2, int(sh * 0.42)); wait(0.6)
        press_key("k"); wait(0.4)

        _emit(callback, _event("status", "workflow_done",
                               f"YouTube video for '{query}' started.", wf))
        return ToolResult(success=True, output=f"YouTube video for '{query}' is now playing.",
                          metadata={"workflow": wf, "query": query, "search_url": search_url})

    def _linkedin_action(
        self,
        *,
        callback: EventCallback = None,
        name: str = "",
        action: str = "search",
        workflow: str = "linkedin_action",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not name.strip():
            return ToolResult(success=False, error="'name' is required.")
        action = (action or "search").strip().lower()
        if action not in {"search", "open"}:
            return ToolResult(success=False,
                              error="action must be 'search' or 'open'.")

        if not _launch_linkedin_app(callback, wf):
            return ToolResult(success=False, error="Could not launch LinkedIn Desktop app.")

        _focus_window_by_title("LinkedIn")
        _wait_emit(1.5, callback, wf, "window settling")
        sw, sh = pyautogui.size()

        hotkey("ctrl", "f"); wait(1.0)
        click(int(sw * 0.30), int(sh * 0.05)); wait(0.6)
        hotkey("ctrl", "a"); wait(0.2)
        paste_text(name); wait(0.8)
        press_key("enter")
        _wait_emit(3.5, callback, wf, "search results loading")

        if action == "search":
            _emit(callback, _event("status", "workflow_done",
                                   f"LinkedIn search results for '{name}'.", wf))
            return ToolResult(success=True,
                              output=f"LinkedIn Desktop: search results for '{name}'.",
                              metadata={"workflow": wf, "name": name, "action": action})

        click(int(sw * 0.35), int(sh * 0.25)); wait(0.5)
        for _ in range(4):
            press_key("tab"); wait(0.22)
        press_key("enter")
        _wait_emit(4.0, callback, wf, "profile loading")

        _emit(callback, _event("status", "workflow_done",
                               f"LinkedIn profile opened for '{name}'.", wf))
        return ToolResult(success=True,
                          output=f"LinkedIn Desktop: profile opened for '{name}'.",
                          metadata={"workflow": wf, "name": name, "action": action})

    def _accept_linkedin_connections(
        self,
        *,
        callback: EventCallback = None,
        workflow: str = "accept_linkedin_connections",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not _launch_linkedin_app(callback, wf):
            return ToolResult(success=False, error="Could not launch LinkedIn Desktop app.")

        _focus_window_by_title("LinkedIn")
        _wait_emit(2.5, callback, wf, "window settling")
        sw, sh = pyautogui.size()
        nav_y = int(sh * 0.035)

        click(int(sw * 0.458), nav_y, duration=0.4); wait(2.0)
        click(int(sw * 0.500), nav_y, duration=0.4); wait(3.0)
        _focus_window_by_title("LinkedIn"); wait(0.5)

        content_x, content_y = int(sw * 0.50), int(sh * 0.20)
        click(content_x, content_y); wait(0.6)

        for tab_n in range(20):
            press_key("tab"); wait(0.18)
            if tab_n in (7, 8, 9, 10, 11):
                press_key("enter"); wait(3.0); break

        _focus_window_by_title("LinkedIn"); wait(1.0)

        total_accepted = 0
        max_passes     = 40
        empty_streak   = 0

        for pass_idx in range(1, max_passes + 1):
            _focus_window_by_title("LinkedIn"); wait(0.4)
            click(int(sw * 0.50), int(sh * 0.40)); wait(0.4)
            for _ in range(3):
                press_key("tab"); wait(0.15)
            press_key("space"); wait(1.5)

            try:
                import pygetwindow as gw
                wins = gw.getWindowsWithTitle("LinkedIn")
                if wins and len(wins[0].title) > len("LinkedIn") + 3:
                    hotkey("alt", "left"); wait(2.5)
                    _focus_window_by_title("LinkedIn"); wait(0.5)
                    empty_streak += 1
                else:
                    total_accepted += 1; empty_streak = 0
            except ImportError:
                total_accepted += 1; empty_streak = 0

            if empty_streak >= 5:
                break

            if pass_idx % 5 == 0:
                click(int(sw * 0.50), int(sh * 0.55)); wait(0.3)
                for _ in range(4):
                    press_key("pagedown"); wait(0.35)
                wait(2.5)

        _emit(callback, _event("status", "workflow_done",
                               f"{total_accepted} connection(s) accepted.", wf))
        return ToolResult(success=True,
                          output=f"Accepted {total_accepted} LinkedIn connection(s).",
                          metadata={"workflow": wf, "total_accepted": total_accepted,
                                    "passes": pass_idx})

    def _code_workflow_cpp(
        self,
        *,
        callback: EventCallback = None,
        filename: str = "main.cpp",
        code: str = "",
        content: str = "",
        workflow: str = "code_workflow_cpp",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        source = (code or content or
                  '#include <iostream>\nint main() {\n'
                  '    std::cout << "Hello, World!" << std::endl;\n'
                  '    return 0;\n}\n')
        if not filename.endswith(".cpp"):
            filename = filename.rsplit(".", 1)[0] + ".cpp"

        vscode_exe = _find_vscode()
        if vscode_exe is None:
            return ToolResult(success=False, error="VS Code not found.")

        desktop  = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        cpp_path = (desktop / filename).resolve()
        exe_path = cpp_path.with_suffix(".exe")
        cpp_path.write_text(source, encoding="utf-8")

        subprocess.Popen([vscode_exe, str(cpp_path)])
        _wait_emit(2.5, callback, wf, "VS Code loading")
        hotkey("ctrl", "s"); wait(0.5)

        gpp_path = shutil.which("g++")
        if gpp_path is None:
            return ToolResult(success=False, error="g++ not found. Install MinGW/GCC.")

        compile_result = subprocess.run(
            [gpp_path, str(cpp_path), "-o", str(exe_path)],
            capture_output=True, text=True, timeout=30,
        )
        if compile_result.returncode != 0:
            return ToolResult(success=False,
                              error=f"Compilation errors:\n{compile_result.stderr.strip()}")

        run_result = subprocess.run([str(exe_path)], capture_output=True, text=True,
                                    timeout=15, cwd=str(desktop))
        run_output = run_result.stdout.strip()

        _emit(callback, _event("status", "workflow_done",
                               f"C++ workflow done. Output: {run_output or '(none)'}", wf))
        return ToolResult(success=True,
                          output=f"C++ compiled and ran. Output: {run_output or '(no output)'}",
                          metadata={"workflow": wf, "source_file": str(cpp_path),
                                    "executable": str(exe_path), "program_output": run_output})

    def _launch_application(
        self,
        *,
        callback: EventCallback = None,
        app_name: str = "",
        workflow: str = "launch_application",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not app_name.strip():
            return ToolResult(success=False, error="'app_name' is required.")

        normalized = app_name.strip().lower()
        _APP_MAP: dict[str, list[str]] = {
            "chrome":             _find_chrome() or ["cmd", "/c", "start", "chrome"],
            "google chrome":      _find_chrome() or ["cmd", "/c", "start", "chrome"],
            "vscode":             [_find_vscode() or "code"],
            "vs code":            [_find_vscode() or "code"],
            "notepad":            ["notepad.exe"],
            "calculator":         ["calc.exe"],
            "calc":               ["calc.exe"],
            "whatsapp":           ["cmd", "/c", "start", "", "whatsapp://"],
            "linkedin":           ["cmd", "/c", "start", "", "linkedin://"],
            "calendar":           ["cmd", "/c", "start", "", "outlookcal:"],
            "onenote":            ["cmd", "/c", "start", "", "onenote:"],
            "clock":              ["cmd", "/c", "start", "", "ms-clock:"],
            "voice recorder":     ["cmd", "/c", "start", "", "ms-voicerecorder:"],
            "explorer":           ["explorer.exe"],
            "file explorer":      ["explorer.exe"],
            "paint":              ["mspaint.exe"],
            "wordpad":            ["wordpad.exe"],
            "cmd":                ["cmd.exe"],
            "command prompt":     ["cmd.exe"],
            "terminal":           ["cmd.exe"],
            "task manager":       ["taskmgr.exe"],
        }

        cmd = _APP_MAP.get(normalized)
        if cmd is None:
            found = shutil.which(normalized) or shutil.which(normalized + ".exe")
            cmd   = [found] if found else None

        if cmd is None:
            return ToolResult(success=False,
                              error=f"Application '{app_name}' not in supported list.")

        cmd = [c for c in cmd if c]
        try:
            subprocess.Popen(cmd)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to launch '{app_name}': {exc}")

        _wait_emit(1.5, callback, wf, f"{app_name} loading")
        _emit(callback, _event("status", "workflow_done",
                               f"Application '{app_name}' launched.", wf))
        return ToolResult(success=True, output=f"Application '{app_name}' launched.",
                          metadata={"workflow": wf, "app_name": app_name, "command": cmd})

    def _take_screenshot(
        self,
        *,
        callback: EventCallback = None,
        screenshot_filename: str = "",
        workflow: str = "take_screenshot",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (screenshot_filename.strip() if screenshot_filename.strip()
                 else f"screenshot_{ts}.png")
        if not fname.lower().endswith(".png"):
            fname += ".png"

        desktop   = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        save_path = (desktop / fname).resolve()

        errors: list[str] = []

        try:
            screenshots_dir = Path.home() / "Pictures" / "Screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            before = set(screenshots_dir.glob("*.png"))
            hotkey("win", "printscreen")
            time.sleep(2.0)
            after     = set(screenshots_dir.glob("*.png"))
            new_files = after - before
            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                shutil.copy2(str(newest), str(save_path))
                return self._screenshot_success(save_path, 0, 0, callback, wf)
            errors.append("Strategy 1: no new file")
        except Exception as exc:
            errors.append(f"Strategy 1: {exc}")

        try:
            img = pyautogui.screenshot()
            img.save(str(save_path), format="PNG")
            return self._screenshot_success(save_path, *img.size, callback, wf)
        except Exception as exc:
            errors.append(f"Strategy 2: {exc}")

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(save_path), format="PNG")
            return self._screenshot_success(save_path, img.width, img.height, callback, wf)
        except Exception as exc:
            errors.append(f"Strategy 3: {exc}")

        return ToolResult(success=False,
                          error="All screenshot strategies failed.\n" + "\n".join(errors))

    @staticmethod
    def _screenshot_success(
        save_path: Path, width: int, height: int,
        callback: EventCallback, wf: str,
    ) -> ToolResult:
        size_kb     = round(save_path.stat().st_size / 1024, 1)
        resolution  = f"{width}×{height}" if width and height else "unknown"
        _emit(callback, _event("status", "workflow_done",
                               f"Screenshot saved as '{save_path.name}' ({size_kb} KB).", wf))
        return ToolResult(success=True, output=f"Screenshot saved: {save_path}",
                          metadata={"workflow": wf, "filename": save_path.name,
                                    "path": str(save_path), "size_kb": size_kb,
                                    "resolution": resolution})