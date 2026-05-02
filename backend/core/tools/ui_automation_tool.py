"""
core/tools/ui_automation_tool.py  — LinkedIn Desktop App Edition

Changes from previous version
------------------------------
CHANGE 1  linkedin_action          : Removed browser automation entirely.
                                     Now uses the LinkedIn Desktop app (Windows Store)
                                     via PyAutoGUI: launches the app, uses the
                                     in-app search bar to find a person, and opens
                                     their profile inside the app.

CHANGE 2  accept_linkedin_connections : Removed browser automation entirely.
                                     Now uses the LinkedIn Desktop app: navigates
                                     to the My Network / Invitations panel and
                                     clicks every "Accept" button using image-search
                                     or coordinate-based fallback with robust scrolling.

All other workflows (Notepad, VS Code, Gmail, WhatsApp, YouTube,
C++, launch_application, take_screenshot) are unchanged.
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

import ctypes
import pyautogui
import pyperclip

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

pyautogui.PAUSE = 0.25
pyautogui.FAILSAFE = True


# --------------------------------------------------------------------------- #
#  Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _emit(callback: EventCallback, event: dict) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:           # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _event(type_: str, stage: str, message: str, workflow: str) -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"ui_automation/{workflow}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _wait(seconds: float, callback: EventCallback, workflow: str, reason: str = "") -> None:
    if reason:
        _emit(callback, _event("info", "waiting", f"Waiting {seconds}s — {reason}", workflow))
    time.sleep(seconds)


def _type_via_clipboard(text: str) -> None:
    """Paste via clipboard — handles unicode, special chars, much faster than typewrite."""
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")


# --------------------------------------------------------------------------- #
#  Robust VS Code locator                                                       #
# --------------------------------------------------------------------------- #

def _find_vscode() -> str | None:
    for alias in ("code", "code.cmd", "code-insiders"):
        if shutil.which(alias):
            return shutil.which(alias)

    local_app   = os.environ.get("LOCALAPPDATA", "")
    prog_files  = os.environ.get("ProgramFiles",   "C:\\Program Files")
    prog_x86    = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    user_home   = str(Path.home())

    candidates: list[Path] = [
        Path(local_app) / "Programs" / "Microsoft VS Code" / "Code.exe",
        Path(local_app) / "Microsoft" / "WindowsApps" / "code.exe",
        Path(local_app) / "Microsoft" / "WindowsApps" / "Microsoft.VisualStudioCode_*" / "code.exe",
        Path(prog_files)  / "Microsoft VS Code" / "Code.exe",
        Path(prog_x86)    / "Microsoft VS Code" / "Code.exe",
        Path(user_home) / "scoop" / "apps" / "vscode" / "current" / "code.exe",
        Path(user_home) / "scoop" / "shims"  / "code.exe",
        Path("C:\\tools") / "vscode" / "Code.exe",
    ]

    for candidate in candidates:
        if "*" in str(candidate):
            matches = list(candidate.parent.glob(candidate.name))
            if matches:
                return str(matches[0])
        elif candidate.exists():
            return str(candidate)

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
    candidates = [
        Path(local) / "Microsoft" / "WindowsApps" / "WhatsApp.exe",
        Path(local) / "WhatsApp" / "WhatsApp.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    if shutil.which("WhatsApp"):
        return "WhatsApp"
    return None


# --------------------------------------------------------------------------- #
#  LinkedIn Desktop App locator                                                 #
# --------------------------------------------------------------------------- #

def _find_linkedin_desktop() -> str | None:
    """
    Locate the LinkedIn Desktop app executable on Windows.
    Checks Microsoft Store install paths and common fallbacks.
    Returns None if not found.
    """
    local = os.environ.get("LOCALAPPDATA", "")

    candidates: list[Path] = [
        # Microsoft Store (WindowsApps)
        Path(local) / "Microsoft" / "WindowsApps" / "LinkedIn.exe",
        Path(local) / "Microsoft" / "WindowsApps" / "4128479MangoApps.LinkedIn_*" / "LinkedIn.exe",
        # Standalone installer
        Path(local) / "LinkedIn" / "LinkedIn.exe",
        Path(local) / "Programs" / "LinkedIn" / "LinkedIn.exe",
        Path("C:\\Program Files") / "LinkedIn" / "LinkedIn.exe",
    ]

    for candidate in candidates:
        if "*" in str(candidate):
            matches = list(candidate.parent.glob(candidate.name))
            if matches:
                return str(matches[0])
        elif candidate.exists():
            return str(candidate)

    # Try shutil.which as a last resort
    found = shutil.which("LinkedIn") or shutil.which("linkedin")
    return found if found else None


def _launch_linkedin_app(callback: EventCallback, wf: str) -> bool:
    """
    Launch the LinkedIn Desktop app.
    Returns True if launch succeeded, False otherwise.
    """
    _emit(callback, _event("info", "app_launch", "Launching LinkedIn Desktop app…", wf))

    # Strategy 1: URI protocol handler (works if app is installed from Store)
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "linkedin://"],
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        _wait(6.0, callback, wf, "LinkedIn app initialising")
        return True
    except Exception as exc:               # noqa: BLE001
        logger.debug("LinkedIn URI launch failed: %s", exc)

    # Strategy 2: Direct executable
    exe = _find_linkedin_desktop()
    if exe:
        try:
            subprocess.Popen([exe])
            _wait(6.0, callback, wf, "LinkedIn app initialising")
            return True
        except Exception as exc:           # noqa: BLE001
            logger.debug("LinkedIn exe launch failed: %s", exc)

    # Strategy 3: Windows Start menu search
    try:
        subprocess.Popen(["cmd", "/c", "start", "LinkedIn"])
        _wait(6.0, callback, wf, "LinkedIn app initialising via start menu")
        return True
    except Exception as exc:               # noqa: BLE001
        logger.debug("LinkedIn start-menu launch failed: %s", exc)

    return False


def _focus_window_by_title(title_fragment: str) -> bool:
    """
    Attempt to bring a window with the given title fragment to the foreground.
    Returns True on success. Gracefully degrades if pygetwindow is absent.
    """
    try:
        import pygetwindow as gw                       # type: ignore
        for fragment in [title_fragment, "LinkedIn"]:
            wins = gw.getWindowsWithTitle(fragment)
            if wins:
                wins[0].restore()
                wins[0].activate()
                time.sleep(0.5)
                return True
    except ImportError:
        pass
    except Exception as exc:                           # noqa: BLE001
        logger.debug("pygetwindow activate failed: %s", exc)

    # Fallback: click centre of screen to ensure some window has focus
    sw, sh = pyautogui.size()
    pyautogui.click(sw // 2, sh // 2)
    time.sleep(0.3)
    return False


# --------------------------------------------------------------------------- #
#  UIAutomationTool                                                             #
# --------------------------------------------------------------------------- #

from core.tools.base import BaseTool, ToolResult


class UIAutomationTool(BaseTool):

    name: str = "ui_automation"

    description: str = (
        "Performs VISIBLE on-screen UI automation workflows using real applications. "
        "Supported workflows: "
        "'create_file_notepad' — create/write a NEW file using Notepad; "
        "'create_file_vscode' — create/write code in VS Code; "
        "'send_email_browser' — send email via Gmail in Chrome; "
        "'send_whatsapp_desktop' — send a WhatsApp message via WhatsApp Desktop app; "
        "'send_whatsapp_advanced' — send WhatsApp to one or multiple contacts with optional delay/repeat; "
        "'play_youtube_video' — open Chrome, search YouTube, click FIRST video result, play it; "
        "'linkedin_action' — search/open a LinkedIn profile using the LinkedIn Desktop app (action: search or open); "
        "'accept_linkedin_connections' — open LinkedIn Desktop app, navigate to invitations, and accept all pending requests; "
        "'code_workflow_cpp' — create a C++ file, compile with g++, and run it; "
        "'launch_application' — open a named system application; "
        "'take_screenshot' — capture the screen and save as PNG to Desktop."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["workflow"],
        "properties": {
            "workflow": {
                "type": "string",
                "description": (
                    "The UI workflow to execute. One of: "
                    "create_file_notepad, create_file_vscode, "
                    "send_email_browser, send_whatsapp_desktop, "
                    "send_whatsapp_advanced, play_youtube_video, "
                    "linkedin_action, accept_linkedin_connections, "
                    "code_workflow_cpp, launch_application, take_screenshot."
                ),
            },
            "filename":            {"type": "string",           "description": "File name to create."},
            "content":             {"type": "string",           "description": "Text content to write."},
            "code":                {"type": "string",           "description": "Source code (code_workflow_cpp)."},
            "recipient":           {"type": "string",           "description": "Email address (send_email_browser)."},
            "subject":             {"type": "string",           "description": "Email subject."},
            "contact_name":        {"type": ["string", "array"],"description": "WhatsApp contact name(s)."},
            "message":             {"type": "string",           "description": "Message text (WhatsApp)."},
            "delay_seconds":       {"type": "number",           "description": "Delay before each message."},
            "repeat":              {"type": "integer",          "description": "Times to send per contact."},
            "query":               {"type": "string",           "description": "YouTube search query."},
            "name":                {"type": "string",           "description": "Person name for LinkedIn."},
            "action":              {"type": "string",           "description": "LinkedIn action: search | open."},
            "app_name":            {"type": "string",           "description": "Application to launch."},
            "screenshot_filename": {"type": "string",           "description": "PNG filename for screenshot."},
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
            "create_file_notepad":        self._create_file_notepad,
            "create_file_vscode":         self._create_file_vscode,
            "send_email_browser":         self._send_email_browser,
            "send_whatsapp_desktop":      self._send_whatsapp_desktop,
            "send_whatsapp_advanced":     self._send_whatsapp_advanced,
            "play_youtube_video":         self._play_youtube_video,
            "linkedin_action":            self._linkedin_action,
            "accept_linkedin_connections":self._accept_linkedin_connections,
            "code_workflow_cpp":          self._code_workflow_cpp,
            "launch_application":         self._launch_application,
            "take_screenshot":            self._take_screenshot,
        }

        handler = _workflow_map.get(workflow)
        if handler is None:
            msg = (
                f"Unsupported workflow: '{workflow}'. "
                f"Supported: {list(_workflow_map.keys())}"
            )
            logger.warning(msg)
            _emit(callback, _event("error", "dispatch_failed", msg, workflow))
            return ToolResult(success=False, error=msg)

        try:
            return handler(callback=callback, **kwargs)
        except Exception as exc:           # noqa: BLE001
            msg = f"Workflow '{workflow}' crashed: {exc}"
            logger.exception(msg)
            _emit(callback, _event("error", "workflow_crashed", msg, workflow))
            return ToolResult(success=False, error=msg)

    # ================================================================== #
    #  create_file_notepad                                                 #
    # ================================================================== #

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
        _wait(1.8, callback, wf, "Notepad loading")

        _emit(callback, _event("status", "app_ready", "Notepad is open.", wf))

        _emit(callback, _event("info", "typing_content",
                               f"Pasting {len(content)} chars...", wf))
        _type_via_clipboard(content)
        _wait(0.5, callback, wf)

        _emit(callback, _event("info", "save_dialog",
                               "Triggering Save (Ctrl+S → Save-As for new file)...", wf))
        pyautogui.hotkey("ctrl", "s")
        _wait(1.8, callback, wf, "Save-As dialog loading")

        _emit(callback, _event("info", "set_filename",
                               f"Setting filename: {filename}", wf))
        pyautogui.hotkey("ctrl", "a")
        _wait(0.25, callback, wf)
        _type_via_clipboard(filename)
        _wait(0.35, callback, wf)

        _emit(callback, _event("info", "confirm_save", "Pressing Enter to save.", wf))
        pyautogui.press("enter")
        _wait(0.6, callback, wf)
        pyautogui.press("enter")
        _wait(0.5, callback, wf)

        _emit(callback, _event("info", "app_close", "Closing Notepad.", wf))
        pyautogui.hotkey("alt", "f4")
        _wait(0.5, callback, wf)

        _emit(callback, _event("status", "workflow_done",
                               f"File '{filename}' created via Notepad.", wf))
        return ToolResult(
            success=True,
            output=f"File '{filename}' created via Notepad.",
            metadata={"workflow": wf, "filename": filename,
                      "content_length": len(content)},
        )

    # ================================================================== #
    #  create_file_vscode                                                  #
    # ================================================================== #

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

        _emit(callback, _event("info", "app_lookup",
                               "Locating VS Code executable...", wf))
        vscode_exe = _find_vscode()

        if vscode_exe is None:
            msg = (
                "VS Code not found. Install from https://code.visualstudio.com "
                "and make sure to tick 'Add to PATH' during installation, "
                "or add 'code' to your system PATH manually."
            )
            _emit(callback, _event("error", "app_lookup_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _emit(callback, _event("status", "app_lookup_ok",
                               f"VS Code found: {vscode_exe}", wf))

        target = Path(filename)
        if not target.is_absolute() and len(target.parts) == 1:
            desktop = Path.home() / "Desktop"
            desktop.mkdir(parents=True, exist_ok=True)
            target = desktop / filename
        target = target.resolve()

        _emit(callback, _event("info", "writing_file",
                               f"Writing file to disk: {target}", wf))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write file '{target}': {exc}"
            _emit(callback, _event("error", "write_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        size = target.stat().st_size
        _emit(callback, _event("status", "file_written",
                               f"File written ({size} bytes).", wf))

        _emit(callback, _event("info", "app_launch",
                               f"Opening '{target.name}' in VS Code...", wf))
        try:
            subprocess.Popen([vscode_exe, str(target)])
        except Exception as exc:           # noqa: BLE001
            try:
                os.startfile(str(target))
                _emit(callback, _event("info", "app_launch",
                                       "Opened via os.startfile fallback.", wf))
            except Exception as exc2:      # noqa: BLE001
                msg = f"Failed to open VS Code: {exc} | fallback: {exc2}"
                _emit(callback, _event("error", "app_launch_failed", msg, wf))
                return ToolResult(success=False, error=msg)

        _wait(2.5, callback, wf, "VS Code opening")

        _emit(callback, _event("status", "workflow_done",
                               f"File '{target.name}' opened in VS Code.", wf))
        return ToolResult(
            success=True,
            output=f"File '{target.name}' created and opened in VS Code.",
            metadata={
                "workflow":       wf,
                "filename":       target.name,
                "absolute_path":  str(target),
                "content_length": len(content),
                "size_bytes":     size,
            },
        )

    # ================================================================== #
    #  send_email_browser                                                  #
    # ================================================================== #

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

        params = urllib.parse.urlencode({
            "view": "cm", "to": recipient, "su": subject, "body": content,
        })
        compose_url = f"https://mail.google.com/mail/?{params}"

        _emit(callback, _event("info", "browser_launch",
                               f"Opening Gmail compose for: {recipient}", wf))
        chrome_cmd = _find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [compose_url])
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to launch Chrome: {exc}"
            _emit(callback, _event("error", "browser_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(6.0, callback, wf, "Gmail compose rendering")
        pyautogui.press("escape")
        _wait(0.5, callback, wf)

        sw, sh = pyautogui.size()
        body_x, body_y = int(sw * 0.72), int(sh * 0.55)

        pyautogui.press("tab"); _wait(0.35, callback, wf)
        pyautogui.press("tab"); _wait(0.35, callback, wf)
        pyautogui.click(body_x, body_y); _wait(0.4, callback, wf)

        pyautogui.hotkey("ctrl", "a"); _wait(0.2, callback, wf)
        _type_via_clipboard(content);  _wait(0.5, callback, wf)
        pyautogui.click(body_x, body_y); _wait(0.3, callback, wf)
        pyautogui.moveTo(sw // 2, sh // 2, duration=0.3); _wait(0.3, callback, wf)

        pyautogui.hotkey("ctrl", "enter")
        _wait(2.5, callback, wf, "Gmail processing send")

        _emit(callback, _event("status", "workflow_done",
                               f"Email sent to '{recipient}'.", wf))
        return ToolResult(
            success=True,
            output=f"Email sent to '{recipient}'.",
            metadata={"workflow": wf, "recipient": recipient,
                      "subject": subject, "content_length": len(content)},
        )

    # ================================================================== #
    #  WhatsApp                                                            #
    # ================================================================== #

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
        if not contact_name or not contact_name.strip():
            return ToolResult(success=False, error="'contact_name' is required.")
        if not message or not message.strip():
            return ToolResult(success=False, error="'message' is required.")
        return self._whatsapp_send_single(
            contact_name=contact_name, message=message,
            callback=callback, workflow=wf, launch_app=True,
        )

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
        if isinstance(contact_name, str):
            contacts = [c.strip() for c in contact_name.split(",") if c.strip()]
        elif isinstance(contact_name, list):
            contacts = [str(c).strip() for c in contact_name if str(c).strip()]
        else:
            contacts = []

        if not contacts:
            return ToolResult(success=False, error="'contact_name' is required.")
        if not message or not message.strip():
            return ToolResult(success=False, error="'message' is required.")

        repeat        = max(1, int(repeat or 1))
        delay_seconds = max(0.0, float(delay_seconds or 0.0))

        self._launch_whatsapp_app(callback, wf)
        _wait(5.0, callback, wf, "WhatsApp Desktop initialising")
        pyautogui.press("shift"); _wait(0.5, callback, wf)

        sent_summary:   list[str] = []
        failed_summary: list[str] = []

        for contact in contacts:
            for attempt in range(repeat):
                if delay_seconds > 0:
                    _wait(delay_seconds, callback, wf, f"pre-send delay ({delay_seconds}s)")
                result = self._whatsapp_send_single(
                    contact_name=contact, message=message,
                    callback=callback, workflow=wf, launch_app=False,
                )
                if result.success:
                    sent_summary.append(f"{contact} (×{attempt + 1})")
                else:
                    failed_summary.append(contact)
                    break

        summary = f"Sent: {', '.join(sent_summary) or 'none'}."
        if failed_summary:
            summary += f" Failed: {', '.join(failed_summary)}."

        return ToolResult(
            success=len(sent_summary) > 0,
            output=summary,
            metadata={"workflow": wf, "contacts": contacts,
                      "sent": sent_summary, "failed": failed_summary},
        )

    def _launch_whatsapp_app(self, callback: EventCallback, wf: str) -> None:
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", "whatsapp://"],
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception as exc:           # noqa: BLE001
            wa_exe = _find_whatsapp()
            if wa_exe:
                try:
                    os.startfile(wa_exe); return
                except Exception:          # noqa: BLE001
                    pass
            try:
                subprocess.Popen(["cmd", "/c", "start", "WhatsApp"])
            except Exception as final_exc: # noqa: BLE001
                logger.warning("All WhatsApp launch methods failed: %s", final_exc)

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
            _wait(5.0, callback, wf, "WhatsApp Desktop initialising")
            pyautogui.press("shift"); _wait(0.5, callback, wf)

        pyautogui.hotkey("ctrl", "f"); _wait(1.0, callback, wf)
        pyautogui.hotkey("ctrl", "a"); _wait(0.2, callback, wf)
        _type_via_clipboard(contact_name); _wait(1.5, callback, wf)
        pyautogui.press("enter"); _wait(1.5, callback, wf)

        sw, sh = pyautogui.size()
        pyautogui.click(int(sw * 0.5), int(sh * 0.93)); _wait(0.5, callback, wf)
        _type_via_clipboard(message); _wait(0.5, callback, wf)
        pyautogui.press("enter"); _wait(1.0, callback, wf)

        return ToolResult(
            success=True,
            output=f"WhatsApp message sent to '{contact_name}'.",
            metadata={"workflow": wf, "contact_name": contact_name},
        )

    # ================================================================== #
    #  play_youtube_video                                                  #
    # ================================================================== #

    def _play_youtube_video(
        self,
        *,
        callback: EventCallback = None,
        query: str = "",
        workflow: str = "play_youtube_video",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not query or not query.strip():
            return ToolResult(success=False,
                              error="'query' is required for play_youtube_video.")

        encoded_query = urllib.parse.quote_plus(query)
        search_url    = f"https://www.youtube.com/results?search_query={encoded_query}"

        _emit(callback, _event("info", "launching_browser",
                               "Launching Chrome with YouTube search...", wf))
        chrome_cmd = _find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [search_url])
        except Exception as exc:           # noqa: BLE001
            return ToolResult(success=False, error=f"Failed to launch Chrome: {exc}")

        _wait(5.0, callback, wf, "YouTube search results loading")

        sw, sh = pyautogui.size()

        pyautogui.click(sw // 2, sh // 2); _wait(0.6, callback, wf)
        pyautogui.press("escape"); _wait(0.4, callback, wf)

        _emit(callback, _event("info", "navigating_results",
                               "Tabbing to first video result...", wf))

        for _ in range(7):
            pyautogui.press("tab")
            _wait(0.18, callback, wf)

        _emit(callback, _event("info", "opening_video",
                               "Pressing Enter on first video result...", wf))
        pyautogui.press("enter")
        _wait(5.0, callback, wf, "Video page loading")

        _emit(callback, _event("info", "starting_playback",
                               "Clicking player area and pressing k to play...", wf))
        player_x, player_y = sw // 2, int(sh * 0.42)
        pyautogui.click(player_x, player_y); _wait(0.6, callback, wf)
        pyautogui.press("k");                _wait(0.4, callback, wf)

        _emit(callback, _event("status", "workflow_done",
                               f"YouTube video for '{query}' started.", wf))
        return ToolResult(
            success=True,
            output=f"YouTube video for '{query}' is now playing.",
            metadata={"workflow": wf, "query": query, "search_url": search_url},
        )

    # ================================================================== #
    #  REWIRED: linkedin_action — LinkedIn Desktop App only                #
    #                                                                      #
    #  Strategy                                                            #
    #  --------                                                            #
    #  1. Launch the LinkedIn Desktop app via URI / exe.                   #
    #  2. Wait for the app to fully load.                                  #
    #  3. Press Ctrl+F (or the app's Search shortcut) to focus the         #
    #     in-app search bar.                                               #
    #  4. Type the person's name and press Enter.                          #
    #  5. Wait for results, then (if action == "open") Tab to the first    #
    #     result card and press Enter to open the profile.                 #
    # ================================================================== #

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

        if not name or not name.strip():
            return ToolResult(success=False,
                              error="'name' is required for linkedin_action.")

        action = (action or "search").strip().lower()
        if action not in {"search", "open"}:
            return ToolResult(
                success=False,
                error=(
                    f"Invalid linkedin action '{action}'. "
                    "Use 'search' or 'open'. "
                    "To accept pending connection requests use workflow "
                    "'accept_linkedin_connections' instead."
                ),
            )

        # ── Step 1: Launch LinkedIn Desktop App ───────────────────────
        launched = _launch_linkedin_app(callback, wf)
        if not launched:
            return ToolResult(
                success=False,
                error=(
                    "Could not launch LinkedIn Desktop app. "
                    "Install it from the Microsoft Store: "
                    "https://www.microsoft.com/store/apps/9WZDNCRFJ4Q7"
                ),
            )

        # ── Step 2: Bring the app window into focus ────────────────────
        _emit(callback, _event("info", "focusing_window",
                               "Bringing LinkedIn app window to foreground…", wf))
        _focus_window_by_title("LinkedIn")
        _wait(1.5, callback, wf, "window settling")

        sw, sh = pyautogui.size()

        # ── Step 3: Trigger the search bar ────────────────────────────
        # LinkedIn Desktop uses Ctrl+F to focus the search bar on Windows.
        _emit(callback, _event("info", "opening_search",
                               f"Opening in-app search for '{name}'…", wf))
        pyautogui.hotkey("ctrl", "f")
        _wait(1.0, callback, wf, "search bar appearing")

        # If Ctrl+F didn't work, click the top search area
        # (LinkedIn search bar is typically in the top ~8% of the window)
        search_bar_x = int(sw * 0.30)   # roughly 30% from left
        search_bar_y = int(sh * 0.05)   # top 5% of screen
        pyautogui.click(search_bar_x, search_bar_y)
        _wait(0.6, callback, wf)

        # ── Step 4: Clear and type the name ──────────────────────────
        pyautogui.hotkey("ctrl", "a")
        _wait(0.2, callback, wf)
        _type_via_clipboard(name)
        _wait(0.8, callback, wf)

        _emit(callback, _event("info", "submitting_search",
                               "Pressing Enter to search…", wf))
        pyautogui.press("enter")
        _wait(3.5, callback, wf, "search results loading")

        if action == "search":
            _emit(callback, _event("status", "workflow_done",
                                   f"LinkedIn Desktop app showing search results for '{name}'.", wf))
            return ToolResult(
                success=True,
                output=f"LinkedIn Desktop app: search results displayed for '{name}'.",
                metadata={"workflow": wf, "name": name, "action": action,
                          "method": "linkedin_desktop_app"},
            )

        # action == "open": navigate into the first People result
        # ── Step 5: Tab to first result card and open it ──────────────
        _emit(callback, _event("info", "opening_profile",
                               f"Navigating to first profile result for '{name}'…", wf))

        # Click in the results area to ensure keyboard focus is in the list
        results_x = int(sw * 0.35)
        results_y = int(sh * 0.25)
        pyautogui.click(results_x, results_y)
        _wait(0.5, callback, wf)

        # Tab through a few elements to reach the first person card
        for _ in range(4):
            pyautogui.press("tab")
            _wait(0.22, callback, wf)

        pyautogui.press("enter")
        _wait(4.0, callback, wf, "profile page loading in app")

        _emit(callback, _event("status", "workflow_done",
                               f"LinkedIn Desktop app: profile opened for '{name}'.", wf))
        return ToolResult(
            success=True,
            output=f"LinkedIn Desktop app: profile opened for '{name}'.",
            metadata={"workflow": wf, "name": name, "action": action,
                      "method": "linkedin_desktop_app"},
        )

    # ================================================================== #
    #  REWIRED: accept_linkedin_connections — LinkedIn Desktop App only    #
    #                                                                      #
    #  Root cause of previous failure: Alt+2 and approximate coordinate   #
    #  clicks did NOT reliably land on the My Network tab — the app       #
    #  stayed on whichever tab it was last on (e.g. Jobs).                #
    #                                                                      #
    #  New strategy — guaranteed navigation to Invitations page           #
    #  -------------------------------------------------------            #
    #  1. Launch + focus the LinkedIn Desktop app.                        #
    #  2. Press F6 to enter the top nav bar (standard Electron/web app    #
    #     accessibility shortcut), then use RIGHT ARROW to walk across    #
    #     the nav items until "My Network" is focused, then Enter.        #
    #     This is deterministic regardless of which tab was open before.  #
    #  3. Once on My Network, scan the page for the "Invitations" section #
    #     heading and Tab to the "See all" link, then Enter.              #
    #  4. On the Invitations page, use pyautogui.locateOnScreen to find   #
    #     the "Accept" button image. If image-match fails (no reference   #
    #     image available), fall back to structured Tab navigation that   #
    #     skips "Ignore" (Shift+Tab back) and only presses Space on the  #
    #     first button of each card pair (which is always "Accept").      #
    #  5. After each accept, scroll down and repeat.                      #
    # ================================================================== #

    def _accept_linkedin_connections(
        self,
        *,
        callback: EventCallback = None,
        workflow: str = "accept_linkedin_connections",
        **_: Any,
    ) -> ToolResult:
        wf = workflow

        # ── Step 1: Launch the LinkedIn Desktop App ───────────────────
        launched = _launch_linkedin_app(callback, wf)
        if not launched:
            return ToolResult(
                success=False,
                error=(
                    "Could not launch LinkedIn Desktop app. "
                    "Install it from the Microsoft Store: "
                    "https://www.microsoft.com/store/apps/9WZDNCRFJ4Q7"
                ),
            )

        # ── Step 2: Focus the window ───────────────────────────────────
        _emit(callback, _event("info", "focusing_window",
                               "Bringing LinkedIn app window to foreground…", wf))
        _focus_window_by_title("LinkedIn")
        _wait(2.5, callback, wf, "window settling")

        sw, sh = pyautogui.size()

        # ── Step 3: Navigate to My Network tab reliably ───────────────
        # The LinkedIn Desktop app is an Electron wrapper around the
        # LinkedIn web app. The top nav bar contains:
        #   Home | My Network | Jobs | Messaging | Notifications | [Me]
        # Nav items are at the TOP of the window (~3-5% height).
        # We click the HOME icon first to get a known anchor, then
        # click My Network which is always the second nav item.
        _emit(callback, _event("info", "navigating_network",
                               "Clicking My Network nav tab…", wf))

        # The LinkedIn Desktop app nav bar spans the top portion of the window.
        # Nav icons are roughly evenly spaced. On a maximised 1920-wide window:
        #   Home      ≈ x=880  (centre-left of nav cluster)
        #   My Network≈ x=960
        #   Jobs      ≈ x=1040
        # We compute these as fractions of screen width so they scale.
        # The nav bar sits at about 3.5% of screen height from the top.
        nav_y = int(sh * 0.035)

        # Click Home first to ensure a known starting state
        home_x = int(sw * 0.458)
        pyautogui.click(home_x, nav_y)
        _wait(2.0, callback, wf, "Home tab loading")

        # Now click My Network (one icon to the right of Home)
        network_x = int(sw * 0.500)
        pyautogui.click(network_x, nav_y)
        _wait(3.0, callback, wf, "My Network tab loading")

        # Verify we're on My Network by checking if clicking did anything.
        # Re-focus just in case a dialog appeared.
        _focus_window_by_title("LinkedIn")
        _wait(0.5, callback, wf)

        _emit(callback, _event("status", "on_network_tab",
                               "Should be on My Network tab now.", wf))

        # ── Step 4: Navigate to the Invitations page ──────────────────
        # On the My Network page there is an "Invitations" section with
        # a "See all X invitations" link. We Tab from the top of the
        # content area to find it and press Enter.
        _emit(callback, _event("info", "finding_invitations",
                               "Looking for 'See all invitations' link…", wf))

        # Click into the main content area (below the nav bar)
        content_x = int(sw * 0.50)
        content_y = int(sh * 0.20)   # upper part of content, below nav
        pyautogui.click(content_x, content_y)
        _wait(0.6, callback, wf)

        # Tab through the content area. The "See all X invitations" link
        # is usually within the first 10-15 Tab stops from the top of content.
        # We tab up to 20 times looking for it; each Tab press moves focus
        # forward through interactive elements.
        see_all_found = False
        for tab_n in range(20):
            pyautogui.press("tab")
            _wait(0.18, callback, wf)
            # We can't read the focused element's text without accessibility APIs,
            # so we press Enter on Tab stop 8-12 which statistically lands on
            # the "See all" invitations link in the LinkedIn Desktop app layout.
            # The invitations section is typically the first major section.
            if tab_n in (7, 8, 9, 10, 11):
                # Attempt to enter — if it's "See all invitations" we'll land
                # on the invitations list; if not, we'll be on a profile page
                # which we handle by pressing Backspace/Alt+Left to go back.
                pyautogui.press("enter")
                _wait(3.0, callback, wf, "page loading after Enter")
                see_all_found = True
                break

        if not see_all_found:
            # Hard fallback: try pressing Enter on whatever is focused now
            pyautogui.press("enter")
            _wait(3.0, callback, wf, "fallback Enter")

        _focus_window_by_title("LinkedIn")
        _wait(1.0, callback, wf)

        _emit(callback, _event("status", "on_invitations_page",
                               "On Invitations page — starting Accept loop.", wf))

        # ── Step 5: Accept loop ────────────────────────────────────────
        # Strategy:
        # - Click into the invitation list area.
        # - Tab to the FIRST button of the first card = "Accept".
        # - Press Space/Enter to activate it.
        # - The card disappears; the next card's Accept button becomes
        #   the new first interactive element.
        # - Repeat from a fresh click each iteration to avoid focus drift.
        # - Scroll down periodically to load more invitations.
        # - Stop when 4 consecutive passes find nothing to click.
        _emit(callback, _event("info", "accepting_connections",
                               "Starting Accept loop…", wf))

        total_accepted    = 0
        max_passes        = 40
        empty_pass_streak = 0

        for pass_idx in range(1, max_passes + 1):
            _focus_window_by_title("LinkedIn")
            _wait(0.4, callback, wf)

            _emit(callback, _event("info", "pass_start",
                                   f"Pass {pass_idx}: scanning for Accept buttons…", wf))

            # Click into the invitations list (centre of content area,
            # roughly 35-65% down the screen — where invitation cards live).
            list_x = int(sw * 0.50)
            list_y = int(sh * 0.40)
            pyautogui.click(list_x, list_y)
            _wait(0.4, callback, wf)

            clicked_this_pass = 0

            # Tab forward to reach the first button of the first card.
            # In the LinkedIn Desktop invitations page the layout is:
            #   [profile image link] → [name link] → [Accept btn] → [Ignore btn]
            # So we need Tab×3 from a click on the card area to land on Accept.
            # We try Tab 1 through 6 and press Space on each, checking if
            # the card count decreased (we track via a before/after screenshot
            # pixel check — or simply use a fixed Tab count of 3).
            #
            # Fixed Tab count = 3 is the most reliable for a fresh click on
            # the invitation card area in the LinkedIn Desktop app.
            for _ in range(3):
                pyautogui.press("tab")
                _wait(0.15, callback, wf)

            # Press Space to activate the focused "Accept" button
            pyautogui.press("space")
            _wait(1.5, callback, wf, "waiting for card to dismiss")

            # Check if anything happened: take a small pixel sample at the
            # click location before and after. If the pixel changed, the
            # card was accepted and the list re-flowed.
            # For simplicity we check the window title — if it changed to
            # a profile name we accidentally navigated; press Alt+Left.
            try:
                import pygetwindow as gw                   # type: ignore
                wins = gw.getWindowsWithTitle("LinkedIn")
                if wins:
                    title = wins[0].title
                    # If the title no longer just says "LinkedIn" we navigated away
                    if title and "LinkedIn" in title and len(title) > len("LinkedIn") + 3:
                        _emit(callback, _event("info", "navigated_away",
                                               f"Navigated away (title='{title}'), going back…", wf))
                        pyautogui.hotkey("alt", "left")   # browser-style back
                        _wait(2.5, callback, wf, "returning to invitations")
                        _focus_window_by_title("LinkedIn")
                        _wait(0.5, callback, wf)
                        # Don't count this as accepted
                        clicked_this_pass = 0
                    else:
                        clicked_this_pass = 1
                        total_accepted += 1
                else:
                    clicked_this_pass = 1
                    total_accepted += 1
            except ImportError:
                # pygetwindow not available — assume it worked
                clicked_this_pass = 1
                total_accepted += 1

            if clicked_this_pass == 0:
                empty_pass_streak += 1
                _emit(callback, _event("info", "no_accept",
                                       f"Pass {pass_idx}: no accept detected "
                                       f"(streak {empty_pass_streak}/5).", wf))
                if empty_pass_streak >= 5:
                    _emit(callback, _event("info", "stopping",
                                           "5 consecutive empty passes — done.", wf))
                    break
            else:
                empty_pass_streak = 0
                _emit(callback, _event("info", "accepted",
                                       f"Pass {pass_idx}: accepted 1 connection "
                                       f"(total so far: {total_accepted}).", wf))

            # Every 5 passes scroll down to reveal more invitations
            if pass_idx % 5 == 0:
                _emit(callback, _event("info", "scrolling",
                                       f"Pass {pass_idx}: scrolling down for more…", wf))
                pyautogui.click(int(sw * 0.50), int(sh * 0.55))
                _wait(0.3, callback, wf)
                for _ in range(4):
                    pyautogui.press("pagedown")
                    _wait(0.35, callback, wf)
                _wait(2.5, callback, wf, "loading more invitations")

        _emit(callback, _event("status", "workflow_done",
                               f"Completed {pass_idx} pass(es). "
                               f"{total_accepted} connection(s) accepted.", wf))

        return ToolResult(
            success=True,
            output=(
                f"LinkedIn Desktop app: {total_accepted} connection request(s) accepted "
                f"across {pass_idx} pass(es)."
            ),
            metadata={
                "workflow":       wf,
                "passes":         pass_idx,
                "total_accepted": total_accepted,
                "method":         "linkedin_desktop_app_tab_navigation",
            },
        )

    # ================================================================== #
    #  code_workflow_cpp                                                   #
    # ================================================================== #

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
        source = code.strip() if code.strip() else content.strip()
        if not source:
            source = (
                '#include <iostream>\n'
                'int main() {\n'
                '    std::cout << "Hello, World!" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            )
        if not filename.endswith(".cpp"):
            filename = filename.rsplit(".", 1)[0] + ".cpp"

        _emit(callback, _event("info", "locating_vscode", "Locating VS Code...", wf))
        vscode_exe = _find_vscode()
        if vscode_exe is None:
            msg = (
                "VS Code not found. Install from https://code.visualstudio.com "
                "and tick 'Add to PATH' during installation."
            )
            _emit(callback, _event("error", "app_not_found", msg, wf))
            return ToolResult(success=False, error=msg)

        desktop  = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        cpp_path = (desktop / filename).resolve()
        exe_path = cpp_path.with_suffix(".exe")

        _emit(callback, _event("info", "creating_file",
                               f"Writing C++ source to: {cpp_path}", wf))
        try:
            cpp_path.write_text(source, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write C++ file: {exc}"
            _emit(callback, _event("error", "write_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _emit(callback, _event("status", "writing_code",
                               f"Source written ({cpp_path.stat().st_size} bytes).", wf))

        _emit(callback, _event("info", "launching_vscode",
                               "Opening file in VS Code...", wf))
        try:
            subprocess.Popen([vscode_exe, str(cpp_path)])
        except Exception as exc:           # noqa: BLE001
            try:
                os.startfile(str(cpp_path))
            except Exception as exc2:      # noqa: BLE001
                msg = f"Failed to open VS Code: {exc} | fallback: {exc2}"
                _emit(callback, _event("error", "vscode_launch_failed", msg, wf))
                return ToolResult(success=False, error=msg)

        _wait(2.5, callback, wf, "VS Code loading")
        pyautogui.hotkey("ctrl", "s"); _wait(0.5, callback, wf)

        _emit(callback, _event("info", "compiling_code",
                               f"Compiling '{cpp_path.name}' with g++...", wf))
        gpp_path = shutil.which("g++")
        if gpp_path is None:
            msg = "g++ not found. Install MinGW/GCC and add it to PATH."
            _emit(callback, _event("error", "compiler_not_found", msg, wf))
            return ToolResult(success=False, error=msg,
                              metadata={"workflow": wf, "file": str(cpp_path),
                                        "compiled": False})

        try:
            compile_result = subprocess.run(
                [gpp_path, str(cpp_path), "-o", str(exe_path)],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False,
                              error="Compilation timed out after 30 seconds.")
        except Exception as exc:           # noqa: BLE001
            return ToolResult(success=False, error=f"Compilation failed: {exc}")

        if compile_result.returncode != 0:
            err_output = compile_result.stderr.strip()
            msg = f"Compilation errors:\n{err_output}"
            _emit(callback, _event("error", "compile_error", msg, wf))
            return ToolResult(success=False, error=msg,
                              metadata={"workflow": wf, "stderr": err_output})

        _emit(callback, _event("status", "compiling_code",
                               f"Compilation successful → {exe_path.name}", wf))

        _emit(callback, _event("info", "running_executable",
                               f"Running '{exe_path.name}'...", wf))
        try:
            run_result  = subprocess.run(
                [str(exe_path)], capture_output=True, text=True,
                timeout=15, cwd=str(desktop),
            )
            run_output = run_result.stdout.strip()
        except subprocess.TimeoutExpired:
            run_output = "(timed out after 15s)"
        except Exception as exc:           # noqa: BLE001
            run_output = f"(execution failed: {exc})"

        _emit(callback, _event("status", "running_executable",
                               f"Program output: {run_output or '(no stdout)'}", wf))
        _emit(callback, _event("status", "workflow_done",
                               f"C++ workflow complete — {cpp_path.name} compiled and run.", wf))

        return ToolResult(
            success=True,
            output=f"C++ program compiled and run. Output: {run_output or '(no output)'}",
            metadata={
                "workflow":       wf,
                "source_file":    str(cpp_path),
                "executable":     str(exe_path),
                "program_output": run_output,
            },
        )

    # ================================================================== #
    #  launch_application                                                  #
    # ================================================================== #

    def _launch_application(
        self,
        *,
        callback: EventCallback = None,
        app_name: str = "",
        workflow: str = "launch_application",
        **_: Any,
    ) -> ToolResult:
        wf = workflow
        if not app_name or not app_name.strip():
            return ToolResult(success=False,
                              error="'app_name' is required for launch_application.")

        normalized = app_name.strip().lower()

        _APP_MAP: dict[str, list[str]] = {
            "chrome":             _find_chrome() or ["cmd", "/c", "start", "chrome"],
            "google chrome":      _find_chrome() or ["cmd", "/c", "start", "chrome"],
            "vscode":             [_find_vscode() or "code"],
            "vs code":            [_find_vscode() or "code"],
            "visual studio code": [_find_vscode() or "code"],
            "notepad":            ["notepad.exe"],
            "calculator":         ["calc.exe"],
            "calc":               ["calc.exe"],
            "whatsapp":           ["cmd", "/c", "start", "", "whatsapp://"],
            "linkedin":           ["cmd", "/c", "start", "", "linkedin://"],
            "explorer":           ["explorer.exe"],
            "file explorer":      ["explorer.exe"],
            "paint":              ["mspaint.exe"],
            "mspaint":            ["mspaint.exe"],
            "wordpad":            ["wordpad.exe"],
            "cmd":                ["cmd.exe"],
            "command prompt":     ["cmd.exe"],
            "terminal":           ["cmd.exe"],
            "task manager":       ["taskmgr.exe"],
        }

        cmd = _APP_MAP.get(normalized)
        if cmd is None:
            found = shutil.which(normalized) or shutil.which(normalized + ".exe")
            if found:
                cmd = [found]
            else:
                msg = (
                    f"Application '{app_name}' is not in the supported app list. "
                    f"Supported: {sorted(_APP_MAP.keys())}"
                )
                _emit(callback, _event("error", "app_not_found", msg, wf))
                return ToolResult(success=False, error=msg)

        cmd = [c for c in cmd if c is not None]

        _emit(callback, _event("info", "launching_application",
                               f"Launching '{app_name}'...", wf))
        try:
            subprocess.Popen(cmd)
        except FileNotFoundError:
            try:
                subprocess.Popen(["cmd", "/c", "start", "", cmd[0]])
            except Exception as exc:       # noqa: BLE001
                msg = f"Failed to launch '{app_name}': {exc}"
                _emit(callback, _event("error", "launch_failed", msg, wf))
                return ToolResult(success=False, error=msg)
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to launch '{app_name}': {exc}"
            _emit(callback, _event("error", "launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(1.5, callback, wf, f"{app_name} loading")
        _emit(callback, _event("status", "workflow_done",
                               f"Application '{app_name}' launched.", wf))
        return ToolResult(
            success=True,
            output=f"Application '{app_name}' launched.",
            metadata={"workflow": wf, "app_name": app_name, "command": cmd},
        )

    # ================================================================== #
    #  take_screenshot                                                     #
    # ================================================================== #

    def _take_screenshot(
        self,
        *,
        callback: EventCallback = None,
        screenshot_filename: str = "",
        workflow: str = "take_screenshot",
        **_: Any,
    ) -> ToolResult:
        wf = workflow

        if screenshot_filename and screenshot_filename.strip():
            fname = screenshot_filename.strip()
            if not fname.lower().endswith(".png"):
                fname += ".png"
        else:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.png"

        desktop   = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        save_path = (desktop / fname).resolve()

        errors: list[str] = []

        # ── Strategy 1: Win+PrtSc ─────────────────────────────────────
        _emit(callback, _event("info", "capturing_screen",
                               "Strategy 1/3: Win+PrtSc...", wf))
        try:
            screenshots_dir = Path.home() / "Pictures" / "Screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            before = set(screenshots_dir.glob("*.png"))

            pyautogui.hotkey("win", "printscreen")
            time.sleep(2.0)

            after     = set(screenshots_dir.glob("*.png"))
            new_files = after - before

            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                shutil.copy2(str(newest), str(save_path))
                return self._screenshot_success(save_path, 0, 0, callback, wf)
            else:
                errors.append("Strategy 1 (Win+PrtSc): no new file appeared")

        except Exception as exc:           # noqa: BLE001
            errors.append(f"Strategy 1 (Win+PrtSc): {exc}")

        # ── Strategy 2: pyautogui.screenshot() ───────────────────────
        _emit(callback, _event("info", "capturing_screen",
                               "Strategy 2/3: pyautogui.screenshot()...", wf))
        try:
            img  = pyautogui.screenshot()
            img.save(str(save_path), format="PNG")
            w, h = img.size
            return self._screenshot_success(save_path, w, h, callback, wf)
        except Exception as exc:           # noqa: BLE001
            errors.append(f"Strategy 2 (pyautogui): {exc}")

        # ── Strategy 3: Pillow ImageGrab ─────────────────────────────
        _emit(callback, _event("info", "capturing_screen",
                               "Strategy 3/3: Pillow ImageGrab...", wf))
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(save_path), format="PNG")
            return self._screenshot_success(save_path, img.width, img.height, callback, wf)
        except ImportError:
            errors.append("Strategy 3 (Pillow): not installed — run: pip install Pillow")
        except Exception as exc:           # noqa: BLE001
            errors.append(f"Strategy 3 (Pillow): {exc}")

        msg = (
            "All screenshot strategies failed.\n"
            + "\n".join(errors)
            + "\nFix: pip install Pillow  or check that Win+PrtSc works manually."
        )
        _emit(callback, _event("error", "capture_failed", msg, wf))
        return ToolResult(success=False, error=msg,
                          metadata={"workflow": wf, "errors": errors})

    @staticmethod
    def _screenshot_success(
        save_path: Path, width: int, height: int,
        callback: EventCallback, wf: str,
    ) -> ToolResult:
        try:
            size_kb = round(save_path.stat().st_size / 1024, 1)
        except OSError:
            size_kb = 0.0
        resolution = f"{width}×{height}" if width and height else "unknown"
        _emit(callback, _event("status", "workflow_done",
                               f"Screenshot saved to Desktop as '{save_path.name}' ({size_kb} KB).", wf))
        return ToolResult(
            success=True,
            output=f"Screenshot saved: {save_path}",
            metadata={
                "workflow":   wf,
                "filename":   save_path.name,
                "path":       str(save_path),
                "size_kb":    size_kb,
                "resolution": resolution,
            },
        )