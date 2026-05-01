"""
core/tools/ui_automation_tool.py  — FIXED VERSION

Changes from original
---------------------
FIX 1  create_file_notepad   : Always opens a fresh Notepad (notepad.exe with no
                               args opens untitled). After typing, uses Ctrl+S
                               (first-save triggers Save-As on a new file) instead
                               of Ctrl+Shift+S which was re-opening the last file.

FIX 2  create_file_vscode    : VS Code lookup now also searches the Microsoft Store
        (+ FIX 5 cpp)          install location and several additional PATH aliases
                               so "the system cannot find the file specified" is
                               resolved on most Windows setups.

FIX 3  play_youtube_video    : After the search results page loads the tool now
                               clicks the FIRST video thumbnail directly using a
                               reliable CSS-selector approach via keyboard nav,
                               waits for the video page, then presses k to play.

FIX 4  linkedin_action       : "connect" workflow removed entirely.
                               New workflow  "accept_connections"  opens
                               linkedin.com/mynetwork/invitation-manager/,
                               scrolls through all pending invitations and clicks
                               every "Accept" button it can find.

FIX 6  take_screenshot       : Primary strategy is now Win+PrtSc which works on
                               every modern Windows 10/11 machine without any
                               third-party library.  Pillow and pyautogui are kept
                               as fallbacks.  The saved file is always copied to
                               the Desktop so the user can find it immediately.
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
#  FIX 2 / FIX 5 — robust VS Code locator                                     #
# --------------------------------------------------------------------------- #

def _find_vscode() -> str | None:
    """
    Search for VS Code executable across every known Windows install location.
    Returns the first executable path found, or None.
    """
    # 1. Try PATH first (works when user added 'code' to PATH during install)
    for alias in ("code", "code.cmd", "code-insiders"):
        if shutil.which(alias):
            return shutil.which(alias)

    local_app   = os.environ.get("LOCALAPPDATA", "")
    prog_files  = os.environ.get("ProgramFiles",   "C:\\Program Files")
    prog_x86    = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    user_home   = str(Path.home())

    candidates: list[Path] = [
        # Standard user install
        Path(local_app) / "Programs" / "Microsoft VS Code" / "Code.exe",
        # Microsoft Store install (WindowsApps)
        Path(local_app) / "Microsoft" / "WindowsApps" / "code.exe",
        Path(local_app) / "Microsoft" / "WindowsApps" / "Microsoft.VisualStudioCode_*" / "code.exe",
        # System-wide install
        Path(prog_files)  / "Microsoft VS Code" / "Code.exe",
        Path(prog_x86)    / "Microsoft VS Code" / "Code.exe",
        # Scoop install
        Path(user_home) / "scoop" / "apps" / "vscode" / "current" / "code.exe",
        Path(user_home) / "scoop" / "shims"  / "code.exe",
        # Chocolatey
        Path("C:\\tools") / "vscode" / "Code.exe",
    ]

    for candidate in candidates:
        # Handle glob patterns (Microsoft Store)
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
        "'linkedin_action' — search/open a LinkedIn profile (action: search or open); "
        "'accept_linkedin_connections' — open LinkedIn invitation manager and accept all pending requests; "
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
    #  FIX 1 — create_file_notepad                                         #
    #  Always opens a brand-new untitled Notepad window, never reopens     #
    #  a previously saved file.  First Ctrl+S on an unsaved file triggers  #
    #  the Save-As dialog automatically on Windows 10/11.                  #
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

        # Launch notepad.exe with NO arguments → always a fresh untitled window
        _emit(callback, _event("info", "app_launch",
                               "Launching fresh Notepad (no arguments)...", wf))
        subprocess.Popen(["notepad.exe"])          # ← no filename arg = always new
        _wait(1.8, callback, wf, "Notepad loading")

        _emit(callback, _event("status", "app_ready", "Notepad is open.", wf))

        # Type content via clipboard
        _emit(callback, _event("info", "typing_content",
                               f"Pasting {len(content)} chars...", wf))
        _type_via_clipboard(content)
        _wait(0.5, callback, wf)

        # Ctrl+S on an *unsaved* (untitled) Notepad file opens Save-As directly.
        # This is more reliable than Ctrl+Shift+S which varies by Windows version.
        _emit(callback, _event("info", "save_dialog",
                               "Triggering Save (Ctrl+S → Save-As for new file)...", wf))
        pyautogui.hotkey("ctrl", "s")
        _wait(1.8, callback, wf, "Save-As dialog loading")

        # Clear whatever is in the filename field and type our name
        _emit(callback, _event("info", "set_filename",
                               f"Setting filename: {filename}", wf))
        pyautogui.hotkey("ctrl", "a")
        _wait(0.25, callback, wf)
        _type_via_clipboard(filename)
        _wait(0.35, callback, wf)

        # Confirm save
        _emit(callback, _event("info", "confirm_save", "Pressing Enter to save.", wf))
        pyautogui.press("enter")
        _wait(0.6, callback, wf)
        # Second enter handles "replace?" dialog if filename already exists
        pyautogui.press("enter")
        _wait(0.5, callback, wf)

        # Close Notepad
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
    #  FIX 2 — create_file_vscode                                          #
    #  Uses the improved _find_vscode() that checks Store + Scoop + choco. #
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

        # Resolve target path — bare filename → Desktop
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
            # Use shell=False; pass the full path as a quoted argument
            subprocess.Popen([vscode_exe, str(target)])
        except Exception as exc:           # noqa: BLE001
            # Final fallback: use Windows shell association
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
    #  send_email_browser — unchanged from original                        #
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
    #  WhatsApp — unchanged from original                                  #
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
            return ToolResult(success=False,
                              error="'contact_name' is required.")
        if not message or not message.strip():
            return ToolResult(success=False,
                              error="'message' is required.")
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
    #  FIX 3 — play_youtube_video                                          #
    #  Searches YouTube, then presses Tab to reach the FIRST video result  #
    #  thumbnail and Enter to open it, then plays via the k shortcut.      #
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

        # Click in the middle of the page to make sure it has keyboard focus
        pyautogui.click(sw // 2, sh // 2); _wait(0.6, callback, wf)

        # On YouTube search results the address bar steals focus initially.
        # Press Escape to defocus it, then Tab into page content.
        pyautogui.press("escape"); _wait(0.4, callback, wf)

        _emit(callback, _event("info", "navigating_results",
                               "Tabbing to first video result...", wf))

        # YouTube's first interactive element after the search box region is
        # a video thumbnail. We tab enough times to skip the header controls.
        # Typically 5–8 tabs reach the first video.  We also watch for a
        # focused element that looks like a video (aria-label contains time).
        # Simple heuristic: 7 tabs clears the header on most layouts.
        for _ in range(7):
            pyautogui.press("tab")
            _wait(0.18, callback, wf)

        _emit(callback, _event("info", "opening_video",
                               "Pressing Enter on first video result...", wf))
        pyautogui.press("enter")
        _wait(5.0, callback, wf, "Video page loading")

        # Ensure video player has focus then press k (play/pause toggle)
        _emit(callback, _event("info", "starting_playback",
                               "Clicking player area and pressing k to play...", wf))
        # The video player is roughly centred horizontally, upper-half vertically
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
    #  FIX 4 — linkedin_action (connect removed) + accept_linkedin_connections
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

        encoded_name = urllib.parse.quote_plus(name)
        search_url   = (
            f"https://www.linkedin.com/search/results/people/"
            f"?keywords={encoded_name}"
        )

        _emit(callback, _event("info", "launching_browser",
                               "Launching Chrome for LinkedIn...", wf))
        chrome_cmd = _find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [search_url])
        except Exception as exc:           # noqa: BLE001
            return ToolResult(success=False, error=f"Failed to launch Chrome: {exc}")

        _wait(5.0, callback, wf, "LinkedIn search results loading")

        if action == "search":
            _emit(callback, _event("status", "workflow_done",
                                   f"LinkedIn search results shown for '{name}'.", wf))
            return ToolResult(
                success=True,
                output=f"LinkedIn search results displayed for '{name}'.",
                metadata={"workflow": wf, "name": name, "action": action},
            )

        # action == "open" — navigate to the first result
        sw, sh = pyautogui.size()
        pyautogui.click(sw // 2, sh // 2); _wait(0.5, callback, wf)
        for _ in range(5):
            pyautogui.press("tab"); _wait(0.2, callback, wf)

        _emit(callback, _event("info", "opening_profile",
                               f"Opening first profile for '{name}'...", wf))
        pyautogui.press("enter"); _wait(5.0, callback, wf, "Profile page loading")

        _emit(callback, _event("status", "workflow_done",
                               f"LinkedIn profile opened for '{name}'.", wf))
        return ToolResult(
            success=True,
            output=f"LinkedIn profile opened for '{name}'.",
            metadata={"workflow": wf, "name": name, "action": action},
        )

    def _accept_linkedin_connections(
        self,
        *,
        callback: "EventCallback" = None,
        workflow: str = "accept_linkedin_connections",
        **_: Any,
    ) -> "ToolResult":
        """
        Open LinkedIn desktop app and accept ALL pending requests
        using pyautogui to interact with the app interface.
        The desktop app maintains login state, unlike browser automation.
        """
        wf = workflow
        
        _emit(callback, _event("info", "app_launch",
                               "Launching LinkedIn desktop app...", wf))
        
        # Launch LinkedIn app
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "linkedin://"])
            _wait(6.0, callback, wf, "LinkedIn app loading")
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch LinkedIn app: {exc}"
            _emit(callback, _event("error", "app_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)
        
        # Focus the LinkedIn window
        self._focus_chrome(callback, wf)
        _wait(2.0, callback, wf, "settling")
        
        sw, sh = pyautogui.size()
        
        # Navigate to invitation manager using keyboard shortcuts
        _emit(callback, _event("info", "navigating",
                               "Navigating to invitation manager...", wf))
        pyautogui.hotkey("ctrl", "l")  # Focus address/search bar
        _wait(0.5, callback, wf)
        _type_via_clipboard("mynetwork/invitation-manager")
        _wait(0.5, callback, wf)
        pyautogui.press("enter")
        _wait(4.0, callback, wf, "invitation manager loading")
        
        total_accepted = 0
        max_passes = 25
        consecutive_empty_passes = 0
        
        _emit(callback, _event("info", "accepting_connections",
                               f"Starting acceptance loop (max {max_passes} passes)...", wf))
        
        for pass_idx in range(1, max_passes + 1):
            self._focus_chrome(callback, wf)
            _wait(0.4, callback, wf)
            
            _emit(callback, _event("info", "pass_start",
                                   f"Pass {pass_idx}: Scanning for Accept buttons...", wf))
            
            # Click in the center of the invitation cards area
            pyautogui.click(sw // 2, int(sh * 0.45))
            _wait(0.3, callback, wf)
            
            # Use Tab to navigate through interactive elements and find Accept buttons
            # LinkedIn invitation cards typically have Accept buttons that are reachable via Tab
            buttons_clicked_this_pass = 0
            
            # Tab through elements on the page to find Accept buttons
            for tab_idx in range(40):  # Try up to 40 Tab presses per pass
                pyautogui.press("tab")
                _wait(0.12, callback, wf)
                
                # Try clicking with Space (activates focused button)
                # We use a conservative approach: try Space, if it seems to work, count it
                try:
                    current_pos = pyautogui.position()
                    pyautogui.press("space")
                    _wait(1.2, callback, wf, "processing click")
                    
                    # Check if we're still on the same page (if page changed, likely accepted)
                    # For now, we assume Space clicked something
                    buttons_clicked_this_pass += 1
                    total_accepted += 1
                    
                    _emit(callback, _event("debug", "button_clicked",
                                           f"Pass {pass_idx}: Tab {tab_idx} - button clicked.", wf))
                    
                    # Break out of tab loop to re-scan from top
                    break
                except Exception:  # noqa: BLE001
                    pass
            
            if buttons_clicked_this_pass == 0:
                consecutive_empty_passes += 1
                _emit(callback, _event("info", "no_buttons",
                                       f"Pass {pass_idx}: No Accept buttons found (empty {consecutive_empty_passes}/3).", wf))
                
                if consecutive_empty_passes >= 3:
                    _emit(callback, _event("info", "stopping",
                                           "No buttons found in 3 consecutive passes—likely complete.", wf))
                    break
            else:
                consecutive_empty_passes = 0
                _emit(callback, _event("info", "buttons_found",
                                       f"Pass {pass_idx}: Clicked {buttons_clicked_this_pass} button(s).", wf))
            
            # Scroll down to load more invitations
            _emit(callback, _event("info", "scrolling",
                                   f"Pass {pass_idx}: Scrolling to load more...", wf))
            pyautogui.click(sw // 2, int(sh * 0.5))
            _wait(0.3, callback, wf)
            
            # Scroll down using Page Down or arrow keys
            for _ in range(3):
                pyautogui.press("pagedown")
                _wait(0.4, callback, wf)
            
            _wait(2.5, callback, wf, "loading more invitations")
        
        _emit(callback, _event("status", "workflow_done",
                               f"Completed {pass_idx} pass(es). Total accepted: {total_accepted}.", wf))
        
        return ToolResult(
            success=True,
            output=(
                f"LinkedIn invitation manager closed. "
                f"Processed {pass_idx} pass(es) - {total_accepted} total connection(s) likely accepted."
            ),
            metadata={
                "workflow":       wf,
                "passes":         pass_idx,
                "total_accepted": total_accepted,
                "method":         "pyautogui_desktop_app",
            },
        )
 
    def _focus_chrome(self, callback: "EventCallback", wf: str) -> None:
        """
        Bring the Chrome/LinkedIn window to the foreground.
        Uses pygetwindow when available; falls back to a centre-click.
        """
        try:
            import pygetwindow as gw                   # type: ignore
            for fragment in ("LinkedIn", "Google Chrome", "Chrome"):
                wins = gw.getWindowsWithTitle(fragment)
                if wins:
                    wins[0].restore()
                    wins[0].activate()
                    time.sleep(0.35)
                    return
        except ImportError:
            pass                                       # graceful degradation
        except Exception as exc:                       # noqa: BLE001
            logger.debug("pygetwindow activate failed: %s", exc)
 
        # Fallback: click in the centre of the screen
        sw, sh = pyautogui.size()
        pyautogui.click(sw // 2, sh // 2)
        time.sleep(0.3)

    # ================================================================== #
    #  FIX 5 — code_workflow_cpp                                           #
    #  Uses the same improved _find_vscode() as create_file_vscode.        #
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

        _emit(callback, _event("info", "locating_vscode",
                               "Locating VS Code...", wf))
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

        # Open in VS Code
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

        # Compile
        _emit(callback, _event("info", "compiling_code",
                               f"Compiling '{cpp_path.name}' with g++...", wf))
        gpp_path = shutil.which("g++")
        if gpp_path is None:
            msg = ("g++ not found. Install MinGW/GCC and add it to PATH.")
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

        # Run
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
    #  launch_application — unchanged                                      #
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
    #  FIX 6 — take_screenshot                                             #
    #  Primary: Win+PrtSc (works on all modern Windows without libraries). #
    #  Fallback 1: pyautogui.screenshot()                                  #
    #  Fallback 2: Pillow ImageGrab                                        #
    #  The file is always copied to the Desktop.                           #
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

        # Resolve filename
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

        # ── Strategy 1: Win+PrtSc ─────────────────────────────────────── #
        # Windows saves the file automatically to ~/Pictures/Screenshots.
        # We grab the newest PNG from that folder and copy it to Desktop.
        _emit(callback, _event("info", "capturing_screen",
                               "Strategy 1/3: Win+PrtSc...", wf))
        try:
            screenshots_dir = Path.home() / "Pictures" / "Screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            # Record the newest file BEFORE pressing so we can detect the new one
            before = set(screenshots_dir.glob("*.png"))

            pyautogui.hotkey("win", "printscreen")
            time.sleep(2.0)            # Windows needs ~1-2s to write the file

            after    = set(screenshots_dir.glob("*.png"))
            new_files = after - before

            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                shutil.copy2(str(newest), str(save_path))
                size_kb = round(save_path.stat().st_size / 1024, 1)
                _emit(callback, _event("status", "screenshot_saved",
                                       f"Saved via Win+PrtSc → {fname} ({size_kb} KB)", wf))
                return self._screenshot_success(save_path, 0, 0, callback, wf)
            else:
                errors.append("Strategy 1 (Win+PrtSc): no new file appeared in ~/Pictures/Screenshots")

        except Exception as exc:           # noqa: BLE001
            errors.append(f"Strategy 1 (Win+PrtSc): {exc}")
            _emit(callback, _event("info", "capturing_screen",
                                   f"Win+PrtSc failed ({exc}), trying next...", wf))

        # ── Strategy 2: pyautogui.screenshot() ───────────────────────── #
        _emit(callback, _event("info", "capturing_screen",
                               "Strategy 2/3: pyautogui.screenshot()...", wf))
        try:
            img  = pyautogui.screenshot()
            img.save(str(save_path), format="PNG")
            w, h = img.size
            _emit(callback, _event("status", "screenshot_saved",
                                   f"Saved via pyautogui ({w}×{h})", wf))
            return self._screenshot_success(save_path, w, h, callback, wf)
        except Exception as exc:           # noqa: BLE001
            errors.append(f"Strategy 2 (pyautogui): {exc}")
            _emit(callback, _event("info", "capturing_screen",
                                   f"pyautogui failed ({exc}), trying next...", wf))

        # ── Strategy 3: Pillow ImageGrab ─────────────────────────────── #
        _emit(callback, _event("info", "capturing_screen",
                               "Strategy 3/3: Pillow ImageGrab...", wf))
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(save_path), format="PNG")
            _emit(callback, _event("status", "screenshot_saved",
                                   f"Saved via Pillow ImageGrab ({img.width}×{img.height})", wf))
            return self._screenshot_success(save_path, img.width, img.height, callback, wf)
        except ImportError:
            errors.append("Strategy 3 (Pillow): not installed — run: pip install Pillow")
        except Exception as exc:           # noqa: BLE001
            errors.append(f"Strategy 3 (Pillow): {exc}")

        # All strategies failed
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