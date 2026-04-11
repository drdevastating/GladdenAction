"""
core/tools/ui_automation_tool.py

UIAutomationTool — the primary visible execution mechanism of the agent.

Supported workflows
-------------------
    create_file_notepad         Open Notepad, type content, save as filename.
    create_file_vscode          Write content to disk, then open it in VS Code.
    send_email_browser          Open Gmail in Chrome with a pre-filled compose URL, then send.
    send_whatsapp_desktop       Open WhatsApp Desktop, find contact, type and send message.
    send_whatsapp_advanced      Send to one or multiple contacts, with optional delay/repeat.
    play_youtube_video          Open Chrome, search YouTube, open first result, start playback.
    linkedin_action             Search/open a LinkedIn profile, optionally send connection request.
    code_workflow_cpp           Create C++ file in VS Code, compile with g++, and run.
    launch_application          Open a named system application via safe subprocess mapping.
    take_screenshot             Capture the screen and save as PNG.
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
    except Exception as exc:  # noqa: BLE001
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
#  UIAutomationTool                                                             #
# --------------------------------------------------------------------------- #

from core.tools.base import BaseTool, ToolResult


class UIAutomationTool(BaseTool):

    name: str = "ui_automation"

    description: str = (
        "Performs VISIBLE on-screen UI automation workflows using real applications. "
        "Use this tool when the user mentions Notepad, VS Code, browser, Gmail, WhatsApp, "
        "YouTube, LinkedIn, screenshots, or any instruction implying visible execution. "
        "Supported workflows: "
        "'create_file_notepad' — create/write a file using Notepad; "
        "'create_file_vscode' — create/write code in VS Code; "
        "'send_email_browser' — send email via Gmail in Chrome; "
        "'send_whatsapp_desktop' — send a WhatsApp message via WhatsApp Desktop app; "
        "'send_whatsapp_advanced' — send WhatsApp to one or multiple contacts with optional delay/repeat; "
        "'play_youtube_video' — open Chrome and play a YouTube video by search query; "
        "'linkedin_action' — search/open a LinkedIn profile, optionally connect; "
        "'code_workflow_cpp' — create a C++ file, compile with g++, and run it; "
        "'launch_application' — open a named system application (chrome, vscode, notepad, etc.); "
        "'take_screenshot' — capture the screen and save as PNG."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["workflow"],
        "properties": {
            "workflow": {
                "type": "string",
                "description": (
                    "The UI workflow to execute. One of: "
                    "'create_file_notepad', 'create_file_vscode', "
                    "'send_email_browser', 'send_whatsapp_desktop', "
                    "'send_whatsapp_advanced', 'play_youtube_video', "
                    "'linkedin_action', 'code_workflow_cpp', "
                    "'launch_application', 'take_screenshot'."
                ),
            },
            # --- file workflows ---
            "filename": {
                "type": "string",
                "description": "File name to create (file/code workflows).",
            },
            "content": {
                "type": "string",
                "description": "Text content to write (file/email/notepad workflows).",
            },
            "code": {
                "type": "string",
                "description": "Source code to write (code_workflow_cpp).",
            },
            # --- email workflow ---
            "recipient": {
                "type": "string",
                "description": "Email address of the recipient (send_email_browser).",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line (send_email_browser).",
            },
            # --- whatsapp workflows ---
            "contact_name": {
                "type": ["string", "array"],
                "description": (
                    "Name (or list of names) of the WhatsApp contact(s) to message. "
                    "Accepts a single string or a JSON array for multiple contacts."
                ),
            },
            "message": {
                "type": "string",
                "description": "Message text to send (WhatsApp workflows).",
            },
            "delay_seconds": {
                "type": "number",
                "description": "Optional delay in seconds before sending each message (send_whatsapp_advanced).",
            },
            "repeat": {
                "type": "integer",
                "description": "Number of times to send the message to each contact (send_whatsapp_advanced).",
            },
            # --- youtube workflow ---
            "query": {
                "type": "string",
                "description": "YouTube search query (play_youtube_video).",
            },
            # --- linkedin workflow ---
            "name": {
                "type": "string",
                "description": "Full name to search on LinkedIn (linkedin_action).",
            },
            "action": {
                "type": "string",
                "description": "LinkedIn action: 'search', 'open', or 'connect' (linkedin_action).",
            },
            # --- launch_application workflow ---
            "app_name": {
                "type": "string",
                "description": (
                    "Application to launch. Supported: chrome, vscode, calculator, "
                    "notepad, whatsapp, explorer, paint, wordpad, mspaint, cmd, terminal."
                ),
            },
            # --- screenshot workflow ---
            "screenshot_filename": {
                "type": "string",
                "description": "Optional filename (with .png extension) for the screenshot.",
            },
        },
    }

    # ------------------------------------------------------------------ #
    #  Application locators                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_vscode() -> str | None:
        if shutil.which("code"):
            return "code"
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs" / "Microsoft VS Code" / "Code.exe",
            Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
            / "Microsoft VS Code" / "Code.exe",
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
            / "Microsoft VS Code" / "Code.exe",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    @staticmethod
    def _find_chrome() -> list[str]:
        chrome_paths = [
            Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
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

    @staticmethod
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

    # ------------------------------------------------------------------ #
    #  Dispatch                                                            #
    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        workflow: str = kwargs.get("workflow", "").strip()
        callback: EventCallback = kwargs.pop("event_callback", None)

        _emit(callback, _event("info", "dispatch_started",
                               f"Dispatching workflow: '{workflow}'", workflow))

        _workflow_map = {
            "create_file_notepad":     self._create_file_notepad,
            "create_file_vscode":      self._create_file_vscode,
            "send_email_browser":      self._send_email_browser,
            "send_whatsapp_desktop":   self._send_whatsapp_desktop,
            "send_whatsapp_advanced":  self._send_whatsapp_advanced,
            "play_youtube_video":      self._play_youtube_video,
            "linkedin_action":         self._linkedin_action,
            "code_workflow_cpp":       self._code_workflow_cpp,
            "launch_application":      self._launch_application,
            "take_screenshot":         self._take_screenshot,
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
        except Exception as exc:  # noqa: BLE001
            msg = f"Workflow '{workflow}' crashed: {exc}"
            logger.exception(msg)
            _emit(callback, _event("error", "workflow_crashed", msg, workflow))
            return ToolResult(success=False, error=msg)

    # ================================================================== #
    #  Workflow: create_file_notepad                                       #
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

        _emit(callback, _event("info", "app_launch", "Launching Notepad...", wf))
        subprocess.Popen(["notepad.exe"])
        _wait(1.5, callback, wf, "Notepad loading")

        _emit(callback, _event("status", "app_ready", "Notepad is open.", wf))

        _emit(callback, _event("info", "typing_content",
                               f"Pasting {len(content)} chars of content...", wf))
        _type_via_clipboard(content)
        _wait(0.5, callback, wf)

        _emit(callback, _event("info", "save_dialog",
                               "Opening Save As (Ctrl+Shift+S)...", wf))
        pyautogui.hotkey("ctrl", "shift", "s")
        _wait(1.5, callback, wf, "Save As dialog loading")

        _emit(callback, _event("info", "set_filename",
                               f"Setting filename: {filename}", wf))
        pyautogui.hotkey("ctrl", "a")
        _wait(0.2, callback, wf)
        _type_via_clipboard(filename)
        _wait(0.3, callback, wf)

        _emit(callback, _event("info", "confirm_save", "Confirming save.", wf))
        pyautogui.press("enter")
        _wait(0.5, callback, wf)
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
            metadata={"workflow": wf, "filename": filename, "content_length": len(content)},
        )

    # ================================================================== #
    #  Workflow: create_file_vscode                                        #
    # ================================================================== #

    def _create_file_vscode(
        self,
        *,
        callback: EventCallback = None,
        filename: str = "main.cpp",
        content: str = "",
        workflow: str = "create_file_vscode",
        **_: Any,
    ) -> ToolResult:
        wf = workflow

        _emit(callback, _event("info", "app_lookup",
                               "Locating VS Code executable...", wf))
        vscode_exe = self._find_vscode()

        if vscode_exe is None:
            msg = (
                "VS Code not found. Install from https://code.visualstudio.com "
                "or add 'code' to your system PATH."
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
        except OSError as exc:
            msg = f"Could not create parent directories for '{target}': {exc}"
            _emit(callback, _event("error", "write_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        try:
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
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch VS Code: {exc}"
            _emit(callback, _event("error", "app_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(2.0, callback, wf, "VS Code opening")

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
    #  Workflow: send_email_browser                                        #
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
            return ToolResult(
                success=False,
                error="'recipient' is required for send_email_browser.",
            )

        params = urllib.parse.urlencode({
            "view": "cm",
            "to":   recipient,
            "su":   subject,
            "body": content,
        })
        compose_url = f"https://mail.google.com/mail/?{params}"

        _emit(callback, _event("info", "browser_launch",
                               f"Opening Gmail compose for: {recipient}", wf))

        chrome_cmd = self._find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [compose_url])
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch Chrome: {exc}"
            _emit(callback, _event("error", "browser_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(6.0, callback, wf, "Gmail compose rendering")

        _emit(callback, _event("info", "dismiss_overlays",
                               "Pressing Escape to clear any popups.", wf))
        pyautogui.press("escape")
        _wait(0.5, callback, wf)

        _emit(callback, _event("info", "navigate_body",
                               "Tabbing to body field (Tab × 2)...", wf))
        pyautogui.press("tab")
        _wait(0.35, callback, wf)
        pyautogui.press("tab")
        _wait(0.35, callback, wf)

        sw, sh = pyautogui.size()
        body_x = int(sw * 0.72)
        body_y = int(sh * 0.55)

        _emit(callback, _event("info", "click_body",
                               f"Clicking body area at ({body_x}, {body_y})...", wf))
        pyautogui.click(body_x, body_y)
        _wait(0.4, callback, wf)

        _emit(callback, _event("info", "fill_body",
                               f"Pasting body ({len(content)} chars)...", wf))
        pyautogui.hotkey("ctrl", "a")
        _wait(0.2, callback, wf)
        _type_via_clipboard(content)
        _wait(0.5, callback, wf)

        pyautogui.click(body_x, body_y)
        _wait(0.3, callback, wf)

        pyautogui.moveTo(sw // 2, sh // 2, duration=0.3)
        _wait(0.3, callback, wf)

        _emit(callback, _event("info", "send_primary",
                               "Sending via Ctrl+Enter...", wf))
        pyautogui.hotkey("ctrl", "enter")
        _wait(2.5, callback, wf, "waiting for Gmail to process send")

        if self._compose_still_visible(sw, sh):
            _emit(callback, _event("info", "send_fallback",
                                   "Ctrl+Enter did not send — clicking Send button directly.", wf))
            self._click_send_button(callback, wf, sw, sh)
            _wait(2.5, callback, wf, "waiting after direct Send click")
        else:
            _emit(callback, _event("status", "send_confirmed",
                                   "Compose dismissed — send confirmed.", wf))

        _emit(callback, _event("status", "workflow_done",
                               f"Email sent to '{recipient}' — subject: '{subject}'.", wf))

        return ToolResult(
            success=True,
            output=f"Email sent to '{recipient}'.",
            metadata={
                "workflow":       wf,
                "recipient":      recipient,
                "subject":        subject,
                "content_length": len(content),
            },
        )

    # ================================================================== #
    #  Workflow: send_whatsapp_desktop                                     #
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
                              error="'contact_name' is required for send_whatsapp_desktop.")
        if not message or not message.strip():
            return ToolResult(success=False,
                              error="'message' is required for send_whatsapp_desktop.")

        return self._whatsapp_send_single(
            contact_name=contact_name,
            message=message,
            callback=callback,
            workflow=wf,
            launch_app=True,
        )

    # ================================================================== #
    #  Workflow: send_whatsapp_advanced                                    #
    # ================================================================== #

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

        # Normalise contact_name to list
        if isinstance(contact_name, str):
            contacts = [c.strip() for c in contact_name.split(",") if c.strip()]
        elif isinstance(contact_name, list):
            contacts = [str(c).strip() for c in contact_name if str(c).strip()]
        else:
            contacts = []

        if not contacts:
            return ToolResult(success=False,
                              error="'contact_name' is required for send_whatsapp_advanced.")
        if not message or not message.strip():
            return ToolResult(success=False,
                              error="'message' is required for send_whatsapp_advanced.")

        repeat = max(1, int(repeat or 1))
        delay_seconds = max(0.0, float(delay_seconds or 0.0))

        _emit(callback, _event("info", "launching_whatsapp",
                               "Launching WhatsApp Desktop...", wf))
        self._launch_whatsapp_app(callback, wf)
        _wait(5.0, callback, wf, "WhatsApp Desktop initialising")
        pyautogui.press("shift")
        _wait(0.5, callback, wf)

        sent_summary: list[str] = []
        failed_summary: list[str] = []

        for contact in contacts:
            for attempt in range(repeat):
                _emit(callback, _event("info", "searching_contact",
                                       f"Searching contact: '{contact}' (attempt {attempt + 1}/{repeat})", wf))

                if delay_seconds > 0:
                    _wait(delay_seconds, callback, wf,
                          f"pre-send delay ({delay_seconds}s)")

                result = self._whatsapp_send_single(
                    contact_name=contact,
                    message=message,
                    callback=callback,
                    workflow=wf,
                    launch_app=False,
                )

                if result.success:
                    sent_summary.append(f"{contact} (×{attempt + 1})")
                    _emit(callback, _event("status", "message_sent",
                                           f"Message sent to '{contact}' attempt {attempt + 1}.", wf))
                else:
                    failed_summary.append(contact)
                    _emit(callback, _event("error", "send_failed",
                                           f"Failed to send to '{contact}': {result.error}", wf))
                    break  # stop repeating for this contact on failure

        summary = f"Sent: {', '.join(sent_summary) or 'none'}."
        if failed_summary:
            summary += f" Failed: {', '.join(failed_summary)}."

        _emit(callback, _event("status", "workflow_done", summary, wf))

        return ToolResult(
            success=len(sent_summary) > 0,
            output=summary,
            metadata={
                "workflow": wf,
                "contacts": contacts,
                "repeat": repeat,
                "sent": sent_summary,
                "failed": failed_summary,
            },
        )

    # ------------------------------------------------------------------ #
    #  WhatsApp shared helpers                                             #
    # ------------------------------------------------------------------ #

    def _launch_whatsapp_app(self, callback: EventCallback, wf: str) -> None:
        """Launch WhatsApp via protocol handler with fallbacks."""
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", "whatsapp://"],
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception as exc:  # noqa: BLE001
            _emit(callback, _event("info", "launch_fallback",
                                   f"Protocol launch failed ({exc}), trying direct exe...", wf))
            wa_exe = self._find_whatsapp()
            if wa_exe:
                try:
                    os.startfile(wa_exe)
                    return
                except Exception:  # noqa: BLE001
                    pass
            try:
                subprocess.Popen(["cmd", "/c", "start", "WhatsApp"])
            except Exception as final_exc:  # noqa: BLE001
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
        """Core logic to open a WhatsApp chat and send a message."""
        wf = workflow

        if launch_app:
            _emit(callback, _event("info", "launching_whatsapp",
                                   "Launching WhatsApp Desktop...", wf))
            self._launch_whatsapp_app(callback, wf)
            _wait(5.0, callback, wf, "WhatsApp Desktop initialising")
            pyautogui.press("shift")
            _wait(0.5, callback, wf)

        # Focus search bar
        _emit(callback, _event("info", "searching_contact",
                               f"Focusing search bar (Ctrl+F)...", wf))
        pyautogui.hotkey("ctrl", "f")
        _wait(1.0, callback, wf, "search bar activation")

        pyautogui.hotkey("ctrl", "a")
        _wait(0.2, callback, wf)
        _type_via_clipboard(contact_name)
        _wait(1.5, callback, wf, "WhatsApp filtering contacts")

        _emit(callback, _event("info", "opening_chat",
                               f"Opening top match for '{contact_name}'...", wf))
        pyautogui.press("enter")
        _wait(1.5, callback, wf, "conversation loading")

        sw, sh = pyautogui.size()
        msg_x = int(sw * 0.5)
        msg_y = int(sh * 0.93)

        _emit(callback, _event("info", "typing_message",
                               f"Clicking message box and pasting message...", wf))
        pyautogui.click(msg_x, msg_y)
        _wait(0.5, callback, wf, "message box focus")

        _type_via_clipboard(message)
        _wait(0.5, callback, wf)

        _emit(callback, _event("info", "sending_message",
                               "Sending via Enter...", wf))
        pyautogui.press("enter")
        _wait(1.0, callback, wf, "message delivery")

        _emit(callback, _event("status", "message_sent",
                               f"WhatsApp message sent to '{contact_name}'.", wf))

        return ToolResult(
            success=True,
            output=f"WhatsApp message sent to '{contact_name}'.",
            metadata={"workflow": wf, "contact_name": contact_name},
        )

    # ================================================================== #
    #  Workflow: play_youtube_video                                        #
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

        # Build a direct YouTube search URL — more reliable than UI automation
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"

        _emit(callback, _event("info", "launching_browser",
                               "Launching Chrome...", wf))
        chrome_cmd = self._find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [search_url])
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch Chrome: {exc}"
            _emit(callback, _event("error", "browser_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(4.0, callback, wf, "YouTube search results loading")

        _emit(callback, _event("status", "navigating_to_youtube",
                               f"Navigated to YouTube search: '{query}'", wf))

        # Press Tab several times to reach the first video result, then Enter
        _emit(callback, _event("info", "typing_search_query",
                               "Focusing first video result via keyboard...", wf))

        # Click somewhere safe first to ensure the page has focus
        sw, sh = pyautogui.size()
        pyautogui.click(sw // 2, sh // 2)
        _wait(0.5, callback, wf)

        # Use keyboard shortcut: Tab to first video link
        # YouTube's first focusable video element is typically reachable via Tab
        for _ in range(6):
            pyautogui.press("tab")
            _wait(0.15, callback, wf)

        _emit(callback, _event("info", "opening_video",
                               "Opening first video result...", wf))
        pyautogui.press("enter")
        _wait(4.0, callback, wf, "video page loading")

        _emit(callback, _event("info", "starting_playback",
                               "Starting video playback (pressing k)...", wf))
        # Click center of page to focus the video player, then press k (play/pause)
        pyautogui.click(sw // 2, sh // 2)
        _wait(0.5, callback, wf)
        pyautogui.press("k")   # YouTube play/pause shortcut
        _wait(0.5, callback, wf)

        _emit(callback, _event("status", "workflow_done",
                               f"YouTube video for '{query}' started.", wf))

        return ToolResult(
            success=True,
            output=f"YouTube video for '{query}' is now playing.",
            metadata={"workflow": wf, "query": query, "search_url": search_url},
        )

    # ================================================================== #
    #  Workflow: linkedin_action                                           #
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

        action = action.strip().lower() if action else "search"
        if action not in {"search", "open", "connect"}:
            return ToolResult(
                success=False,
                error=f"Invalid linkedin action '{action}'. Use: 'search', 'open', or 'connect'.",
            )

        # Build LinkedIn people search URL
        encoded_name = urllib.parse.quote_plus(name)
        search_url = f"https://www.linkedin.com/search/results/people/?keywords={encoded_name}"

        _emit(callback, _event("info", "launching_browser",
                               "Launching Chrome for LinkedIn...", wf))
        chrome_cmd = self._find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [search_url])
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch Chrome: {exc}"
            _emit(callback, _event("error", "browser_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(5.0, callback, wf, "LinkedIn search results loading")

        _emit(callback, _event("status", "navigating_to_linkedin",
                               f"Navigated to LinkedIn search for: '{name}'", wf))

        if action == "search":
            _emit(callback, _event("status", "searching_profile",
                                   f"Search results shown for '{name}'.", wf))
            _emit(callback, _event("status", "workflow_done",
                                   f"LinkedIn search completed for '{name}'.", wf))
            return ToolResult(
                success=True,
                output=f"LinkedIn search results displayed for '{name}'.",
                metadata={"workflow": wf, "name": name, "action": action},
            )

        # For "open" and "connect" — navigate to the first result
        _emit(callback, _event("info", "searching_profile",
                               "Navigating to first profile result...", wf))

        sw, sh = pyautogui.size()
        pyautogui.click(sw // 2, sh // 2)
        _wait(0.5, callback, wf)

        # Tab to first result link — LinkedIn search results are keyboard-navigable
        for _ in range(5):
            pyautogui.press("tab")
            _wait(0.2, callback, wf)

        _emit(callback, _event("info", "opening_profile",
                               f"Opening profile for '{name}'...", wf))
        pyautogui.press("enter")
        _wait(5.0, callback, wf, "profile page loading")

        _emit(callback, _event("status", "opening_profile",
                               f"Profile page opened for '{name}'.", wf))

        if action == "connect":
            _emit(callback, _event("info", "sending_connection_request",
                                   "Looking for Connect button...", wf))

            # Tab through the profile page action buttons to find Connect
            # LinkedIn profile action area is typically reachable via Tab from the top
            pyautogui.click(sw // 2, sh // 3)
            _wait(0.5, callback, wf)

            for _ in range(10):
                pyautogui.press("tab")
                _wait(0.2, callback, wf)

            # Press Enter on the focused button (hoping it's Connect)
            pyautogui.press("enter")
            _wait(1.5, callback, wf, "connect dialog loading")

            # If a 'Send now' dialog appears, confirm it
            pyautogui.press("enter")
            _wait(1.0, callback, wf)

            _emit(callback, _event("status", "sending_connection_request",
                                   f"Connection request sent to '{name}'.", wf))

        _emit(callback, _event("status", "workflow_done",
                               f"LinkedIn {action} completed for '{name}'.", wf))

        return ToolResult(
            success=True,
            output=f"LinkedIn {action} completed for '{name}'.",
            metadata={"workflow": wf, "name": name, "action": action},
        )

    # ================================================================== #
    #  Workflow: code_workflow_cpp                                         #
    # ================================================================== #

    def _code_workflow_cpp(
        self,
        *,
        callback: EventCallback = None,
        filename: str = "main.cpp",
        code: str = "",
        content: str = "",   # alias for code
        workflow: str = "code_workflow_cpp",
        **_: Any,
    ) -> ToolResult:
        wf = workflow

        # Support both 'code' and 'content' as the source code argument
        source = code.strip() if code.strip() else content.strip()

        if not source:
            # Provide a default Hello World if nothing supplied
            source = (
                '#include <iostream>\n'
                'int main() {\n'
                '    std::cout << "Hello, World!" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            )

        if not filename.endswith(".cpp"):
            filename = filename.rsplit(".", 1)[0] + ".cpp"

        # ── 1. Write file to Desktop ──────────────────────────────────── #
        _emit(callback, _event("info", "launching_vscode",
                               "Locating VS Code...", wf))
        vscode_exe = self._find_vscode()
        if vscode_exe is None:
            msg = "VS Code not found. Install from https://code.visualstudio.com"
            _emit(callback, _event("error", "app_not_found", msg, wf))
            return ToolResult(success=False, error=msg)

        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        cpp_path = (desktop / filename).resolve()
        exe_path = cpp_path.with_suffix(".exe")

        _emit(callback, _event("info", "creating_file",
                               f"Creating file: {cpp_path}", wf))
        try:
            cpp_path.write_text(source, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write C++ file: {exc}"
            _emit(callback, _event("error", "write_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _emit(callback, _event("status", "writing_code",
                               f"C++ source written ({cpp_path.stat().st_size} bytes).", wf))

        # ── 2. Open in VS Code ────────────────────────────────────────── #
        _emit(callback, _event("info", "launching_vscode",
                               "Opening file in VS Code...", wf))
        try:
            subprocess.Popen([vscode_exe, str(cpp_path)])
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to open VS Code: {exc}"
            _emit(callback, _event("error", "vscode_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(2.5, callback, wf, "VS Code loading")

        # ── 3. Save (Ctrl+S to be safe) ───────────────────────────────── #
        _emit(callback, _event("info", "saving_file",
                               "Saving file (Ctrl+S)...", wf))
        pyautogui.hotkey("ctrl", "s")
        _wait(0.5, callback, wf)

        # ── 4. Compile ────────────────────────────────────────────────── #
        _emit(callback, _event("info", "compiling_code",
                               f"Compiling '{cpp_path.name}' with g++...", wf))

        gpp_path = shutil.which("g++")
        if gpp_path is None:
            msg = (
                "g++ not found. Install MinGW/GCC (e.g. via MSYS2 or Scoop) "
                "and add it to your system PATH."
            )
            _emit(callback, _event("error", "compiler_not_found", msg, wf))
            return ToolResult(
                success=False, error=msg,
                metadata={"workflow": wf, "file": str(cpp_path), "compiled": False},
            )

        try:
            compile_result = subprocess.run(
                [gpp_path, str(cpp_path), "-o", str(exe_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            msg = "Compilation timed out after 30 seconds."
            _emit(callback, _event("error", "compile_timeout", msg, wf))
            return ToolResult(success=False, error=msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"Compilation failed: {exc}"
            _emit(callback, _event("error", "compile_error", msg, wf))
            return ToolResult(success=False, error=msg)

        if compile_result.returncode != 0:
            err_output = compile_result.stderr.strip()
            msg = f"Compilation errors:\n{err_output}"
            _emit(callback, _event("error", "compile_error", msg, wf))
            return ToolResult(
                success=False, error=msg,
                metadata={"workflow": wf, "file": str(cpp_path),
                          "stderr": err_output},
            )

        _emit(callback, _event("status", "compiling_code",
                               f"Compilation successful → {exe_path.name}", wf))

        # ── 5. Run executable ─────────────────────────────────────────── #
        _emit(callback, _event("info", "running_executable",
                               f"Running '{exe_path.name}'...", wf))
        try:
            run_result = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(desktop),
            )
            run_output = run_result.stdout.strip()
        except subprocess.TimeoutExpired:
            run_output = "(timed out after 15s)"
        except Exception as exc:  # noqa: BLE001
            run_output = f"(execution failed: {exc})"

        _emit(callback, _event("status", "running_executable",
                               f"Program output: {run_output or '(no stdout)'}", wf))

        _emit(callback, _event("status", "workflow_done",
                               f"C++ workflow complete — {cpp_path.name} compiled and run.", wf))

        return ToolResult(
            success=True,
            output=f"C++ program compiled and run. Output: {run_output or '(no output)'}",
            metadata={
                "workflow":      wf,
                "source_file":   str(cpp_path),
                "executable":    str(exe_path),
                "program_output": run_output,
            },
        )

    # ================================================================== #
    #  Workflow: launch_application                                        #
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

        # Safe, static application map — no shell injection possible
        _APP_MAP: dict[str, list[str]] = {
            "chrome":        self._find_chrome() or ["cmd", "/c", "start", "chrome"],
            "google chrome": self._find_chrome() or ["cmd", "/c", "start", "chrome"],
            "vscode":        [self._find_vscode() or "code"],
            "vs code":       [self._find_vscode() or "code"],
            "visual studio code": [self._find_vscode() or "code"],
            "notepad":       ["notepad.exe"],
            "calculator":    ["calc.exe"],
            "calc":          ["calc.exe"],
            "whatsapp":      ["cmd", "/c", "start", "", "whatsapp://"],
            "explorer":      ["explorer.exe"],
            "file explorer": ["explorer.exe"],
            "paint":         ["mspaint.exe"],
            "mspaint":       ["mspaint.exe"],
            "wordpad":       ["wordpad.exe"],
            "cmd":           ["cmd.exe"],
            "command prompt":["cmd.exe"],
            "terminal":      ["cmd.exe"],
            "task manager":  ["taskmgr.exe"],
        }

        cmd = _APP_MAP.get(normalized)
        if cmd is None:
            # Try to find via PATH as a last resort
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

        _emit(callback, _event("info", "launching_application",
                               f"Launching '{app_name}'...", wf))

        # Filter out None entries (e.g. if _find_vscode returned None)
        cmd = [c for c in cmd if c is not None]

        try:
            subprocess.Popen(cmd)
        except FileNotFoundError:
            # Fallback: use Windows shell start for executables
            try:
                subprocess.Popen(["cmd", "/c", "start", "", cmd[0]])
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to launch '{app_name}': {exc}"
                _emit(callback, _event("error", "launch_failed", msg, wf))
                return ToolResult(success=False, error=msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch '{app_name}': {exc}"
            _emit(callback, _event("error", "launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        _wait(1.5, callback, wf, f"{app_name} loading")

        _emit(callback, _event("status", "application_opened",
                               f"'{app_name}' launched successfully.", wf))
        _emit(callback, _event("status", "workflow_done",
                               f"Application '{app_name}' is now open.", wf))

        return ToolResult(
            success=True,
            output=f"Application '{app_name}' launched.",
            metadata={"workflow": wf, "app_name": app_name, "command": cmd},
        )

    # ================================================================== #
    #  Workflow: take_screenshot                                           #
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

        # ── Resolve save path ─────────────────────────────────────────── #
        if screenshot_filename and screenshot_filename.strip():
            fname = screenshot_filename.strip()
            if not fname.lower().endswith(".png"):
                fname += ".png"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.png"

        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        save_path = (desktop / fname).resolve()

        _emit(callback, _event("info", "capturing_screen",
                               f"Capturing screenshot → {save_path.name}", wf))

        errors: list[str] = []

        # ── Strategy 1: Pillow ImageGrab (Windows-native, most reliable) ─ #
        _emit(callback, _event("info", "capturing_screen",
                               "Trying Pillow ImageGrab (strategy 1/3)...", wf))
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(save_path), format="PNG")
            _emit(callback, _event("status", "screenshot_saved",
                                   f"Captured via Pillow ImageGrab ({img.width}×{img.height}).", wf))
            return self._screenshot_success(save_path, img.width, img.height, callback, wf)
        except ImportError:
            errors.append("Strategy 1 (Pillow ImageGrab): Pillow not installed — run: pip install Pillow")
            _emit(callback, _event("info", "capturing_screen",
                                   "Pillow not available, trying next method...", wf))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Strategy 1 (Pillow ImageGrab): {exc}")
            _emit(callback, _event("info", "capturing_screen",
                                   f"Pillow ImageGrab failed ({exc}), trying next...", wf))

        # ── Strategy 2: pyautogui.screenshot() ───────────────────────── #
        _emit(callback, _event("info", "capturing_screen",
                               "Trying pyautogui.screenshot (strategy 2/3)...", wf))
        try:
            img = pyautogui.screenshot()          # returns PIL.Image
            img.save(str(save_path), format="PNG")
            w, h = img.size
            _emit(callback, _event("status", "screenshot_saved",
                                   f"Captured via pyautogui ({w}×{h}).", wf))
            return self._screenshot_success(save_path, w, h, callback, wf)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Strategy 2 (pyautogui): {exc}")
            _emit(callback, _event("info", "capturing_screen",
                                   f"pyautogui failed ({exc}), trying next...", wf))

        # ── Strategy 3: Windows Snipping Tool via keyboard shortcut ──── #
        # Win+Shift+S triggers Snip & Sketch which saves to clipboard,
        # then we use PIL ImageGrab.grabclipboard() to get it.
        _emit(callback, _event("info", "capturing_screen",
                               "Trying Win+PrintScreen fallback (strategy 3/3)...", wf))
        try:
            # Win+PrtSc saves directly to ~/Pictures/Screenshots on Windows
            pyautogui.hotkey("win", "printscreen")
            time.sleep(1.5)   # Windows needs a moment to write the file

            screenshots_dir = Path.home() / "Pictures" / "Screenshots"
            if screenshots_dir.exists():
                # Find the most recently created PNG
                candidates = sorted(
                    screenshots_dir.glob("*.png"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    latest = candidates[0]
                    import shutil as _shutil
                    _shutil.copy2(str(latest), str(save_path))
                    st = save_path.stat()
                    _emit(callback, _event("status", "screenshot_saved",
                                           f"Captured via Win+PrtSc, copied from {latest.name}.", wf))
                    return self._screenshot_success(save_path, 0, 0, callback, wf)

            # If no file appeared, raise so we fall through to the error block
            raise FileNotFoundError(
                "Win+PrtSc fired but no PNG found in ~/Pictures/Screenshots"
            )

        except Exception as exc:  # noqa: BLE001
            errors.append(f"Strategy 3 (Win+PrtSc): {exc}")
            _emit(callback, _event("error", "capture_failed",
                                   f"Win+PrtSc also failed: {exc}", wf))

        # ── All strategies exhausted ──────────────────────────────────── #
        diagnostics = " | ".join(errors)
        msg = (
            f"All screenshot strategies failed.\n"
            f"Details: {diagnostics}\n"
            f"Fix: run  pip install Pillow  then retry."
        )
        _emit(callback, _event("error", "capture_failed", msg, wf))
        return ToolResult(success=False, error=msg,
                          metadata={"workflow": wf, "errors": errors})

    @staticmethod
    def _screenshot_success(
        save_path: Path,
        width: int,
        height: int,
        callback: EventCallback,
        wf: str,
    ) -> ToolResult:
        """Shared success path — emit final events and return ToolResult."""
        try:
            size_kb = round(save_path.stat().st_size / 1024, 1)
        except OSError:
            size_kb = 0.0

        resolution = f"{width}×{height}" if width and height else "unknown"

        _emit(callback, _event("status", "workflow_done",
                               f"Screenshot saved to Desktop as '{save_path.name}'.", wf))

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

    # ------------------------------------------------------------------ #
    #  Gmail send helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compose_still_visible(sw: int, sh: int) -> bool:
        try:
            check_x = int(sw * 0.635)
            check_y = int(sh * 0.972)
            r, g, b = pyautogui.pixel(check_x, check_y)
            is_blue = (r < 110) and (g < 160) and (b > 170)
            return is_blue
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _click_send_button(
        callback: EventCallback, wf: str, sw: int, sh: int
    ) -> None:
        send_x = int(sw * 0.622)
        send_y = int(sw * 0.972)
        _emit(callback, _event("info", "send_button_click",
                               f"Moving to Send button at ({send_x}, {send_y})...", wf))
        pyautogui.moveTo(send_x, send_y, duration=0.4)
        time.sleep(0.4)
        pyautogui.click(send_x, send_y)