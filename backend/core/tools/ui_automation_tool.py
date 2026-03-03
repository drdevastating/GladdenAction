"""
core/tools/ui_automation_tool.py

UIAutomationTool — the primary visible execution mechanism of the agent.

Supported workflows
-------------------
    create_file_notepad     Open Notepad, type content, save as filename.
    create_file_vscode      Write content to disk, then open it in VS Code.
    send_email_browser      Open Gmail in Chrome with a pre-filled compose URL, then send.
    send_whatsapp_desktop   Open WhatsApp Desktop, find contact, type and send message.

VS Code strategy (create_file_vscode)
--------------------------------------
Instead of fighting VS Code's command palette and Save-As dialog (which are
notoriously timing-sensitive), we:
  1. Write the file directly to disk at the desired path.
  2. Open it in VS Code with:  code <path>
This is instant, reliable, and gives the user the file open and ready to edit.

Gmail strategy (send_email_browser)
------------------------------------
Tab-based navigation inside Gmail compose is unreliable because the compose
window is an async iframe and Chrome's focus order shifts depending on whether
CC/BCC rows are visible.  Instead we use Gmail's pre-filled compose URL:

    https://mail.google.com/mail/?view=cm&to=EMAIL&su=SUBJECT&body=BODY

Gmail reads these params and pre-populates To and Subject before rendering.

Send strategy — why Ctrl+Enter alone is not enough
---------------------------------------------------
The Gmail Send button has a small DROPDOWN ARROW immediately to its right
that opens "Schedule send". When the mouse drifts near it during rendering,
the dropdown pops open and STEALS keyboard focus from the compose iframe.
Ctrl+Enter is then swallowed by the popup instead of sending the message.

Fix — layered, focus-safe send sequence
----------------------------------------
1. Tab to body, then CLICK the body area (explicit iframe focus).
2. Press Escape to close any stray dropdown (safe no-op if nothing is open).
3. Click body AGAIN (Escape can blur the iframe).
4. Move the mouse to the centre of the screen — away from the Send/dropdown.
5. Ctrl+Enter — works whenever the compose iframe has keyboard focus.
6. Wait 2 s. Check if compose is still visible via pixel colour at the Send
   button position. Blue pixel ≈ button still present ≈ compose still open.
7. If compose is still open: click the LEFT HALF of the Send button directly
   (avoiding the dropdown arrow on the right half).

WhatsApp Desktop strategy (send_whatsapp_desktop)
--------------------------------------------------
Uses hybrid keyboard + mouse navigation for reliability:
  1. Launch WhatsApp Desktop via Windows Shell protocol (whatsapp://) to support
     both UWP (Microsoft Store) and standalone installations.
  2. Wait for the app to fully render.
  3. Focus the search bar with Ctrl+F (universal WhatsApp shortcut).
  4. Type the contact name — WhatsApp filters the list in real time.
  5. Press Enter to open the top matching conversation.
  6. Click the message input box directly (bottom-center) — Tab is unreliable.
  7. Type the message via clipboard paste (handles unicode safely).
  8. Press Enter to send.

UWP App Launch Strategy
------------------------
Microsoft Store apps (UWP) like WhatsApp cannot be launched directly via
os.startfile() or subprocess.Popen() on the .exe path. They require protocol
handlers. WhatsApp provides the 'whatsapp://' protocol which Windows resolves
to the correct app installation regardless of Store vs standalone installer.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import urllib.parse
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
    from datetime import datetime, timezone
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
        "or any instruction implying visible execution. "
        "Supported workflows: "
        "'create_file_notepad' — create/write a file using Notepad; "
        "'create_file_vscode' — create/write code in VS Code; "
        "'send_email_browser' — send email via Gmail in Chrome; "
        "'send_whatsapp_desktop' — send a WhatsApp message via WhatsApp Desktop app."
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
                    "'send_email_browser', 'send_whatsapp_desktop'."
                ),
            },
            "filename": {
                "type": "string",
                "description": "File name to create (file workflows).",
            },
            "content": {
                "type": "string",
                "description": "Text content to write (file/email workflows).",
            },
            "recipient": {
                "type": "string",
                "description": "Email address of the recipient (send_email_browser).",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line (send_email_browser).",
            },
            "contact_name": {
                "type": "string",
                "description": (
                    "Name of the WhatsApp contact to message (send_whatsapp_desktop). "
                    "Must match the display name in WhatsApp exactly or partially."
                ),
            },
            "message": {
                "type": "string",
                "description": "Message text to send (send_whatsapp_desktop).",
            },
        },
    }

    # ------------------------------------------------------------------ #
    #  Application locators                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_vscode() -> str | None:
        """Return the VS Code executable path, or None if not found."""
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
                logger.info("VS Code found: %s", p)
                return str(p)
        return None

    @staticmethod
    def _find_chrome() -> list[str]:
        """Return a command list that launches Chrome."""
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
        """
        Return the WhatsApp Desktop executable path, or None if not found.

        Checks, in priority order:
          1. Microsoft Store installation (WindowsApps — modern path)
          2. Classic WhatsApp standalone installer path
          3. PATH (unlikely but included for completeness)

        Notes
        -----
        The Microsoft Store version of WhatsApp is installed under:
            %LOCALAPPDATA%\\Microsoft\\WindowsApps\\WhatsApp.exe
        This is a thin launcher stub that works correctly with os.startfile()
        and subprocess.Popen — no special invocation needed.

        The legacy standalone installer places the binary at:
            %LOCALAPPDATA%\\WhatsApp\\WhatsApp.exe
        
        However, for UWP apps, we should use the protocol handler (whatsapp://)
        instead of direct execution, which is handled in the launch logic.
        """
        local = os.environ.get("LOCALAPPDATA", "")

        candidates = [
            # Microsoft Store stub (most common on Windows 11)
            Path(local) / "Microsoft" / "WindowsApps" / "WhatsApp.exe",
            # Legacy standalone installer
            Path(local) / "WhatsApp" / "WhatsApp.exe",
        ]

        for p in candidates:
            if p.exists():
                logger.info("WhatsApp found: %s", p)
                return str(p)

        # Fallback: check PATH
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
            "create_file_notepad":    self._create_file_notepad,
            "create_file_vscode":     self._create_file_vscode,
            "send_email_browser":     self._send_email_browser,
            "send_whatsapp_desktop":  self._send_whatsapp_desktop,
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
        pyautogui.press("enter")   # handle "overwrite?" prompt if shown
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
        """
        Write the file directly to disk, then open it in VS Code.

        Bare filenames (no directory) are placed on the Desktop so the
        user can find them easily. VS Code is opened with `code <path>`.
        """
        wf = workflow

        # ── 1. Locate VS Code ─────────────────────────────────────────── #
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

        # ── 2. Resolve target path ────────────────────────────────────── #
        target = Path(filename)
        if not target.is_absolute() and len(target.parts) == 1:
            desktop = Path.home() / "Desktop"
            desktop.mkdir(parents=True, exist_ok=True)
            target = desktop / filename
        target = target.resolve()

        _emit(callback, _event("info", "writing_file",
                               f"Writing file to disk: {target}", wf))

        # ── 3. Create parent directories ──────────────────────────────── #
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"Could not create parent directories for '{target}': {exc}"
            _emit(callback, _event("error", "write_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        # ── 4. Write content ──────────────────────────────────────────── #
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write file '{target}': {exc}"
            _emit(callback, _event("error", "write_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        size = target.stat().st_size
        _emit(callback, _event("status", "file_written",
                               f"File written ({size} bytes).", wf))

        # ── 5. Open in VS Code ────────────────────────────────────────── #
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

        # ── 1. Build pre-filled Gmail compose URL ─────────────────────── #
        params = urllib.parse.urlencode({
            "view": "cm",
            "to":   recipient,
            "su":   subject,
            "body": content,
        })
        compose_url = f"https://mail.google.com/mail/?{params}"

        _emit(callback, _event("info", "browser_launch",
                               f"Opening Gmail compose for: {recipient}", wf))

        # ── 2. Launch Chrome ──────────────────────────────────────────── #
        chrome_cmd = self._find_chrome()
        try:
            subprocess.Popen(chrome_cmd + [compose_url])
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to launch Chrome: {exc}"
            _emit(callback, _event("error", "browser_launch_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        # ── 3. Wait for Gmail compose to render ───────────────────────── #
        _wait(6.0, callback, wf, "Gmail compose rendering")

        # ── 4. Escape any dropdown / tooltip ─────────────────────────── #
        _emit(callback, _event("info", "dismiss_overlays",
                               "Pressing Escape to clear any popups.", wf))
        pyautogui.press("escape")
        _wait(0.5, callback, wf)

        # ── 5. Tab to body field ──────────────────────────────────────── #
        _emit(callback, _event("info", "navigate_body",
                               "Tabbing to body field (Tab × 2)...", wf))
        pyautogui.press("tab")
        _wait(0.35, callback, wf)
        pyautogui.press("tab")
        _wait(0.35, callback, wf)

        # ── 6. Click body area — explicit iframe focus ────────────────── #
        sw, sh = pyautogui.size()
        body_x = int(sw * 0.72)
        body_y = int(sh * 0.55)

        _emit(callback, _event("info", "click_body",
                               f"Clicking body area at ({body_x}, {body_y})...", wf))
        pyautogui.click(body_x, body_y)
        _wait(0.4, callback, wf)

        # ── 7. Paste body content ─────────────────────────────────────── #
        _emit(callback, _event("info", "fill_body",
                               f"Pasting body ({len(content)} chars)...", wf))
        pyautogui.hotkey("ctrl", "a")
        _wait(0.2, callback, wf)
        _type_via_clipboard(content)
        _wait(0.5, callback, wf)

        # ── 8. Re-click body to restore focus after Ctrl+A ────────────── #
        pyautogui.click(body_x, body_y)
        _wait(0.3, callback, wf)

        # ── 9. Park mouse in screen centre — away from Send dropdown ──── #
        pyautogui.moveTo(sw // 2, sh // 2, duration=0.3)
        _wait(0.3, callback, wf)

        # ── 10. Primary send: Ctrl+Enter ──────────────────────────────── #
        _emit(callback, _event("info", "send_primary",
                               "Sending via Ctrl+Enter...", wf))
        pyautogui.hotkey("ctrl", "enter")
        _wait(2.5, callback, wf, "waiting for Gmail to process send")

        # ── 11. Check if compose is still open ────────────────────────── #
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
        """
        Send a WhatsApp message via the WhatsApp Desktop app.

        Strategy
        --------
        Uses hybrid keyboard + mouse approach for maximum reliability:
        - Keyboard shortcuts for search and contact selection
        - Mouse click to focus the message input field (Tab is unreliable)
        - Clipboard paste for message content (handles unicode)
        - Enter key to send

        Keyboard shortcuts used
        -----------------------
        Ctrl+F      → Focus the search / new-chat bar (universal WA Desktop shortcut)
        Enter       → Open the top search result (the matched conversation)
        Ctrl+V      → Paste message content from clipboard
        Enter       → Send the message

        Why we click instead of Tab
        ----------------------------
        WhatsApp Desktop's focus order is inconsistent. After opening a chat,
        Tab may land on the voice message button, attachment button, or other
        elements instead of the message input box. Clicking the bottom-center
        of the screen reliably focuses the message input field.

        Assumptions
        -----------
        - WhatsApp Desktop is installed (Microsoft Store or standalone).
        - The user is already logged in.
        - The contact_name matches (at least partially) a name in WhatsApp contacts.
        - The app opens maximised so the layout is stable.

        Parameters
        ----------
        contact_name : str
            Display name (or prefix) of the WhatsApp contact.
        message : str
            Message body to send.
        """
        wf = workflow

        # ── Guard: required fields ────────────────────────────────────── #
        if not contact_name or not contact_name.strip():
            msg = "'contact_name' is required for send_whatsapp_desktop."
            _emit(callback, _event("error", "validation_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        if not message or not message.strip():
            msg = "'message' is required for send_whatsapp_desktop."
            _emit(callback, _event("error", "validation_failed", msg, wf))
            return ToolResult(success=False, error=msg)

        # ── 1. Locate WhatsApp executable (for logging purposes) ──────── #
        _emit(callback, _event("info", "app_lookup",
                               "Checking for WhatsApp Desktop installation...", wf))

        whatsapp_exe = self._find_whatsapp()

        if whatsapp_exe is None:
            logger.warning("WhatsApp executable not found in standard locations. "
                         "Will attempt protocol launch anyway.")
            _emit(callback, _event("info", "app_lookup_fallback",
                                   "WhatsApp not found via standard paths — using protocol handler.", wf))
        else:
            _emit(callback, _event("status", "app_lookup_ok",
                                   f"WhatsApp found: {whatsapp_exe}", wf))

        # ── 2. Launch WhatsApp via protocol handler ───────────────────── #
        _emit(callback, _event("info", "launching_whatsapp",
                               "Launching WhatsApp Desktop via protocol handler...", wf))
        try:
            # Primary method: Use whatsapp:// protocol
            # This works for both UWP (Microsoft Store) and standalone installations
            # Windows resolves the protocol to the correct app automatically
            subprocess.Popen(
                ["cmd", "/c", "start", "", "whatsapp://"],
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            _emit(callback, _event("status", "launch_success",
                                   "WhatsApp protocol handler invoked successfully.", wf))
        except Exception as exc:  # noqa: BLE001
            # Fallback 1: Try direct execution if we found an exe path
            _emit(callback, _event("info", "launch_fallback_1",
                                   f"Protocol launch failed ({exc}), trying direct execution...", wf))
            try:
                if whatsapp_exe and Path(whatsapp_exe).exists():
                    os.startfile(whatsapp_exe)
                    _emit(callback, _event("status", "launch_fallback_1_success",
                                           "Direct execution succeeded.", wf))
                else:
                    raise FileNotFoundError("No WhatsApp executable path available.")
            except Exception as inner_exc:  # noqa: BLE001
                # Fallback 2: Try Windows shell association as last resort
                _emit(callback, _event("info", "launch_fallback_2",
                                       f"Direct execution failed ({inner_exc}), trying shell association...", wf))
                try:
                    subprocess.Popen(["cmd", "/c", "start", "WhatsApp"])
                    _emit(callback, _event("status", "launch_fallback_2_success",
                                           "Shell association launch succeeded.", wf))
                except Exception as final_exc:  # noqa: BLE001
                    msg = (
                        f"All WhatsApp launch methods failed. "
                        f"Protocol: {exc} | Direct: {inner_exc} | Shell: {final_exc}. "
                        f"Please install WhatsApp Desktop from https://www.whatsapp.com/download "
                        f"or the Microsoft Store."
                    )
                    _emit(callback, _event("error", "app_launch_failed", msg, wf))
                    return ToolResult(success=False, error=msg)

        # ── 3. Wait for app to fully load ─────────────────────────────── #
        _emit(callback, _event("info", "waiting_for_app_ready",
                               "Waiting for WhatsApp Desktop to load (5 s)...", wf))
        _wait(5.0, callback, wf, "WhatsApp Desktop initialising")

        # Bring window to foreground — Alt+Tab is unreliable across focus managers.
        # Pressing a neutral key ensures the OS acknowledges the window.
        pyautogui.press("shift")   # innocuous key to capture focus
        _wait(0.5, callback, wf)

        # ── 4. Focus the search bar (Ctrl+F) ──────────────────────────── #
        _emit(callback, _event("info", "focusing_search_bar",
                               "Focusing WhatsApp search bar via Ctrl+F...", wf))
        pyautogui.hotkey("ctrl", "f")
        _wait(1.0, callback, wf, "search bar activation")

        # ── 5. Type contact name ──────────────────────────────────────── #
        _emit(callback, _event("info", "typing_contact_name",
                               f"Typing contact name: '{contact_name}'...", wf))
        # Clear any pre-existing search text, then type
        pyautogui.hotkey("ctrl", "a")
        _wait(0.2, callback, wf)
        # Use clipboard paste to handle unicode and special characters reliably
        _type_via_clipboard(contact_name)
        _wait(1.5, callback, wf, "WhatsApp filtering contacts")

        # ── 6. Select the top matching contact (Enter) ────────────────── #
        _emit(callback, _event("info", "selecting_contact",
                               f"Selecting top match for '{contact_name}' via Enter...", wf))
        pyautogui.press("enter")
        _wait(1.5, callback, wf, "conversation loading")

        # ── 7. Click the message input field ──────────────────────────── #
        # Tab navigation is unreliable because WhatsApp's UI order varies.
        # Instead, we click the message input area directly (bottom-center).
        sw, sh = pyautogui.size()
        message_box_x = int(sw * 0.5)   # Center horizontally
        message_box_y = int(sh * 0.93)  # Near bottom (message input area)
        
        _emit(callback, _event("info", "clicking_message_box",
                               f"Clicking message input at ({message_box_x}, {message_box_y})...", wf))
        pyautogui.click(message_box_x, message_box_y)
        _wait(0.5, callback, wf, "message box focus")

        # ── 8. Type the message ───────────────────────────────────────── #
        _emit(callback, _event("info", "typing_message",
                               f"Pasting message ({len(message)} chars)...", wf))
        _type_via_clipboard(message)
        _wait(0.5, callback, wf)

        # ── 9. Send the message (Enter) ───────────────────────────────── #
        _emit(callback, _event("info", "sending_message",
                               "Sending message via Enter key...", wf))
        pyautogui.press("enter")
        _wait(1.0, callback, wf, "message delivery confirmation")

        # ── 10. Done ──────────────────────────────────────────────────── #
        _emit(callback, _event("status", "message_sent",
                               f"WhatsApp message sent to '{contact_name}'.", wf))
        _emit(callback, _event("status", "workflow_done",
                               f"send_whatsapp_desktop completed for contact='{contact_name}'.", wf))

        return ToolResult(
            success=True,
            output=f"WhatsApp message sent to '{contact_name}'.",
            metadata={
                "workflow":        wf,
                "contact_name":    contact_name,
                "message_length":  len(message),
            },
        )

    # ------------------------------------------------------------------ #
    #  Gmail send helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compose_still_visible(sw: int, sh: int) -> bool:
        """
        Pixel-colour heuristic: is the Gmail Send button still on screen?

        The Send button is a distinctive blue (#1A73E8 family).
        We sample a pixel at the expected button position.
        If it's blue → compose is still open.
        If it's white/grey/gone → compose was dismissed → send succeeded.
        """
        try:
            check_x = int(sw * 0.635)
            check_y = int(sh * 0.972)
            r, g, b = pyautogui.pixel(check_x, check_y)
            is_blue = (r < 110) and (g < 160) and (b > 170)
            logger.debug(
                "Send-button pixel check (%d,%d) = (%d,%d,%d) → still_open=%s",
                check_x, check_y, r, g, b, is_blue,
            )
            return is_blue
        except Exception as exc:  # noqa: BLE001
            logger.warning("_compose_still_visible failed: %s — assuming closed.", exc)
            return False

    @staticmethod
    def _click_send_button(
        callback: EventCallback, wf: str, sw: int, sh: int
    ) -> None:
        """
        Click the LEFT HALF of the Gmail Send button to avoid the dropdown arrow.
        """
        send_x = int(sw * 0.622)
        send_y = int(sw * 0.972)

        _emit(callback, _event("info", "send_button_click",
                               f"Moving to Send button at ({send_x}, {send_y})...", wf))

        pyautogui.moveTo(send_x, send_y, duration=0.4)
        time.sleep(0.4)
        pyautogui.click(send_x, send_y)