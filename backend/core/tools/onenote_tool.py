"""
core/tools/onenote_tool.py

OneNoteTool — Microsoft OneNote automation via PyAutoGUI + Pyperclip.

Typing strategy
---------------
  page title  → slow_type()   (short, tab-header updates per keystroke)
  body text   → paste_text()  when len > 80 chars, else slow_type()
  code blocks → paste_text()  always (preserve indentation perfectly)

Actions
-------
  open              Open OneNote (launches app)
  create_note       Create a new page with title + body content
  add_to_page       Add content to the currently open/active page
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
    click, focus_app, hotkey, paste_text, press_key, slow_type, wait,
)

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

_PASTE_THRESHOLD = 80  # chars — above this use paste_text, below use slow_type


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, action: str = "onenote") -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"onenote/{action}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _launch_onenote() -> bool:
    """Launch OneNote via URI, shutil.which, or Start-menu fallback."""
    for cmd in [
        ["cmd", "/c", "start", "", "onenote:"],
        ["cmd", "/c", "start", "OneNote"],
    ]:
        try:
            subprocess.Popen(
                cmd, shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("OneNote launch failed (%s): %s", cmd[0], exc)

    # Start-menu search
    try:
        pyautogui.hotkey("win")
        time.sleep(0.7)
        slow_type("onenote", interval=0.08)
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("OneNote Start-menu launch failed: %s", exc)
    return False


def _type_content(content: str, cb: EventCallback, action: str) -> None:
    """
    Intelligently choose between slow_type and paste_text based on content length
    and presence of special characters.
    """
    if not content:
        return
    use_paste = (
        len(content) > _PASTE_THRESHOLD
        or "\n" in content
        or any(ord(c) > 127 for c in content)
    )
    mode = "paste" if use_paste else "slow"
    _emit(cb, _event("info", "content_input",
                      f"Adding content via {mode} mode ({len(content)} chars)…", action))
    if use_paste:
        paste_text(content)
    else:
        slow_type(content, interval=0.045)
    wait(0.3)


class OneNoteTool(BaseTool):
    """
    Automates Microsoft OneNote to open the app and manage notes/pages.
    Combines PyAutoGUI (actions) + Pyperclip (content) for reliability + visibility.
    """

    name = "onenote"

    description = (
        "Automates Microsoft OneNote to open the app and create/edit notes. "
        "Actions: "
        "'open' — open OneNote; "
        "'create_note' — create a new page with title and content; "
        "'add_to_page' — add content to the currently visible page."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "create_note", "add_to_page"],
            },
            "title": {
                "type": "string",
                "description": "Page title (used with create_note).",
            },
            "content": {
                "type": "string",
                "description": "Note body / content. Long content is pasted automatically.",
            },
            "section": {
                "type": "string",
                "description": "OneNote section name to create the page in (optional).",
            },
        },
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)
        action = kwargs.get("action", "").strip().lower()

        dispatch = {
            "open":         self._action_open,
            "create_note":  self._action_create_note,
            "add_to_page":  self._action_add_to_page,
        }

        if action not in dispatch:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Must be: {list(dispatch.keys())}",
            )

        try:
            return dispatch[action](cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"OneNoteTool/{action} crashed: {exc}"
            logger.exception(msg)
            _emit(cb, _event("error", "tool_crashed", msg, action))
            return ToolResult(success=False, error=msg)

    def _action_open(self, *, cb: EventCallback, **_: Any) -> ToolResult:
        action = "open"
        _emit(cb, _event("info", "launching", "Launching Microsoft OneNote…", action))
        if not _launch_onenote():
            return ToolResult(success=False, error="Failed to launch OneNote.")
        wait(4.0, "OneNote initialising")
        focus_app("OneNote")
        wait(0.8)
        _emit(cb, _event("status", "onenote_open", "OneNote is open.", action))
        return ToolResult(success=True, output="OneNote opened.",
                          metadata={"action": action})

    def _action_create_note(
        self,
        *,
        cb: EventCallback,
        title: str = "New Note",
        content: str = "",
        section: str = "",
        **_: Any,
    ) -> ToolResult:
        action = "create_note"

        _emit(cb, _event("info", "launching", "Launching OneNote to create note…", action))
        if not _launch_onenote():
            return ToolResult(success=False, error="Failed to launch OneNote.")
        wait(4.0, "OneNote initialising")
        focus_app("OneNote")
        wait(0.8)

        # Create new page: Ctrl+N in OneNote adds a page
        _emit(cb, _event("info", "new_page", "Creating new page (Ctrl+N)…", action))
        hotkey("ctrl", "n")
        wait(1.5, "New page loading")

        # Title field is at top — it has focus automatically on new page
        _emit(cb, _event("info", "typing_title",
                          f"Typing title: '{title}'…", action))
        hotkey("ctrl", "a")
        wait(0.15)
        # SLOW TYPE: title is short, typing visible in tab label
        slow_type(title, interval=0.07)
        wait(0.35)

        # Tab / Enter to move to page body
        press_key("enter")
        wait(0.5)

        # Add body content
        if content:
            _type_content(content, cb, action)

        # Save (OneNote auto-saves, but Ctrl+S triggers manual save)
        hotkey("ctrl", "s")
        wait(0.6)

        _emit(cb, _event("status", "note_created",
                          f"OneNote page '{title}' created.", action))
        return ToolResult(
            success=True,
            output=f"OneNote page '{title}' created with {len(content)} chars of content.",
            metadata={"action": action, "title": title, "content_length": len(content)},
        )

    def _action_add_to_page(
        self,
        *,
        cb: EventCallback,
        content: str = "",
        **_: Any,
    ) -> ToolResult:
        action = "add_to_page"
        if not content:
            return ToolResult(success=False, error="'content' is required for add_to_page.")

        _emit(cb, _event("info", "focusing", "Focusing OneNote page body…", action))
        focus_app("OneNote")
        wait(0.8)

        sw, sh = pyautogui.size()
        # Click in the main content area (centre of page)
        click(int(sw * 0.55), int(sh * 0.55), duration=0.35)
        wait(0.4)

        # Move to end of existing content
        hotkey("ctrl", "end")
        wait(0.25)
        press_key("enter")
        wait(0.2)

        _type_content(content, cb, action)
        hotkey("ctrl", "s")
        wait(0.5)

        _emit(cb, _event("status", "content_added",
                          f"Added {len(content)} chars to active OneNote page.", action))
        return ToolResult(
            success=True,
            output=f"Content added to OneNote page ({len(content)} chars).",
            metadata={"action": action, "content_length": len(content)},
        )