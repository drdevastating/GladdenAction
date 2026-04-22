"""
core/tools/code_edit_tool.py

CodeEditTool — Claude Code-style surgical file editing.

Inspired by Claude Code's core edit primitive: instead of overwriting
whole files, this tool reads with line numbers, applies targeted
str_replace diffs, and inserts at specific line positions.

Actions
-------
    read_file       Read a file with line numbers (+ optional range).
    edit_file       Replace a unique string occurrence (str_replace diff).
    insert_lines    Insert text before a given line number.
    create_file     Create a new file (fails if exists unless overwrite=True).
    delete_file     Delete a single file (protected-path aware).

Security
--------
    - Protected paths from SystemControlTool are reused (no Windows system dirs).
    - No shell execution — pure Python pathlib only.
    - Max file size: 2 MB for reads to avoid accidental binary dumps.

Event contract
--------------
    type: info | status | error
    stage: <snake_case>
    message: <human readable>
    tool: code_edit/<action>
    timestamp: ISO-8601
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

_MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MB

_PROTECTED_PREFIXES = (
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/lib", "/boot", "/sys", "/proc", "/root",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, action: str) -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"code_edit/{action}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _is_protected(path: Path) -> bool:
    s = str(path).lower()
    if s == str(path.anchor).lower():
        return True
    return any(s.startswith(p) for p in _PROTECTED_PREFIXES) or "appdata" in s


def _resolve(raw: str) -> Path | None:
    try:
        return Path(raw).resolve()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------

class CodeEditTool(BaseTool):
    """
    Surgical file read/edit/insert — Claude Code style.

    This is the workhorse for agentic code-editing workflows:
    read → understand → diff → verify.
    """

    name = "code_edit"

    description = (
        "Claude Code-style surgical file editing. "
        "Actions: "
        "'read_file' — read a file with line numbers (use start_line/end_line for ranges); "
        "'edit_file' — replace a unique string in a file (old_str → new_str); "
        "'insert_lines' — insert text before a specific line number; "
        "'create_file' — create a new file with content; "
        "'delete_file' — delete a single file. "
        "Use this for all code editing tasks. Always read_file first, then edit_file."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["action", "path"],
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: read_file | edit_file | insert_lines | create_file | delete_file",
            },
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the target file.",
            },
            "old_str": {
                "type": "string",
                "description": "(edit_file) The exact string to replace. Must appear exactly once in the file.",
            },
            "new_str": {
                "type": "string",
                "description": "(edit_file) The replacement string.",
            },
            "content": {
                "type": "string",
                "description": "(create_file | insert_lines) Content to write or insert.",
            },
            "insert_line": {
                "type": "integer",
                "description": "(insert_lines) Insert BEFORE this 1-based line number.",
            },
            "start_line": {
                "type": "integer",
                "description": "(read_file) First line to return (1-based, inclusive).",
            },
            "end_line": {
                "type": "integer",
                "description": "(read_file) Last line to return (1-based, inclusive).",
            },
            "overwrite": {
                "type": "boolean",
                "description": "(create_file) Overwrite if file exists. Default false.",
            },
        },
    }

    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)
        action = kwargs.get("action", "").strip().lower()
        path_raw = kwargs.get("path", "").strip()

        dispatch = {
            "read_file":    self._read_file,
            "edit_file":    self._edit_file,
            "insert_lines": self._insert_lines,
            "create_file":  self._create_file,
            "delete_file":  self._delete_file,
        }

        if action not in dispatch:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Must be one of: {list(dispatch.keys())}",
            )
        if not path_raw:
            return ToolResult(success=False, error="'path' is required.")

        path = _resolve(path_raw)
        if path is None:
            return ToolResult(success=False, error=f"Cannot resolve path: '{path_raw}'")

        try:
            return dispatch[action](path=path, cb=cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = f"code_edit/{action} crashed: {exc}"
            logger.exception(msg)
            _emit(cb, _event("error", "tool_crashed", msg, action))
            return ToolResult(success=False, error=msg)

    # ================================================================== #
    #  read_file                                                           #
    # ================================================================== #

    def _read_file(
        self,
        *,
        path: Path,
        cb: EventCallback,
        start_line: int = 0,
        end_line: int = 0,
        **_: Any,
    ) -> ToolResult:
        action = "read_file"

        _emit(cb, _event("info", "reading_file", f"Reading '{path.name}'…", action))

        if not path.exists():
            return ToolResult(success=False, error=f"File not found: '{path}'")
        if not path.is_file():
            return ToolResult(success=False, error=f"Path is not a file: '{path}'")
        if path.stat().st_size > _MAX_READ_BYTES:
            return ToolResult(
                success=False,
                error=f"File too large ({path.stat().st_size // 1024} KB > 2 MB limit).",
            )

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot read '{path}': {exc}")

        lines = raw.splitlines()
        total = len(lines)

        # Apply range if requested
        lo = max(1, int(start_line)) if start_line else 1
        hi = min(total, int(end_line)) if end_line else total

        selected = lines[lo - 1 : hi]

        # Build output with line numbers (like `cat -n`)
        numbered = "\n".join(f"{lo + i:>6}\t{line}" for i, line in enumerate(selected))

        _emit(cb, _event(
            "status", "file_read",
            f"Read {len(selected)} line(s) (of {total} total) from '{path.name}'.",
            action,
        ))

        return ToolResult(
            success=True,
            output=numbered,
            metadata={
                "path": str(path),
                "total_lines": total,
                "returned_lines": len(selected),
                "start_line": lo,
                "end_line": hi,
            },
        )

    # ================================================================== #
    #  edit_file  (str_replace diff)                                      #
    # ================================================================== #

    def _edit_file(
        self,
        *,
        path: Path,
        cb: EventCallback,
        old_str: str = "",
        new_str: str = "",
        **_: Any,
    ) -> ToolResult:
        action = "edit_file"

        if not old_str:
            return ToolResult(success=False, error="'old_str' is required for edit_file.")

        _emit(cb, _event("info", "reading_for_edit", f"Reading '{path.name}' for edit…", action))

        if not path.exists():
            return ToolResult(success=False, error=f"File not found: '{path}'")
        if _is_protected(path):
            return ToolResult(success=False, error=f"Path '{path}' is protected.")

        try:
            original = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot read '{path}': {exc}")

        count = original.count(old_str)
        if count == 0:
            return ToolResult(
                success=False,
                error=(
                    f"'old_str' not found in '{path.name}'. "
                    "Ensure it exactly matches the file content (including whitespace)."
                ),
            )
        if count > 1:
            return ToolResult(
                success=False,
                error=(
                    f"'old_str' appears {count} times in '{path.name}'. "
                    "It must appear exactly once for a safe replacement."
                ),
            )

        updated = original.replace(old_str, new_str, 1)

        _emit(cb, _event("info", "applying_edit", f"Applying edit to '{path.name}'…", action))

        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot write '{path}': {exc}")

        lines_before = original.count("\n")
        lines_after  = updated.count("\n")
        delta        = lines_after - lines_before

        _emit(cb, _event(
            "status", "edit_applied",
            f"Edit applied to '{path.name}' (lines: {lines_before} → {lines_after}, Δ{delta:+d}).",
            action,
        ))

        return ToolResult(
            success=True,
            output=f"Edit applied to '{path.name}'.",
            metadata={
                "path": str(path),
                "lines_before": lines_before,
                "lines_after": lines_after,
                "delta_lines": delta,
            },
        )

    # ================================================================== #
    #  insert_lines                                                        #
    # ================================================================== #

    def _insert_lines(
        self,
        *,
        path: Path,
        cb: EventCallback,
        content: str = "",
        insert_line: int = 0,
        **_: Any,
    ) -> ToolResult:
        action = "insert_lines"

        if not content:
            return ToolResult(success=False, error="'content' is required for insert_lines.")
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: '{path}'")
        if _is_protected(path):
            return ToolResult(success=False, error=f"Path '{path}' is protected.")

        try:
            original = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot read '{path}': {exc}")

        lines = original.splitlines(keepends=True)
        total = len(lines)

        # Default to end-of-file
        pos = max(0, min(int(insert_line) - 1, total)) if insert_line else total

        new_lines_text = content if content.endswith("\n") else content + "\n"
        new_lines = new_lines_text.splitlines(keepends=True)

        result_lines = lines[:pos] + new_lines + lines[pos:]

        _emit(cb, _event("info", "inserting_lines",
                          f"Inserting {len(new_lines)} line(s) at line {pos + 1} in '{path.name}'…",
                          action))

        try:
            path.write_text("".join(result_lines), encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot write '{path}': {exc}")

        _emit(cb, _event("status", "lines_inserted",
                          f"Inserted {len(new_lines)} line(s) before line {pos + 1} in '{path.name}'.",
                          action))

        return ToolResult(
            success=True,
            output=f"Inserted {len(new_lines)} line(s) into '{path.name}'.",
            metadata={"path": str(path), "insert_before_line": pos + 1, "lines_added": len(new_lines)},
        )

    # ================================================================== #
    #  create_file                                                         #
    # ================================================================== #

    def _create_file(
        self,
        *,
        path: Path,
        cb: EventCallback,
        content: str = "",
        overwrite: bool = False,
        **_: Any,
    ) -> ToolResult:
        action = "create_file"

        if path.exists() and not overwrite:
            return ToolResult(
                success=False,
                error=f"File already exists: '{path}'. Pass overwrite=true to replace it.",
            )
        if _is_protected(path):
            return ToolResult(success=False, error=f"Path '{path}' is protected.")

        _emit(cb, _event("info", "creating_file", f"Creating '{path.name}'…", action))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot create '{path}': {exc}")

        size = path.stat().st_size
        _emit(cb, _event("status", "file_created",
                          f"Created '{path.name}' ({size} bytes).", action))

        return ToolResult(
            success=True,
            output=str(path),
            metadata={"path": str(path), "size_bytes": size},
        )

    # ================================================================== #
    #  delete_file                                                         #
    # ================================================================== #

    def _delete_file(
        self,
        *,
        path: Path,
        cb: EventCallback,
        **_: Any,
    ) -> ToolResult:
        action = "delete_file"

        if not path.exists():
            return ToolResult(success=False, error=f"File not found: '{path}'")
        if not path.is_file():
            return ToolResult(success=False, error=f"Not a file: '{path}'")
        if _is_protected(path):
            return ToolResult(success=False, error=f"Path '{path}' is protected.")

        _emit(cb, _event("info", "deleting_file", f"Deleting '{path.name}'…", action))

        try:
            path.unlink()
        except OSError as exc:
            return ToolResult(success=False, error=f"Cannot delete '{path}': {exc}")

        _emit(cb, _event("status", "file_deleted", f"Deleted '{path.name}'.", action))

        return ToolResult(
            success=True,
            output=str(path),
            metadata={"deleted_path": str(path)},
        )