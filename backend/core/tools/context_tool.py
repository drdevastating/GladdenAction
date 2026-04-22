"""
core/tools/context_tool.py

ContextTool — project scanning, file discovery, and grep search.

Inspired by Claude Code's context discovery: scans the project tree,
finds files by pattern, and searches file contents with regex/plain text.

Actions
-------
    find_files      Find files by name pattern (glob-style).
    grep            Search file contents by regex or plain text.
    list_directory  List contents of a directory.
    get_tree        Get a tree view of a directory structure (summary).

Event contract
--------------
    type: info | status | error
    stage: <snake_case>
    message: <human readable>
    tool: context/<action>
    timestamp: ISO-8601
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

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
        "tool":      f"context/{action}",
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


class ContextTool(BaseTool):
    """
    Project context discovery — file finding and content search.

    Provides agents with visibility into the project structure and contents.
    """

    name = "context"
    description = (
        "Discover project files and search contents. "
        "Actions: find_files (glob pattern), grep (regex search), "
        "list_directory, get_tree"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["find_files", "grep", "list_directory", "get_tree"],
                "description": "The action to perform.",
            },
            "path": {
                "type": "string",
                "description": "Root directory path (defaults to current working directory).",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern for find_files, or regex/text for grep.",
            },
            "regex": {
                "type": "boolean",
                "description": "If True, treat pattern as regex; otherwise plain text search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default: 100).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, event_callback: EventCallback = None):
        """Initialize ContextTool with optional event callback."""
        self.event_callback = event_callback

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute a context discovery action.

        Supported actions:
          - find_files    : Find files by name pattern
          - grep          : Search file contents
          - list_directory: List directory contents
          - get_tree      : Get tree view of directory

        Args:
            action       : The action to perform
            path         : Root directory (default: current dir)
            pattern      : Search pattern (for find_files / grep)
            regex        : If True, use regex for grep
            max_results  : Maximum results to return

        Returns:
            ToolResult with success/output or error.
        """
        action = kwargs.get("action", "").strip()
        path_str = kwargs.get("path", ".")
        pattern = kwargs.get("pattern", "")
        use_regex = kwargs.get("regex", False)
        max_results = kwargs.get("max_results", 100)

        # Resolve and validate path
        root = _resolve(path_str)
        if not root:
            error_msg = f"Invalid path: {path_str}"
            _emit(self.event_callback, _event("error", "validation", error_msg, action))
            return ToolResult(success=False, error=error_msg)

        if _is_protected(root):
            error_msg = f"Protected path: {root}"
            _emit(self.event_callback, _event("error", "validation", error_msg, action))
            return ToolResult(success=False, error=error_msg)

        if not root.is_dir():
            error_msg = f"Not a directory: {root}"
            _emit(self.event_callback, _event("error", "validation", error_msg, action))
            return ToolResult(success=False, error=error_msg)

        try:
            if action == "find_files":
                return self._find_files(root, pattern, max_results)
            elif action == "grep":
                return self._grep(root, pattern, use_regex, max_results)
            elif action == "list_directory":
                return self._list_directory(root, max_results)
            elif action == "get_tree":
                return self._get_tree(root, max_results)
            else:
                error_msg = f"Unknown action: {action}"
                _emit(
                    self.event_callback,
                    _event("error", "validation", error_msg, action),
                )
                return ToolResult(success=False, error=error_msg)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{action} failed: {exc}"
            logger.exception("ContextTool.execute failed")
            _emit(self.event_callback, _event("error", "execution", error_msg, action))
            return ToolResult(success=False, error=error_msg)

    def _find_files(self, root: Path, pattern: str, max_results: int) -> ToolResult:
        """Find files matching a glob pattern."""
        if not pattern:
            pattern = "*"

        results = []
        try:
            for item in root.rglob(pattern):
                if len(results) >= max_results:
                    break
                if item.is_file():
                    results.append(str(item.relative_to(root)))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"glob search failed: {exc}")

        output = "\n".join(sorted(results)) if results else "(no matches)"
        _emit(
            self.event_callback,
            _event(
                "info",
                "find_files",
                f"Found {len(results)} file(s) matching {pattern!r}",
                "find_files",
            ),
        )
        return ToolResult(success=True, output=output, metadata={"count": len(results)})

    def _grep(
        self, root: Path, pattern: str, use_regex: bool, max_results: int
    ) -> ToolResult:
        """Search file contents for a pattern."""
        if not pattern:
            return ToolResult(success=False, error="pattern is required for grep")

        results = []
        compiled_pattern = None

        if use_regex:
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                return ToolResult(success=False, error=f"invalid regex: {exc}")

        try:
            for filepath in root.rglob("*"):
                if len(results) >= max_results:
                    break
                if not filepath.is_file():
                    continue

                # Skip binary files
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if use_regex:
                                if compiled_pattern.search(line):
                                    rel_path = str(filepath.relative_to(root))
                                    results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                            else:
                                if pattern.lower() in line.lower():
                                    rel_path = str(filepath.relative_to(root))
                                    results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                except (UnicodeDecodeError, OSError):
                    pass

        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"grep search failed: {exc}")

        output = "\n".join(results) if results else "(no matches)"
        _emit(
            self.event_callback,
            _event(
                "info",
                "grep",
                f"Found {len(results)} match(es) for {pattern!r}",
                "grep",
            ),
        )
        return ToolResult(success=True, output=output, metadata={"count": len(results)})

    def _list_directory(self, root: Path, max_results: int) -> ToolResult:
        """List contents of a directory."""
        try:
            items = []
            for item in sorted(root.iterdir()):
                if len(items) >= max_results:
                    break
                item_type = "DIR" if item.is_dir() else "FILE"
                items.append(f"{item_type:4}  {item.name}")

            output = "\n".join(items) if items else "(empty)"
            _emit(
                self.event_callback,
                _event(
                    "info",
                    "list_directory",
                    f"Listed {len(items)} item(s)",
                    "list_directory",
                ),
            )
            return ToolResult(success=True, output=output, metadata={"count": len(items)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"list_directory failed: {exc}")

    def _get_tree(self, root: Path, max_results: int, depth: int = 0) -> ToolResult:
        """Get a tree view of the directory structure."""
        lines = []
        visited = set()

        def _traverse(path: Path, prefix: str, d: int):
            if len(lines) >= max_results or d > 4:
                return

            try:
                items = sorted(path.iterdir())
            except PermissionError:
                return

            for i, item in enumerate(items):
                if item in visited or len(lines) >= max_results:
                    continue

                visited.add(item)
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "

                lines.append(prefix + current_prefix + item.name)

                if item.is_dir():
                    _traverse(item, prefix + next_prefix, d + 1)

        lines.append(root.name + "/")
        _traverse(root, "", 0)

        output = "\n".join(lines)
        _emit(
            self.event_callback,
            _event("info", "get_tree", f"Generated tree with {len(lines)} line(s)", "get_tree"),
        )
        return ToolResult(success=True, output=output, metadata={"lines": len(lines)})
