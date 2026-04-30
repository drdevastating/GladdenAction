"""
core/tools/nl_command_tool.py

NLCommandTool — Natural Language → Shell Command Converter

Converts plain-English descriptions into Windows CMD / PowerShell commands
using the Groq LLM, then executes them through the existing ShellTool safety
whitelist.  Users never need to remember complex command syntax again.

Examples
--------
  "show me all files modified today"
      → dir /od /tw  (or Get-ChildItem … | sort)
  "find all python files recursively"
      → dir /s /b *.py
  "check which port 8000 is being used by"
      → netstat -ano | findstr :8000
  "kill the process on port 3000"
      → for /f "tokens=5" %a in ('netstat -aon ^| findstr :3000') do taskkill /PID %a /F
  "show disk usage summary"
      → wmic logicaldisk get size,freespace,caption
  "list all environment variables"
      → set
  "compress the logs folder into a zip"
      → powershell Compress-Archive -Path logs -DestinationPath logs.zip

Safety
------
  The translated command is passed through the same ShellTool whitelist before
  execution.  If the command is blocked, the tool explains why and suggests
  a safe alternative.  Commands can also be shown to the user for confirmation
  without executing (preview_only=True).

Event contract
--------------
  type: info | status | error
  stage: translating | command_ready | executing | blocked | done
  tool: nl_command/<mode>
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from groq import Groq

from core.tools.base import BaseTool, ToolResult
from core.tools.shell_tool import ShellTool, _ALLOWED_COMMANDS, _BLOCKED_ARG_PATTERNS

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
#  LLM prompt for command generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Windows command-line engineer. Your ONLY job is to convert
a plain-English description into the best possible Windows shell command.

STRICT RULES:
1. Output ONLY the raw command — no explanation, no markdown, no code fences,
   no preamble, no trailing period.
2. Prefer CMD commands when they are simpler; use PowerShell syntax when CMD
   cannot do the job cleanly (prefix PowerShell-only commands with "powershell ").
3. Never use commands that delete system files, modify the registry with
   destructive intent, or require elevated privileges that wouldn't normally
   be available.
4. If the request is ambiguous, produce the safest, most broadly useful variant.
5. Use only commands from this allowed list (base command must appear here):
   python, python3, pip, pip3, node, npm, npx, yarn, g++, gcc, cc, c++,
   clang, clang++, make, java, javac, mvn, gradle, go, cargo, rustc, dotnet,
   git, ls, dir, cat, type, head, tail, wc, find, which, where, echo, pwd,
   black, isort, flake8, pylint, mypy, eslint, prettier, tsc, pytest, jest,
   mocha, netstat, tasklist, taskkill, ipconfig, ping, curl, wmic, xcopy,
   robocopy, powershell, set, cls, tree, attrib, fc, comp, sfc, chkdsk,
   diskpart (read-only queries only).
6. For multi-step operations, chain with && or use a single powershell oneliner.
7. Output exactly one line.

EXAMPLES:
  Request: show all files modified today
  Output:  powershell Get-ChildItem -Recurse | Where-Object {$_.LastWriteTime -gt (Get-Date).Date}

  Request: find python files recursively from current directory
  Output:  dir /s /b *.py

  Request: which process is using port 8080
  Output:  netstat -ano | findstr :8080

  Request: count lines in all .py files
  Output:  powershell (Get-ChildItem -Recurse -Filter *.py | Get-Content | Measure-Object -Line).Lines

  Request: show git log as one line per commit for last 20 commits
  Output:  git log --oneline -20

  Request: list running node processes
  Output:  tasklist /fi "imagename eq node.exe"

  Request: compress folder dist into dist.zip
  Output:  powershell Compress-Archive -Path dist -DestinationPath dist.zip -Force

  Request: show top 10 largest files in current directory
  Output:  powershell Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 10 Name,Length
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str) -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      "nl_command",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _check_command_safety(command: str) -> tuple[bool, str]:
    """
    Run the same checks as ShellTool against the translated command.
    Returns (is_safe, reason_if_blocked).
    """
    try:
        parts = shlex.split(command)
    except ValueError:
        # Might be a powershell one-liner with complex quoting — check base
        parts = command.split()

    if not parts:
        return False, "Empty command."

    base = parts[0].lower().rstrip(".exe")

    # powershell prefix — check the sub-command
    if base in ("powershell", "pwsh") and len(parts) > 1:
        # The overall powershell invocation is allowed; content check below
        pass
    elif base not in _ALLOWED_COMMANDS and base not in {
        # Extra CMD built-ins that are read-only / safe
        "netstat", "tasklist", "taskkill", "ipconfig", "ping", "curl",
        "wmic", "xcopy", "robocopy", "set", "cls", "tree", "attrib",
        "fc", "comp", "sfc", "chkdsk",
    }:
        return False, (
            f"Base command '{base}' is not in the allowed list. "
            "Ask for a different approach."
        )

    full_lower = command.lower()
    for bad in _BLOCKED_ARG_PATTERNS:
        if bad in full_lower:
            return False, f"Command contains blocked pattern: '{bad}'"

    return True, ""


# ---------------------------------------------------------------------------


class NLCommandTool(BaseTool):
    """
    Natural Language → Shell Command converter + executor.

    The user describes what they want in plain English; this tool translates
    it to the best CMD/PowerShell command, shows it for confirmation, and
    (optionally) executes it through ShellTool's safety layer.
    """

    name = "nl_command"

    description = (
        "Convert plain-English descriptions into Windows CMD / PowerShell "
        "commands and optionally execute them. "
        "Use this when the user says things like: "
        "'how do I …', 'give me the command to …', 'run a command that …', "
        "'find files that …', 'check what process is on port …', "
        "'show me disk usage', 'compress folder X', etc. "
        "Parameters: "
        "'request' — the natural-language description (required); "
        "'execute' — whether to run the command after translating (default true); "
        "'working_dir' — directory to run in (optional); "
        "'preview_only' — if true, show command but don't run it (default false)."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["request"],
        "properties": {
            "request": {
                "type": "string",
                "description": "Plain-English description of the command you want.",
            },
            "execute": {
                "type": "boolean",
                "description": "Whether to execute the translated command. Default true.",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for execution (optional).",
            },
            "preview_only": {
                "type": "boolean",
                "description": "If true, translate but do not execute. Default false.",
            },
        },
    }

    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self._client = Groq(api_key=api_key) if api_key else None
        self._shell  = ShellTool()

    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)

        request      = kwargs.get("request", "").strip()
        do_execute   = kwargs.get("execute", True)
        working_dir  = kwargs.get("working_dir", "")
        preview_only = kwargs.get("preview_only", False)

        if not request:
            return ToolResult(success=False, error="'request' is required.")

        if self._client is None:
            return ToolResult(
                success=False,
                error="GROQ_API_KEY not set — cannot translate command.",
            )

        # ── Step 1: Translate ──────────────────────────────────────────
        _emit(cb, _event("info", "translating", f"Translating: {request}…"))

        command = self._translate(request)

        if not command:
            return ToolResult(
                success=False,
                error="LLM did not produce a valid command. Try rephrasing.",
            )

        # Strip any accidental markdown fences the model might add
        command = re.sub(r"^```[a-z]*\n?", "", command, flags=re.IGNORECASE)
        command = re.sub(r"\n?```$", "", command).strip()

        _emit(cb, _event("status", "command_ready",
                          f"Translated command: {command}"))

        # ── Step 2: Safety check ───────────────────────────────────────
        is_safe, reason = _check_command_safety(command)

        if not is_safe:
            _emit(cb, _event("error", "blocked",
                              f"Command blocked: {reason}"))
            return ToolResult(
                success=False,
                error=f"Translated command was blocked by safety filter: {reason}",
                metadata={"translated_command": command, "blocked_reason": reason},
            )

        # ── Step 3: Preview-only or execute ───────────────────────────
        if preview_only or not do_execute:
            _emit(cb, _event("status", "done",
                              f"Preview only — command not executed: {command}"))
            return ToolResult(
                success=True,
                output=f"Command (not executed): {command}",
                metadata={
                    "translated_command": command,
                    "executed": False,
                    "request": request,
                },
            )

        _emit(cb, _event("info", "executing",
                          f"Executing: {command}"))

        shell_kwargs: dict[str, Any] = {"command": command}
        if working_dir:
            shell_kwargs["working_dir"] = working_dir
        if cb is not None:
            shell_kwargs["event_callback"] = cb

        result = self._shell.execute(**shell_kwargs)

        if result.success:
            _emit(cb, _event("status", "done",
                              f"Command completed. Output: {str(result.output)[:120]}"))
        else:
            _emit(cb, _event("error", "done",
                              f"Command failed: {result.error}"))

        # Attach translated command to metadata for transparency
        meta = dict(result.metadata or {})
        meta["translated_command"] = command
        meta["request"]            = request

        return ToolResult(
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=meta,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _translate(self, request: str) -> str | None:
        try:
            response = self._client.chat.completions.create(
                model=_DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": request},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            text = (response.choices[0].message.content or "").strip()
            return text if text else None
        except Exception as exc:  # noqa: BLE001
            logger.error("NLCommandTool._translate failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return "<NLCommandTool>"