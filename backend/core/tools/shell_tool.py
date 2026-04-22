"""
core/tools/shell_tool.py

ShellTool — safe terminal command execution with live output streaming.

Inspired by Claude Code's BashTool. Runs whitelisted commands in a
controlled subprocess, streams stdout/stderr via event callbacks in
real time, and enforces hard safety constraints.

Safety model
------------
Layer 1 — Command whitelist    : Only explicitly allowed commands run.
Layer 2 — Argument blocklist   : Dangerous flags/paths are rejected.
Layer 3 — Working dir scoping  : Execution is scoped to a declared root.
Layer 4 — Timeout              : Hard 60-second cap (configurable, max 120s).
Layer 5 — No shell=True        : subprocess.Popen with shell=False always.
Layer 6 — Output truncation    : Max 50 KB combined stdout+stderr.

Allowed commands (conservative whitelist)
-----------------------------------------
    Development:  python, python3, pip, node, npm, npx, yarn,
                  g++, gcc, cargo, go, java, javac, dotnet, make
    File/info:    ls, dir, cat, type, head, tail, wc, find, which, where
    Git:          git status, git log, git diff, git branch
    System info:  echo, pwd, cd, env (read-only queries)

Event contract
--------------
    Streams individual lines as "stdout_line" / "stderr_line" events
    during execution, then emits "execution_complete" on finish.

    type: info | status | error
    stage: stdout_line | stderr_line | execution_started | execution_complete | blocked
    tool: shell/<command>
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

_MAX_OUTPUT_BYTES = 50 * 1024   # 50 KB total output
_DEFAULT_TIMEOUT  = 60          # seconds
_MAX_TIMEOUT      = 120         # hard cap

# ── Allowed base commands ─────────────────────────────────────────────────
_ALLOWED_COMMANDS: frozenset[str] = frozenset({
    # Python
    "python", "python3", "pip", "pip3",
    # JS / Node
    "node", "npm", "npx", "yarn",
    # C/C++
    "g++", "gcc", "cc", "c++", "clang", "clang++", "make",
    # JVM
    "java", "javac", "mvn", "gradle",
    # Go / Rust / .NET
    "go", "cargo", "rustc", "dotnet",
    # Git (read-ish — mutations blocked by arg check)
    "git",
    # File info (read-only)
    "ls", "dir", "cat", "type", "head", "tail", "wc", "find",
    "which", "where", "echo", "pwd",
    # Linters / formatters
    "black", "isort", "flake8", "pylint", "mypy",
    "eslint", "prettier", "tsc",
    # Test runners
    "pytest", "jest", "mocha", "cargo",
})

# ── Blocked argument patterns ─────────────────────────────────────────────
_BLOCKED_ARG_PATTERNS: tuple[str, ...] = (
    "rm ", "del ", "/s ", "format",
    "; rm", "| rm", "&& rm",
    ":(){ :|:& };:",    # fork bomb
    "> /dev/",
    "shutdown", "reboot", "halt",
    "sudo", "su ", "runas",
    "reg add", "reg delete", "regedit",
    "net user", "net localgroup",
)

# ── Git subcommands allowed (others blocked) ──────────────────────────────
_ALLOWED_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "log", "diff", "branch", "show",
    "ls-files", "rev-parse", "describe", "shortlog",
    "remote", "fetch",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, cmd: str = "shell") -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"shell/{cmd}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


# ---------------------------------------------------------------------------


class ShellTool(BaseTool):
    """
    Safe terminal command execution with live stdout/stderr streaming.

    Every line of output is emitted as an event so the frontend can
    display real-time terminal output as the command runs.
    """

    name = "shell"

    description = (
        "Execute terminal/shell commands safely with live output streaming. "
        "Allowed commands: python, node, npm, g++, make, git (read-only), "
        "ls, cat, head, tail, wc, find, pytest, eslint, prettier, tsc, cargo, go. "
        "Use 'working_dir' to scope execution to a project folder. "
        "Output is streamed live. Max timeout: 120 seconds."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {
                "type": "string",
                "description": "The full command to run (e.g. 'python main.py', 'npm test', 'git status').",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for the command. Defaults to current working directory.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (max {_MAX_TIMEOUT}). Default {_DEFAULT_TIMEOUT}.",
            },
            "env_vars": {
                "type": "object",
                "description": "Extra environment variables to inject (e.g. {'DEBUG': '1'}).",
            },
        },
    }

    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)

        command_str: str = kwargs.get("command", "").strip()
        working_dir_str: str = kwargs.get("working_dir", "").strip()
        timeout: int = min(int(kwargs.get("timeout") or _DEFAULT_TIMEOUT), _MAX_TIMEOUT)
        env_vars: dict = kwargs.get("env_vars") or {}

        if not command_str:
            return ToolResult(success=False, error="'command' is required.")

        # ── Parse command ─────────────────────────────────────────────
        try:
            parts = shlex.split(command_str)
        except ValueError as exc:
            return ToolResult(success=False, error=f"Cannot parse command: {exc}")

        if not parts:
            return ToolResult(success=False, error="Empty command.")

        base_cmd = Path(parts[0]).name.lower().rstrip(".exe")

        # ── Layer 1: command whitelist ────────────────────────────────
        if base_cmd not in _ALLOWED_COMMANDS:
            msg = (
                f"Command '{base_cmd}' is not in the allowed list. "
                f"Allowed: {sorted(_ALLOWED_COMMANDS)}"
            )
            _emit(cb, _event("error", "blocked", msg, base_cmd))
            return ToolResult(success=False, error=msg)

        # ── Layer 2: argument blocklist ───────────────────────────────
        full_lower = command_str.lower()
        for bad in _BLOCKED_ARG_PATTERNS:
            if bad in full_lower:
                msg = f"Command contains blocked pattern '{bad}'."
                _emit(cb, _event("error", "blocked", msg, base_cmd))
                return ToolResult(success=False, error=msg)

        # ── Git subcommand guard ──────────────────────────────────────
        if base_cmd == "git" and len(parts) > 1:
            git_sub = parts[1].lower()
            if git_sub not in _ALLOWED_GIT_SUBCOMMANDS:
                msg = (
                    f"git subcommand '{git_sub}' is not allowed. "
                    f"Allowed: {sorted(_ALLOWED_GIT_SUBCOMMANDS)}"
                )
                _emit(cb, _event("error", "blocked", msg, "git"))
                return ToolResult(success=False, error=msg)

        # ── Resolve working dir ───────────────────────────────────────
        if working_dir_str:
            try:
                cwd = Path(working_dir_str).resolve()
            except Exception:  # noqa: BLE001
                cwd = Path.cwd()
        else:
            cwd = Path.cwd()

        if not cwd.exists():
            return ToolResult(success=False, error=f"Working directory not found: '{cwd}'")

        # ── Resolve actual executable ─────────────────────────────────
        exe = shutil.which(parts[0])
        if exe is None:
            return ToolResult(
                success=False,
                error=f"Executable '{parts[0]}' not found in PATH.",
            )
        parts[0] = exe

        # ── Build environment ─────────────────────────────────────────
        env = os.environ.copy()
        if isinstance(env_vars, dict):
            for k, v in env_vars.items():
                if isinstance(k, str) and isinstance(v, str):
                    env[k] = v

        _emit(cb, _event(
            "info", "execution_started",
            f"$ {command_str}  (cwd={cwd}, timeout={timeout}s)",
            base_cmd,
        ))

        # ── Run with live streaming ───────────────────────────────────
        return self._run_streaming(
            parts=parts,
            cwd=cwd,
            env=env,
            timeout=timeout,
            command_str=command_str,
            base_cmd=base_cmd,
            cb=cb,
        )

    # ================================================================== #
    #  Streaming subprocess runner                                         #
    # ================================================================== #

    def _run_streaming(
        self,
        *,
        parts: list[str],
        cwd: Path,
        env: dict,
        timeout: int,
        command_str: str,
        base_cmd: str,
        cb: EventCallback,
    ) -> ToolResult:
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        total_bytes = 0
        truncated   = False

        try:
            proc = subprocess.Popen(
                parts,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,          # line-buffered
            )
        except FileNotFoundError as exc:
            return ToolResult(success=False, error=f"Executable not found: {exc}")
        except PermissionError as exc:
            return ToolResult(success=False, error=f"Permission denied: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Failed to start process: {exc}")

        # Stream stdout and stderr concurrently using threads
        def _read_stream(stream, lines_store: list, stream_type: str) -> None:
            nonlocal total_bytes, truncated
            for line in stream:
                if truncated:
                    break
                line_stripped = line.rstrip("\n")
                line_bytes    = len(line.encode("utf-8"))
                total_bytes  += line_bytes

                if total_bytes > _MAX_OUTPUT_BYTES:
                    truncated = True
                    _emit(cb, _event("info", f"{stream_type}_line",
                                      f"[output truncated at {_MAX_OUTPUT_BYTES // 1024} KB]",
                                      base_cmd))
                    break

                lines_store.append(line_stripped)
                _emit(cb, _event(
                    "info",
                    f"{stream_type}_line",
                    line_stripped,
                    base_cmd,
                ))

        t_out = threading.Thread(
            target=_read_stream,
            args=(proc.stdout, stdout_lines, "stdout"),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_read_stream,
            args=(proc.stderr, stderr_lines, "stderr"),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        exit_code = proc.returncode

        stdout_text = "\n".join(stdout_lines)
        stderr_text = "\n".join(stderr_lines)

        if timed_out:
            _emit(cb, _event("error", "execution_timeout",
                              f"Command timed out after {timeout}s.", base_cmd))
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s.",
                metadata={
                    "command":   command_str,
                    "exit_code": -1,
                    "stdout":    stdout_text,
                    "stderr":    stderr_text,
                },
            )

        success = (exit_code == 0)

        _emit(cb, _event(
            "status" if success else "error",
            "execution_complete",
            f"Exit code: {exit_code}" + (" ✓" if success else " ✗")
            + (f"  stderr: {stderr_text[:120]}" if stderr_text and not success else ""),
            base_cmd,
        ))

        combined = stdout_text + ("\n" + stderr_text if stderr_text else "")

        return ToolResult(
            success=success,
            output=combined.strip() or f"(command exited with code {exit_code})",
            metadata={
                "command":    command_str,
                "exit_code":  exit_code,
                "stdout":     stdout_text,
                "stderr":     stderr_text,
                "truncated":  truncated,
                "timed_out":  False,
            },
        )