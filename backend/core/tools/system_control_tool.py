"""
core/tools/system_control_tool.py

Production-Level SystemControlTool
====================================
A secure, domain-based OS capability engine that enforces strict safety
validation, emits structured execution events at every stage, and explicitly
rejects all unsafe instructions with no shell execution at any point.

Architecture
------------
Single tool, three domains, strict per-domain action whitelists:

    Domain       Allowed actions
    ──────────── ──────────────────────────────────────────────────────
    process      list · inspect · kill
    system       cpu_usage · memory_usage · disk_usage · uptime
    filesystem   list_directory · file_info · create_directory
                 rename_file · delete_file (restricted)

Security model (layered)
------------------------
Layer 1 — Domain whitelist     : Unknown domain  → reject immediately.
Layer 2 — Action whitelist     : Per-domain; unknown action → reject.
Layer 3 — Protected processes  : By exact lowercase name AND by PID (0, 4,
                                  and current Python PID); kill is blocked.
Layer 4 — Protected paths      : C:\\Windows, Program Files, Program Files (x86),
                                  System32, AppData, and any root-drive path;
                                  any mutating filesystem operation is blocked.
Layer 5 — No shell execution   : All operations use Python stdlib + psutil only.
Layer 6 — Security events      : Violations emit "security_violation_detected"
                                  before returning ToolResult(success=False).

Event contract (stable schema, all fields always present)
----------------------------------------------------------
    {
        "type":      "info" | "status" | "error" | "security",
        "stage":     "<snake_case_stage_name>",
        "message":   "<human-readable description>",
        "tool":      "system_control/<domain>",
        "timestamp": "<ISO-8601 UTC>",
    }

Dependencies
------------
    pip install psutil>=5.9.0       # already in requirements.txt

Example valid command mappings
------------------------------
    Natural-language instruction            domain        action            target / options
    ─────────────────────────────────────   ───────────   ───────────────   ──────────────────────────────────────
    "List top 5 RAM consumers"              process       list              options={"sort_by":"memory","limit":5}
    "List processes sorted by CPU"          process       list              options={"sort_by":"cpu","limit":10}
    "Inspect chrome.exe"                    process       inspect           target="chrome.exe"
    "Inspect PID 1234"                      process       inspect           target="1234"
    "Kill notepad.exe"                      process       kill              target="notepad.exe"
    "Kill PID 5678"                         process       kill              target="5678"
    "What is my CPU usage?"                 system        cpu_usage         —
    "Check memory stats"                    system        memory_usage      —
    "Disk usage for C drive"                system        disk_usage        target="C:\\"
    "All disk partitions"                   system        disk_usage        —
    "How long has the PC been on?"          system        uptime            —
    "List files in Documents"               filesystem    list_directory    target="C:\\Users\\me\\Documents"
    "Info about report.pdf"                 filesystem    file_info         target="C:\\Users\\me\\report.pdf"
    "Create folder MyProject on Desktop"    filesystem    create_directory  target="C:\\Users\\me\\Desktop\\MyProject"
    "Rename notes.txt to todo.txt"          filesystem    rename_file       target="C:\\Users\\me\\notes.txt" options={"new_name":"todo.txt"}
    "Delete temp.log"                       filesystem    delete_file       target="C:\\Users\\me\\temp.log"
"""

from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────── #
EventCallback = Optional[Callable[[dict], None]]


# ============================================================================ #
#  Constants — security boundaries                                              #
# ============================================================================ #

# Protected process names — exact match on proc.name().lower()
_PROTECTED_PROCESS_NAMES: frozenset[str] = frozenset({
    "system",
    "explorer.exe",
    "wininit.exe",
    "csrss.exe",
    "services.exe",
    "lsass.exe",
    "smss.exe",
    # Linux / macOS equivalents
    "systemd",
    "launchd",
    "init",
    "kernel_task",
})

# Protected PIDs — always blocked regardless of name
_PROTECTED_PIDS: frozenset[int] = frozenset({0, 4})

# Protected path prefixes — resolved & lowercased; any path starting with
# one of these strings is considered protected.
_PROTECTED_PATH_PREFIXES: tuple[str, ...] = (
    # Windows
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    # Linux / macOS system locations
    "/etc",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/lib",
    "/boot",
    "/sys",
    "/proc",
    "/root",
    "/private/etc",
    "/private/var",
)

# Filesystem actions that write/delete (require path-safety check)
_MUTATING_FS_ACTIONS: frozenset[str] = frozenset({
    "create_directory",
    "rename_file",
    "delete_file",
})

# Per-domain action whitelists
_DOMAIN_ACTION_WHITELIST: dict[str, frozenset[str]] = {
    "process": frozenset({"list", "inspect", "kill"}),
    "system":  frozenset({"cpu_usage", "memory_usage", "disk_usage", "uptime"}),
    "filesystem": frozenset({
        "list_directory",
        "file_info",
        "create_directory",
        "rename_file",
        "delete_file",
    }),
}


# ============================================================================ #
#  Module-level helpers                                                         #
# ============================================================================ #

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_event(
    *,
    type: str,      # noqa: A002
    stage: str,
    message: str,
    domain: str = "system_control",
) -> dict:
    """Return a fully-formed, schema-stable event dict."""
    return {
        "type":      type,
        "stage":     stage,
        "message":   message,
        "tool":      f"system_control/{domain}",
        "timestamp": _utc_now(),
    }


def _emit(callback: EventCallback, event: dict) -> None:
    """Safely call the event callback; swallows and logs any consumer error."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:          # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _security_violation(
    callback: EventCallback,
    domain: str,
    reason: str,
) -> ToolResult:
    """
    Emit a security_violation_detected event and return the standard
    'Operation not permitted.' ToolResult.  Called for every blocked action.
    """
    _emit(callback, _make_event(
        type="security",
        stage="security_violation_detected",
        message=f"Security violation in domain '{domain}': {reason}",
        domain=domain,
    ))
    logger.warning("SECURITY VIOLATION [%s]: %s", domain, reason)
    return ToolResult(success=False, error="Operation not permitted.")


# ============================================================================ #
#  Path safety helpers                                                          #
# ============================================================================ #

def _resolve_path_safe(raw: str) -> Path | None:
    """Resolve a path string; return None if unparseable."""
    try:
        return Path(raw).resolve()
    except Exception:                  # noqa: BLE001
        return None


def _is_protected_path(path: Path) -> bool:
    """
    Return True if *path* falls under any protected prefix, is a bare
    root drive, or contains an 'appdata' component.

    Rules
    -----
    1. Root drives / filesystem root (parts count ≤ 1, or path == anchor)
       are always blocked to prevent mass-operations.
    2. Any resolved path starting with a _PROTECTED_PATH_PREFIXES entry
       (case-insensitive) is blocked.
    3. Any path component named "appdata" (case-insensitive) is blocked —
       covers C:\\Users\\<any>\\AppData\\...
    """
    try:
        if path == path.parent or str(path) == str(path.anchor):
            return True
        if len(path.parts) <= 1:
            return True
    except Exception:                  # noqa: BLE001
        return True

    path_str = str(path).lower()

    for prefix in _PROTECTED_PATH_PREFIXES:
        if path_str.startswith(prefix):
            return True

    # AppData anywhere in the hierarchy
    if "appdata" in path_str:
        return True

    return False


# ============================================================================ #
#  Process safety helpers                                                       #
# ============================================================================ #

def _is_protected_process(pid: int, name: str) -> bool:
    """Return True if this PID / name must never be terminated."""
    if pid in _PROTECTED_PIDS:
        return True
    if pid == os.getpid():          # never kill ourselves
        return True
    if name.lower() in _PROTECTED_PROCESS_NAMES:
        return True
    return False


def _find_processes_by_target(target: str) -> list[psutil.Process]:
    """
    Return all matching processes.

    If *target* is a pure integer string it is treated as a PID lookup;
    otherwise it is matched as a case-insensitive substring of the process name.
    """
    target = target.strip()
    results: list[psutil.Process] = []

    if target.isdigit():
        pid = int(target)
        try:
            results.append(psutil.Process(pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return results

    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if proc.info["name"] and target.lower() in proc.info["name"].lower():
                results.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return results


# ============================================================================ #
#  SystemControlTool                                                            #
# ============================================================================ #

class SystemControlTool(BaseTool):
    """
    Production-level, secure OS capability engine.

    Three domains, strict action whitelists, comprehensive path and process
    protection, no shell execution, structured events at every stage.
    """

    name: str = "system_control"

    description: str = (
        "Provides safe OS-level capabilities including process management, "
        "system metrics, and controlled filesystem operations using strict "
        "validation and predefined action whitelists. "
        "Domains: 'process' (list/inspect/kill), "
        "'system' (cpu_usage/memory_usage/disk_usage/uptime), "
        "'filesystem' (list_directory/file_info/create_directory/rename_file/delete_file)."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["domain", "action"],
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Capability domain. One of: 'process', 'system', 'filesystem'."
                ),
            },
            "action": {
                "type": "string",
                "description": (
                    "Action to perform within the domain. "
                    "process: list | inspect | kill. "
                    "system: cpu_usage | memory_usage | disk_usage | uptime. "
                    "filesystem: list_directory | file_info | create_directory | "
                    "rename_file | delete_file."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "Target of the action: process name or PID (process domain), "
                    "file/directory path (filesystem domain), "
                    "mount point for disk_usage (system domain)."
                ),
            },
            "options": {
                "type": "object",
                "description": (
                    "Extra parameters. "
                    "process/list: {sort_by: 'memory'|'cpu'|'pid', limit: int}. "
                    "filesystem/rename_file: {new_name: str}."
                ),
            },
        },
    }

    # ================================================================== #
    #  Top-level dispatch                                                  #
    # ================================================================== #

    def execute(self, **kwargs: Any) -> ToolResult:                          # noqa: C901
        """
        Main entry point — validates domain + action then dispatches.

        Stages emitted (always):
            dispatch_started → domain_validated → action_validated
            → <domain-specific stages> → operation_completed | error
        """
        callback: EventCallback = kwargs.pop("event_callback", None)

        # ── Required-field validation ──────────────────────────────────
        missing = self.validate_inputs(kwargs)
        if missing:
            return ToolResult(
                success=False,
                error=f"Missing required input(s): {', '.join(missing)}",
            )

        domain:  str  = kwargs.get("domain",  "").strip().lower()
        action:  str  = kwargs.get("action",  "").strip().lower()
        target:  str  = kwargs.get("target",  "") or ""
        options: dict = kwargs.get("options", {}) or {}

        _emit(callback, _make_event(
            type="info",
            stage="dispatch_started",
            message=(
                f"Received request — domain='{domain}'  "
                f"action='{action}'  target='{target}'"
            ),
            domain=domain or "system_control",
        ))

        if not domain:
            return ToolResult(success=False, error="'domain' must not be empty.")
        if not action:
            return ToolResult(success=False, error="'action' must not be empty.")

        # ── Layer 1: domain whitelist ──────────────────────────────────
        if domain not in _DOMAIN_ACTION_WHITELIST:
            msg = (
                f"Unsupported domain: '{domain}'. "
                f"Supported: {sorted(_DOMAIN_ACTION_WHITELIST.keys())}"
            )
            _emit(callback, _make_event(
                type="error", stage="domain_validation_failed",
                message=msg, domain=domain,
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="domain_validated",
            message=f"Domain '{domain}' is valid.",
            domain=domain,
        ))

        # ── Layer 2: action whitelist ──────────────────────────────────
        allowed = _DOMAIN_ACTION_WHITELIST[domain]
        if action not in allowed:
            msg = (
                f"Unsupported action '{action}' in domain '{domain}'. "
                f"Allowed: {sorted(allowed)}"
            )
            _emit(callback, _make_event(
                type="error", stage="action_validation_failed",
                message=msg, domain=domain,
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="action_validated",
            message=f"Action '{action}' is valid in domain '{domain}'.",
            domain=domain,
        ))

        # ── Dispatch ───────────────────────────────────────────────────
        try:
            if domain == "process":
                return self._process_dispatch(
                    action=action, target=target, options=options, callback=callback,
                )
            if domain == "system":
                return self._system_dispatch(
                    action=action, target=target, callback=callback,
                )
            if domain == "filesystem":
                return self._filesystem_dispatch(
                    action=action, target=target, options=options, callback=callback,
                )
        except Exception as exc:                               # noqa: BLE001
            msg = f"Unhandled exception in domain '{domain}': {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="handler_crashed",
                message=msg, domain=domain,
            ))
            return ToolResult(success=False, error=msg)

        return ToolResult(
            success=False,
            error=f"Internal routing error for domain '{domain}'.",
        )

    # ================================================================== #
    #  DOMAIN: process                                                     #
    # ================================================================== #

    def _process_dispatch(
        self, *, action: str, target: str, options: dict, callback: EventCallback,
    ) -> ToolResult:
        if action == "list":
            return self._process_list(options=options, callback=callback)
        if action == "inspect":
            return self._process_inspect(target=target, callback=callback)
        if action == "kill":
            return self._process_kill(target=target, callback=callback)
        raise AssertionError(f"Unhandled process action: {action!r}")

    # ── process / list ─────────────────────────────────────────────────

    def _process_list(self, *, options: dict, callback: EventCallback) -> ToolResult:
        """
        List running processes.

        Options
        -------
        sort_by : "memory" | "cpu" | "pid"  (default: "memory")
        limit   : int ≥ 1                    (default: 10)
        """
        sort_by: str = str(options.get("sort_by", "memory")).lower()
        limit: int   = options.get("limit", 10)
        if not isinstance(limit, int) or limit < 1:
            limit = 10
        if sort_by not in {"memory", "cpu", "pid"}:
            sort_by = "memory"

        _emit(callback, _make_event(
            type="info", stage="collecting_processes",
            message=f"Collecting processes (sort_by={sort_by!r}, limit={limit})…",
            domain="process",
        ))

        try:
            procs: list[dict] = []
            for proc in psutil.process_iter(attrs=["pid", "name", "memory_info", "cpu_percent"]):
                try:
                    info = proc.info
                    if info["memory_info"] is None:
                        continue
                    procs.append({
                        "pid":          info["pid"],
                        "name":         info["name"] or "",
                        "memory_bytes": info["memory_info"].rss,
                        "cpu_percent":  info["cpu_percent"] or 0.0,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to enumerate processes: {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="collection_failed", message=msg, domain="process",
            ))
            return ToolResult(success=False, error=msg)

        total = len(procs)
        _emit(callback, _make_event(
            type="status", stage="processes_collected",
            message=f"Collected {total} process(es).",
            domain="process",
        ))

        _emit(callback, _make_event(
            type="info", stage="sorting_processes",
            message=f"Sorting by '{sort_by}'…",
            domain="process",
        ))

        sort_key = {
            "memory": lambda p: p["memory_bytes"],
            "cpu":    lambda p: p["cpu_percent"],
            "pid":    lambda p: p["pid"],
        }[sort_by]
        procs.sort(key=sort_key, reverse=(sort_by != "pid"))

        output = [
            {
                "pid":         p["pid"],
                "name":        p["name"],
                "memory_mb":   round(p["memory_bytes"] / (1024 * 1024), 2),
                "cpu_percent": round(p["cpu_percent"], 2),
            }
            for p in procs[:limit]
        ]

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"Returned {len(output)} of {total} process(es).",
            domain="process",
        ))
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "domain": "process", "action": "list",
                "total": total, "returned": len(output),
                "sort_by": sort_by, "limit": limit,
            },
        )

    # ── process / inspect ──────────────────────────────────────────────

    def _process_inspect(self, *, target: str, callback: EventCallback) -> ToolResult:
        """
        Return detailed snapshot for the first process matching *target*.
        Reports: cpu%, memory_mb, threads, status, create_time.
        """
        if not target or not target.strip():
            return ToolResult(
                success=False,
                error="'target' (process name or PID) is required for process/inspect.",
            )

        _emit(callback, _make_event(
            type="info", stage="searching_process",
            message=f"Searching for process matching '{target}'…",
            domain="process",
        ))

        matches = _find_processes_by_target(target)
        if not matches:
            msg = f"No process found matching target: '{target}'"
            _emit(callback, _make_event(
                type="error", stage="process_not_found", message=msg, domain="process",
            ))
            return ToolResult(success=False, error=msg)

        proc = matches[0]

        _emit(callback, _make_event(
            type="info", stage="collecting_process_info",
            message=f"Collecting detailed info for PID {proc.pid}…",
            domain="process",
        ))

        try:
            with proc.oneshot():
                name        = proc.name()
                status      = proc.status()
                cpu_pct     = round(proc.cpu_percent(interval=0.15), 2)
                mem_info    = proc.memory_info()
                mem_mb      = round(mem_info.rss / (1024 * 1024), 2)
                num_threads = proc.num_threads()
                created     = datetime.fromtimestamp(proc.create_time()).isoformat()

            output = {
                "name":        name,
                "pid":         proc.pid,
                "status":      status,
                "cpu_percent": cpu_pct,
                "memory_mb":   mem_mb,
                "num_threads": num_threads,
                "create_time": created,
            }
        except psutil.NoSuchProcess:
            msg = f"Process (PID {proc.pid}) terminated during inspection."
            _emit(callback, _make_event(
                type="error", stage="process_gone", message=msg, domain="process",
            ))
            return ToolResult(success=False, error=msg)
        except psutil.AccessDenied:
            msg = f"Access denied inspecting PID {proc.pid}."
            _emit(callback, _make_event(
                type="error", stage="access_denied", message=msg, domain="process",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"Inspection complete for '{output['name']}' (PID {output['pid']}).",
            domain="process",
        ))
        return ToolResult(
            success=True, output=output,
            metadata={"domain": "process", "action": "inspect"},
        )

    # ── process / kill ─────────────────────────────────────────────────

    def _process_kill(self, *, target: str, callback: EventCallback) -> ToolResult:
        """
        Terminate ALL processes matching *target* (name substring or PID).

        Includes full child-process tree termination.
        Protected processes are skipped with a security_violation_detected event.
        Returns a detailed killed / skipped / failed summary.
        """
        if not target or not target.strip():
            return ToolResult(
                success=False,
                error="'target' (process name or PID) is required for process/kill.",
            )

        _emit(callback, _make_event(
            type="info", stage="searching_processes",
            message=f"Searching for all processes matching '{target}'…",
            domain="process",
        ))

        matches = _find_processes_by_target(target)
        if not matches:
            msg = f"No process found matching target: '{target}'"
            _emit(callback, _make_event(
                type="error", stage="process_not_found", message=msg, domain="process",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="processes_found",
            message=f"Found {len(matches)} instance(s) matching '{target}'.",
            domain="process",
        ))

        killed:  list[dict] = []
        skipped: list[dict] = []
        failed:  list[dict] = []

        for proc in matches:
            pid = proc.pid
            try:
                name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                name = "<unknown>"

            # ── Layer 3: protected process check ──────────────────────
            if _is_protected_process(pid, name):
                _security_violation(
                    callback, "process",
                    f"'{name}' (PID {pid}) is a protected system process.",
                )
                skipped.append({"pid": pid, "name": name, "reason": "protected"})
                continue

            # ── Collect process tree (children first, root last) ───────
            _emit(callback, _make_event(
                type="info", stage="collecting_process_tree",
                message=f"Collecting process tree for '{name}' (PID {pid})…",
                domain="process",
            ))

            try:
                children = proc.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                children = []

            tree: list[psutil.Process] = children + [proc]

            # ── Terminate every member of the tree ────────────────────
            for p in tree:
                try:
                    p_pid  = p.pid
                    p_name = p.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

                if _is_protected_process(p_pid, p_name):
                    skipped.append({"pid": p_pid, "name": p_name, "reason": "protected child"})
                    _emit(callback, _make_event(
                        type="info", stage="skipping_protected_child",
                        message=f"Skipping protected child '{p_name}' (PID {p_pid}).",
                        domain="process",
                    ))
                    continue

                _emit(callback, _make_event(
                    type="info", stage="terminating_process",
                    message=f"Terminating '{p_name}' (PID {p_pid})…",
                    domain="process",
                ))

                try:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        logger.warning("PID %d did not exit gracefully — force-killing.", p_pid)
                        p.kill()
                        p.wait(timeout=2)

                    killed.append({"pid": p_pid, "name": p_name})
                    _emit(callback, _make_event(
                        type="status", stage="process_terminated",
                        message=f"Terminated '{p_name}' (PID {p_pid}).",
                        domain="process",
                    ))

                except psutil.NoSuchProcess:
                    killed.append({"pid": p_pid, "name": p_name})
                    _emit(callback, _make_event(
                        type="status", stage="process_already_gone",
                        message=f"PID {p_pid} was already gone.",
                        domain="process",
                    ))

                except psutil.AccessDenied:
                    failed.append({"pid": p_pid, "name": p_name, "reason": "access denied"})
                    _emit(callback, _make_event(
                        type="error", stage="access_denied",
                        message=(
                            f"Access denied terminating '{p_name}' (PID {p_pid}). "
                            "Try running as administrator."
                        ),
                        domain="process",
                    ))

                except Exception as exc:               # noqa: BLE001
                    failed.append({"pid": p_pid, "name": p_name, "reason": str(exc)})
                    _emit(callback, _make_event(
                        type="error", stage="termination_failed",
                        message=f"Failed to terminate '{p_name}' (PID {p_pid}): {exc}",
                        domain="process",
                    ))

        # ── Summary ────────────────────────────────────────────────────
        parts = []
        if killed:
            parts.append(f"{len(killed)} terminated")
        if skipped:
            parts.append(f"{len(skipped)} skipped (protected)")
        if failed:
            parts.append(f"{len(failed)} failed (access denied)")
        summary = "; ".join(parts) + "." if parts else "Nothing was killed."

        _emit(callback, _make_event(
            type="status", stage="kill_summary",
            message=summary,
            domain="process",
        ))

        return ToolResult(
            success=len(killed) > 0,
            output=summary,
            metadata={
                "domain": "process", "action": "kill",
                "target": target,
                "killed": killed, "skipped": skipped, "failed": failed,
            },
        )

    # ================================================================== #
    #  DOMAIN: system                                                      #
    # ================================================================== #

    def _system_dispatch(
        self, *, action: str, target: str, callback: EventCallback,
    ) -> ToolResult:
        if action == "cpu_usage":
            return self._system_cpu(callback=callback)
        if action == "memory_usage":
            return self._system_memory(callback=callback)
        if action == "disk_usage":
            return self._system_disk(target=target, callback=callback)
        if action == "uptime":
            return self._system_uptime(callback=callback)
        raise AssertionError(f"Unhandled system action: {action!r}")

    # ── system / cpu_usage ─────────────────────────────────────────────

    def _system_cpu(self, *, callback: EventCallback) -> ToolResult:
        _emit(callback, _make_event(
            type="info", stage="collecting_cpu_metrics",
            message="Collecting CPU usage (1-second interval)…",
            domain="system",
        ))
        try:
            overall = psutil.cpu_percent(interval=1)
            per_cpu = psutil.cpu_percent(interval=1, percpu=True)
            freq    = psutil.cpu_freq()
            output  = {
                "overall_percent": round(overall, 2),
                "per_cpu_percent": [round(x, 2) for x in per_cpu],
                "logical_cores":   psutil.cpu_count(logical=True),
                "physical_cores":  psutil.cpu_count(logical=False),
                "frequency_mhz":   round(freq.current, 1) if freq else None,
            }
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to collect CPU metrics: {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="collection_failed", message=msg, domain="system",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=(
                f"CPU: {output['overall_percent']}% overall across "
                f"{output['logical_cores']} logical core(s)."
            ),
            domain="system",
        ))
        return ToolResult(
            success=True, output=output,
            metadata={"domain": "system", "action": "cpu_usage"},
        )

    # ── system / memory_usage ──────────────────────────────────────────

    def _system_memory(self, *, callback: EventCallback) -> ToolResult:
        _emit(callback, _make_event(
            type="info", stage="collecting_memory_metrics",
            message="Collecting virtual memory and swap metrics…",
            domain="system",
        ))
        try:
            mem  = psutil.virtual_memory()
            swap = psutil.swap_memory()
            gb   = 1024 ** 3
            output = {
                "total_gb":      round(mem.total      / gb, 2),
                "available_gb":  round(mem.available  / gb, 2),
                "used_gb":       round(mem.used       / gb, 2),
                "percent":       round(mem.percent,    2),
                "swap_total_gb": round(swap.total     / gb, 2),
                "swap_used_gb":  round(swap.used      / gb, 2),
                "swap_percent":  round(swap.percent,   2),
            }
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to collect memory metrics: {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="collection_failed", message=msg, domain="system",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=(
                f"Memory: {output['percent']}% used "
                f"({output['used_gb']} GB / {output['total_gb']} GB). "
                f"Swap: {output['swap_percent']}%."
            ),
            domain="system",
        ))
        return ToolResult(
            success=True, output=output,
            metadata={"domain": "system", "action": "memory_usage"},
        )

    # ── system / disk_usage ────────────────────────────────────────────

    def _system_disk(self, *, target: str, callback: EventCallback) -> ToolResult:
        _emit(callback, _make_event(
            type="info", stage="collecting_disk_metrics",
            message="Collecting disk usage…",
            domain="system",
        ))
        gb = 1024 ** 3
        try:
            if target:
                usage  = psutil.disk_usage(target)
                output: Any = {
                    "path":     target,
                    "total_gb": round(usage.total / gb, 2),
                    "used_gb":  round(usage.used  / gb, 2),
                    "free_gb":  round(usage.free  / gb, 2),
                    "percent":  round(usage.percent, 2),
                }
            else:
                output = []
                for part in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        output.append({
                            "device":     part.device,
                            "mountpoint": part.mountpoint,
                            "fstype":     part.fstype,
                            "total_gb":   round(usage.total / gb, 2),
                            "used_gb":    round(usage.used  / gb, 2),
                            "free_gb":    round(usage.free  / gb, 2),
                            "percent":    round(usage.percent, 2),
                        })
                    except (PermissionError, OSError):
                        continue
        except FileNotFoundError:
            msg = f"Disk path not found: '{target}'"
            _emit(callback, _make_event(
                type="error", stage="path_not_found", message=msg, domain="system",
            ))
            return ToolResult(success=False, error=msg)
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to collect disk metrics: {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="collection_failed", message=msg, domain="system",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message="Disk usage collected.",
            domain="system",
        ))
        return ToolResult(
            success=True, output=output,
            metadata={"domain": "system", "action": "disk_usage"},
        )

    # ── system / uptime ────────────────────────────────────────────────

    def _system_uptime(self, *, callback: EventCallback) -> ToolResult:
        _emit(callback, _make_event(
            type="info", stage="collecting_uptime",
            message="Collecting system uptime and boot time…",
            domain="system",
        ))
        try:
            boot_ts    = psutil.boot_time()
            boot_dt    = datetime.fromtimestamp(boot_ts)
            uptime_sec = time.time() - boot_ts
            days    = int(uptime_sec // 86400)
            hours   = int((uptime_sec % 86400) // 3600)
            minutes = int((uptime_sec % 3600)  // 60)
            output  = {
                "boot_time":        boot_dt.isoformat(),
                "uptime_seconds":   int(uptime_sec),
                "uptime_formatted": f"{days}d {hours}h {minutes}m",
                "platform":         platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
            }
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to collect uptime: {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="collection_failed", message=msg, domain="system",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"Uptime: {output['uptime_formatted']}.",
            domain="system",
        ))
        return ToolResult(
            success=True, output=output,
            metadata={"domain": "system", "action": "uptime"},
        )

    # ================================================================== #
    #  DOMAIN: filesystem                                                  #
    # ================================================================== #

    def _filesystem_dispatch(
        self, *, action: str, target: str, options: dict, callback: EventCallback,
    ) -> ToolResult:
        if action == "list_directory":
            return self._fs_list_directory(target=target, callback=callback)
        if action == "file_info":
            return self._fs_file_info(target=target, callback=callback)
        if action == "create_directory":
            return self._fs_create_directory(target=target, callback=callback)
        if action == "rename_file":
            return self._fs_rename_file(target=target, options=options, callback=callback)
        if action == "delete_file":
            return self._fs_delete_file(target=target, callback=callback)
        raise AssertionError(f"Unhandled filesystem action: {action!r}")

    # ── shared: resolve + guard path ───────────────────────────────────

    def _resolve_and_guard(
        self,
        raw_target: str,
        action: str,
        callback: EventCallback,
        *,
        must_exist:    bool = True,
        must_be_file:  bool = False,
        must_be_dir:   bool = False,
    ) -> tuple[Path | None, ToolResult | None]:
        """
        Resolve path, enforce existence/type checks, and block protected paths
        for mutating actions.

        Returns (resolved_path, None) on success.
        Returns (None, ToolResult) on any failure.
        """
        if not raw_target or not raw_target.strip():
            return None, ToolResult(
                success=False,
                error=f"'target' (path) is required for filesystem/{action}.",
            )

        _emit(callback, _make_event(
            type="info", stage="resolving_path",
            message=f"Resolving path: '{raw_target}'",
            domain="filesystem",
        ))

        path = _resolve_path_safe(raw_target)
        if path is None:
            msg = f"Invalid or unparseable path: '{raw_target}'"
            _emit(callback, _make_event(
                type="error", stage="path_invalid", message=msg, domain="filesystem",
            ))
            return None, ToolResult(success=False, error=msg)

        # ── Layer 4: protected-path check (mutating actions only) ──────
        if action in _MUTATING_FS_ACTIONS:
            _emit(callback, _make_event(
                type="info", stage="checking_path_safety",
                message=f"Checking path safety for mutating action '{action}'…",
                domain="filesystem",
            ))
            if _is_protected_path(path):
                return None, _security_violation(
                    callback, "filesystem",
                    f"Path '{path}' is protected. Action '{action}' is not permitted.",
                )
            _emit(callback, _make_event(
                type="status", stage="path_safety_confirmed",
                message="Path is not protected — mutating action permitted.",
                domain="filesystem",
            ))

        if must_exist and not path.exists():
            msg = f"Path does not exist: '{path}'"
            _emit(callback, _make_event(
                type="error", stage="path_not_found", message=msg, domain="filesystem",
            ))
            return None, ToolResult(success=False, error=msg)

        if must_be_file and path.exists() and not path.is_file():
            msg = f"Path is not a file: '{path}'"
            _emit(callback, _make_event(
                type="error", stage="not_a_file", message=msg, domain="filesystem",
            ))
            return None, ToolResult(success=False, error=msg)

        if must_be_dir and path.exists() and not path.is_dir():
            msg = f"Path is not a directory: '{path}'"
            _emit(callback, _make_event(
                type="error", stage="not_a_directory", message=msg, domain="filesystem",
            ))
            return None, ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="path_resolved",
            message=f"Path resolved: '{path}'",
            domain="filesystem",
        ))
        return path, None

    # ── filesystem / list_directory ────────────────────────────────────

    def _fs_list_directory(self, *, target: str, callback: EventCallback) -> ToolResult:
        path, err = self._resolve_and_guard(
            target, "list_directory", callback,
            must_exist=True, must_be_dir=True,
        )
        if err:
            return err

        _emit(callback, _make_event(
            type="info", stage="scanning_directory",
            message=f"Scanning directory '{path}'…",
            domain="filesystem",
        ))

        try:
            entries = []
            for item in path.iterdir():             # type: ignore[union-attr]
                try:
                    st = item.stat()
                    entries.append({
                        "name":       item.name,
                        "type":       "directory" if item.is_dir() else "file",
                        "size_bytes": st.st_size if item.is_file() else 0,
                        "modified":   datetime.fromtimestamp(st.st_mtime).isoformat(),
                    })
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            msg = f"Permission denied accessing '{path}'."
            _emit(callback, _make_event(
                type="error", stage="permission_denied", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to scan directory '{path}': {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="scan_failed", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"Directory scanned: {len(entries)} item(s).",
            domain="filesystem",
        ))
        return ToolResult(
            success=True, output=entries,
            metadata={
                "domain": "filesystem", "action": "list_directory",
                "path": str(path), "item_count": len(entries),
            },
        )

    # ── filesystem / file_info ─────────────────────────────────────────

    def _fs_file_info(self, *, target: str, callback: EventCallback) -> ToolResult:
        path, err = self._resolve_and_guard(
            target, "file_info", callback, must_exist=True,
        )
        if err:
            return err

        _emit(callback, _make_event(
            type="info", stage="collecting_file_info",
            message=f"Collecting metadata for '{path.name}'…",   # type: ignore[union-attr]
            domain="filesystem",
        ))

        try:
            st = path.stat()                                      # type: ignore[union-attr]
            output = {
                "name":       path.name,                          # type: ignore[union-attr]
                "path":       str(path),
                "type":       "directory" if path.is_dir() else "file",  # type: ignore[union-attr]
                "size_bytes": st.st_size,
                "size_mb":    round(st.st_size / (1024 * 1024), 4),
                "created":    datetime.fromtimestamp(st.st_ctime).isoformat(),
                "modified":   datetime.fromtimestamp(st.st_mtime).isoformat(),
                "accessed":   datetime.fromtimestamp(st.st_atime).isoformat(),
            }
            if path.is_file():                                    # type: ignore[union-attr]
                output["extension"] = path.suffix                # type: ignore[union-attr]
        except PermissionError:
            msg = f"Permission denied accessing '{path}'."
            _emit(callback, _make_event(
                type="error", stage="permission_denied", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)
        except Exception as exc:           # noqa: BLE001
            msg = f"Failed to collect file info for '{path}': {exc}"
            logger.exception(msg)
            _emit(callback, _make_event(
                type="error", stage="collection_failed", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"File info collected for '{output['name']}'.",
            domain="filesystem",
        ))
        return ToolResult(
            success=True, output=output,
            metadata={"domain": "filesystem", "action": "file_info"},
        )

    # ── filesystem / create_directory ──────────────────────────────────

    def _fs_create_directory(self, *, target: str, callback: EventCallback) -> ToolResult:
        path, err = self._resolve_and_guard(
            target, "create_directory", callback, must_exist=False,
        )
        if err:
            return err

        _emit(callback, _make_event(
            type="info", stage="creating_directory",
            message=f"Creating directory '{path}'…",
            domain="filesystem",
        ))

        try:
            path.mkdir(parents=True, exist_ok=True)             # type: ignore[union-attr]
        except PermissionError:
            msg = f"Permission denied creating '{path}'."
            _emit(callback, _make_event(
                type="error", stage="permission_denied", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)
        except OSError as exc:
            msg = f"Failed to create directory '{path}': {exc}"
            _emit(callback, _make_event(
                type="error", stage="creation_failed", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"Directory created: '{path}'.",
            domain="filesystem",
        ))
        return ToolResult(
            success=True, output=str(path),
            metadata={
                "domain": "filesystem", "action": "create_directory",
                "path": str(path),
            },
        )

    # ── filesystem / rename_file ───────────────────────────────────────

    def _fs_rename_file(
        self, *, target: str, options: dict, callback: EventCallback,
    ) -> ToolResult:
        """
        Rename a file or directory within the same parent directory.

        options["new_name"] must be a bare filename (no path separators).
        """
        new_name: str = str(options.get("new_name", "")).strip()
        if not new_name:
            return ToolResult(
                success=False,
                error="options['new_name'] is required for filesystem/rename_file.",
            )
        if "/" in new_name or "\\" in new_name:
            return ToolResult(
                success=False,
                error="options['new_name'] must be a filename only (no path separators).",
            )

        path, err = self._resolve_and_guard(
            target, "rename_file", callback, must_exist=True,
        )
        if err:
            return err

        new_path = path.parent / new_name                        # type: ignore[union-attr]

        if _is_protected_path(new_path):
            return _security_violation(
                callback, "filesystem",
                f"Destination path '{new_path}' is protected.",
            )

        if new_path.exists():
            msg = f"Destination already exists: '{new_path}'. Rename aborted."
            _emit(callback, _make_event(
                type="error", stage="destination_exists", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="info", stage="renaming_file",
            message=f"Renaming '{path.name}' → '{new_name}'…",   # type: ignore[union-attr]
            domain="filesystem",
        ))

        try:
            path.rename(new_path)                                # type: ignore[union-attr]
        except PermissionError:
            msg = f"Permission denied renaming '{path}'."
            _emit(callback, _make_event(
                type="error", stage="permission_denied", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)
        except OSError as exc:
            msg = f"Failed to rename '{path}' → '{new_path}': {exc}"
            _emit(callback, _make_event(
                type="error", stage="rename_failed", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"Renamed '{path.name}' → '{new_name}'.",    # type: ignore[union-attr]
            domain="filesystem",
        ))
        return ToolResult(
            success=True, output=str(new_path),
            metadata={
                "domain": "filesystem", "action": "rename_file",
                "old_path": str(path), "new_path": str(new_path),
            },
        )

    # ── filesystem / delete_file ───────────────────────────────────────

    def _fs_delete_file(self, *, target: str, callback: EventCallback) -> ToolResult:
        """
        Delete a SINGLE FILE only — directories are explicitly rejected.

        Protected paths are always blocked via _resolve_and_guard (Layer 4).
        """
        path, err = self._resolve_and_guard(
            target, "delete_file", callback,
            must_exist=True, must_be_file=True,
        )
        if err:
            return err

        _emit(callback, _make_event(
            type="info", stage="deleting_file",
            message=f"Deleting file '{path}'…",
            domain="filesystem",
        ))

        try:
            path.unlink()                                        # type: ignore[union-attr]
        except PermissionError:
            msg = f"Permission denied deleting '{path}'."
            _emit(callback, _make_event(
                type="error", stage="permission_denied", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)
        except OSError as exc:
            msg = f"Failed to delete '{path}': {exc}"
            _emit(callback, _make_event(
                type="error", stage="deletion_failed", message=msg, domain="filesystem",
            ))
            return ToolResult(success=False, error=msg)

        _emit(callback, _make_event(
            type="status", stage="operation_completed",
            message=f"File deleted: '{path}'.",
            domain="filesystem",
        ))
        return ToolResult(
            success=True, output=str(path),
            metadata={
                "domain": "filesystem", "action": "delete_file",
                "deleted_path": str(path),
            },
        )