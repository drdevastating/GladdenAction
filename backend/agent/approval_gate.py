"""
agent/approval_gate.py

ApprovalGate — "show plan, auto-run unless user intervenes"

This implements the autonomy model:
    1. The Planner generates a multi-step plan.
    2. The plan is shown to the user via the event callback.
    3. A countdown begins (default 8 seconds).
    4. If the user does nothing → auto-approved, execution begins.
    5. If the user rejects (via the API) → execution is cancelled.
    6. For DESTRUCTIVE steps (delete, kill, format, shell mutations)
       the timeout is longer (15s) and the UI shows a stronger warning.

Integration
-----------
    The ApprovalGate wraps the AutonomousController / multi-step
    execution in agent.py.  It inserts approval events into the
    existing event_callback stream — no frontend changes required
    to get the events, but the frontend needs a /approve and /reject
    endpoint to send signals back.

Event schema additions (stable)
--------------------------------
    {
        "type":    "approval_required",
        "stage":   "plan_approval" | "step_approval",
        "message": "<plan summary>",
        "plan":    [ {step, tool, arguments, description}, … ],   # plan_approval only
        "step":    <int>,          # step_approval only
        "tool":    "approval_gate",
        "destructive": true|false,
        "auto_approve_in": <seconds>,
        "timestamp": "<ISO-8601>"
    }

    {
        "type":    "approval_decision",
        "stage":   "approved" | "rejected" | "auto_approved",
        "message": "<reason>",
        "tool":    "approval_gate",
        "timestamp": "<ISO-8601>"
    }

Usage
-----
    gate = ApprovalGate(
        auto_approve_delay   = 8,    # normal steps
        destructive_delay    = 15,   # destructive steps
    )

    # When a plan arrives:
    approved = gate.request_plan_approval(plan_steps, event_callback)

    # When a single step arrives in an auto loop:
    approved = gate.request_step_approval(step_idx, tool, arguments, event_callback)

    # From the API/UI thread (e.g. POST /reject):
    gate.reject("User clicked Cancel")

    # From the API/UI thread (e.g. POST /approve):
    gate.approve()
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

# ── Destructive action detection ─────────────────────────────────────────

_DESTRUCTIVE_TOOLS = frozenset({"shell"})

_DESTRUCTIVE_WORKFLOWS = frozenset({
    "delete_file",
    "delete_file",
    "kill",
})

_DESTRUCTIVE_SHELL_PATTERNS = (
    "rm ", "del ", "rmdir", "unlink",
    "drop table", "truncate",
    "kill ", "terminate",
    "format",
)


def _is_destructive(tool: str, arguments: dict) -> bool:
    if tool in _DESTRUCTIVE_TOOLS:
        cmd = (arguments.get("command") or "").lower()
        return any(p in cmd for p in _DESTRUCTIVE_SHELL_PATTERNS)

    action = (arguments.get("action") or "").lower()
    workflow = (arguments.get("workflow") or "").lower()
    domain = (arguments.get("domain") or "").lower()

    if action in _DESTRUCTIVE_WORKFLOWS:
        return True
    if workflow in _DESTRUCTIVE_WORKFLOWS:
        return True
    if domain == "process" and action == "kill":
        return True

    return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _describe_step(tool: str, arguments: dict) -> str:
    """Generate a short human-readable description of what a step will do."""
    workflow = arguments.get("workflow", "")
    action   = arguments.get("action", "")
    domain   = arguments.get("domain", "")
    command  = arguments.get("command", "")
    path     = arguments.get("path", "")
    target   = arguments.get("target", "")

    if tool == "shell":
        return f"Run: `{command}`"
    if tool == "code_edit":
        if action == "read_file":
            return f"Read file: {path}"
        if action == "edit_file":
            return f"Edit file: {path}"
        if action == "create_file":
            return f"Create file: {path}"
        if action == "delete_file":
            return f"Delete file: {path}"
        if action == "insert_lines":
            return f"Insert lines into: {path}"
    if tool == "context":
        if action == "scan_project":
            return f"Scan project: {path or '.'}"
        if action == "grep_files":
            return f"Search for: {arguments.get('pattern', '?')}"
        return f"Read context: {path or '.'}"
    if tool == "ui_automation":
        return f"UI: {workflow or action}"
    if tool == "system_control":
        if domain == "process":
            return f"Process {action}: {target}"
        if domain == "filesystem":
            return f"Filesystem {action}: {target}"
        return f"System {action}"
    return f"{tool}: {action or workflow or command or '?'}"


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("approval_gate event_callback raised: %s", exc)


# ---------------------------------------------------------------------------


class ApprovalGate:
    """
    Intercepts planned actions and gives the user a window to reject them.

    Thread-safe: approval/rejection can be signalled from any thread
    (e.g. a FastAPI request handler).
    """

    def __init__(
        self,
        auto_approve_delay: float = 8.0,
        destructive_delay:  float = 15.0,
    ) -> None:
        self._auto_approve_delay  = auto_approve_delay
        self._destructive_delay   = destructive_delay

        self._decision_event = threading.Event()
        self._approved: Optional[bool] = None
        self._reason: str = ""
        self._lock = threading.Lock()

    # ================================================================== #
    #  Signal methods — called from API endpoints                         #
    # ================================================================== #

    def approve(self, reason: str = "User approved") -> None:
        with self._lock:
            self._approved = True
            self._reason   = reason
        self._decision_event.set()

    def reject(self, reason: str = "User rejected") -> None:
        with self._lock:
            self._approved = False
            self._reason   = reason
        self._decision_event.set()

    def reset(self) -> None:
        """Reset gate for the next approval request."""
        with self._lock:
            self._approved = None
            self._reason   = ""
        self._decision_event.clear()

    @property
    def is_pending(self) -> bool:
        with self._lock:
            return self._approved is None and not self._decision_event.is_set()

    # ================================================================== #
    #  Plan-level approval (shown before any steps run)                   #
    # ================================================================== #

    def request_plan_approval(
        self,
        steps: list[dict],
        event_callback: EventCallback = None,
    ) -> bool:
        """
        Emit the full plan and wait for approval/rejection/timeout.

        Returns True if the plan should proceed, False if cancelled.
        """
        self.reset()

        has_destructive = any(
            _is_destructive(s.get("tool", ""), s.get("arguments", {}))
            for s in steps
        )
        delay = self._destructive_delay if has_destructive else self._auto_approve_delay

        # Build human-readable plan summary
        plan_summary = []
        for i, step in enumerate(steps, start=1):
            tool = step.get("tool", "?")
            args = step.get("arguments", {})
            desc = _describe_step(tool, args)
            destructive = _is_destructive(tool, args)
            plan_summary.append({
                "step": i,
                "tool": tool,
                "description": desc,
                "destructive": destructive,
            })

        _emit(event_callback, {
            "type":             "approval_required",
            "stage":            "plan_approval",
            "message":          f"Ready to execute {len(steps)} step(s). Auto-approving in {delay:.0f}s…",
            "plan":             plan_summary,
            "destructive":      has_destructive,
            "auto_approve_in":  delay,
            "tool":             "approval_gate",
            "timestamp":        _utc_now(),
        })

        logger.info(
            "ApprovalGate: plan with %d step(s) presented (destructive=%s, delay=%.0fs)",
            len(steps), has_destructive, delay,
        )

        return self._wait_for_decision(delay, event_callback)

    # ================================================================== #
    #  Step-level approval (during autonomous execution)                  #
    # ================================================================== #

    def request_step_approval(
        self,
        step_idx: int,
        tool: str,
        arguments: dict,
        event_callback: EventCallback = None,
    ) -> bool:
        """
        Request approval for a single autonomous step.

        Returns True if the step should proceed.
        """
        self.reset()

        destructive = _is_destructive(tool, arguments)
        delay = self._destructive_delay if destructive else self._auto_approve_delay
        desc  = _describe_step(tool, arguments)

        _emit(event_callback, {
            "type":             "approval_required",
            "stage":            "step_approval",
            "message":          f"Step {step_idx}: {desc}. Auto-approving in {delay:.0f}s…",
            "step":             step_idx,
            "tool":             "approval_gate",
            "step_tool":        tool,
            "step_description": desc,
            "destructive":      destructive,
            "auto_approve_in":  delay,
            "timestamp":        _utc_now(),
        })

        return self._wait_for_decision(delay, event_callback)

    # ================================================================== #
    #  Internal                                                            #
    # ================================================================== #

    def _wait_for_decision(
        self,
        delay: float,
        event_callback: EventCallback,
    ) -> bool:
        signalled = self._decision_event.wait(timeout=delay)

        with self._lock:
            approved = self._approved
            reason   = self._reason

        if not signalled:
            # Timeout → auto-approve
            _emit(event_callback, {
                "type":      "approval_decision",
                "stage":     "auto_approved",
                "message":   f"No response in {delay:.0f}s — auto-approved.",
                "tool":      "approval_gate",
                "timestamp": _utc_now(),
            })
            logger.info("ApprovalGate: auto-approved after %.0fs timeout.", delay)
            return True

        if approved:
            _emit(event_callback, {
                "type":      "approval_decision",
                "stage":     "approved",
                "message":   reason or "Approved.",
                "tool":      "approval_gate",
                "timestamp": _utc_now(),
            })
            logger.info("ApprovalGate: approved — %s", reason)
            return True
        else:
            _emit(event_callback, {
                "type":      "approval_decision",
                "stage":     "rejected",
                "message":   reason or "Rejected.",
                "tool":      "approval_gate",
                "timestamp": _utc_now(),
            })
            logger.info("ApprovalGate: rejected — %s", reason)
            return False

    def __repr__(self) -> str:
        return (
            f"<ApprovalGate delay={self._auto_approve_delay}s "
            f"destructive_delay={self._destructive_delay}s>"
        )