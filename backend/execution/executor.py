"""
execution/executor.py

The ToolExecutor is the single gateway through which all tool invocations
must pass. No layer above it (agent, API routes, WebSocket handlers) should
ever call tool.execute() directly.

Responsibilities
----------------
- Resolve a tool name to a registered BaseTool instance.
- Validate inputs before execution.
- Run the tool and surface a standardised ToolResult.
- Catch and wrap any unexpected runtime exceptions so callers never
  receive a raw Python exception from tool code.
- Emit structured execution events at every key stage via an optional
  event_callback. This decouples monitoring from execution logic and
  prepares the system for future real-time streaming (WebSocket, SSE, etc.).

Event contract
--------------
Each event is a plain dict with a stable schema:

    {
        "type":      "info" | "status" | "error",
        "stage":     "<stage_name>",
        "message":   "<human-readable description>",
        "tool":      "<tool_name>",
        "timestamp": "<ISO-8601 UTC timestamp>"
    }

Stages emitted (in order of a successful execution):
    tool_lookup_started
    tool_lookup_completed
    validation_started
    validation_failed       ← only when required inputs are missing
    execution_started
    execution_completed
    execution_failed        ← only when an unhandled exception is raised
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.tools.base import ToolResult
from core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Type alias for the callback — keeps signatures readable
EventCallback = Optional[Callable[[dict], None]]


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_event(
    *,
    type: str,          # noqa: A002  (shadowing built-in intentionally for clarity)
    stage: str,
    message: str,
    tool: str,
) -> dict:
    """
    Build a fully-formed event dict.

    All fields are always present so consumers never have to guard against
    missing keys.
    """
    return {
        "type":      type,
        "stage":     stage,
        "message":   message,
        "tool":      tool,
        "timestamp": _now(),
    }


class ToolExecutor:
    """
    Decoupled execution gateway for all registered tools.

    Parameters
    ----------
    registry : ToolRegistry
        The registry instance that holds all available tools.
        Injected at construction time so the executor is fully testable
        in isolation with a custom registry.

    Example
    -------
    registry = ToolRegistry()
    registry.register(FileCreationTool())

    executor = ToolExecutor(registry)

    # Without events (original behaviour, fully preserved)
    result = executor.execute("file_creation", filename="out.txt", content="hi")

    # With events
    def on_event(event: dict) -> None:
        print(event)

    result = executor.execute(
        "file_creation",
        event_callback=on_event,
        filename="out.txt",
        content="hi",
    )
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _emit(callback: EventCallback, event: dict) -> None:
        """
        Safely fire the callback with the event dict.

        - Does nothing when callback is None.
        - Swallows and logs any exception raised inside the callback so that
          a buggy consumer can never crash the executor.
        """
        if callback is None:
            return
        try:
            callback(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("event_callback raised an exception: %s", exc)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        tool_name: str,
        *,
        event_callback: EventCallback = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute a registered tool by name.

        Parameters
        ----------
        tool_name : str
            The `name` attribute of the target tool as registered.
        event_callback : callable, optional
            A function that accepts a single ``dict`` event argument.
            Called at each stage of execution. Safe to omit.
        **kwargs : Any
            Input parameters forwarded verbatim to the tool.

        Returns
        -------
        ToolResult
            Always returned — never raises.
        """

        # ── Stage 1: tool_lookup_started ──────────────────────────────── #
        logger.info("Executor received request → tool=%r  inputs=%s",
                    tool_name, list(kwargs.keys()))

        self._emit(callback=event_callback, event=_make_event(
            type="info",
            stage="tool_lookup_started",
            message=f"Looking up tool '{tool_name}' in the registry.",
            tool=tool_name,
        ))

        tool = self._registry.get_or_none(tool_name)

        if tool is None:
            msg = (
                f"Tool '{tool_name}' is not registered. "
                f"Available: {self._registry.list_names()}"
            )
            logger.warning(msg)
            self._emit(callback=event_callback, event=_make_event(
                type="error",
                stage="tool_lookup_failed",
                message=msg,
                tool=tool_name,
            ))
            return ToolResult(success=False, error=msg)

        # ── Stage 2: tool_lookup_completed ────────────────────────────── #
        self._emit(callback=event_callback, event=_make_event(
            type="status",
            stage="tool_lookup_completed",
            message=f"Tool '{tool_name}' found successfully.",
            tool=tool_name,
        ))

        # ── Stage 3: validation_started ───────────────────────────────── #
        self._emit(callback=event_callback, event=_make_event(
            type="info",
            stage="validation_started",
            message=f"Validating inputs for tool '{tool_name}'.",
            tool=tool_name,
        ))

        missing = tool.validate_inputs(kwargs)

        if missing:
            msg = f"Missing required input(s) for '{tool_name}': {', '.join(missing)}"
            logger.warning(msg)

            # ── Stage 3a: validation_failed ───────────────────────────── #
            self._emit(callback=event_callback, event=_make_event(
                type="error",
                stage="validation_failed",
                message=msg,
                tool=tool_name,
            ))
            return ToolResult(success=False, error=msg)

        # ── Stage 4: execution_started ────────────────────────────────── #
        self._emit(callback=event_callback, event=_make_event(
            type="status",
            stage="execution_started",
            message=f"Executing tool '{tool_name}'.",
            tool=tool_name,
        ))

        try:
            result = tool.execute(**kwargs)

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            msg = f"Unexpected error in tool '{tool_name}': {exc}"
            logger.error("%s\n%s", msg, tb)

            # ── Stage 4a: execution_failed ────────────────────────────── #
            self._emit(callback=event_callback, event=_make_event(
                type="error",
                stage="execution_failed",
                message=msg,
                tool=tool_name,
            ))
            return ToolResult(
                success=False,
                error=msg,
                metadata={"traceback": tb},
            )

        # ── Stage 5: execution_completed ──────────────────────────────── #
        log_fn = logger.info if result.success else logger.warning
        log_fn(
            "Tool %r finished — success=%s  output=%r",
            tool_name, result.success, result.output,
        )

        self._emit(callback=event_callback, event=_make_event(
            type="status" if result.success else "error",
            stage="execution_completed",
            message=(
                f"Tool '{tool_name}' completed successfully. Output: {result.output}"
                if result.success
                else f"Tool '{tool_name}' returned a failure: {result.error}"
            ),
            tool=tool_name,
        ))

        return result

    # ------------------------------------------------------------------ #
    #  Introspection helpers                                               #
    # ------------------------------------------------------------------ #

    def available_tools(self) -> list[str]:
        """Return the names of all tools the executor can dispatch to."""
        return self._registry.list_names()

    def tool_metadata(self) -> list[dict]:
        """Return structured metadata for every available tool."""
        return self._registry.list_metadata()

    def __repr__(self) -> str:
        return f"<ToolExecutor tools={self.available_tools()}>"