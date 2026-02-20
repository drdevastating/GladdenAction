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
- (Future) Provide a single place to add cross-cutting concerns such as
  logging, metrics, rate-limiting, timeouts, and retry logic.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from core.tools.base import ToolResult
from core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
    result = executor.execute("file_creation", filename="out.txt", content="hi")
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a registered tool by name.

        Steps
        -----
        1. Resolve the tool from the registry (unknown name → failure result).
        2. Validate required inputs via the tool's own schema (missing fields
           → failure result, no exception raised).
        3. Call tool.execute(**kwargs).
        4. Catch any unhandled exception from tool code and convert it to a
           failure ToolResult so the caller always gets a structured response.

        Parameters
        ----------
        tool_name : str
            The `name` attribute of the target tool as registered.
        **kwargs : Any
            Input parameters forwarded verbatim to the tool.

        Returns
        -------
        ToolResult
            Always returned — never raises.
        """

        # --- 1. Resolve tool -------------------------------------------- #
        logger.info("Executor received request → tool=%r  inputs=%s",
                    tool_name, list(kwargs.keys()))

        tool = self._registry.get_or_none(tool_name)
        if tool is None:
            msg = (
                f"Tool {tool_name!r} is not registered. "
                f"Available: {self._registry.list_names()}"
            )
            logger.warning(msg)
            return ToolResult(success=False, error=msg)

        # --- 2. Validate inputs ----------------------------------------- #
        missing = tool.validate_inputs(kwargs)
        if missing:
            msg = f"Missing required input(s) for {tool_name!r}: {', '.join(missing)}"
            logger.warning(msg)
            return ToolResult(success=False, error=msg)

        # --- 3. Execute ------------------------------------------------- #
        try:
            result = tool.execute(**kwargs)

        # --- 4. Safety net ---------------------------------------------- #
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            msg = f"Unexpected error in tool {tool_name!r}: {exc}"
            logger.error("%s\n%s", msg, tb)
            return ToolResult(
                success=False,
                error=msg,
                metadata={"traceback": tb},
            )

        log_fn = logger.info if result.success else logger.warning
        log_fn(
            "Tool %r finished — success=%s  output=%r",
            tool_name, result.success, result.output,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Introspection helpers                                               #
    # ------------------------------------------------------------------ #

    def available_tools(self) -> list[str]:
        """Return the names of all tools the executor can dispatch to."""
        return self._registry.list_names()

    def tool_metadata(self) -> list[dict]:
        """
        Return structured metadata for every available tool.
        This will be consumed by the agent layer to build the LLM tool list.
        """
        return self._registry.list_metadata()

    def __repr__(self) -> str:
        return f"<ToolExecutor tools={self.available_tools()}>"