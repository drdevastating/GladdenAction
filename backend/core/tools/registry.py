"""
core/tools/registry.py

Central registry for all tools in the system.
Tools are registered once at startup (or dynamically at runtime) and
looked up by name when the agent wants to execute them.

Design goals:
  - Single source of truth for available tools.
  - Dynamic registration: tools can be added at any time.
  - Clean retrieval API consumed by the future Executor layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Maintains a name → tool instance mapping.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(FileCreationTool())
    tool = registry.get("file_creation")
    result = tool.execute(filename="test.txt", content="hello")
    """

    def __init__(self) -> None:
        self._tools: dict[str, "BaseTool"] = {}

    # ------------------------------------------------------------------ #
    #  Registration                                                        #
    # ------------------------------------------------------------------ #

    def register(self, tool: "BaseTool") -> None:
        """
        Register a tool instance.

        Raises:
            ValueError: If a tool with the same name is already registered
                        (prevents silent overwrites).
        """
        if not tool.name:
            raise ValueError(
                f"Tool {tool.__class__.__name__!r} has no `name` defined."
            )

        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. "
                "Use `force_register` if you intend to overwrite it."
            )

        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def force_register(self, tool: "BaseTool") -> None:
        """
        Register a tool, silently replacing any existing tool with the same name.
        Useful during development / hot-reloading scenarios.
        """
        if not tool.name:
            raise ValueError(
                f"Tool {tool.__class__.__name__!r} has no `name` defined."
            )
        self._tools[tool.name] = tool
        logger.info("Force-registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry by name."""
        if name not in self._tools:
            raise KeyError(f"No tool named {name!r} is registered.")
        del self._tools[name]
        logger.info("Unregistered tool: %s", name)

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> "BaseTool":
        """
        Retrieve a registered tool by name.

        Raises:
            KeyError: If the tool is not found.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool {name!r} not found. "
                f"Available tools: {self.list_names()}"
            )
        return self._tools[name]

    def get_or_none(self, name: str) -> "BaseTool | None":
        """Return the tool or None if not found (no exception)."""
        return self._tools.get(name)

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def list_names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def list_metadata(self) -> list[dict]:
        """
        Return metadata for all registered tools.
        This is what the future agent layer will pass to the LLM so it
        can reason about which tool to call.
        """
        return [tool.get_metadata() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.list_names()}>"


# --------------------------------------------------------------------------- #
#  Module-level singleton                                                       #
#                                                                               #
#  Import this anywhere: `from core.tools.registry import registry`            #
#  All layers share the same registry instance.                                 #
# --------------------------------------------------------------------------- #
registry = ToolRegistry()