"""
core/tools/base.py

Defines the abstract base class for all tools in the system.
Every tool must inherit from BaseTool and implement its interface.
This enforces a consistent contract across all tool implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    Standardized result returned by every tool execution.

    Attributes:
        success:  Whether the tool executed without error.
        output:   The primary return value (string, path, data, etc.).
        error:    Human-readable error message if success is False.
        metadata: Optional dict for extra context (e.g. file size, duration).
    """

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.success:
            return f"ToolResult(success=True, output={self.output!r})"
        return f"ToolResult(success=False, error={self.error!r})"


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Each concrete tool must declare:
      - name        : unique snake_case identifier used by the registry & agent.
      - description : plain-English explanation used by the LLM to choose tools.
      - input_schema: JSON-Schema-style dict describing accepted parameters.

    The only method a subclass must implement is `execute(**kwargs)`.
    """

    # ------------------------------------------------------------------ #
    #  Subclasses MUST override these three class-level attributes         #
    # ------------------------------------------------------------------ #
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Concrete interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the provided keyword arguments.

        Args:
            **kwargs: Parameters defined in `input_schema`.

        Returns:
            ToolResult indicating success/failure and carrying output data.
        """
        ...

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> dict[str, Any]:
        """
        Return a structured metadata dict suitable for agent / LLM consumption.
        Mirrors the MCP-inspired tool descriptor format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def validate_inputs(self, kwargs: dict[str, Any]) -> list[str]:
        """
        Basic validation: checks that all required fields from input_schema
        are present in the provided kwargs.

        Returns a list of missing field names (empty list = valid).
        """
        required = [
            key
            for key, spec in self.input_schema.get("properties", {}).items()
            if key in self.input_schema.get("required", [])
        ]
        return [field for field in required if field not in kwargs]

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"