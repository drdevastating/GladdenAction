"""
core/tools/file_creation_tool.py

A concrete tool that creates a file with specified content.
Demonstrates the full BaseTool contract and serves as a reference
implementation for future tools.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class FileCreationTool(BaseTool):
    """
    Creates a text file at the given path with the provided content.

    Input parameters
    ----------------
    filename : str
        Name (or relative path) of the file to create.
        Example: "output.txt" or "reports/summary.md"
    content  : str
        Text content to write into the file.
    overwrite: bool  (optional, default False)
        If False, raises an error when the file already exists.
        If True, silently overwrites.
    """

    name: str = "file_creation"

    description: str = (
        "Creates a new text file with the specified filename and content. "
        "Returns the absolute path of the created file."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["filename", "content"],
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name or relative path of the file to create.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write into the file.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite an existing file. Defaults to False.",
                "default": False,
            },
        },
    }

    # ------------------------------------------------------------------ #
    #  Core execution                                                      #
    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the file creation.

        Steps
        -----
        1. Validate required inputs.
        2. Resolve the target path.
        3. Guard against accidental overwrite.
        4. Create parent directories if needed.
        5. Write content and return a structured result.
        """

        # --- 1. Input validation ----------------------------------------
        missing = self.validate_inputs(kwargs)
        if missing:
            return ToolResult(
                success=False,
                error=f"Missing required input(s): {', '.join(missing)}",
            )

        filename: str = kwargs["filename"]
        content: str = kwargs["content"]
        overwrite: bool = kwargs.get("overwrite", False)

        if not filename.strip():
            return ToolResult(success=False, error="'filename' must not be empty.")

        # --- 2. Resolve path --------------------------------------------
        target_path = Path(filename).resolve()

        # --- 3. Overwrite guard -----------------------------------------
        if target_path.exists() and not overwrite:
            return ToolResult(
                success=False,
                error=(
                    f"File already exists: {target_path}. "
                    "Pass overwrite=True to replace it."
                ),
            )

        # --- 4. Create parent directories if necessary ------------------
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Could not create parent directories: {exc}",
            )

        # --- 5. Write file ----------------------------------------------
        try:
            target_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.exception("FileCreationTool failed to write %s", target_path)
            return ToolResult(
                success=False,
                error=f"Failed to write file: {exc}",
            )

        file_size = target_path.stat().st_size
        logger.info("Created file: %s (%d bytes)", target_path, file_size)

        return ToolResult(
            success=True,
            output=str(target_path),
            metadata={
                "filename": target_path.name,
                "absolute_path": str(target_path),
                "size_bytes": file_size,
                "encoding": "utf-8",
            },
        )