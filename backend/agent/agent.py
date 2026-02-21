"""
agent/agent.py

The Agent is the reasoning layer of the system. It sits above the Executor
and is the only layer that communicates with the LLM.

Responsibilities
----------------
- Build a structured prompt that exposes available tools to the model.
- Send the user instruction to Llama via the Groq API.
- Safely parse the model's JSON response.
- Delegate execution to ToolExecutor and return the result.
- Never execute tools directly — always goes through the Executor.

What this layer is NOT responsible for
---------------------------------------
- Knowing how tools work internally (that's BaseTool's job).
- Performing validation (that's the Executor's job).
- Storing conversation history (multi-turn reasoning — future phase).

API note
--------
Groq's API is fully OpenAI-compatible. We use the official `groq` Python SDK
which mirrors the OpenAI client interface exactly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq

from core.tools.base import ToolResult
from core.tools.registry import ToolRegistry
from execution.executor import ToolExecutor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# --------------------------------------------------------------------------- #
#  Prompt templates                                                             #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are an AI agent that controls a computer by calling tools.

You will be given:
1. A list of available tools with their names, descriptions, and input schemas.
2. A user instruction.

Your job is to decide which single tool to call and with what arguments.

RULES:
- Respond ONLY with a single valid JSON object. No explanation, no markdown, no code fences.
- The JSON must have exactly two keys: "tool" and "arguments".
- "tool" must be the exact tool name from the list.
- "arguments" must be an object matching the tool's input schema.
- If no tool is appropriate, respond with: {"tool": null, "arguments": {}}

RESPONSE FORMAT:
{"tool": "<tool_name>", "arguments": {<key>: <value>, ...}}
"""

_USER_PROMPT_TEMPLATE = """\
AVAILABLE TOOLS:
{tool_listing}

USER INSTRUCTION:
{instruction}
"""


def _build_tool_listing(metadata: list[dict]) -> str:
    """Render tool metadata into a readable block for the prompt."""
    lines: list[str] = []
    for i, tool in enumerate(metadata, start=1):
        lines.append(f"{i}. Tool name: {tool['name']}")
        lines.append(f"   Description: {tool['description']}")
        props = tool["input_schema"].get("properties", {})
        required = tool["input_schema"].get("required", [])
        if props:
            lines.append("   Arguments:")
            for arg_name, spec in props.items():
                req_marker = " (required)" if arg_name in required else " (optional)"
                arg_type = spec.get("type", "any")
                arg_desc = spec.get("description", "")
                lines.append(f"     - {arg_name} [{arg_type}]{req_marker}: {arg_desc}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _extract_json(text: str) -> str:
    """
    Extract a JSON object from the model response even if it wrapped the
    output in markdown fences despite instructions not to.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text.strip()


# --------------------------------------------------------------------------- #
#  Agent                                                                        #
# --------------------------------------------------------------------------- #

class Agent:
    """
    Single-step reasoning agent backed by Groq + Llama 3.3.

    Parameters
    ----------
    registry   : ToolRegistry   -- provides tool metadata for prompt building.
    executor   : ToolExecutor   -- dispatches the tool call decided by the model.
    api_key    : str            -- Groq API key from console.groq.com.
    model_name : str            -- Groq model identifier. Defaults to llama-3.3-70b-versatile.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        api_key: str,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._model_name = model_name

        # Official Groq Python SDK — mirrors OpenAI client interface
        self._client = Groq(api_key=api_key)

        logger.info(
            "Agent initialised — model=%r  tools=%s",
            model_name,
            registry.list_names(),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, instruction: str) -> ToolResult:
        """
        Process a natural-language instruction end-to-end.

        Steps
        -----
        1. Build a prompt exposing available tools + the user instruction.
        2. Send to Llama via Groq's chat completions endpoint.
        3. Safely parse the JSON tool-call decision from the response.
        4. Validate the decision structure.
        5. Delegate execution to ToolExecutor and return ToolResult.

        Returns
        -------
        ToolResult -- always returned, never raises.
        """
        if not instruction.strip():
            return ToolResult(success=False, error="Instruction must not be empty.")

        # --- 1. Build prompt -------------------------------------------- #
        tool_metadata = self._registry.list_metadata()
        tool_listing = _build_tool_listing(tool_metadata)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            tool_listing=tool_listing,
            instruction=instruction,
        )

        logger.info("Sending instruction to Groq/Llama: %r", instruction[:120])

        # --- 2. Call Groq ----------------------------------------------- #
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,      # deterministic tool selection
                max_tokens=512,     # tool calls are short JSON blobs
            )
            raw_text: str = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            msg = f"Groq API call failed: {exc}"
            logger.error(msg)
            return ToolResult(success=False, error=msg)

        logger.debug("Groq raw response: %s", raw_text)

        # --- 3. Parse JSON ---------------------------------------------- #
        json_str = _extract_json(raw_text)
        try:
            decision: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            msg = (
                f"Model returned invalid JSON: {exc}\n"
                f"Raw response was:\n{raw_text}"
            )
            logger.error(msg)
            return ToolResult(success=False, error=msg, metadata={"raw": raw_text})

        # --- 4. Validate decision structure ----------------------------- #
        if not isinstance(decision, dict):
            return ToolResult(
                success=False,
                error=f"Expected a JSON object, got: {type(decision).__name__}",
                metadata={"raw": raw_text},
            )

        tool_name = decision.get("tool")
        arguments  = decision.get("arguments", {})

        if tool_name is None:
            return ToolResult(
                success=False,
                error="Model responded with tool=null — no suitable tool for this instruction.",
                metadata={"raw": raw_text},
            )

        if not isinstance(arguments, dict):
            return ToolResult(
                success=False,
                error=f"'arguments' must be a JSON object, got: {type(arguments).__name__}",
                metadata={"raw": raw_text},
            )

        logger.info(
            "Model decision → tool=%r  arguments=%s",
            tool_name,
            list(arguments.keys()),
        )

        # --- 5. Execute via Executor ------------------------------------ #
        return self._executor.execute(tool_name, **arguments)

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"<Agent model={self._model_name!r}  "
            f"tools={self._registry.list_names()}>"
        )