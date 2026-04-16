"""
agent/planner.py

The Planner sits between the Agent and the Executor.
It receives a user instruction and returns a structured multi-step
execution plan using the same Groq / Llama 3.3 backend.

Responsibilities
----------------
- Decide whether an instruction requires multiple steps.
- Decompose complex instructions into an ordered list of tool calls.
- Return a strict JSON plan that the Agent can iterate over.
- Fall back gracefully (return None) so the Agent can use its existing
  single-step path without any change to the Executor, Registry or tools.

Plan format (always validated before returning)
-----------------------------------------------
{
    "steps": [
        {
            "tool":      "<tool_name>",
            "arguments": { ... }
        },
        ...
    ]
}

Rules enforced by the prompt
-----------------------------
- Only tools registered in the system may be used.
- Every workflow / domain / action value must be valid.
- Each step must be self-contained (no cross-step variable references).
- Single-step instructions must still return a one-step plan so the
  Agent can use a uniform execution loop.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from groq import Groq

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
#  Planner system prompt
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """\
You are a planning engine for an AI desktop-automation agent.

Your ONLY job is to decompose a user instruction into an ordered list of
tool calls and return them as a single JSON object. You must NEVER produce
any text outside the JSON object — no explanation, no markdown, no code fences.

════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS & VALID VALUES
════════════════════════════════════════════════════════════════════

── TOOL 1: ui_automation ──────────────────────────────────────────
Use for anything that involves visible on-screen interaction.

  workflow values (use EXACTLY these strings):
    create_file_notepad      → write a file via Notepad
    create_file_vscode       → write/open code in VS Code
    send_email_browser       → send e-mail via Gmail in Chrome
    send_whatsapp_desktop    → single WhatsApp message
    send_whatsapp_advanced   → multiple contacts / repeat / delay
    play_youtube_video       → search & play a YouTube video
    linkedin_action          → search/open/connect on LinkedIn
    code_workflow_cpp        → create + compile + run a C++ file
    launch_application       → open a named application
    take_screenshot          → capture screen as PNG

  argument keys per workflow:
    create_file_notepad   : filename (str), content (str)
    create_file_vscode    : filename (str), content (str)
    send_email_browser    : recipient (str), subject (str), content (str)
    send_whatsapp_desktop : contact_name (str), message (str)
    send_whatsapp_advanced: contact_name (str|list), message (str),
                            delay_seconds (float), repeat (int)
    play_youtube_video    : query (str)
    linkedin_action       : name (str), action ("search"|"open"|"connect")
    code_workflow_cpp     : filename (str), code (str)
    launch_application    : app_name (str)
    take_screenshot       : screenshot_filename (str, optional)

── TOOL 2: system_control ─────────────────────────────────────────
Use for process management, system metrics and filesystem operations.

  domain + action combinations:
    process / list       → options: {sort_by: "memory"|"cpu"|"pid", limit: int}
    process / inspect    → target: "<name or PID>"
    process / kill       → target: "<name or PID>"
    system  / cpu_usage
    system  / memory_usage
    system  / disk_usage → target: "<path>" (optional)
    system  / uptime
    filesystem / list_directory  → target: "<path>"
    filesystem / file_info       → target: "<path>"
    filesystem / create_directory→ target: "<path>"
    filesystem / rename_file     → target: "<path>", options: {new_name: str}
    filesystem / delete_file     → target: "<path>"

════════════════════════════════════════════════════════════════════
PLANNING RULES
════════════════════════════════════════════════════════════════════

1. Think about what sequence of actions achieves the instruction.
2. Map each action to exactly one tool call.
3. Order steps so each one can execute independently (no dependencies).
4. For a single-action instruction, return exactly one step.
5. Never invent tool names, workflow names, domain names or action names.
6. For ui_automation steps, always include the "workflow" key in arguments.
7. For system_control steps, always include "domain" and "action" keys.
8. Return ONLY the JSON object — nothing else.

════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (strict)
════════════════════════════════════════════════════════════════════

{
  "steps": [
    {
      "tool": "<tool_name>",
      "arguments": { <key-value pairs> }
    }
  ]
}

════════════════════════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════════════════════════

Instruction: "Open YouTube and play a system design video"
{
  "steps": [
    {
      "tool": "ui_automation",
      "arguments": {
        "workflow": "play_youtube_video",
        "query": "system design interview"
      }
    }
  ]
}

Instruction: "Open Chrome, then take a screenshot"
{
  "steps": [
    {
      "tool": "ui_automation",
      "arguments": { "workflow": "launch_application", "app_name": "chrome" }
    },
    {
      "tool": "ui_automation",
      "arguments": { "workflow": "take_screenshot" }
    }
  ]
}

Instruction: "Show CPU usage and memory stats"
{
  "steps": [
    {
      "tool": "system_control",
      "arguments": { "domain": "system", "action": "cpu_usage" }
    },
    {
      "tool": "system_control",
      "arguments": { "domain": "system", "action": "memory_usage" }
    }
  ]
}

Instruction: "Open WhatsApp and send a message to Alice saying hello"
{
  "steps": [
    {
      "tool": "ui_automation",
      "arguments": {
        "workflow": "send_whatsapp_desktop",
        "contact_name": "Alice",
        "message": "Hello!"
      }
    }
  ]
}

Instruction: "Create a C++ hello world program and run it"
{
  "steps": [
    {
      "tool": "ui_automation",
      "arguments": {
        "workflow": "code_workflow_cpp",
        "filename": "hello.cpp",
        "code": "#include <iostream>\\nint main() {\\n    std::cout << \\"Hello, World!\\" << std::endl;\\n    return 0;\\n}"
      }
    }
  ]
}

Instruction: "List the top 5 RAM-consuming processes, then kill notepad.exe"
{
  "steps": [
    {
      "tool": "system_control",
      "arguments": {
        "domain": "process",
        "action": "list",
        "options": { "sort_by": "memory", "limit": 5 }
      }
    },
    {
      "tool": "system_control",
      "arguments": {
        "domain": "process",
        "action": "kill",
        "target": "notepad.exe"
      }
    }
  ]
}
"""

_PLANNER_USER_TEMPLATE = "Instruction: {instruction}"

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VALID_TOOLS = {"ui_automation", "system_control", "file_creation"}

_VALID_UI_WORKFLOWS = {
    "create_file_notepad", "create_file_vscode", "send_email_browser",
    "send_whatsapp_desktop", "send_whatsapp_advanced", "play_youtube_video",
    "linkedin_action", "code_workflow_cpp", "launch_application", "take_screenshot",
}

_VALID_SC_DOMAINS  = {"process", "system", "filesystem"}
_VALID_SC_ACTIONS  = {
    "list", "inspect", "kill",
    "cpu_usage", "memory_usage", "disk_usage", "uptime",
    "list_directory", "file_info", "create_directory", "rename_file", "delete_file",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> str:
    """Pull the first {...} block from model output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    return text.strip()


def _validate_plan(plan: dict) -> list[str]:
    """
    Return a list of validation errors.
    Empty list means the plan is valid.
    """
    errors: list[str] = []

    if not isinstance(plan, dict):
        return ["Plan is not a JSON object."]

    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return ["'steps' must be a non-empty list."]

    for i, step in enumerate(steps):
        prefix = f"Step {i + 1}"

        if not isinstance(step, dict):
            errors.append(f"{prefix}: step is not an object.")
            continue

        tool = step.get("tool", "")
        args = step.get("arguments", {})

        if tool not in _VALID_TOOLS:
            errors.append(
                f"{prefix}: unknown tool '{tool}'. "
                f"Must be one of {sorted(_VALID_TOOLS)}."
            )
            continue

        if not isinstance(args, dict):
            errors.append(f"{prefix}: 'arguments' must be a JSON object.")
            continue

        if tool == "ui_automation":
            wf = args.get("workflow", "")
            if wf not in _VALID_UI_WORKFLOWS:
                errors.append(
                    f"{prefix}: unknown workflow '{wf}'. "
                    f"Must be one of {sorted(_VALID_UI_WORKFLOWS)}."
                )

        elif tool == "system_control":
            domain = args.get("domain", "")
            action = args.get("action", "")
            if domain not in _VALID_SC_DOMAINS:
                errors.append(
                    f"{prefix}: unknown domain '{domain}'. "
                    f"Must be one of {sorted(_VALID_SC_DOMAINS)}."
                )
            if action not in _VALID_SC_ACTIONS:
                errors.append(
                    f"{prefix}: unknown action '{action}'. "
                    f"Must be one of {sorted(_VALID_SC_ACTIONS)}."
                )

    return errors


# ─────────────────────────────────────────────────────────────────────────────
#  Planner class
# ─────────────────────────────────────────────────────────────────────────────

class Planner:
    """
    Converts a natural-language instruction into a validated multi-step plan.

    Usage
    -----
    planner = Planner(api_key="gsk_...")
    plan = planner.create_plan("Open Chrome and take a screenshot")
    # plan == {
    #   "steps": [
    #     {"tool": "ui_automation", "arguments": {"workflow": "launch_application", "app_name": "chrome"}},
    #     {"tool": "ui_automation", "arguments": {"workflow": "take_screenshot"}},
    #   ]
    # }

    Returns None if planning fails after retries, allowing the caller to
    fall back to the original single-step Agent path.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = _DEFAULT_MODEL,
        max_retries: int = 2,
    ) -> None:
        self._client     = Groq(api_key=api_key)
        self._model_name = model_name
        self._max_retries = max_retries
        logger.info("Planner initialised — model=%r", model_name)

    # ── Public API ────────────────────────────────────────────────────── #

    def create_plan(self, instruction: str) -> dict | None:
        """
        Generate and validate an execution plan for *instruction*.

        Returns
        -------
        dict
            A validated plan dict with a 'steps' list, or None on failure.
        """
        if not instruction.strip():
            logger.warning("Planner received empty instruction — skipping.")
            return None

        last_error: str = ""

        for attempt in range(1, self._max_retries + 1):
            logger.info(
                "Planner attempt %d/%d for: %r",
                attempt, self._max_retries, instruction[:100],
            )

            raw_text = self._call_llm(instruction)
            if raw_text is None:
                last_error = "LLM call failed."
                continue

            plan, error = self._parse_and_validate(raw_text)
            if plan is not None:
                logger.info(
                    "Planner produced %d step(s) on attempt %d.",
                    len(plan["steps"]), attempt,
                )
                return plan

            last_error = error
            logger.warning(
                "Planner attempt %d failed validation: %s", attempt, error
            )

        logger.error(
            "Planner exhausted %d retries. Last error: %s",
            self._max_retries, last_error,
        )
        return None

    # ── Private helpers ───────────────────────────────────────────────── #

    def _call_llm(self, instruction: str) -> str | None:
        """Call Groq and return raw text, or None on error."""
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                    {"role": "user",   "content": _PLANNER_USER_TEMPLATE.format(
                        instruction=instruction,
                    )},
                ],
                temperature=0,
                max_tokens=1024,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Planner LLM call failed: %s", exc)
            return None

    def _parse_and_validate(self, raw_text: str) -> tuple[dict | None, str]:
        """
        Parse JSON from *raw_text* and validate it.

        Returns (plan_dict, "") on success.
        Returns (None, error_message) on failure.
        """
        json_str = _extract_json(raw_text)

        try:
            plan = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc}  |  raw={raw_text[:200]}"

        errors = _validate_plan(plan)
        if errors:
            return None, "Validation errors: " + "; ".join(errors)

        return plan, ""

    def __repr__(self) -> str:
        return f"<Planner model={self._model_name!r}>"