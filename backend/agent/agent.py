"""
agent/agent.py

The Agent is the reasoning layer of the system. It sits above the Executor
and is the only layer that communicates with the LLM.

Responsibilities
----------------
- Build a structured prompt that exposes available tools to the model.
- Bias the LLM toward the ui_automation tool when the user instruction
  implies visible, on-screen execution.
- PRE-PASS: When the instruction implies composing content (WhatsApp message,
  email body, file content, Notepad text, VS Code file), call the LLM first
  to generate that content from the user's intent description, then inject
  the generated content into the instruction before the tool-dispatch call.
- Send the (enriched) instruction to Llama 3.3 via the Groq API.
- Safely parse the model's JSON response.
- Delegate execution to ToolExecutor and return the result.
- Never execute tools directly — always goes through the Executor.

What this layer is NOT responsible for
---------------------------------------
- Knowing how tools work internally (that's BaseTool's job).
- Performing validation (that's the Executor's job).
- Storing conversation history (multi-turn reasoning — future phase).
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
#  Prompt templates — Tool Dispatch                                             #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are an AI agent that controls a computer by calling tools.

You will be given:
1. A list of available tools with their names, descriptions, and input schemas.
2. A user instruction.

Your job is to decide which single tool to call and with what arguments.

== TOOL SELECTION RULES ==

RULE 1 — PREFER VISIBLE UI AUTOMATION:
If the user instruction mentions or implies any of the following, you MUST select
the "ui_automation" tool:
  - Notepad, text editor, .txt file creation via an app
  - VS Code, code editor, writing code, creating source files (.cpp, .py, .js, etc.)
  - Browser, Chrome, Gmail, email, sending mail
  - "open", "launch", "type into", "using notepad", "in VS Code", "send email"
  - WhatsApp, messaging, sending a message to a contact

RULE 2 — MATCH THE WORKFLOW ARGUMENT:
When you select "ui_automation", you must also pick the correct "workflow" value:
  - "create_file_notepad"   → user wants to create/write a file using Notepad
  - "create_file_vscode"    → user wants to create/write code in VS Code
  - "send_email_browser"    → send email via Gmail in Chrome (pre-filled compose URL)
  - "send_whatsapp_desktop" → send a WhatsApp message via WhatsApp Desktop

RULE 3 — DIRECT API TOOLS (fallback only):
Use "file_creation" only when the user explicitly asks for a direct/silent file
operation with no mention of a visible app.

== RESPONSE FORMAT ==

Respond ONLY with a single valid JSON object. No explanation, no markdown, no code fences.
The JSON must have exactly two keys: "tool" and "arguments".
"tool" must be the exact tool name from the list.
"arguments" must be an object matching the tool's input schema.
If no tool is appropriate, respond with: {"tool": null, "arguments": {}}

EXAMPLES:

User: "Open Notepad and write Hello World in a file called hello.txt"
Response: {"tool": "ui_automation", "arguments": {"workflow": "create_file_notepad", "filename": "hello.txt", "content": "Hello World"}}

User: "Create a C++ Hello World program in VS Code"
Response: {"tool": "ui_automation", "arguments": {"workflow": "create_file_vscode", "filename": "main.cpp", "content": "#include <iostream>\\nint main() {\\n    std::cout << \\"Hello, World!\\" << std::endl;\\n    return 0;\\n}"}}

User: "Send an email to john@example.com saying the meeting is at 3pm"
Response: {"tool": "ui_automation", "arguments": {"workflow": "send_email_browser", "recipient": "john@example.com", "subject": "Meeting Update", "content": "The meeting is at 3pm."}}

User: "Send a WhatsApp message to Alice asking about her weekend"
Response: {"tool": "ui_automation", "arguments": {"workflow": "send_whatsapp_desktop", "contact_name": "Alice", "message": "Hey Alice! Hope you had a great weekend. How did it go?"}}

User: "Create a file called notes.txt with the content buy milk"
Response: {"tool": "file_creation", "arguments": {"filename": "notes.txt", "content": "buy milk"}}
"""

_USER_PROMPT_TEMPLATE = """\
AVAILABLE TOOLS:
{tool_listing}

USER INSTRUCTION:
{instruction}
"""

# --------------------------------------------------------------------------- #
#  Prompt templates — Content Generation Pre-Pass                              #
# --------------------------------------------------------------------------- #

_CONTENT_GEN_SYSTEM_PROMPT = """\
You are a helpful assistant that composes natural, human-like written content.
You will receive a description of what content to write (e.g. "a warm greeting
asking about someone's health", "a professional email about a project update",
"a Python script that reads a CSV file", etc.).

Your job is to produce ONLY the final content — no preamble, no explanation,
no quotes around your answer, no markdown unless appropriate for the medium.

Rules:
- For messages and emails: write in a warm, natural, human tone. Do not be robotic.
- For code: write complete, working code with appropriate comments.
- For document/note content: write clear, well-structured prose.
- Keep responses appropriately concise unless the description implies length.
- Do NOT include subject lines in your output — only the body/message/code.
"""

# Keywords that signal the instruction describes *intent* rather than
# providing *verbatim* content. When any of these appear, we run the
# content-generation pre-pass.
_INTENT_SIGNALS = [
    # intent verbs
    "ask", "asking", "tell", "telling", "wish", "wishing", "greet", "greeting",
    "remind", "reminding", "invite", "inviting", "congratulate", "congratulating",
    "inform", "informing", "request", "requesting", "thank", "thanking",
    "apologise", "apologize", "apologising", "apologizing",
    "suggest", "suggesting", "recommend", "recommending",
    "explain", "explaining", "describe", "describing",
    # vague content descriptors
    "about his", "about her", "about their", "about the", "about my",
    "in general", "general", "well-being", "wellbeing", "health",
    "how he is", "how she is", "how they are", "how are you",
    "catch up", "checking in", "check in", "follow up", "follow-up",
    # code generation signals
    "write a", "write an", "create a", "implement", "build a",
    "a program that", "a script that", "a function that",
    "hello world",
]

# Which tools and workflows may need content generation
_CONTENT_WORKFLOWS = {
    "send_whatsapp_desktop": "message",
    "send_email_browser":    "content",
    "create_file_notepad":   "content",
    "create_file_vscode":    "content",
}
_CONTENT_TOOLS = {"file_creation": "content"}


def _needs_content_generation(instruction: str) -> bool:
    """
    Heuristic: does the instruction *describe intent* rather than
    providing the verbatim content to write?

    Returns True when any intent signal is found in the lowercased instruction.
    """
    lowered = instruction.lower()
    return any(signal in lowered for signal in _INTENT_SIGNALS)


def _build_content_gen_prompt(instruction: str, medium: str) -> str:
    """
    Build the user-facing prompt for the content-generation pre-pass.

    Parameters
    ----------
    instruction : str   The original user instruction.
    medium      : str   One of "whatsapp_message", "email_body", "file_content",
                        "code_file", "note_content".
    """
    medium_hints = {
        "whatsapp_message": (
            "Write a WhatsApp message based on this intent. "
            "Keep it casual, warm, and conversational (2–5 sentences max)."
        ),
        "email_body": (
            "Write only the email BODY (not the subject line) based on this intent. "
            "Be professional but friendly. 2–4 short paragraphs at most."
        ),
        "file_content": (
            "Write the plain-text file content based on this intent. "
            "Format it clearly and concisely."
        ),
        "code_file": (
            "Write complete, working code based on this intent. "
            "Include brief comments. Output ONLY the code, no markdown fences."
        ),
        "note_content": (
            "Write clear, well-structured note content based on this intent."
        ),
    }
    hint = medium_hints.get(medium, "Write appropriate content based on this intent.")
    return f"{hint}\n\nUser intent: {instruction}"


def _detect_medium(instruction: str, workflow: str | None) -> str:
    """Map a workflow name (or instruction keywords) to a content-gen medium."""
    if workflow == "send_whatsapp_desktop":
        return "whatsapp_message"
    if workflow == "send_email_browser":
        return "email_body"
    if workflow == "create_file_vscode":
        return "code_file"
    if workflow == "create_file_notepad":
        return "note_content"
    # file_creation tool
    lowered = instruction.lower()
    if any(kw in lowered for kw in [".py", ".js", ".cpp", ".ts", ".java", "code", "script", "program"]):
        return "code_file"
    return "file_content"


# --------------------------------------------------------------------------- #
#  Tool listing builder                                                         #
# --------------------------------------------------------------------------- #

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

    New capability — Content Generation Pre-Pass
    --------------------------------------------
    Before dispatching a tool call, the agent checks whether the user's
    instruction *describes intent* (e.g. "greet him and ask about his health")
    rather than providing verbatim content. If so, a second LLM call is made
    that generates the actual message/email/file content. The generated content
    is then injected into the instruction so the tool-dispatch call receives
    ready-to-use text.

    Parameters
    ----------
    registry   : ToolRegistry   -- provides tool metadata for prompt building.
    executor   : ToolExecutor   -- dispatches the tool call decided by the model.
    api_key    : str            -- Groq API key from console.groq.com.
    model_name : str            -- Groq model identifier.
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
        2. Send to Llama 3.3 via Groq's chat completions endpoint.
        3. Safely parse the JSON tool-call decision from the response.
        4. Validate the decision structure.
        5. [NEW] If the instruction describes intent rather than content,
           run a content-generation pre-pass to produce the actual text,
           then inject it into the decision arguments.
        6. Delegate execution to ToolExecutor and return ToolResult.

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

        # --- 2. Call Groq (tool dispatch) -------------------------------- #
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=512,
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

        # --- 5. Content-generation pre-pass ----------------------------- #
        arguments = self._maybe_generate_content(
            instruction=instruction,
            tool_name=tool_name,
            arguments=arguments,
        )

        # --- 6. Execute via Executor ------------------------------------ #
        return self._executor.execute(tool_name, **arguments)

    # ------------------------------------------------------------------ #
    #  Content-generation pre-pass                                         #
    # ------------------------------------------------------------------ #

    def _maybe_generate_content(
        self,
        *,
        instruction: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        If the instruction describes *intent* rather than verbatim content,
        call the LLM to generate the actual content and inject it into the
        arguments dict before tool execution.

        The method is side-effect free — it always returns a (possibly
        updated) copy of *arguments*.

        Supported cases
        ---------------
        ui_automation / send_whatsapp_desktop  → generates arguments["message"]
        ui_automation / send_email_browser     → generates arguments["content"]
        ui_automation / create_file_notepad    → generates arguments["content"]
        ui_automation / create_file_vscode     → generates arguments["content"]
        file_creation                          → generates arguments["content"]
        """
        if not _needs_content_generation(instruction):
            logger.debug("Content-gen pre-pass skipped — instruction appears verbatim.")
            return arguments

        # Determine which argument key holds the content
        workflow = arguments.get("workflow", "")
        content_key: str | None = None

        if tool_name == "ui_automation":
            content_key = _CONTENT_WORKFLOWS.get(workflow)
        elif tool_name == "file_creation":
            content_key = _CONTENT_TOOLS.get(tool_name)

        if content_key is None:
            # Not a content-bearing tool — skip
            return arguments

        # Check if the model already produced decent content (>20 chars,
        # not just a placeholder). If so, don't overwrite it.
        existing = arguments.get(content_key, "")
        if isinstance(existing, str) and len(existing.strip()) > 20:
            logger.debug(
                "Content-gen pre-pass skipped — model already supplied content (%d chars).",
                len(existing.strip()),
            )
            return arguments

        # Determine medium for the prompt hint
        medium = _detect_medium(instruction, workflow if tool_name == "ui_automation" else None)

        logger.info(
            "Content-gen pre-pass triggered — tool=%r  workflow=%r  content_key=%r  medium=%r",
            tool_name, workflow, content_key, medium,
        )

        # Call LLM for content generation
        generated = self._generate_content(instruction, medium)

        if generated:
            logger.info(
                "Content-gen pre-pass produced %d chars for key %r.",
                len(generated), content_key,
            )
            # Return a shallow copy with the generated content injected
            updated = dict(arguments)
            updated[content_key] = generated
            return updated

        # If generation failed, fall back to original arguments
        logger.warning("Content-gen pre-pass produced no output — keeping original arguments.")
        return arguments

    def _generate_content(self, instruction: str, medium: str) -> str | None:
        """
        Make a focused LLM call to produce written content from the
        user's intent description.

        Returns the generated string, or None on failure.
        """
        user_prompt = _build_content_gen_prompt(instruction, medium)

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _CONTENT_GEN_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,   # slightly creative for natural prose
                max_tokens=512,
            )
            text = (response.choices[0].message.content or "").strip()
            return text if text else None
        except Exception as exc:  # noqa: BLE001
            logger.error("Content-gen LLM call failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"<Agent model={self._model_name!r}  "
            f"tools={self._registry.list_names()}>"
        )