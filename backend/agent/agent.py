"""
agent/agent.py  (updated — multi-step Planner integration)

Changes from the previous version
----------------------------------
ADDED:
  - Optional Planner dependency (injected at construction time).
  - _run_plan()  : iterates over plan steps, emitting step events.
  - _emit_event(): emits planning/step lifecycle events via the executor's
                   event callback mechanism.
  - run() now calls the Planner first; falls back to the original single-step
    path if planning is disabled, fails, or produces a single generic step.

NOT CHANGED:
  - ToolExecutor  — untouched.
  - ToolRegistry  — untouched.
  - BaseTool      — untouched.
  - UIAutomationTool / SystemControlTool — untouched.
  - Content-generation pre-pass — preserved and still applied per-step.
  - All existing prompt templates — untouched.

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
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from groq import Groq

from core.tools.base import ToolResult
from agent.planner import Planner
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
  - YouTube, play video, watch video, search video
  - LinkedIn, profile, connection request, connect with someone
  - C++, compile, run program, g++, executable
  - Screenshot, capture screen, take a screenshot
  - Opening apps/applications by name

RULE 2 — MATCH THE WORKFLOW ARGUMENT:
When you select "ui_automation", pick the correct "workflow" value:

  File creation:
    "create_file_notepad"     → user wants to create/write a file using Notepad
    "create_file_vscode"      → user wants to create/write code in VS Code

  Email:
    "send_email_browser"      → send email via Gmail in Chrome

  WhatsApp:
    "send_whatsapp_desktop"   → send a single WhatsApp message to one contact
    "send_whatsapp_advanced"  → send to multiple contacts, repeat messages, or use delay

  YouTube:
    "play_youtube_video"      → search YouTube and play a video (query required)

  LinkedIn:
    "linkedin_action"         → search/open a LinkedIn profile; optionally connect

  Code:
    "code_workflow_cpp"       → create a C++ file, compile with g++, run it

  System:
    "launch_application"      → open a named application (chrome, vscode, calculator, etc.)
    "take_screenshot"         → capture the screen and save as PNG

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

User: "Send WhatsApp messages to Alice and Bob saying Happy Birthday"
Response: {"tool": "ui_automation", "arguments": {"workflow": "send_whatsapp_advanced", "contact_name": ["Alice", "Bob"], "message": "Happy Birthday! 🎉", "repeat": 1}}

User: "Play a YouTube video about system design interviews"
Response: {"tool": "ui_automation", "arguments": {"workflow": "play_youtube_video", "query": "system design interviews"}}

User: "Open Elon Musk's LinkedIn profile"
Response: {"tool": "ui_automation", "arguments": {"workflow": "linkedin_action", "name": "Elon Musk", "action": "open"}}

User: "Send a connection request to Sundar Pichai on LinkedIn"
Response: {"tool": "ui_automation", "arguments": {"workflow": "linkedin_action", "name": "Sundar Pichai", "action": "connect"}}

User: "Create and run a C++ program that prints Fibonacci numbers"
Response: {"tool": "ui_automation", "arguments": {"workflow": "code_workflow_cpp", "filename": "fibonacci.cpp", "code": "#include <iostream>\\nint main() {\\n    int a=0,b=1;\\n    for(int i=0;i<10;i++){std::cout<<a<<' ';int c=a+b;a=b;b=c;}\\n    return 0;\\n}"}}

User: "Open Chrome"
Response: {"tool": "ui_automation", "arguments": {"workflow": "launch_application", "app_name": "chrome"}}

User: "Open the calculator"
Response: {"tool": "ui_automation", "arguments": {"workflow": "launch_application", "app_name": "calculator"}}

User: "Take a screenshot"
Response: {"tool": "ui_automation", "arguments": {"workflow": "take_screenshot"}}

User: "Take a screenshot and save it as desktop_capture.png"
Response: {"tool": "ui_automation", "arguments": {"workflow": "take_screenshot", "screenshot_filename": "desktop_capture.png"}}

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

_INTENT_SIGNALS = [
    "ask", "asking", "tell", "telling", "wish", "wishing", "greet", "greeting",
    "remind", "reminding", "invite", "inviting", "congratulate", "congratulating",
    "inform", "informing", "request", "requesting", "thank", "thanking",
    "apologise", "apologize", "apologising", "apologizing",
    "suggest", "suggesting", "recommend", "recommending",
    "explain", "explaining", "describe", "describing",
    "about his", "about her", "about their", "about the", "about my",
    "in general", "general", "well-being", "wellbeing", "health",
    "how he is", "how she is", "how they are", "how are you",
    "catch up", "checking in", "check in", "follow up", "follow-up",
    "write a", "write an", "create a", "implement", "build a",
    "a program that", "a script that", "a function that",
    "hello world",
]

_CONTENT_WORKFLOWS = {
    "send_whatsapp_desktop":   "message",
    "send_whatsapp_advanced":  "message",
    "send_email_browser":      "content",
    "create_file_notepad":     "content",
    "create_file_vscode":      "content",
    "code_workflow_cpp":       "code",
}
_CONTENT_TOOLS = {"file_creation": "content"}


def _needs_content_generation(instruction: str) -> bool:
    lowered = instruction.lower()
    return any(signal in lowered for signal in _INTENT_SIGNALS)


def _build_content_gen_prompt(instruction: str, medium: str) -> str:
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
    if workflow in ("send_whatsapp_desktop", "send_whatsapp_advanced"):
        return "whatsapp_message"
    if workflow == "send_email_browser":
        return "email_body"
    if workflow in ("create_file_vscode", "code_workflow_cpp"):
        return "code_file"
    if workflow == "create_file_notepad":
        return "note_content"
    lowered = instruction.lower()
    if any(kw in lowered for kw in [".py", ".js", ".cpp", ".ts", ".java", "code", "script", "program"]):
        return "code_file"
    return "file_content"


# --------------------------------------------------------------------------- #
#  Tool listing builder                                                         #
# --------------------------------------------------------------------------- #

def _build_tool_listing(metadata: list[dict]) -> str:
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
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return text.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
#  Agent                                                                        #
# --------------------------------------------------------------------------- #

class Agent:
    """
    Single-step (and now multi-step) reasoning agent backed by Groq + Llama 3.3.

    Construction
    ------------
    Pass a Planner instance to enable multi-step planning.
    If planner=None the agent behaves exactly as before.

    Execution flow (with Planner)
    -----------------------------
    1. Planner.create_plan(instruction)  →  plan with N steps
    2. For each step: content-gen pre-pass → executor.execute()
    3. If planner fails → fall back to original single-step path

    Execution flow (without Planner / fallback)
    -------------------------------------------
    1. LLM decides single tool call
    2. content-gen pre-pass
    3. executor.execute()
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        api_key: str,
        model_name: str = _DEFAULT_MODEL,
        planner: "Planner | None" = None,     # type: Planner | None  (avoid circular import)
    ) -> None:
        self._registry   = registry
        self._executor   = executor
        self._model_name = model_name
        self._client     = Groq(api_key=api_key)
        self._planner    = planner

        logger.info(
            "Agent initialised — model=%r  tools=%s  planner=%s",
            model_name,
            registry.list_names(),
            "enabled" if planner else "disabled",
        )

    # ── Public API ────────────────────────────────────────────────────── #

    def run(self, instruction: str) -> ToolResult:
        """
        Execute *instruction*.

        Tries the Planner path first (if a Planner is attached).
        Falls back to the original single-step path on any failure.
        """
        if not instruction.strip():
            return ToolResult(success=False, error="Instruction must not be empty.")

        # ── Multi-step path ────────────────────────────────────────────── #
        if self._planner is not None:
            self._emit_event("info", "planning_started",
                             f"Planner is decomposing: {instruction[:80]}")
            try:
                plan = self._planner.create_plan(instruction)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Planner raised unexpectedly: %s — falling back.", exc)
                plan = None

            if plan is not None and isinstance(plan.get("steps"), list):
                step_count = len(plan["steps"])
                self._emit_event(
                    "status", "planning_completed",
                    f"Plan ready: {step_count} step(s).",
                )
                result = self._run_plan(instruction, plan)

                # Emit top-level finished event
                self._emit_event(
                    "status" if result.success else "error",
                    "execution_finished",
                    f"All steps done — success={result.success}",
                )
                return result

            logger.warning("Planner returned no valid plan — falling back to single-step.")
            self._emit_event("info", "planning_fallback",
                             "Planner produced no plan; using single-step execution.")

        # ── Single-step fallback (original path — unchanged) ──────────── #
        return self._run_single_step(instruction)

    # ── Multi-step execution ──────────────────────────────────────────── #

    def _run_plan(self, instruction: str, plan: dict) -> ToolResult:
        """
        Iterate over plan["steps"] and execute each one.

        Stops on the first failure and returns that step's ToolResult.
        On full success returns the last step's ToolResult.
        """
        steps: list[dict] = plan["steps"]
        last_result: ToolResult | None = None

        for idx, step in enumerate(steps):
            step_num   = idx + 1
            tool_name  = step.get("tool", "")
            arguments  = dict(step.get("arguments", {}))

            self._emit_event(
                "info", "step_started",
                f"Step {step_num}/{len(steps)}: {tool_name} "
                f"({arguments.get('workflow') or arguments.get('action', '')})",
            )

            # Apply the content-gen pre-pass so generated content still works
            arguments = self._maybe_generate_content(
                instruction=instruction,
                tool_name=tool_name,
                arguments=arguments,
            )

            last_result = self._executor.execute(tool_name, **arguments)

            if last_result.success:
                self._emit_event(
                    "status", "step_completed",
                    f"Step {step_num}/{len(steps)} succeeded: {last_result.output}",
                )
            else:
                self._emit_event(
                    "error", "step_failed",
                    f"Step {step_num}/{len(steps)} failed: {last_result.error}",
                )
                return last_result  # stop on first failure

        # All steps succeeded
        return last_result or ToolResult(
            success=False,
            error="Plan had no steps to execute.",
        )

    # ── Original single-step path (preserved verbatim) ───────────────── #

    def _run_single_step(self, instruction: str) -> ToolResult:
        """
        Original Agent execution logic — unchanged from the pre-planner version.
        """
        tool_metadata = self._registry.list_metadata()
        tool_listing  = _build_tool_listing(tool_metadata)
        user_prompt   = _USER_PROMPT_TEMPLATE.format(
            tool_listing=tool_listing,
            instruction=instruction,
        )

        logger.info("Sending instruction to Groq/Llama: %r", instruction[:120])

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

        arguments = self._maybe_generate_content(
            instruction=instruction,
            tool_name=tool_name,
            arguments=arguments,
        )

        return self._executor.execute(tool_name, **arguments)

    # ── Content-generation pre-pass (unchanged) ───────────────────────── #

    def _maybe_generate_content(
        self,
        *,
        instruction: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not _needs_content_generation(instruction):
            logger.debug("Content-gen pre-pass skipped — instruction appears verbatim.")
            return arguments

        workflow = arguments.get("workflow", "")
        content_key: str | None = None

        if tool_name == "ui_automation":
            content_key = _CONTENT_WORKFLOWS.get(workflow)
        elif tool_name == "file_creation":
            content_key = _CONTENT_TOOLS.get(tool_name)

        if content_key is None:
            return arguments

        existing = arguments.get(content_key, "")
        if isinstance(existing, str) and len(existing.strip()) > 20:
            logger.debug(
                "Content-gen pre-pass skipped — model already supplied content (%d chars).",
                len(existing.strip()),
            )
            return arguments

        medium = _detect_medium(instruction, workflow if tool_name == "ui_automation" else None)

        logger.info(
            "Content-gen pre-pass triggered — tool=%r  workflow=%r  content_key=%r  medium=%r",
            tool_name, workflow, content_key, medium,
        )

        generated = self._generate_content(instruction, medium)

        if generated:
            logger.info(
                "Content-gen pre-pass produced %d chars for key %r.",
                len(generated), content_key,
            )
            updated = dict(arguments)
            updated[content_key] = generated
            return updated

        logger.warning("Content-gen pre-pass produced no output — keeping original arguments.")
        return arguments

    def _generate_content(self, instruction: str, medium: str) -> str | None:
        user_prompt = _build_content_gen_prompt(instruction, medium)

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _CONTENT_GEN_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            text = (response.choices[0].message.content or "").strip()
            return text if text else None
        except Exception as exc:  # noqa: BLE001
            logger.error("Content-gen LLM call failed: %s", exc)
            return None

    # ── Internal event emitter ────────────────────────────────────────── #

    def _emit_event(self, type_: str, stage: str, message: str) -> None:
        """
        Emit a lifecycle event through the executor's patched execute path.

        The executor may have been monkey-patched with an event_callback by
        the REPL / API layer (see main.py / api.py).  We reach the callback
        by calling a no-op execution — instead we fire the event directly
        through any registered callback stored on the executor instance.

        Because we cannot call executor.execute() without a real tool, we
        store the last-known callback on self after each patched execute call.
        A simpler, zero-coupling approach is used here: we emit to the logger
        and also store a callback reference that the REPL/API can register.
        """
        event = {
            "type":      type_,
            "stage":     stage,
            "message":   message,
            "tool":      "agent/planner",
            "timestamp": _utc_now(),
        }
        logger.info("[%s] %s — %s", type_.upper(), stage, message)

        # Fire through registered callback if one has been set
        cb = getattr(self, "_event_callback", None)
        if cb is not None:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("_event_callback raised: %s", exc)

    def register_event_callback(self, callback) -> None:
        """
        Register a callback to receive Agent-level planning events.

        The REPL (main.py) and API layer (api.py) can call this so that
        planning_started / planning_completed / step_started / step_completed
        events appear in the same stream as tool-execution events.
        """
        self._event_callback = callback

    def clear_event_callback(self) -> None:
        """Remove the registered event callback."""
        self._event_callback = None

    def __repr__(self) -> str:
        return (
            f"<Agent model={self._model_name!r}  "
            f"tools={self._registry.list_names()}  "
            f"planner={'on' if self._planner else 'off'}>"
        )