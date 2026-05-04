"""
agent/agent.py  (updated — nl_command tool integration)

Added RULE 4: Natural Language → Shell Command routing via the nl_command tool.
All other behaviour is unchanged.
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

_AUTO_PREFIX = "auto:"

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
  - LinkedIn, profile, connection request, accept connections, pending invitations
  - C++, compile, run program, g++, executable
  - Screenshot, capture screen, take a screenshot
  - Opening apps/applications by name
  - Google Calendar, add event, create event, schedule, calendar, meeting, appointment

RULE 2 — MATCH THE WORKFLOW ARGUMENT:
When you select "ui_automation", pick the correct "workflow" value:

  File creation:
    "create_file_notepad"     → user wants to create/write a file using Notepad
    "create_file_vscode"      → user wants to create/write code in VS Code

  Email:
    "send_email_browser"      → send email via Gmail in Chrome

  Calendar:
    "create_event_browser"    → create a Google Calendar event via Chrome browser
                                (args: event_title required; event_date as YYYY-MM-DD,
                                 event_start_time as HH:MM 24h, event_end_time as HH:MM 24h)

  WhatsApp:
    "send_whatsapp_desktop"   → send a single WhatsApp message to one contact
    "send_whatsapp_advanced"  → send to multiple contacts, repeat messages, or use delay

  YouTube:
    "play_youtube_video"      → search YouTube and play the first video result (query required)

  LinkedIn:
    "linkedin_action"               → search or open a LinkedIn profile (action: "search" or "open" ONLY)
    "accept_linkedin_connections"   → open LinkedIn invitation manager and accept ALL pending connection requests (no extra args needed)

  Code:
    "code_workflow_cpp"       → create a C++ file, compile with g++, run it

  System:
    "launch_application"      → open a named application (chrome, vscode, calculator, etc.)
    "take_screenshot"         → capture the screen and save as PNG to Desktop

RULE 3 — DIRECT API TOOLS (fallback only):
Use "file_creation" only when the user explicitly asks for a direct/silent file
operation with no mention of a visible app.

RULE 4 — NATURAL LANGUAGE → COMMANDS  (nl_command tool):
Select "nl_command" when the user's message matches ANY of these patterns:

  SHELL commands:
  - "how do I [verb] [thing] in terminal / cmd / powershell / command line"
  - "give me the command to …"
  - "what command …" / "what's the command for …"
  - "run a command that …"
  - "find files that …" / "find all [files/folders] …"
  - "check what process is on port …" / "which process is using port …"
  - "show me disk usage / memory usage / network stats" via a command
  - "compress / zip / archive [folder/files]"
  - "kill the process on port …" / "terminate process …" via command
  - "list all [running processes / open ports / env vars / services]" via CLI
  - "count [lines/files/words] …"
  - "I always forget the command for …"
  - System monitoring, file searching, or network diagnostics via CLI

  GIT commands:
  - "git command to …" / "how do I … in git"
  - "undo / revert / reset" a commit, push, or change via git
  - "show git log …" / "search git history …"
  - "stash", "cherry-pick", "squash", "rebase", "bisect"
  - "list branches sorted by …" / "find commits by …"
  - "what will be pushed" / "show diff between branches"
  - "amend last commit" / "create a signed tag"
  - Anything involving commit history, branches, remotes, or git workflows

  SQL queries:
  - "write a SQL query to …" / "SQL to find / count / update …"
  - "how do I join … in SQL" / "query to get all … where …"
  - "create a table …" / "add an index on …"
  - "SQL for pagination" / "SQL to delete duplicates"
  - Anything described as a database query, DML, or DDL

  REGEX patterns:
  - "regex / regular expression to match …"
  - "pattern to validate / extract / parse …"
  - "give me a regex for emails / URLs / phone numbers / dates"
  - "regex with lookahead / lookbehind / capture group"
  - "does [string] match [pattern]" / "explain this regex: …"

  DOCKER commands:
  - "docker command to …" / "how do I … in docker"
  - "show running containers / images / volumes"
  - "tail logs for container …" / "exec into container"
  - "docker compose …" / "inspect container network"

  NETWORK commands:
  - "check if [host] is reachable" / "ping …"
  - "trace route to …" / "DNS lookup for …"
  - "find my public IP" / "show listening ports"
  - "flush DNS cache" / "check SSL certificate"
  - Anything about network connectivity, DNS, or HTTP checks

RULE 4 ARGUMENT MAPPING for nl_command:
  "request"      → the full natural-language description (required)
  "category"     → auto-detected; override with: shell | git | sql | regex | docker | network
  "execute"      → true by default; false/preview_only=true when user says
                   "just show me", "what would the command be", "don't run", "preview"
  "preview_only" → true when user says "just show the command", "don't execute"
  "working_dir"  → fill if user specifies a folder context

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

User: "Add a meeting called Team Sync on 2025-06-15 from 10:00 to 11:00"
Response: {"tool": "ui_automation", "arguments": {"workflow": "create_event_browser", "event_title": "Team Sync", "event_date": "2025-06-15", "event_start_time": "10:00", "event_end_time": "11:00"}}

User: "Schedule a dentist appointment on 2025-07-20 at 9am to 9:30am on Google Calendar"
Response: {"tool": "ui_automation", "arguments": {"workflow": "create_event_browser", "event_title": "Dentist Appointment", "event_date": "2025-07-20", "event_start_time": "09:00", "event_end_time": "09:30"}}

User: "Send a WhatsApp message to Alice asking about her weekend"
Response: {"tool": "ui_automation", "arguments": {"workflow": "send_whatsapp_desktop", "contact_name": "Alice", "message": "Hey Alice! Hope you had a great weekend. How did it go?"}}

User: "Play a YouTube video about system design interviews"
Response: {"tool": "ui_automation", "arguments": {"workflow": "play_youtube_video", "query": "system design interviews"}}

User: "Open Chrome"
Response: {"tool": "ui_automation", "arguments": {"workflow": "launch_application", "app_name": "chrome"}}

User: "Take a screenshot"
Response: {"tool": "ui_automation", "arguments": {"workflow": "take_screenshot"}}

User: "Create a file called notes.txt with the content buy milk"
Response: {"tool": "file_creation", "arguments": {"filename": "notes.txt", "content": "buy milk"}}

User: "I always forget how to find which process is using port 8080"
Response: {"tool": "nl_command", "arguments": {"request": "find which process is using port 8080", "execute": true}}

User: "Give me the command to list all python files recursively"
Response: {"tool": "nl_command", "arguments": {"request": "list all python files recursively", "preview_only": true}}

User: "What's the command to compress the dist folder into a zip?"
Response: {"tool": "nl_command", "arguments": {"request": "compress the dist folder into a zip", "preview_only": true}}

User: "Run a command to show top 10 largest files in the current directory"
Response: {"tool": "nl_command", "arguments": {"request": "show top 10 largest files in current directory", "execute": true}}

User: "How do I check disk usage on Windows?"
Response: {"tool": "nl_command", "arguments": {"request": "check disk usage on Windows", "preview_only": true}}

User: "What git command undoes the last commit without losing my changes?"
Response: {"tool": "nl_command", "arguments": {"request": "undo the last git commit without losing changes", "preview_only": true}}

User: "What command shows all environment variables?"
Response: {"tool": "nl_command", "arguments": {"request": "show all environment variables", "execute": true}}

User: "Check what's listening on port 3000"
Response: {"tool": "nl_command", "arguments": {"request": "check what process is listening on port 3000", "execute": true}}

User: "Show me top 5 processes by RAM usage"
Response: {"tool": "system_control", "arguments": {"domain": "process", "action": "list", "options": {"sort_by": "memory", "limit": 5}}}

User: "What is my CPU usage?"
Response: {"tool": "system_control", "arguments": {"domain": "system", "action": "cpu_usage"}}

User: "What git command squashes the last 4 commits?"
Response: {"tool": "nl_command", "arguments": {"request": "squash the last 4 commits", "category": "git", "preview_only": true}}

User: "Write a SQL query to find users who signed up in the last 30 days but never placed an order"
Response: {"tool": "nl_command", "arguments": {"request": "find users who signed up in the last 30 days but never placed an order", "category": "sql", "preview_only": true}}

User: "Give me a regex to validate email addresses"
Response: {"tool": "nl_command", "arguments": {"request": "validate email addresses", "category": "regex", "preview_only": true}}

User: "Docker command to tail logs for the web container"
Response: {"tool": "nl_command", "arguments": {"request": "tail logs for the web container", "category": "docker", "execute": true}}

User: "How do I find my public IP from the terminal?"
Response: {"tool": "nl_command", "arguments": {"request": "find my public IP address from the terminal", "category": "network", "execute": true}}

User: "Run git log with a graph showing the last 20 commits"
Response: {"tool": "nl_command", "arguments": {"request": "show git log with graph for last 20 commits", "category": "git", "execute": true}}

User: "SQL query to count orders grouped by status"
Response: {"tool": "nl_command", "arguments": {"request": "count orders grouped by status", "category": "sql", "preview_only": true}}

User: "Regex to extract all URLs from a string"
Response: {"tool": "nl_command", "arguments": {"request": "extract all URLs from a string", "category": "regex", "preview_only": true}}

User: "What git command lists branches sorted by most recent commit?"
Response: {"tool": "nl_command", "arguments": {"request": "list branches sorted by most recent commit date", "category": "git", "preview_only": true}}

User: "Show me the docker command to see disk usage"
Response: {"tool": "nl_command", "arguments": {"request": "show docker disk usage", "category": "docker", "preview_only": true}}
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
    # git
    "commit", "squash", "rebase", "stash", "cherry", "bisect", "reflog",
    "amend", "branch", "merge", "undo commit",
    # sql
    "sql", "query", "select", "insert", "update", "join", "group by",
    "where clause", "create table",
    # regex
    "regex", "regular expression", "pattern to match", "pattern for",
    "match emails", "match urls", "validate",
    # docker
    "docker", "container", "image logs",
    # network
    "trace route", "dns lookup", "ping ",
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
        props    = tool["input_schema"].get("properties", {})
        required = tool["input_schema"].get("required", [])
        if props:
            lines.append("   Arguments:")
            for arg_name, spec in props.items():
                req_marker = " (required)" if arg_name in required else " (optional)"
                arg_type   = spec.get("type", "any")
                arg_desc   = spec.get("description", "")
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


# --------------------------------------------------------------------------- #
#  Agent                                                                        #
# --------------------------------------------------------------------------- #

class Agent:
    """
    Single-step reasoning agent backed by Groq + Llama 3.3.

    Autonomous mode
    ---------------
    Prefix the instruction with "AUTO:" to engage the AutonomousController.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        api_key: str,
        planner,
        model_name: str = _DEFAULT_MODEL,
        autonomous_max_steps: int = 8,
        autonomous_use_perception: bool = False,
    ) -> None:
        self._registry   = registry
        self._executor   = executor
        self._model_name = model_name
        self._planner    = planner
        self._client     = Groq(api_key=api_key)

        self._autonomous_max_steps      = autonomous_max_steps
        self._autonomous_use_perception = autonomous_use_perception
        self._auto_controller           = None

        logger.info(
            "Agent initialised — model=%r  tools=%s",
            model_name,
            registry.list_names(),
        )

    # ------------------------------------------------------------------ #
    #  AutonomousController accessor (lazy init)                          #
    # ------------------------------------------------------------------ #

    def _get_auto_controller(self):
        if self._auto_controller is None:
            from .autonomous_controller import AutonomousController
            self._auto_controller = AutonomousController(
                registry       = self._registry,
                executor       = self._executor,
                groq_client    = self._client,
                model_name     = self._model_name,
                max_steps      = self._autonomous_max_steps,
                use_perception = self._autonomous_use_perception,
            )
            logger.info("AutonomousController initialised (lazy).")
        return self._auto_controller

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(self, instruction: str, event_callback=None) -> ToolResult:
        if not instruction.strip():
            return ToolResult(success=False, error="Instruction must not be empty.")

        if instruction.strip().lower().startswith(_AUTO_PREFIX):
            goal = instruction.strip()[len(_AUTO_PREFIX):].strip()
            if not goal:
                return ToolResult(
                    success=False,
                    error="AUTO: prefix provided but goal is empty.",
                )
            logger.info("Routing to AutonomousController — goal=%r", goal[:80])
            return self._get_auto_controller().run(
                goal,
                event_callback=event_callback,
            )

        return self._run_deterministic(instruction, event_callback)

    # ------------------------------------------------------------------ #
    #  Deterministic flow                                                  #
    # ------------------------------------------------------------------ #

    def _run_deterministic(self, instruction: str, event_callback=None) -> ToolResult:
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
        except Exception as exc:           # noqa: BLE001
            msg = f"Groq API call failed: {exc}"
            logger.error(msg)
            return ToolResult(success=False, error=msg)

        logger.debug("Groq raw response: %s", raw_text)

        json_str = _extract_json(raw_text)
        try:
            decision: dict = json.loads(json_str)
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
        arguments = decision.get("arguments", {})

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

        if event_callback is not None:
            return self._executor.execute(
                tool_name,
                event_callback=event_callback,
                **arguments,
            )
        return self._executor.execute(tool_name, **arguments)

    # ------------------------------------------------------------------ #
    #  Content generation pre-pass                                        #
    # ------------------------------------------------------------------ #

    def _maybe_generate_content(
        self,
        *,
        instruction: str,
        tool_name: str,
        arguments: dict,
    ) -> dict:
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
        except Exception as exc:           # noqa: BLE001
            logger.error("Content-gen LLM call failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return (
            f"<Agent model={self._model_name!r}  "
            f"tools={self._registry.list_names()}>"
        )