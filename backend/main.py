"""
main.py

Interactive REPL entry point — GladdenAction v5 (Claude Code features).

New tools registered
--------------------
    code_edit   — surgical file read/edit/insert (str_replace diff)
    context     — project scan, find_files, grep_files, file_summary
    shell       — live terminal execution (python, npm, g++, pytest, git…)

Approval gate
-------------
    Plans are shown before execution. Auto-approved after 8s.
    Type 'reject' at the REPL prompt to cancel the pending plan.
    Type 'approve' to immediately proceed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

from agent.agent import Agent
from agent.approval_gate import ApprovalGate
from agent.planner import Planner
from core.tools import FileCreationTool
from core.tools.code_edit_tool import CodeEditTool
from core.tools.context_tool import ContextTool
from core.tools.registry import ToolRegistry
from core.tools.shell_tool import ShellTool
from core.tools.system_control_tool import SystemControlTool
from core.tools.ui_automation_tool import UIAutomationTool
from execution.executor import ToolExecutor


# ============================================================================ #
#  Console event callback                                                       #
# ============================================================================ #

_COLOURS = {
    "info":              "\033[94m",   # blue
    "status":            "\033[92m",   # green
    "error":             "\033[91m",   # red
    "security":          "\033[95m",   # magenta
    "approval_required": "\033[93m",   # yellow
    "approval_decision": "\033[96m",   # cyan
    "reset":             "\033[0m",
}


def console_event_callback(event: dict) -> None:
    etype  = event.get("type", "info")
    colour = _COLOURS.get(etype, _COLOURS["info"])
    reset  = _COLOURS["reset"]

    # Special rendering for approval events
    if etype == "approval_required":
        stage = event.get("stage", "")
        msg   = event.get("message", "")
        print(f"\n  {colour}{'═'*60}{reset}")
        print(f"  {colour}⚠  APPROVAL REQUIRED  — {stage}{reset}")
        print(f"  {colour}{msg}{reset}")
        if "plan" in event:
            for step in event["plan"]:
                marker = "🔴" if step.get("destructive") else "🔵"
                print(f"    {marker} Step {step['step']}: {step['description']}")
        print(f"  {colour}  Type 'approve' to proceed immediately, 'reject' to cancel.{reset}")
        print(f"  {colour}{'═'*60}{reset}\n")
        return

    if etype == "approval_decision":
        stage = event.get("stage", "")
        icons = {"approved": "✅", "rejected": "❌", "auto_approved": "⏱ "}
        icon  = icons.get(stage, "•")
        print(f"  {colour}{icon} {event.get('message', '')}{reset}\n")
        return

    print(
        f"  {colour}[{etype.upper():8}]{reset} "
        f"stage={event.get('stage',''):<32} "
        f"tool={event.get('tool',''):<36} "
        f"@ {event.get('timestamp','')}\n"
        f"             └─ {event.get('message','')}"
    )


# ============================================================================ #
#  Bootstrap                                                                    #
# ============================================================================ #

def build_agent() -> tuple[Agent, ApprovalGate]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable is not set.")
        print("        Get your key at https://console.groq.com")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(UIAutomationTool())
    registry.register(FileCreationTool())
    registry.register(SystemControlTool())
    registry.register(CodeEditTool())       # surgical file editing
    registry.register(ContextTool())        # project awareness
    registry.register(ShellTool())          # live terminal execution

    executor = ToolExecutor(registry)
    planner  = Planner(api_key=api_key)
    gate     = ApprovalGate(auto_approve_delay=8.0, destructive_delay=15.0)

    agent = Agent(
        registry=registry,
        executor=executor,
        api_key=api_key,
        planner=planner,
    )

    logger.info("Agent v5 ready — tools: %s", registry.list_names())
    return agent, gate


# ============================================================================ #
#  REPL                                                                         #
# ============================================================================ #

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║  GladdenAction v5 — Claude Code Features Enabled                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Code Editing (like Claude Code)                                     ║
║    "Read main.py and fix the bug on line 42"                         ║
║    "Scan my project and add type hints to all functions"             ║
║    "Find all TODO comments in the codebase"                          ║
║    "Refactor the Agent class to use async"                           ║
║                                                                      ║
║  Terminal Execution (live output)                                    ║
║    "Run pytest and show me the failures"                             ║
║    "npm install and then npm run build"                              ║
║    "Compile and run my C++ file"                                     ║
║    "Run git status and git log --oneline -10"                        ║
║                                                                      ║
║  Project Awareness                                                   ║
║    "Scan my project structure"                                       ║
║    "Find all Python files in src/"                                   ║
║    "Search for 'TODO' across all files"                              ║
║    "Summarise what executor.py does"                                 ║
║                                                                      ║
║  UI Automation (unchanged)                                           ║
║    "Open Notepad and write a shopping list"                          ║
║    "Send an email to bob@example.com"                                ║
║                                                                      ║
║  Approval Gate                                                       ║
║    Plans auto-approve after 8s (15s for destructive actions)         ║
║    Type 'approve' or 'reject' to control pending plans               ║
║                                                                      ║
║  Commands: tools | approve | reject | planner on/off | quit         ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _print_result(result) -> None:
    print()
    if result.success:
        print("  ✅  Success")
        print(f"  Output   : {result.output}")
        if result.metadata:
            print(f"  Metadata : {result.metadata}")
    else:
        print("  ❌  Failed")
        print(f"  Error    : {result.error}")


def repl(agent: Agent, gate: ApprovalGate) -> None:
    print(BANNER)
    print(f"  Agent : {agent}")
    print()

    while True:
        try:
            raw = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        if raw.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if raw.lower() == "tools":
            print("\n  Registered tools:")
            for name in agent._registry.list_names():
                print(f"    • {name}")
            print()
            continue

        if raw.lower() == "approve":
            gate.approve("User typed 'approve'")
            print("  ✅  Approved.")
            continue

        if raw.lower() == "reject":
            gate.reject("User typed 'reject'")
            print("  ❌  Rejected — plan cancelled.")
            continue

        if raw.lower() == "planner on":
            if agent._planner is None:
                api_key = os.environ.get("GROQ_API_KEY", "").strip()
                agent._planner = Planner(api_key=api_key)
            print("  Planner: ON")
            continue

        if raw.lower() == "planner off":
            agent._planner = None
            print("  Planner: OFF  (single-step mode)")
            continue

        # ── Patch executor with callback ──────────────────────────── #
        original_execute = agent._executor.execute

        def execute_with_callback(tool_name, **kwargs):
            return original_execute(
                tool_name,
                event_callback=console_event_callback,
                **kwargs,
            )

        agent._executor.execute = execute_with_callback

        print()
        print("  ── Execution events ─────────────────────────────────────────")
        result = agent.run(raw, event_callback=console_event_callback)
        print("  ─────────────────────────────────────────────────────────────")
        _print_result(result)

        agent._executor.execute = original_execute
        print()


# ============================================================================ #
#  Entry point                                                                  #
# ============================================================================ #

if __name__ == "__main__":
    agent, gate = build_agent()
    repl(agent, gate)