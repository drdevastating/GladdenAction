"""
main.py

Interactive REPL entry point for the AI Agent backend.

Registered tools
----------------
    ui_automation    — visible desktop UI workflows (Notepad, VS Code, Gmail, WhatsApp)
    file_creation    — direct/silent file writes
    system_control   — secure OS capability engine (process / system / filesystem)
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

from agent.agent import Agent
from core.tools import FileCreationTool
from core.tools.registry import ToolRegistry
from core.tools.system_control_tool import SystemControlTool
from core.tools.ui_automation_tool import UIAutomationTool
from execution.executor import ToolExecutor


# ============================================================================ #
#  Console event callback                                                       #
# ============================================================================ #

_COLOURS = {
    "info":     "\033[94m",   # blue
    "status":   "\033[92m",   # green
    "error":    "\033[91m",   # red
    "security": "\033[95m",   # magenta
    "reset":    "\033[0m",
}


def console_event_callback(event: dict) -> None:
    etype  = event.get("type", "info")
    colour = _COLOURS.get(etype, "")
    reset  = _COLOURS["reset"]
    print(
        f"  {colour}[{etype.upper():8}]{reset} "
        f"stage={event['stage']:<32} "
        f"tool={event['tool']:<36} "
        f"@ {event['timestamp']}\n"
        f"             └─ {event['message']}"
    )


# ============================================================================ #
#  Bootstrap                                                                    #
# ============================================================================ #

def build_agent() -> Agent:
    """Wire up the full tool stack and return a ready Agent."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable is not set.")
        print("        Get your key at https://console.groq.com")
        print("        Then:  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(UIAutomationTool())   # primary: visible UI workflows
    registry.register(FileCreationTool())   # fallback: silent file writes
    registry.register(SystemControlTool())  # secure OS control engine

    executor = ToolExecutor(registry)
    agent    = Agent(registry=registry, executor=executor, api_key=api_key)

    logger.info("Agent ready — tools: %s", registry.list_names())
    return agent


# ============================================================================ #
#  REPL                                                                         #
# ============================================================================ #

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║         AI Agent — Jarvis Mode  (Groq + Secure OS Control)          ║
╠══════════════════════════════════════════════════════════════════════╣
║  UI Automation                                                       ║
║    "Open Notepad and write a shopping list"                          ║
║    "Create a C++ Hello World in VS Code"                             ║
║    "Send an email to you@example.com saying hello"                   ║
║                                                                      ║
║  Process Control                                                     ║
║    "Show top 5 processes by RAM"                                     ║
║    "Inspect chrome.exe"                                              ║
║    "Kill notepad.exe"                                                ║
║                                                                      ║
║  System Metrics                                                      ║
║    "What is my CPU usage?"                                           ║
║    "Check memory stats"                                              ║
║    "Show disk usage for C:"                                          ║
║    "How long has the PC been running?"                               ║
║                                                                      ║
║  Filesystem                                                          ║
║    "List files in my Documents folder"                               ║
║    "Get info about report.pdf"                                       ║
║    "Create a folder called MyProject on the Desktop"                 ║
║    "Rename notes.txt to todo.txt"                                    ║
║    "Delete temp.log"                                                 ║
║                                                                      ║
║  Commands:  tools | quit | exit                                      ║
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


def repl(agent: Agent) -> None:
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

        # Inject event callback for this run
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
        result = agent.run(raw)
        print("  ─────────────────────────────────────────────────────────────")
        _print_result(result)

        agent._executor.execute = original_execute
        print()


# ============================================================================ #
#  Entry point                                                                  #
# ============================================================================ #

if __name__ == "__main__":
    agent = build_agent()
    repl(agent)