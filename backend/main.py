"""
main.py

Interactive entry point for the AI Agent backend.
Loads the full stack (Registry → Executor → Agent) and drops into a
terminal REPL where you can give natural-language instructions to the agent.

Setup
-----
    # Windows
    set GROQ_API_KEY=your_key_here
    python main.py

    # PowerShell
    $env:GROQ_API_KEY="your_key_here"
    python main.py

    # macOS / Linux
    export GROQ_API_KEY=your_key_here
    python main.py

Type  'quit' or 'exit'  to stop.
Type  'tools'           to list registered tools.
"""

import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# --------------------------------------------------------------------------- #
#  Imports                                                                      #
# --------------------------------------------------------------------------- #
from core.tools import FileCreationTool
from core.tools.registry import ToolRegistry
from execution.executor import ToolExecutor
from agent.agent import Agent


# --------------------------------------------------------------------------- #
#  Event callback                                                               #
#                                                                               #
#  This is a plain Python function — no WebSockets, no FastAPI, no threads.    #
#  Later you will swap this out for a WebSocket sender or an SSE emitter.      #
#  The executor doesn't care: it just calls callback(event).                   #
# --------------------------------------------------------------------------- #

# ANSI colour codes for terminal readability
_COLOURS = {
    "info":   "\033[94m",   # blue
    "status": "\033[92m",   # green
    "error":  "\033[91m",   # red
    "reset":  "\033[0m",
}

def console_event_callback(event: dict) -> None:
    """
    Print a structured execution event to the terminal.

    This is the local stand-in for what will later become a WebSocket
    broadcast or Server-Sent Event push.

    Expected event shape
    --------------------
    {
        "type":      "info" | "status" | "error",
        "stage":     "<stage_name>",
        "message":   "<human-readable message>",
        "tool":      "<tool_name>",
        "timestamp": "<ISO-8601 UTC timestamp>"
    }
    """
    colour = _COLOURS.get(event.get("type", "info"), "")
    reset  = _COLOURS["reset"]

    print(
        f"  {colour}[{event['type'].upper():6}]{reset} "
        f"stage={event['stage']:<26} "
        f"tool={event['tool']:<20} "
        f"@ {event['timestamp']}\n"
        f"           └─ {event['message']}"
    )


# --------------------------------------------------------------------------- #
#  Bootstrap                                                                    #
# --------------------------------------------------------------------------- #

def build_agent() -> Agent:
    """Wire up the full stack and return a ready Agent."""

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable is not set.")
        print("        Get your key at https://console.groq.com")
        print("        Then set it:  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(FileCreationTool())

    executor = ToolExecutor(registry)
    agent    = Agent(registry=registry, executor=executor, api_key=api_key)

    return agent


# --------------------------------------------------------------------------- #
#  REPL                                                                         #
# --------------------------------------------------------------------------- #

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         AI Agent Backend — Interactive Mode (Groq)          ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                  ║
║    tools        → list available tools                      ║
║    quit / exit  → exit                                      ║
║    <anything else> → sent to the agent as an instruction    ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_result(result) -> None:
    """Pretty-print the final ToolResult after all events have fired."""
    print()
    if result.success:
        print(f"  ✅  Success")
        print(f"  Output   : {result.output}")
        if result.metadata:
            print(f"  Metadata : {result.metadata}")
    else:
        print(f"  ❌  Failed")
        print(f"  Error    : {result.error}")


def repl(agent: Agent) -> None:
    print(BANNER)
    print(f"  Agent   : {agent}")
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

        # ---------------------------------------------------------------- #
        # Run the agent.                                                    #
        #                                                                   #
        # The agent calls executor.execute() internally. We monkey-patch   #
        # the executor so that every execute() call automatically receives  #
        # our callback — without touching agent.py at all.                 #
        #                                                                   #
        # When WebSockets arrive, replace console_event_callback with a    #
        # function that does: await websocket.send_json(event)             #
        # ---------------------------------------------------------------- #

        # Wrap executor.execute so the callback is always injected
        original_execute = agent._executor.execute

        def execute_with_callback(tool_name, **kwargs):
            return original_execute(
                tool_name,
                event_callback=console_event_callback,
                **kwargs,
            )

        agent._executor.execute = execute_with_callback

        print()
        print("  ── Execution events ─────────────────────────────────────")
        result = agent.run(raw)
        print("  ─────────────────────────────────────────────────────────")
        print_result(result)

        # Restore original so the executor stays clean between calls
        agent._executor.execute = original_execute
        print()


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    agent = build_agent()
    repl(agent)