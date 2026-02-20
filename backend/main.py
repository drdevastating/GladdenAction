"""
main.py

Interactive entry point for the AI Agent backend.
Loads the full stack (Registry → Executor → Agent) and drops into a
terminal REPL where you can give natural-language instructions to the agent.

Setup
-----
    # Windows
    set XAI_API_KEY=your_key_here
    python main.py

    # PowerShell
    $env:XAI_API_KEY="your_key_here"
    python main.py

    # macOS / Linux
    export XAI_API_KEY=your_key_here
    python main.py

Type  'quit' or 'exit'  to stop.
Type  'tools'           to list registered tools.
"""

import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()  # reads .env from the project root before anything else runs

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
#  Bootstrap                                                                    #
# --------------------------------------------------------------------------- #

def build_agent() -> Agent:
    """Wire up the full stack and return a ready Agent."""

    # 1. API key
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] XAI_API_KEY environment variable is not set.")
        print("        Set it and re-run:  set XAI_API_KEY=your_key_here")
        sys.exit(1)

    # 2. Registry — register all tools here
    registry = ToolRegistry()
    registry.register(FileCreationTool())

    # 3. Executor
    executor = ToolExecutor(registry)

    # 4. Agent
    agent = Agent(registry=registry, executor=executor, api_key=api_key)

    return agent


# --------------------------------------------------------------------------- #
#  REPL                                                                         #
# --------------------------------------------------------------------------- #

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            AI Agent Backend — Interactive Mode             ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                  ║
║    tools        → list available tools                      ║
║    quit / exit  → exit                                      ║
║    <anything else> → sent to the agent as an instruction    ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_result(result) -> None:
    """Pretty-print a ToolResult."""
    if result.success:
        print(f"\n  ✅  Success")
        print(f"  Output   : {result.output}")
        if result.metadata:
            print(f"  Metadata : {result.metadata}")
    else:
        print(f"\n  ❌  Failed")
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

        # Send to agent
        print()
        result = agent.run(raw)
        print_result(result)
        print()


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    agent = build_agent()
    repl(agent)