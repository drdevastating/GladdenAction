"""
main.py

Manual integration test for the Tool Abstraction + Execution layers.
All tool calls now go through ToolExecutor — no layer calls tool.execute() directly.

Usage
-----
    python main.py
"""

import logging
from pathlib import Path

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
from core.tools import FileCreationTool, ToolResult
from core.tools.registry import ToolRegistry
from execution.executor import ToolExecutor


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def print_section(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def assert_result(label: str, result: ToolResult, expect_success: bool) -> None:
    status = "✅ PASS" if result.success == expect_success else "❌ FAIL"
    print(f"  {status}  [{label}]")
    if result.success:
        print(f"         output   : {result.output}")
        print(f"         metadata : {result.metadata}")
    else:
        print(f"         error    : {result.error}")


# --------------------------------------------------------------------------- #
#  Bootstrap: registry + executor                                               #
# --------------------------------------------------------------------------- #

def build_executor() -> ToolExecutor:
    """
    Compose the registry and executor.
    In production this wiring will live in an app factory / DI container.
    """
    registry = ToolRegistry()
    registry.register(FileCreationTool())
    executor = ToolExecutor(registry)
    print(f"\n  Executor ready: {executor}")
    return executor


# --------------------------------------------------------------------------- #
#  Test suite                                                                   #
# --------------------------------------------------------------------------- #

def test_executor_introspection(executor: ToolExecutor) -> None:
    print_section("1 · Executor — introspection")
    print(f"  Available tools : {executor.available_tools()}")
    for m in executor.tool_metadata():
        print(f"  Tool descriptor : name={m['name']!r}  "
              f"desc={m['description'][:55]!r}…")
    print("  ✅ PASS  [executor introspection]")


def test_successful_creation(executor: ToolExecutor) -> None:
    print_section("2 · Executor → FileCreationTool — successful creation")
    result = executor.execute("file_creation",
                              filename="test_output/hello.txt",
                              content="Hello from the executor layer!\n")
    assert_result("create hello.txt", result, expect_success=True)
    if result.success:
        assert Path(result.output).exists(), "File not on disk!"
        print("  ✅ PASS  [file exists on disk]")


def test_overwrite_guard(executor: ToolExecutor) -> None:
    print_section("3 · Executor → FileCreationTool — overwrite guard (expect failure)")
    result = executor.execute("file_creation",
                              filename="test_output/hello.txt",
                              content="Should be blocked")
    assert_result("overwrite blocked", result, expect_success=False)


def test_explicit_overwrite(executor: ToolExecutor) -> None:
    print_section("4 · Executor → FileCreationTool — overwrite=True (expect success)")
    result = executor.execute("file_creation",
                              filename="test_output/hello.txt",
                              content="Updated via executor ✓\n",
                              overwrite=True)
    assert_result("overwrite allowed", result, expect_success=True)
    if result.success:
        content = Path(result.output).read_text(encoding="utf-8")
        assert "Updated via executor" in content
        print("  ✅ PASS  [file content updated on disk]")


def test_missing_inputs(executor: ToolExecutor) -> None:
    print_section("5 · Executor — missing required inputs (expect failure)")
    result = executor.execute("file_creation", filename="ghost.txt")
    assert_result("missing content", result, expect_success=False)


def test_unknown_tool(executor: ToolExecutor) -> None:
    print_section("6 · Executor — unknown tool name (expect failure)")
    result = executor.execute("nonexistent_tool", foo="bar")
    assert_result("unknown tool", result, expect_success=False)


def test_empty_filename(executor: ToolExecutor) -> None:
    print_section("7 · Executor → FileCreationTool — empty filename (expect failure)")
    result = executor.execute("file_creation", filename="  ", content="data")
    assert_result("empty filename", result, expect_success=False)


def test_nested_path_creation(executor: ToolExecutor) -> None:
    print_section("8 · Executor → FileCreationTool — nested directories")
    result = executor.execute("file_creation",
                              filename="test_output/deep/nested/report.txt",
                              content="Deep file created via executor.\n")
    assert_result("nested path", result, expect_success=True)
    if result.success:
        assert Path(result.output).exists()
        print("  ✅ PASS  [nested directories created on disk]")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║      AI Agent Backend — Execution Layer Test Suite        ║")
    print("╚════════════════════════════════════════════════════════════╝")

    executor = build_executor()

    test_executor_introspection(executor)
    test_successful_creation(executor)
    test_overwrite_guard(executor)
    test_explicit_overwrite(executor)
    test_missing_inputs(executor)
    test_unknown_tool(executor)
    test_empty_filename(executor)
    test_nested_path_creation(executor)

    print("\n" + "═" * 62)
    print("  All tests completed.")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()