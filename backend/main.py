"""
main.py

Manual integration test for the tool abstraction layer.
Run this script to verify that the registry and FileCreationTool work
correctly before moving on to LLM / FastAPI integration.

Usage
-----
    python main.py
"""

import logging
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Logging setup  (shows INFO messages from tools/registry in the terminal)    #
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
from core.tools import FileCreationTool, ToolResult, registry


# --------------------------------------------------------------------------- #
#  Helper                                                                       #
# --------------------------------------------------------------------------- #

def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def assert_result(label: str, result: ToolResult, expect_success: bool) -> None:
    status = "✅ PASS" if result.success == expect_success else "❌ FAIL"
    print(f"  {status}  [{label}]")
    if result.success:
        print(f"         output   : {result.output}")
        print(f"         metadata : {result.metadata}")
    else:
        print(f"         error    : {result.error}")


# --------------------------------------------------------------------------- #
#  Test suite                                                                   #
# --------------------------------------------------------------------------- #

def test_registry() -> None:
    print_section("1 · Registry — registration & introspection")

    tool = FileCreationTool()
    registry.register(tool)

    print(f"  Registry: {registry}")
    print(f"  Registered tools: {registry.list_names()}")

    metadata = registry.list_metadata()
    print(f"  Tool metadata snapshot:")
    for m in metadata:
        print(f"    name        : {m['name']}")
        print(f"    description : {m['description']}")
        print(f"    input_schema: {m['input_schema']}")

    assert "file_creation" in registry.list_names(), "file_creation not found!"
    print("  ✅ PASS  [registry contains file_creation]")


def test_file_creation_success() -> None:
    print_section("2 · FileCreationTool — successful creation")

    tool = registry.get("file_creation")
    result = tool.execute(filename="test_output/hello.txt", content="Hello, AI Agent!\n")
    assert_result("create hello.txt", result, expect_success=True)

    if result.success:
        assert Path(result.output).exists(), "File not found on disk!"
        print("  ✅ PASS  [file exists on disk]")


def test_file_creation_overwrite_guard() -> None:
    print_section("3 · FileCreationTool — overwrite guard (expect failure)")

    tool = registry.get("file_creation")
    # Try creating the same file again without overwrite=True → should fail
    result = tool.execute(filename="test_output/hello.txt", content="Should not overwrite")
    assert_result("overwrite blocked", result, expect_success=False)


def test_file_creation_with_overwrite() -> None:
    print_section("4 · FileCreationTool — explicit overwrite=True (expect success)")

    tool = registry.get("file_creation")
    result = tool.execute(
        filename="test_output/hello.txt",
        content="Overwritten content ✓\n",
        overwrite=True,
    )
    assert_result("overwrite allowed", result, expect_success=True)

    if result.success:
        content = Path(result.output).read_text(encoding="utf-8")
        assert "Overwritten" in content
        print("  ✅ PASS  [file content updated on disk]")


def test_missing_required_inputs() -> None:
    print_section("5 · FileCreationTool — missing required inputs (expect failure)")

    tool = registry.get("file_creation")
    result = tool.execute(filename="something.txt")  # 'content' omitted
    assert_result("missing content", result, expect_success=False)


def test_empty_filename() -> None:
    print_section("6 · FileCreationTool — empty filename (expect failure)")

    tool = registry.get("file_creation")
    result = tool.execute(filename="   ", content="data")
    assert_result("empty filename", result, expect_success=False)


def test_duplicate_registry_registration() -> None:
    print_section("7 · Registry — duplicate registration (expect ValueError)")

    try:
        registry.register(FileCreationTool())  # already registered
        print("  ❌ FAIL  [no error raised for duplicate]")
    except ValueError as exc:
        print(f"  ✅ PASS  [ValueError raised: {exc}]")


def test_unknown_tool_lookup() -> None:
    print_section("8 · Registry — unknown tool lookup (expect KeyError)")

    try:
        registry.get("nonexistent_tool")
        print("  ❌ FAIL  [no error raised]")
    except KeyError as exc:
        print(f"  ✅ PASS  [KeyError raised: {exc}]")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║         AI Agent Backend — Tool Layer Test Suite        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_registry()
    test_file_creation_success()
    test_file_creation_overwrite_guard()
    test_file_creation_with_overwrite()
    test_missing_required_inputs()
    test_empty_filename()
    test_duplicate_registry_registration()
    test_unknown_tool_lookup()

    print("\n" + "═" * 60)
    print("  All tests completed.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()