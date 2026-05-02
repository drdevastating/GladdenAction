"""
core/tools/__init__.py  (updated — v7 new tools registered)

Changes:
    + CalendarTool
    + OneNoteTool
    + ClockTool
    + RecorderTool
"""

from core.tools.base import BaseTool, ToolResult
from core.tools.calendar_tool import CalendarTool
from core.tools.clock_tool import ClockTool
from core.tools.file_creation_tool import FileCreationTool
from core.tools.onenote_tool import OneNoteTool
from core.tools.recorder_tool import RecorderTool
from core.tools.registry import ToolRegistry, registry

__all__ = [
    "BaseTool",
    "ToolResult",
    "CalendarTool",
    "ClockTool",
    "FileCreationTool",
    "OneNoteTool",
    "RecorderTool",
    "ToolRegistry",
    "registry",
]


# ═══════════════════════════════════════════════════════════════════════
# UPDATED main.py bootstrap — replace build_agent() with this version
# ═══════════════════════════════════════════════════════════════════════
#
#  from core.tools.calendar_tool  import CalendarTool
#  from core.tools.onenote_tool   import OneNoteTool
#  from core.tools.clock_tool     import ClockTool
#  from core.tools.recorder_tool  import RecorderTool
#
#  def build_agent() -> tuple[Agent, ApprovalGate]:
#      ...
#      registry.register(UIAutomationTool())
#      registry.register(FileCreationTool())
#      registry.register(SystemControlTool())
#      registry.register(CodeEditTool())
#      registry.register(ContextTool())
#      registry.register(ShellTool())
#      registry.register(NLCommandTool())
#      registry.register(CalendarTool())    # ← NEW
#      registry.register(OneNoteTool())     # ← NEW
#      registry.register(ClockTool())       # ← NEW
#      registry.register(RecorderTool())    # ← NEW
#      ...