"""Windows automation tools package"""

# Tool Registry
from .registry import (
    ToolRegistry,
    initialize_tool_registry,
    get_tool_registry,
    get_tool,
    get_web_tools,
    get_shell_tools,
    get_file_tools,
    get_gui_tools,
)

# Advanced tools
from .web_search import RobustWebSearchTool, AdvancedWebTool
from .shell import AdvancedPowerShellTool, AdvancedShellTool
from .file_ops import AdvancedFileReadTool
from .windows.gui import AdvancedGUIAutomationTool, ProGUIAutomationTool

# Legacy Windows tools
from .windows.automation import (
    type_text,
    click,
    hotkey,
    activate_window,
    get_focused_window
)

__all__ = [
    # Tool Registry
    'ToolRegistry',
    'initialize_tool_registry',
    'get_tool_registry',
    'get_tool',
    'get_web_tools',
    'get_shell_tools',
    'get_file_tools',
    'get_gui_tools',
    # Advanced tools
    'RobustWebSearchTool',
    'AdvancedWebTool',
    'AdvancedPowerShellTool',
    'AdvancedShellTool',
    'AdvancedFileReadTool',
    'AdvancedGUIAutomationTool',
    'ProGUIAutomationTool',
    # Legacy functions
    'type_text',
    'click',
    'hotkey',
    'activate_window',
    'get_focused_window',
]
