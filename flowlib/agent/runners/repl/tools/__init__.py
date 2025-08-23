"""REPL Tools Package - opencode style with flowlib conventions.

This package contains tool implementations adapted from opencode's approach,
maintaining CLAUDE.md principles with fail-fast validation and comprehensive
safety features including:

- Path validation and CWD restrictions
- Binary file detection  
- Permission system for file operations
- Enhanced error messages with suggestions
- Comprehensive parameter validation
"""

from .base import (
    REPLTool,
    ToolResult, 
    ToolResultStatus,
    ToolExecutionContext,
    ToolRegistry,
    tool_registry
)

# Import individual tools for registration
from .read import ReadTool
from .write import WriteTool  
from .edit import EditTool
from .bash import BashTool
from .grep import GrepTool
from .glob import GlobTool
from .ls import ListTool

__all__ = [
    "REPLTool",
    "ToolResult", 
    "ToolResultStatus", 
    "ToolExecutionContext",
    "ToolRegistry",
    "tool_registry",
    "ReadTool",
    "WriteTool",
    "EditTool", 
    "BashTool",
    "GrepTool",
    "GlobTool",
    "ListTool"
]

def create_default_registry() -> ToolRegistry:
    """Create a tool registry with all default tools registered.
    
    This function provides a clean way to get a fully configured tool registry
    without triggering circular imports during module loading.
    
    Returns:
        ToolRegistry: Registry with all default tools loaded
    """
    return tool_registry  # Tools are auto-registered in ToolRegistry.__init__()