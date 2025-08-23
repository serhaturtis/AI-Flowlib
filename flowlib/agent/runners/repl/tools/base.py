"""Base classes for REPL tools - opencode style with flowlib conventions.

This module adapts opencode's tool system to flowlib's architecture,
maintaining CLAUDE.md principles with fail-fast validation.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Protocol, runtime_checkable
from pydantic import BaseModel, Field, ConfigDict


class ToolExecutionContext(BaseModel):
    """Context for tool execution."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)
    
    session_id: Optional[str] = Field(default=None, description="Session ID")
    message_id: Optional[str] = Field(default=None, description="Message ID") 
    call_id: Optional[str] = Field(default=None, description="Tool call ID")
    working_directory: str = Field(default_factory=lambda: os.getcwd(), description="Current working directory")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")


class ToolResultStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class ToolResult(BaseModel):
    """Tool execution result."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)
    
    status: ToolResultStatus = Field(..., description="Execution status")
    content: Any = Field(..., description="Result content")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def success(cls, content: Any, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        """Create a successful result."""
        return cls(
            status=ToolResultStatus.SUCCESS,
            content=content,
            metadata=metadata or {}
        )
    
    @classmethod
    def error(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        """Create an error result."""
        return cls(
            status=ToolResultStatus.ERROR,
            content=None,
            error=error,
            metadata=metadata or {}
        )
    
    @classmethod
    def warning(cls, content: Any, error: str, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        """Create a warning result.""" 
        return cls(
            status=ToolResultStatus.WARNING,
            content=content,
            error=error,
            metadata=metadata or {}
        )


@runtime_checkable
class ToolInterface(Protocol):
    """Protocol defining tool interface."""
    
    id: str
    description: str
    
    async def init(self) -> Dict[str, Any]:
        """Initialize tool and return schema."""
        ...
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute tool with parameters and context."""
        ...


class REPLTool(ABC):
    """Base class for REPL tools - opencode style."""
    
    def __init__(self, tool_id: str):
        self.id = tool_id
        if not self.__doc__:
            raise ValueError(f"Tool {tool_id} must have a docstring")
        self.description = self.__doc__.strip()
    
    @abstractmethod 
    async def init(self) -> Dict[str, Any]:
        """Initialize tool and return parameter schema."""
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute tool with validated parameters."""
        pass
    
    def _validate_file_path(self, file_path: str, context: ToolExecutionContext, allow_create: bool = False) -> str:
        """Validate file path for safety - opencode style validation.
        
        Args:
            file_path: The file path to validate
            context: Tool execution context
            allow_create: Whether to allow non-existent files
            
        Returns:
            str: Absolute path if valid
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.working_directory, file_path)
        
        file_path = os.path.abspath(file_path)
        
        # Check if path is within working directory (security constraint)
        working_dir = os.path.abspath(context.working_directory)
        if not file_path.startswith(working_dir):
            raise ValueError(f"File {file_path} is not in the current working directory {working_dir}")
        
        # Check if file exists (unless creation is allowed)
        if not allow_create and not os.path.exists(file_path):
            # Try to find similar files for helpful error message
            parent_dir = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            
            if os.path.exists(parent_dir):
                similar_files = []
                for entry in os.listdir(parent_dir):
                    if filename.lower() in entry.lower() or entry.lower() in filename.lower():
                        similar_files.append(os.path.join(parent_dir, entry))
                
                if similar_files:
                    suggestions = "\n".join(similar_files[:3])
                    raise ValueError(f"File not found: {file_path}\n\nDid you mean one of these?\n{suggestions}")
            
            raise ValueError(f"File not found: {file_path}")
        
        return file_path
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if file is binary using opencode's approach."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(512)  # Read first 512 bytes
                if b'\0' in chunk:  # Null bytes indicate binary
                    return True
                # Check for high ratio of non-text bytes
                text_chars = set(range(32, 127)) | {9, 10, 13}  # Printable + tab, newline, CR
                non_text = sum(1 for byte in chunk if byte not in text_chars)
                return non_text / len(chunk) > 0.3 if chunk else False
        except Exception:
            return True  # Assume binary if we can't read it
    
    def _is_image_file(self, file_path: str) -> Optional[str]:
        """Check if file is an image and return type."""
        ext = os.path.splitext(file_path)[1].lower()
        image_extensions = {
            '.png': 'PNG',
            '.jpg': 'JPEG', 
            '.jpeg': 'JPEG',
            '.gif': 'GIF',
            '.bmp': 'BMP',
            '.tiff': 'TIFF',
            '.webp': 'WebP',
            '.svg': 'SVG'
        }
        return image_extensions.get(ext)


class ToolRegistry:
    """Registry for REPL tools - simplified opencode pattern."""
    
    def __init__(self):
        self.tools: Dict[str, REPLTool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all tools."""
        # Import and register tools
        from .read import ReadTool
        from .write import WriteTool  
        from .edit import EditTool
        from .bash import BashTool
        from .grep import GrepTool
        from .glob import GlobTool
        from .ls import ListTool
        
        tools = [
            ReadTool("read"),
            WriteTool("write"), 
            EditTool("edit"),
            BashTool("bash"),
            GrepTool("grep"),
            GlobTool("glob"),
            ListTool("ls")
        ]
        
        for tool in tools:
            self.register(tool)
    
    def register(self, tool: REPLTool):
        """Register a tool."""
        self.tools[tool.id] = tool
    
    def get(self, tool_id: str) -> Optional[REPLTool]:
        """Get tool by ID."""
        return self.tools.get(tool_id)
    
    def list_tools(self) -> List[str]:
        """List all tool IDs."""
        return list(self.tools.keys())
    
    async def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        schemas = []
        for tool in self.tools.values():
            schema = await tool.init()
            schemas.append({
                "id": tool.id,
                "description": tool.description,
                **schema
            })
        return schemas
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute a tool."""
        tool = self.get(tool_id)
        if not tool:
            return ToolResult.error(f"Tool '{tool_id}' not found")
        
        try:
            start_time = time.time()
            result = await tool.execute(parameters, context)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Add execution time to metadata
            if not result.metadata:
                result.metadata = {}
            result.metadata["execution_time_ms"] = execution_time
            
            return result
        except Exception as e:
            return ToolResult.error(f"Tool execution failed: {str(e)}")


# Global tool registry instance
tool_registry = ToolRegistry()