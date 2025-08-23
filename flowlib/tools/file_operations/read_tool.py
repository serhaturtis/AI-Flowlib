"""File reading tool implementation.

This module provides a concrete tool implementation for reading file contents,
demonstrating the @tool decorator pattern and flowlib's tool calling system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from flowlib.core.decorators.decorators import tool
from flowlib.core.models import StrictBaseModel
from flowlib.providers.tools.models import ToolExecutionContext


@tool(
    name="read",
    description="Read and return the contents of a file from the filesystem"
)
class ReadTool:
    """Tool for reading file contents with path validation and error handling."""
    
    class Parameters(StrictBaseModel):
        """Strictly validated parameters for file reading."""
        file_path: str
        encoding: Optional[str] = "utf-8"
        max_lines: Optional[int] = None
        start_line: Optional[int] = None
    
    async def execute(self, params: Parameters, context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
        """Execute file reading with validation and safety checks.
        
        Args:
            params: Validated parameters for reading
            context: Execution context (working directory, metadata)
            
        Returns:
            Dict containing file contents and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file access denied
            UnicodeDecodeError: If encoding issues
        """
        # Resolve file path (handle relative paths with context)
        file_path = Path(params.file_path)
        
        if context and context.working_directory and not file_path.is_absolute():
            file_path = Path(context.working_directory) / file_path
        
        # Validate file exists and is readable
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for file: {file_path}")
        
        # Read file with optional line limits
        try:
            with open(file_path, 'r', encoding=params.encoding) as f:
                if params.max_lines or params.start_line:
                    # Read specific line range
                    lines = f.readlines()
                    
                    start = (params.start_line - 1) if params.start_line else 0
                    end = start + params.max_lines if params.max_lines else None
                    
                    selected_lines = lines[start:end]
                    content = ''.join(selected_lines)
                    
                    return {
                        "content": content,
                        "file_path": str(file_path.absolute()),
                        "encoding": params.encoding,
                        "total_lines": len(lines),
                        "lines_read": len(selected_lines),
                        "start_line": start + 1,
                        "file_size_bytes": file_path.stat().st_size
                    }
                else:
                    # Read entire file
                    content = f.read()
                    line_count = content.count('\n') + 1 if content else 0
                    
                    return {
                        "content": content,
                        "file_path": str(file_path.absolute()),
                        "encoding": params.encoding,
                        "total_lines": line_count,
                        "lines_read": line_count,
                        "file_size_bytes": file_path.stat().st_size
                    }
                    
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding, e.object, e.start, e.end,
                f"Failed to decode file {file_path} with encoding {params.encoding}: {e.reason}"
            )