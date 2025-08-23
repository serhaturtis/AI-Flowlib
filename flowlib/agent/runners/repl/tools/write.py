"""Write tool - opencode style implementation with flowlib conventions."""

import os
from typing import Dict, Any
from .base import REPLTool, ToolResult, ToolExecutionContext
from ..permissions import request_permission


class WriteTool(REPLTool):
    """Write content to files with permission checks."""
    
    async def init(self) -> Dict[str, Any]:
        """Initialize write tool schema."""
        return {
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"],
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute write operation."""
        file_path = parameters["file_path"]
        content = parameters["content"]
        
        try:
            # Validate and resolve file path (allow creation)
            file_path = self._validate_file_path(file_path, context, allow_create=True)
            
            # Check if file exists
            file_exists = os.path.exists(file_path)
            
            # Request permission for write operation
            permission_title = f"Write to {'existing' if file_exists else 'new'} file: {file_path}"
            permission_details = f"Content length: {len(content)} characters"
            
            granted = await request_permission(
                operation_type="write",
                title=permission_title,
                details=permission_details,
                session_id=context.session_id,
                message_id=context.message_id,
                call_id=context.call_id,
                metadata={
                    "file_path": file_path,
                    "file_exists": file_exists,
                    "content_length": len(content)
                }
            )
            
            if not granted:
                return ToolResult.error("Write operation denied by user")
            
            # Create parent directories if needed
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file info
            file_stat = os.stat(file_path)
            
            metadata = {
                "file_path": file_path,
                "bytes_written": file_stat.st_size,
                "lines_written": content.count('\\n') + (1 if content and not content.endswith('\\n') else 0),
                "file_existed": file_exists,
                "parent_dirs_created": parent_dir and not os.path.exists(parent_dir)
            }
            
            action = "overwrote" if file_exists else "created"
            result_message = f"Successfully {action} file {file_path} ({file_stat.st_size} bytes)"
            
            return ToolResult.success(result_message, metadata)
            
        except PermissionError:
            return ToolResult.error(f"Permission denied writing to file: {file_path}")
        except OSError as e:
            return ToolResult.error(f"OS error writing file: {str(e)}")
        except Exception as e:
            return ToolResult.error(f"Error writing file: {str(e)}")