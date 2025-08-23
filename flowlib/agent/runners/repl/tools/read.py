"""Read tool - opencode style implementation with flowlib conventions."""

import os
from typing import Dict, Any
from .base import REPLTool, ToolResult, ToolExecutionContext


class ReadTool(REPLTool):
    """Read file contents with safety checks and binary detection."""
    
    async def init(self) -> Dict[str, Any]:
        """Initialize read tool schema."""
        return {
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from (0-based)",
                        "minimum": 0
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "The number of lines to read (defaults to 2000)",
                        "minimum": 1,
                        "maximum": 10000
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute read operation."""
        file_path = parameters["file_path"]
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", 2000)
        
        try:
            # Validate and resolve file path
            file_path = self._validate_file_path(file_path, context, allow_create=False)
            
            # Check if file is an image
            image_type = self._is_image_file(file_path)
            if image_type:
                return ToolResult.error(f"This is an image file of type: {image_type}\\nUse a different tool to process images")
            
            # Check if file is binary
            if self._is_binary_file(file_path):
                return ToolResult.error(f"Cannot read binary file: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\\n')
            total_lines = len(lines)
            
            # Apply offset and limit
            if offset >= total_lines:
                return ToolResult.error(f"Offset {offset} is beyond file length ({total_lines} lines)")
            
            end_line = min(offset + limit, total_lines)
            selected_lines = lines[offset:end_line]
            
            # Truncate very long lines (opencode approach)
            MAX_LINE_LENGTH = 2000
            truncated_lines = []
            for line in selected_lines:
                if len(line) > MAX_LINE_LENGTH:
                    truncated_lines.append(line[:MAX_LINE_LENGTH] + "...")
                else:
                    truncated_lines.append(line)
            
            # Format output with line numbers (cat -n style)
            output_lines = []
            for i, line in enumerate(truncated_lines, start=offset + 1):
                output_lines.append(f"{i:6}\\t{line}")
            
            result_content = "\\n".join(output_lines)
            
            # Add metadata
            metadata = {
                "file_path": file_path,
                "total_lines": total_lines,
                "lines_read": len(selected_lines),
                "offset": offset,
                "truncated": any("..." in line for line in truncated_lines)
            }
            
            if end_line < total_lines:
                metadata["more_available"] = True
                metadata["next_offset"] = end_line
            
            return ToolResult.success(result_content, metadata)
            
        except FileNotFoundError as e:
            return ToolResult.error(f"File not found: {file_path}")
        except PermissionError:
            return ToolResult.error(f"Permission denied reading file: {file_path}")
        except UnicodeDecodeError:
            return ToolResult.error(f"Cannot decode file {file_path} - it may be binary or use unsupported encoding")
        except Exception as e:
            return ToolResult.error(f"Error reading file: {str(e)}")