"""Edit tool - opencode style implementation with flowlib conventions."""

import os
from typing import Dict, Any
from .base import REPLTool, ToolResult, ToolExecutionContext
from ..permissions import request_permission


class EditTool(REPLTool):
    """Edit files with exact string replacement and diff generation."""
    
    async def init(self) -> Dict[str, Any]:
        """Initialize edit tool schema."""
        return {
            "parameters": {
                "type": "object", 
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "new_string": {
                        "type": "string", 
                        "description": "The text to replace it with (must be different from old_string)"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences of old_string (default false)",
                        "default": False
                    }
                },
                "required": ["file_path", "old_string", "new_string"],
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute edit operation."""
        file_path = parameters["file_path"]
        old_string = parameters["old_string"]
        new_string = parameters["new_string"]
        replace_all = parameters.get("replace_all", False)
        
        try:
            # Validation
            if old_string == new_string:
                return ToolResult.error("old_string and new_string must be different")
            
            # Validate and resolve file path
            file_path = self._validate_file_path(file_path, context, allow_create=False)
            
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content_old = f.read()
            
            # Handle empty old_string (file creation case)
            if old_string == "":
                if content_old:
                    return ToolResult.error("Cannot use empty old_string on non-empty file")
                content_new = new_string
            else:
                # Check if old_string exists in file
                if old_string not in content_old:
                    return ToolResult.error(f"String not found in file: {old_string[:100]}...")
                
                # Count occurrences
                occurrence_count = content_old.count(old_string)
                
                if not replace_all and occurrence_count > 1:
                    return ToolResult.error(
                        f"String occurs {occurrence_count} times in file. "
                        "Use replace_all=true to replace all occurrences, or provide a more specific string."
                    )
                
                # Perform replacement
                if replace_all:
                    content_new = content_old.replace(old_string, new_string)
                else:
                    content_new = content_old.replace(old_string, new_string, 1)
            
            # Generate diff (simplified version of opencode's approach)
            diff = self._generate_diff(file_path, content_old, content_new)
            
            # Request permission
            permission_title = f"Edit file: {file_path}"
            permission_details = f"Replace {'all occurrences' if replace_all else '1 occurrence'} of text"
            
            granted = await request_permission(
                operation_type="edit",
                title=permission_title,
                details=permission_details,
                session_id=context.session_id,
                message_id=context.message_id,
                call_id=context.call_id,
                metadata={
                    "file_path": file_path,
                    "old_string": old_string[:100],
                    "new_string": new_string[:100], 
                    "replace_all": replace_all,
                    "diff": diff[:500]  # Truncated diff for metadata
                }
            )
            
            if not granted:
                return ToolResult.error("Edit operation denied by user")
            
            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_new)
            
            # Calculate changes
            old_lines = len(content_old.split('\\n'))
            new_lines = len(content_new.split('\\n'))
            replacements_made = occurrence_count if replace_all else 1
            
            metadata = {
                "file_path": file_path,
                "replacements_made": replacements_made,
                "old_lines": old_lines,
                "new_lines": new_lines,
                "lines_changed": abs(new_lines - old_lines),
                "diff": diff,
                "replace_all": replace_all
            }
            
            result_message = f"Successfully edited {file_path}\\n"
            result_message += f"Made {replacements_made} replacement(s)\\n"
            result_message += f"Lines: {old_lines} → {new_lines}\\n\\n"
            result_message += f"Diff:\\n{diff}"
            
            return ToolResult.success(result_message, metadata)
            
        except FileNotFoundError:
            return ToolResult.error(f"File not found: {file_path}")
        except PermissionError:
            return ToolResult.error(f"Permission denied editing file: {file_path}")
        except UnicodeDecodeError:
            return ToolResult.error(f"Cannot decode file {file_path} - it may be binary")
        except Exception as e:
            return ToolResult.error(f"Error editing file: {str(e)}")
    
    def _generate_diff(self, filename: str, old_content: str, new_content: str) -> str:
        """Generate a simple diff between old and new content."""
        if old_content == new_content:
            return "No changes"
        
        old_lines = old_content.split('\\n')
        new_lines = new_content.split('\\n')
        
        # Simple diff - just show some context around changes
        diff_lines = []
        diff_lines.append(f"--- {filename}")
        diff_lines.append(f"+++ {filename}")
        
        # Find first and last differing lines
        first_diff = 0
        last_diff = max(len(old_lines), len(new_lines)) - 1
        
        for i in range(min(len(old_lines), len(new_lines))):
            if old_lines[i] != new_lines[i]:
                first_diff = i
                break
        
        for i in range(min(len(old_lines), len(new_lines)) - 1, -1, -1):
            if i < len(old_lines) and i < len(new_lines) and old_lines[i] != new_lines[i]:
                last_diff = i
                break
        
        # Add context around changes
        context = 3
        start = max(0, first_diff - context)
        end = min(max(len(old_lines), len(new_lines)), last_diff + context + 1)
        
        diff_lines.append(f"@@ -{start+1},{end-start} +{start+1},{end-start} @@")
        
        for i in range(start, end):
            if i < len(old_lines) and i < len(new_lines):
                if old_lines[i] == new_lines[i]:
                    diff_lines.append(f" {old_lines[i]}")
                else:
                    diff_lines.append(f"-{old_lines[i]}")
                    diff_lines.append(f"+{new_lines[i]}")
            elif i < len(old_lines):
                diff_lines.append(f"-{old_lines[i]}")
            elif i < len(new_lines):
                diff_lines.append(f"+{new_lines[i]}")
        
        return "\\n".join(diff_lines)