"""Edit tool implementation."""

import os
import shutil
from pathlib import Path
from typing import Any
from ...models import ToolResult, ToolExecutionContext, ToolStatus
from ...decorators import tool
from .models import EditParameters, EditResult


@tool(name="edit", category="filesystem", description="Edit files by finding and replacing text with backup support")
class EditTool:
    """Tool for editing files with find/replace operations."""
    
    def get_name(self) -> str:
        """Get tool name."""
        return "edit"
    
    def get_description(self) -> str:
        """Get tool description."""
        return "Edit files by finding and replacing text with backup support"
    
    async def execute(
        self, 
        todo: Any,  # TodoItem with task description
        context: ToolExecutionContext  # Execution context
    ) -> ToolResult:
        """Execute file edit operations."""
        
        # Generate parameters from todo content
        try:
            params = await self._generate_parameters(todo, context)
        except Exception as e:
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"Failed to generate parameters: {str(e)}"
            )
        
        # Resolve file path
        file_path = Path(params.file_path)
        
        if context and context.working_directory and not file_path.is_absolute():
            file_path = Path(context.working_directory) / file_path
        
        # Check if file exists
        if not file_path.exists():
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"File not found: {params.file_path}",
                file_path=params.file_path
            )
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"Path is not a file: {params.file_path}",
                file_path=params.file_path
            )
        
        # Check read/write permission
        if not os.access(file_path, os.R_OK | os.W_OK):
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied for file: {params.file_path}",
                file_path=params.file_path
            )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding=params.encoding) as f:
                content = f.read()
            
            original_content = content
            
            # Create backup if requested
            backup_path = None
            if params.backup:
                backup_path = str(file_path) + ".bak"
                shutil.copy2(file_path, backup_path)
            
            # Apply edit operations
            operations_applied = 0
            total_replacements = 0
            
            for operation in params.operations:
                if operation.old_text not in content:
                    continue  # Skip if text not found
                
                if operation.replace_all:
                    # Replace all occurrences
                    count = content.count(operation.old_text)
                    content = content.replace(operation.old_text, operation.new_text)
                    total_replacements += count
                else:
                    # Replace only first occurrence
                    if operation.old_text in content:
                        content = content.replace(operation.old_text, operation.new_text, 1)
                        total_replacements += 1
                
                operations_applied += 1
            
            # Count modified lines
            original_lines = original_content.splitlines()
            new_lines = content.splitlines()
            lines_modified = 0
            
            max_lines = max(len(original_lines), len(new_lines))
            for i in range(max_lines):
                orig_line = original_lines[i] if i < len(original_lines) else ""
                new_line = new_lines[i] if i < len(new_lines) else ""
                if orig_line != new_line:
                    lines_modified += 1
            
            # Write modified content back to file
            with open(file_path, 'w', encoding=params.encoding) as f:
                f.write(content)
            
            return EditResult(
                status=ToolStatus.SUCCESS,
                message=f"Successfully applied {operations_applied} edit operations to {params.file_path}",
                file_path=params.file_path,
                operations_applied=operations_applied,
                total_replacements=total_replacements,
                backup_path=backup_path,
                lines_modified=lines_modified
            )
            
        except UnicodeDecodeError as e:
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"Encoding error reading file: {str(e)}. Try a different encoding.",
                file_path=params.file_path
            )
        
        except UnicodeEncodeError as e:
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"Encoding error writing file: {str(e)}. Try a different encoding.",
                file_path=params.file_path
            )
        
        except PermissionError:
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied editing file: {params.file_path}",
                file_path=params.file_path
            )
        
        except OSError as e:
            return EditResult(
                status=ToolStatus.ERROR,
                message=f"OS error editing file: {str(e)}",
                file_path=params.file_path
            )
    
    async def _generate_parameters(self, todo: Any, context: ToolExecutionContext) -> EditParameters:
        """Generate EditParameters from todo content using flow."""
        from flowlib.flows.registry.registry import flow_registry
        from .flow import EditParameterGenerationInput
        
        # Get the parameter generation flow class
        flow_instance = flow_registry.get("edit-parameter-generation")
        
        
        # Extract task content from todo
        task_content = todo.content if hasattr(todo, 'content') else str(todo)
        
        # Create flow input
        flow_input = EditParameterGenerationInput(
            task_content=task_content,
            working_directory=context.working_directory or "."
        )
        
        # Execute flow to generate parameters
        result = await flow_instance.run_pipeline(flow_input)
        
        return result.parameters