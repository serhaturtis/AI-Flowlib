"""Write tool implementation."""

import shutil
from pathlib import Path
from typing import cast
from ...models import ToolResult, ToolExecutionContext, ToolStatus
from ...decorators import tool
from .models import WriteParameters, WriteResult
from flowlib.agent.components.task.models import TodoItem


@tool(name="write", tool_category="software", description="Write content to files with backup and directory creation options")
class WriteTool:
    """Tool for writing content to files."""
    
    def get_name(self) -> str:
        """Get tool name."""
        return "write"
    
    def get_description(self) -> str:
        """Get tool description."""
        return "Write content to files with backup and directory creation options"
    
    async def execute(
        self, 
        todo: TodoItem,
        context: ToolExecutionContext  # Execution context
    ) -> ToolResult:
        """Execute file write operation."""
        
        # Generate parameters from todo content
        try:
            params = await self._generate_parameters(todo, context)
        except Exception as e:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"Failed to generate parameters: {str(e)}"
            )
        
        # Resolve file path
        file_path = Path(params.file_path)
        
        if context and context.working_directory and not file_path.is_absolute():
            file_path = Path(context.working_directory) / file_path
        
        try:
            # Create parent directories if needed
            created_directories = False
            if params.create_directories and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                created_directories = True
            
            # Create backup if file exists and backup is requested
            backup_path = None
            if params.backup and file_path.exists():
                backup_path = str(file_path) + ".bak"
                shutil.copy2(file_path, backup_path)
            
            # Write content to file
            with open(file_path, 'w', encoding=params.encoding) as f:
                f.write(params.content)
            
            # Calculate stats
            file_size = file_path.stat().st_size
            lines_written = params.content.count('\n') + 1 if params.content else 0
            
            return WriteResult(
                status=ToolStatus.SUCCESS,
                message=f"Successfully wrote {lines_written} lines to {params.file_path}",
                file_path=params.file_path,
                bytes_written=file_size,
                lines_written=lines_written,
                backup_path=backup_path,
                created_directories=created_directories
            )
            
        except PermissionError:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied writing to file: {params.file_path}",
                file_path=params.file_path
            )
        
        except OSError as e:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"OS error writing file: {str(e)}",
                file_path=params.file_path
            )
        
        except UnicodeEncodeError as e:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"Encoding error writing file: {str(e)}. Try a different encoding.",
                file_path=params.file_path
            )
    
    async def _generate_parameters(self, todo: TodoItem, context: ToolExecutionContext) -> WriteParameters:
        """Generate WriteParameters from todo content using flow."""
        from flowlib.flows.registry.registry import flow_registry
        from .flow import WriteParameterGenerationInput, WriteParameterGenerationFlow

        # Get the parameter generation flow class
        flow_obj = flow_registry.get("write-parameter-generation")
        if flow_obj is None:
            raise RuntimeError("Write parameter generation flow not found in registry")
        flow_instance = cast(WriteParameterGenerationFlow, flow_obj)
        
        
        # Extract task content from todo
        task_content = todo.content if hasattr(todo, 'content') else str(todo)
        
        # Create flow input
        flow_input = WriteParameterGenerationInput(
            task_content=task_content,
            working_directory=context.working_directory or "."
        )
        
        # Execute flow to generate parameters
        result = await flow_instance.run_pipeline(flow_input)

        return cast(WriteParameters, result.parameters)