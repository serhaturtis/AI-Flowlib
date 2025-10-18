"""Read tool implementation."""

import os
from pathlib import Path
from typing import cast

from ...decorators import tool
from ...models import TodoItem, ToolExecutionContext, ToolResult, ToolStatus
from .models import ReadParameters, ReadResult


@tool(name="read", tool_category="software", description="Read contents of a file with optional line range selection")
class ReadTool:
    """Tool for reading file contents with optional line range selection."""

    def get_name(self) -> str:
        """Get tool name."""
        return "read"

    def get_description(self) -> str:
        """Get tool description."""
        return "Read contents of a file with optional line range selection"

    async def execute(
        self,
        todo: TodoItem,  # TodoItem with task description
        context: ToolExecutionContext  # Execution context
    ) -> ToolResult:
        """Execute file read operation."""

        # Generate parameters from todo content
        try:
            params = await self._generate_parameters(todo, context)
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"Failed to generate parameters: {str(e)}"
            )

        # Resolve file path
        file_path = Path(params.file_path)

        if context and context.working_directory and not file_path.is_absolute():
            file_path = Path(context.working_directory) / file_path

        # Check if file exists
        if not file_path.exists():
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"File not found: {params.file_path}",
                file_path=params.file_path
            )

        # Check if it's a file (not directory)
        if not file_path.is_file():
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Path is not a file: {params.file_path}",
                file_path=params.file_path
            )

        # Check read permission
        if not os.access(file_path, os.R_OK):
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied reading file: {params.file_path}",
                file_path=params.file_path
            )

        try:
            # Get file size
            file_size = file_path.stat().st_size

            # Read file content with optional line range
            with open(file_path, 'r', encoding=params.encoding) as f:
                if params.start_line == 1 and params.line_count == -1:
                    # Read entire file
                    content = f.read()
                    lines_read = content.count('\n') + 1 if content else 0
                    total_lines = lines_read
                else:
                    # Read specific line range
                    lines = f.readlines()
                    total_lines = len(lines)

                    start = (params.start_line - 1) if params.start_line else 0
                    end = start + params.line_count if params.line_count else None

                    selected_lines = lines[start:end]
                    content = ''.join(selected_lines)
                    lines_read = len(selected_lines)

            return ReadResult(
                status=ToolStatus.SUCCESS,
                message=f"Successfully read {lines_read} lines from {params.file_path}",
                content=content,
                file_path=params.file_path,
                lines_read=lines_read,
                total_lines=total_lines,
                file_size=file_size,
                encoding_used=params.encoding
            )

        except UnicodeDecodeError as e:
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Encoding error reading file: {str(e)}. Try a different encoding.",
                file_path=params.file_path,
                file_size=file_path.stat().st_size if file_path.exists() else None
            )

        except PermissionError:
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied reading file: {params.file_path}",
                file_path=params.file_path
            )

        except OSError as e:
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"OS error reading file: {str(e)}",
                file_path=params.file_path
            )

    async def _generate_parameters(self, todo: TodoItem, context: ToolExecutionContext) -> ReadParameters:
        """Generate ReadParameters from todo content using flow.
        
        Args:
            todo: TodoItem with task description
            context: Execution context
            
        Returns:
            ReadParameters for tool execution
        """
        from flowlib.flows.registry.registry import flow_registry

        from .flow import ReadParameterGenerationFlow, ReadParameterGenerationInput

        # Get the parameter generation flow class
        flow_obj = flow_registry.get("read-parameter-generation")
        if flow_obj is None:
            raise RuntimeError("Read parameter generation flow not found in registry")
        flow_instance = cast(ReadParameterGenerationFlow, flow_obj)


        # Extract task content from todo
        task_content = todo.content if hasattr(todo, 'content') else str(todo)

        # FIX: Create flow input with full conversation context
        flow_input = ReadParameterGenerationInput(
            task_content=task_content,
            working_directory=context.working_directory or ".",
            conversation_history=context.conversation_history,  # Full conversation context
            original_user_message=context.original_user_message  # Original user intent
        )

        # Execute flow to generate parameters
        result = await flow_instance.run_pipeline(flow_input)

        return cast(ReadParameters, result.parameters)
