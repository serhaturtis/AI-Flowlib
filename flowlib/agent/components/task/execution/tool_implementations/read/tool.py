"""Read tool implementation."""

import os
from pathlib import Path

from ....core.todo import TodoItem
from ...decorators import tool
from ...models import ToolExecutionContext, ToolResult, ToolStatus
from .models import ReadParameters, ReadResult


@tool(
    parameter_type=ReadParameters,
    name="read",
    tool_category="generic",
    description="Read contents of a file with optional line range selection",
)
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
        todo: TodoItem,
        params: ReadParameters,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute file read operation.

        Reads file and returns content in cat -n format with line numbers.
        By default, reads up to 2000 lines from beginning.
        Lines longer than 2000 characters are truncated.

        Args:
            todo: TodoItem describing the task
            params: Validated ReadParameters instance
            context: Tool execution context

        Returns:
            ReadResult with formatted file contents or error
        """
        # Resolve file path
        file_path = Path(params.file_path)

        # Require absolute paths for clarity and safety
        if not file_path.is_absolute():
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"file_path must be absolute, not relative: {params.file_path}",
                file_path=params.file_path,
            )

        # Check if file exists
        if not file_path.exists():
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"File not found: {params.file_path}",
                file_path=params.file_path,
            )

        # Check if it's a file (not directory)
        if not file_path.is_file():
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Path is not a file: {params.file_path}",
                file_path=params.file_path,
            )

        # Check read permission
        if not os.access(file_path, os.R_OK):
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied reading file: {params.file_path}",
                file_path=params.file_path,
            )

        try:
            # Get file size
            file_size = file_path.stat().st_size

            # Read all lines from file
            with open(file_path, encoding="utf-8") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Check if file is empty
            is_empty = file_size > 0 and total_lines == 0

            # Apply offset and limit (default to 2000 lines from beginning)
            offset = params.offset if params.offset is not None else 0
            limit = params.limit if params.limit is not None else 2000

            # Select line range
            end = offset + limit
            selected_lines = all_lines[offset:end]
            lines_read = len(selected_lines)

            # Format as cat -n (line numbers starting at 1)
            # Line numbers are based on actual file line numbers, not selection
            formatted_lines = []
            for i, line in enumerate(selected_lines):
                line_num = offset + i + 1  # 1-based line numbering
                line_content = line.rstrip('\n\r')  # Remove trailing newlines

                # Truncate lines longer than 2000 characters
                if len(line_content) > 2000:
                    line_content = line_content[:2000]

                # Format: "  line_num→content"
                formatted_lines.append(f"{line_num:6d}→{line_content}")

            content = "\n".join(formatted_lines)

            # Build success message
            message = f"Successfully read {lines_read} lines from {params.file_path}"
            if is_empty:
                message += " (WARNING: File exists but has empty contents)"

            return ReadResult(
                status=ToolStatus.SUCCESS,
                message=message,
                content=content,
                file_path=params.file_path,
                lines_read=lines_read,
                total_lines=total_lines,
                file_size=file_size,
                is_empty=is_empty,
            )

        except UnicodeDecodeError as e:
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"File is not valid UTF-8: {str(e)}. Cannot read file.",
                file_path=params.file_path,
                file_size=file_path.stat().st_size if file_path.exists() else None,
            )

        except PermissionError:
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied reading file: {params.file_path}",
                file_path=params.file_path,
            )

        except OSError as e:
            return ReadResult(
                status=ToolStatus.ERROR,
                message=f"OS error reading file: {str(e)}",
                file_path=params.file_path,
            )
