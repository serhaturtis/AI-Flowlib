"""Write tool implementation."""

from pathlib import Path

from ....core.todo import TodoItem
from ...decorators import tool
from ...models import ToolExecutionContext, ToolResult, ToolStatus
from .models import WriteParameters, WriteResult


@tool(
    parameter_type=WriteParameters,
    name="write",
    tool_category="generic",
    description="Write content to files. This tool will overwrite existing files. If file exists, you MUST use Read tool first. NEVER create documentation files (*.md, README) unless explicitly requested. ALWAYS prefer editing existing files.",
)
class WriteTool:
    """Tool for writing content to files.

    Enforces best practices:
    - Requires Read tool to be used first for existing files
    - Prevents creating documentation files unless explicitly requested
    - Prefers editing over creating new files
    """

    def get_name(self) -> str:
        """Get tool name."""
        return "write"

    def get_description(self) -> str:
        """Get tool description."""
        return "Write content to files (use Read first for existing files, prefer Edit over Write)"

    async def execute(
        self,
        todo: TodoItem,
        params: WriteParameters,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute file write operation.

        Enforces best practices:
        - Fails if file exists (must use Read first)
        - Fails if trying to create documentation files
        - Requires absolute paths

        Args:
            todo: TodoItem describing the task
            params: Validated WriteParameters instance
            context: Tool execution context

        Returns:
            WriteResult with operation status or error
        """
        # Resolve file path
        file_path = Path(params.file_path)

        # Require absolute paths
        if not file_path.is_absolute():
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"file_path must be absolute, not relative: {params.file_path}",
                file_path=params.file_path,
            )

        # Check if file exists - fail fast to enforce Read-first
        if file_path.exists():
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"This tool will overwrite the existing file if there is one at the provided path. If this is an existing file, you MUST use the Read tool first to read the file's contents. File: {params.file_path}",
                file_path=params.file_path,
            )

        # Detect documentation files and fail (unless explicitly creating)
        file_name = file_path.name.lower()
        is_documentation_file = (
            file_name.endswith('.md') or
            file_name.startswith('readme') or
            file_name == 'readme'
        )

        if is_documentation_file:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User. File: {params.file_path}",
                file_path=params.file_path,
            )

        try:
            # Create parent directories if they don't exist
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(params.content)

            # Calculate stats
            file_size = file_path.stat().st_size
            lines_written = params.content.count("\n") + 1 if params.content else 0

            return WriteResult(
                status=ToolStatus.SUCCESS,
                message=f"Successfully wrote {lines_written} lines to {params.file_path}",
                file_path=params.file_path,
                bytes_written=file_size,
                lines_written=lines_written,
            )

        except PermissionError:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"Permission denied writing to file: {params.file_path}",
                file_path=params.file_path,
            )

        except OSError as e:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"OS error writing file: {str(e)}",
                file_path=params.file_path,
            )

        except UnicodeEncodeError as e:
            return WriteResult(
                status=ToolStatus.ERROR,
                message=f"File content cannot be encoded as UTF-8: {str(e)}",
                file_path=params.file_path,
            )
