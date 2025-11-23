"""Workspace management endpoints."""

import os
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from pydantic import BaseModel, Field

from server.core.workspace import WorkspaceNotConfiguredError, get_workspace_path, set_workspace_path

router = APIRouter()


class WorkspacePathResponse(BaseModel):
    """Workspace path response."""

    path: str = Field(description="Current workspace path (absolute)")
    message: str | None = Field(
        default=None, description="Optional message about workspace configuration"
    )


class WorkspacePathRequest(BaseModel):
    """Workspace path request."""

    path: str = Field(min_length=1, description="Workspace path to set (absolute or relative)")


class DirectoryEntry(BaseModel):
    """Directory entry in a listing."""

    name: str = Field(description="Entry name")
    path: str = Field(description="Absolute path to entry")
    is_directory: bool = Field(description="True if entry is a directory")
    readable: bool = Field(description="True if entry is readable")


class DirectoryListingResponse(BaseModel):
    """Response for directory listing."""

    path: str = Field(description="Absolute path of listed directory")
    parent: str | None = Field(default=None, description="Parent directory path, or None if at root")
    entries: list[DirectoryEntry] = Field(description="Directory entries (directories first, then files)")


@router.get("/path", response_model=WorkspacePathResponse)
async def get_workspace_path_endpoint() -> WorkspacePathResponse:
    """Get the current workspace path.

    Returns:
        Current workspace path

    Raises:
        503: If workspace path is not configured
    """
    try:
        path = get_workspace_path()
        return WorkspacePathResponse(path=path)
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.put("/path", response_model=WorkspacePathResponse)
async def set_workspace_path_endpoint(request: WorkspacePathRequest) -> WorkspacePathResponse:
    """Set the workspace path.

    **Note**: Changing the workspace path requires restarting the server
    for the changes to take effect, as services are initialized at startup.

    Args:
        request: Workspace path request containing the new path

    Returns:
        Confirmation with the new workspace path

    Raises:
        400: If path is invalid
        500: If workspace path cannot be saved
    """
    try:
        set_workspace_path(request.path, save_to_env=True)
        workspace_path = get_workspace_path()
        return WorkspacePathResponse(
            path=workspace_path,
            message="Workspace path updated. Please restart the server for changes to take effect.",
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/browse", response_model=DirectoryListingResponse)
async def browse_directory(path: str | None = Query(default=None, description="Directory path to browse")) -> DirectoryListingResponse:
    """Browse directory contents on the server.

    Args:
        path: Directory path to browse (absolute or relative). If None, starts from home directory.

    Returns:
        DirectoryListingResponse with directory contents

    Raises:
        400: If path is invalid or not accessible
        404: If directory doesn't exist
    """
    def _list_directory(directory_path: str | None) -> DirectoryListingResponse:
        """List directory contents (synchronous helper)."""
        if directory_path is None or directory_path == "":
            # Start from home directory
            start_path = Path.home()
        else:
            start_path = Path(directory_path).expanduser().resolve()

        if not start_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {start_path}")

        if not start_path.is_dir():
            raise ValueError(f"Path is not a directory: {start_path}")

        if not start_path.is_absolute():
            raise ValueError(f"Path must be absolute: {start_path}")

        # List entries
        directories: list[DirectoryEntry] = []
        files: list[DirectoryEntry] = []

        try:
            for entry in sorted(start_path.iterdir()):
                try:
                    is_dir = entry.is_dir()
                    readable = os.access(entry, os.R_OK)
                    
                    entry_data = DirectoryEntry(
                        name=entry.name,
                        path=str(entry),
                        is_directory=is_dir,
                        readable=readable,
                    )

                    if is_dir:
                        directories.append(entry_data)
                    else:
                        files.append(entry_data)
                except (OSError, PermissionError):
                    # Skip entries we can't access
                    continue
        except PermissionError as exc:
            raise ValueError(f"Permission denied reading directory: {start_path}") from exc

        # Determine parent path
        parent_path: str | None = None
        try:
            if start_path.parent != start_path:  # Not at filesystem root
                parent_path = str(start_path.parent)
        except (OSError, PermissionError):
            pass

        # Combine: directories first, then files
        entries = directories + files

        return DirectoryListingResponse(
            path=str(start_path),
            parent=parent_path,
            entries=entries,
        )

    try:
        result = await run_in_threadpool(_list_directory, path)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ValueError, PermissionError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


class CreateDirectoryRequest(BaseModel):
    """Request to create a directory."""

    parent_path: str = Field(description="Parent directory path where to create the new directory")
    name: str = Field(min_length=1, description="Name of the new directory to create")


class CreateDirectoryResponse(BaseModel):
    """Response after creating a directory."""

    path: str = Field(description="Absolute path of the created directory")
    message: str = Field(description="Success message")


@router.post("/create-directory", response_model=CreateDirectoryResponse)
async def create_directory(request: CreateDirectoryRequest) -> CreateDirectoryResponse:
    """Create a new directory.

    Args:
        request: Request containing parent path and directory name

    Returns:
        CreateDirectoryResponse with the created directory path

    Raises:
        400: If path is invalid or directory already exists
        403: If permission denied
    """
    def _create_directory(parent_path: str, name: str) -> CreateDirectoryResponse:
        """Create directory (synchronous helper)."""
        parent = Path(parent_path).expanduser().resolve()

        if not parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")

        if not parent.is_dir():
            raise ValueError(f"Parent path is not a directory: {parent}")

        if not parent.is_absolute():
            raise ValueError(f"Parent path must be absolute: {parent}")

        # Validate directory name (no path separators, no special characters that would cause issues)
        if "/" in name or "\\" in name:
            raise ValueError("Directory name cannot contain path separators")

        # Check if parent is writable
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Permission denied: cannot create directory in {parent}")

        new_dir = parent / name

        if new_dir.exists():
            raise ValueError(f"Directory already exists: {new_dir}")

        try:
            new_dir.mkdir(parents=False, exist_ok=False)
            return CreateDirectoryResponse(
                path=str(new_dir),
                message=f"Directory '{name}' created successfully",
            )
        except OSError as exc:
            raise ValueError(f"Failed to create directory: {exc}") from exc

    try:
        result = await run_in_threadpool(_create_directory, request.parent_path, request.name)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
