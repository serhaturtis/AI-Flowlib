"""Project management endpoints."""

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from server.core.workspace import WorkspaceNotConfiguredError, get_workspace_path
from server.models.projects import (
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectMetadata,
    ProjectValidationIssue,
    ProjectValidationResponse,
)
from server.services.project_service import ProjectService

router = APIRouter()


def _get_project_service() -> ProjectService:
    """Get project service with current workspace path."""
    try:
        workspace_path = get_workspace_path()
        return ProjectService(workspace_path)
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/", response_model=ProjectListResponse)
async def list_projects() -> ProjectListResponse:
    """List all projects."""
    service = _get_project_service()
    projects = await run_in_threadpool(service.list_projects)
    return ProjectListResponse(projects=projects, total=len(projects))


@router.post("/", response_model=ProjectMetadata, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreateRequest) -> ProjectMetadata:
    """Create a new project."""
    service = _get_project_service()
    try:
        project = await run_in_threadpool(service.create_project, payload)
        return project
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/{project_id}", response_model=ProjectMetadata)
async def get_project(project_id: str) -> ProjectMetadata:
    """Get project details."""
    service = _get_project_service()
    try:
        return await run_in_threadpool(service.get_project, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/{project_id}")
async def update_project(project_id: str) -> None:
    """Update project (not yet implemented)."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not yet implemented")


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str) -> None:
    """Delete project permanently."""
    service = _get_project_service()
    try:
        await run_in_threadpool(service.delete_project, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/{project_id}/validate")
async def validate_project(project_id: str) -> ProjectValidationResponse:
    """Validate project structure."""
    service = _get_project_service()
    try:
        result = await run_in_threadpool(service.validate_project, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    issues = [
        ProjectValidationIssue(path=issue.path, message=issue.message) for issue in result.issues
    ]
    return ProjectValidationResponse(is_valid=result.is_valid, issues=issues)

