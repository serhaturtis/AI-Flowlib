"""Diff endpoints."""

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from server.core.workspace import WorkspaceNotConfiguredError, get_workspace_path
from server.models.diff import (
    ConfigApplyRequest,
    ConfigApplyResponse,
    ConfigDiffRequest,
    ConfigDiffResponse,
)
from server.services.config_service import ConfigService
from server.services.diff_service import ConfigValidationError, DiffService

router = APIRouter()


def _get_diff_services() -> tuple[ConfigService, DiffService]:
    """Get diff services with current workspace path."""
    try:
        workspace_path = get_workspace_path()
        config_service = ConfigService(workspace_path)
        diff_service = DiffService(workspace_path, config_service)
        return config_service, diff_service
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/configs", response_model=ConfigDiffResponse)
async def diff_config(payload: ConfigDiffRequest) -> ConfigDiffResponse:
    """Generate a unified diff for a configuration file."""
    _, diff_service = _get_diff_services()
    try:
        return await run_in_threadpool(diff_service.diff_config, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/configs/apply", response_model=ConfigApplyResponse)
async def apply_config(payload: ConfigApplyRequest) -> ConfigApplyResponse:
    """Apply edits to a configuration file with validation."""
    _, diff_service = _get_diff_services()
    try:
        return await run_in_threadpool(diff_service.apply_config, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": str(exc),
                "issues": [issue.model_dump() for issue in exc.issues],
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

