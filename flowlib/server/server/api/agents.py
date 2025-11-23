"""Agent runtime endpoints."""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool

from server.core.config import settings
from server.core.workspace import WorkspaceNotConfiguredError, get_workspace_path
from server.models.agents import (
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatusResponse,
    ReplInputRequest,
    ReplSessionCreateRequest,
    ReplSessionResponse,
)
from server.services.agent_service import (
    AgentRunNotFound,
    AgentService,
    ReplSessionNotFound,
)

# Constants
DEFAULT_HISTORY_LIMIT = 100
MAX_HISTORY_LIMIT = 1000

router = APIRouter()


def _get_agent_service() -> AgentService:
    """Get agent service with current workspace path."""
    try:
        workspace_path = get_workspace_path()
        return AgentService(workspace_path)
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/", response_model=list[str])
async def list_agents(project_id: str) -> list[str]:
    """List registered agent configuration names."""
    service = _get_agent_service()
    try:
        return await run_in_threadpool(service.list_agents, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/run", response_model=AgentRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_agent(payload: AgentRunRequest) -> AgentRunResponse:
    """Start agent execution."""
    service = _get_agent_service()
    try:
        return await service.start_run(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=AgentRunStatusResponse)
async def get_run_status(run_id: str) -> AgentRunStatusResponse:
    """Get status for a run."""
    service = _get_agent_service()
    try:
        return await service.get_run_status(run_id)
    except AgentRunNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/runs", response_model=list[AgentRunStatusResponse])
async def list_runs() -> list[AgentRunStatusResponse]:
    """List all runs (in-memory)."""
    service = _get_agent_service()
    return await service.list_runs()


@router.get("/runs/history", response_model=list[AgentRunStatusResponse])
async def list_run_history(
    limit: int = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT)
) -> list[AgentRunStatusResponse]:
    """List persisted run history (most recent first)."""
    service = _get_agent_service()
    return service.list_run_history(limit=limit)


@router.post("/runs/{run_id}/stop", response_model=AgentRunStatusResponse)
async def stop_run(run_id: str) -> AgentRunStatusResponse:
    """Stop an active run."""
    service = _get_agent_service()
    try:
        return await service.stop_run(run_id)
    except AgentRunNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/repl/sessions", response_model=ReplSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_repl_session(payload: ReplSessionCreateRequest) -> ReplSessionResponse:
    service = _get_agent_service()
    try:
        session = await service.create_repl_session(payload.project_id, payload.agent_config_name)
        return ReplSessionResponse(
            session_id=session.session_id,
            project_id=session.project_id,
            agent_config_name=session.agent_name,
            created_at=session.created_at,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete("/repl/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_repl_session(session_id: str) -> None:
    service = _get_agent_service()
    try:
        await service.close_repl_session(session_id)
    except ReplSessionNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/repl/sessions/{session_id}/input", status_code=status.HTTP_202_ACCEPTED)
async def send_repl_input(session_id: str, payload: ReplInputRequest) -> None:
    service = _get_agent_service()
    try:
        await service.send_repl_input(session_id, payload.message)
    except ReplSessionNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.websocket("/repl/sessions/{session_id}/events")
async def repl_events(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    service = _get_agent_service()
    try:
        queue = service.get_repl_event_queue(session_id)
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except ReplSessionNotFound:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except WebSocketDisconnect:
        pass

