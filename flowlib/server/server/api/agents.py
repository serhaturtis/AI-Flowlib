"""Agent runtime endpoints."""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool

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


@router.get("/runs/{run_id}/streaming")
async def get_run_streaming_status(run_id: str) -> dict:
    """Check if a run has an active event stream.

    This endpoint helps clients determine whether to open a WebSocket
    connection for live event streaming.

    Returns:
        JSON object with:
        - run_id: The run identifier
        - streaming: True if events are being streamed
        - subscriber_count: Number of active WebSocket subscribers
    """
    service = _get_agent_service()

    # Verify run exists
    try:
        await service.get_run_status(run_id)
    except AgentRunNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    event_manager = service.get_event_manager()
    is_streaming = event_manager.is_run_active(run_id)
    subscriber_count = event_manager.get_subscriber_count(run_id) if is_streaming else 0

    return {
        "run_id": run_id,
        "streaming": is_streaming,
        "subscriber_count": subscriber_count,
    }


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


@router.websocket("/runs/{run_id}/events")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    """Stream real-time events from an agent run via WebSocket.

    This endpoint provides live streaming of agent activity events including:
    - Agent activity (planning, execution, reflection, etc.)
    - Run lifecycle events (started, completed, failed, cancelled)
    - Error events

    The connection remains open until the run completes or the client disconnects.

    Args:
        websocket: WebSocket connection
        run_id: Run identifier to stream events from

    Protocol:
        - Client connects and receives events as JSON messages
        - Each event follows the RunEvent schema with event_id, run_id,
          event_type, timestamp, and data fields
        - Connection closes automatically when run completes
        - Client can disconnect at any time
    """
    import logging
    ws_logger = logging.getLogger(__name__)

    await websocket.accept()
    ws_logger.info("WebSocket accepted for run %s", run_id)

    service = _get_agent_service()

    # Check if run is actively streaming
    is_streaming = service.is_run_streaming(run_id)
    ws_logger.info("Run %s streaming status: %s", run_id, is_streaming)

    if not is_streaming:
        # Run not active on this worker - check if it exists at all
        try:
            run_status = await service.get_run_status(run_id)
            ws_logger.info("Run %s exists but not streaming, status: %s", run_id, run_status.status.value)
            # Run exists but not streaming - it may have completed or be on another worker
            await websocket.send_json({
                "error": "run_not_streaming",
                "message": f"Run '{run_id}' is not actively streaming events. "
                           f"Current status: {run_status.status.value}",
                "run_status": run_status.model_dump(mode="json"),
            })
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except AgentRunNotFound:
            ws_logger.warning("Run %s not found", run_id)
            await websocket.send_json({
                "error": "run_not_found",
                "message": f"Run '{run_id}' not found",
            })
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    # Subscribe to run events
    event_manager = service.get_event_manager()
    ws_logger.info("Subscribing to events for run %s", run_id)

    try:
        event_count = 0
        async for event in event_manager.subscribe(run_id):
            event_count += 1
            ws_logger.info("Sending event #%d to client: %s", event_count, event.event_type)
            await websocket.send_json(event.model_dump(mode="json"))
        ws_logger.info("Event stream ended for run %s, sent %d events", run_id, event_count)
    except ValueError as exc:
        # Run stream closed or not found
        ws_logger.error("Stream error for run %s: %s", run_id, exc)
        await websocket.send_json({
            "error": "stream_error",
            "message": str(exc),
        })
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except WebSocketDisconnect:
        # Client disconnected - normal termination
        ws_logger.info("Client disconnected from run %s", run_id)

