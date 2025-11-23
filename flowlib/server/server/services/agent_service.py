"""Agent execution service.

This service uses the database as the single source of truth for run state,
ensuring consistency across multiple worker processes in production deployments.

In-memory state is kept minimal:
- Only active task references for cancellation (can't be stored in DB)
- No caching of run status or metadata
- All queries read directly from database

Multi-worker behavior:
- Any worker can query run status (reads from DB)
- Only the worker that created a run can cancel its task
- Run state updates are immediately persisted to DB
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from flowlib.agent.core.models.messages import (
    AgentMessage,
    AgentMessagePriority,
    AgentMessageType,
)
from flowlib.agent.core.thread_manager import AgentThreadPoolManager
from flowlib.agent.execution.strategy import ExecutionMode
from flowlib.agent.launcher import AgentLauncher, build_agent_config
from flowlib.core.project.project import Project
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.registry.registry import resource_registry

from server.core.registry_lock import registry_lock
from server.models.agents import (
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
    AgentRunStatusResponse,
)
from server.persistence.db import get_session
from server.persistence.models import RunHistory

logger = logging.getLogger(__name__)


@dataclass
class ActiveTask:
    """Lightweight record for active task management.

    Only stores the asyncio.Task reference needed for cancellation.
    All other state (status, timestamps, etc.) is stored in the database.
    """
    run_id: str
    task: asyncio.Task[Any]


@dataclass
class ReplSession:
    session_id: str
    project_id: str
    agent_name: str
    agent_id: str
    manager: AgentThreadPoolManager
    event_queue: asyncio.Queue[dict[str, Any]]
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    active: bool = True


class AgentRunNotFound(Exception):
    """Raised when a run id is unknown."""


class ReplSessionNotFound(Exception):
    """Raised when a REPL session is unknown."""


class AgentService:
    """Manage agent execution using AgentLauncher or interactive sessions.

    Uses database as single source of truth for run state.
    In-memory storage is minimal: only active task references for cancellation.
    """

    def __init__(self, projects_root: str = "./projects") -> None:
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        # Only store active task references (can't be stored in DB)
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._repl_sessions: dict[str, ReplSession] = {}
        self._lock = asyncio.Lock()

    def list_agents(self, project_id: str) -> list[str]:
        """List all agent configurations for a project."""
        project_path = self._resolve_project_path(project_id)
        with registry_lock:
            try:
                resource_registry.clear()
                project = Project(str(project_path))
                project.initialize()
                project.load_configurations()
                agents = resource_registry.get_by_type(ResourceType.AGENT_CONFIG)
                return sorted(agents.keys())
            finally:
                resource_registry.clear()

    async def start_run(self, payload: AgentRunRequest) -> AgentRunResponse:
        """Start a new agent run.

        Creates database record and background task. State is persisted to DB
        immediately, making it visible to all workers. Only the task reference
        is kept in memory for cancellation.
        """
        mode = ExecutionMode(payload.mode)
        run_id = str(uuid.uuid4())

        # Create database record first (single source of truth)
        with get_session() as session:
            db_run = RunHistory(
                run_id=run_id,
                project_id=payload.project_id,
                agent_name=payload.agent_config_name,
                mode=mode.value if hasattr(mode, "value") else str(mode),
                status=AgentRunStatus.PENDING,
                started_at=None,
                finished_at=None,
                message=None,
            )
            session.add(db_run)
            session.commit()

        try:
            # Create async task for execution
            task = asyncio.create_task(
                self._execute_run(run_id, payload.project_id, payload.agent_config_name, mode, payload.execution_config)
            )

            # Store only task reference for cancellation
            async with self._lock:
                self._active_tasks[run_id] = task

            return AgentRunResponse(run_id=run_id, status=AgentRunStatus.PENDING)

        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
            # Clean up database record on task creation failure
            logger.exception("Failed to start run %s", run_id)
            with get_session() as session:
                db_run = session.get(RunHistory, run_id)
                if db_run:
                    db_run.status = AgentRunStatus.FAILED
                    db_run.message = f"Task creation failed: {exc}"
                    db_run.finished_at = datetime.now(tz=timezone.utc)
                    session.commit()
            raise RuntimeError(f"Failed to start agent run: {exc}") from exc

    async def get_run_status(self, run_id: str) -> AgentRunStatusResponse:
        """Get run status from database (single source of truth).

        Any worker can query any run's status. This read goes directly to
        the database, ensuring consistency across all workers.
        """
        with get_session() as session:
            db_run = session.get(RunHistory, run_id)
            if not db_run:
                raise AgentRunNotFound(f"Run '{run_id}' not found")

            return AgentRunStatusResponse(
                run_id=db_run.run_id,
                status=db_run.status,
                started_at=db_run.started_at,
                finished_at=db_run.finished_at,
                message=db_run.message,
            )

    async def list_runs(self, limit: int = 100) -> list[AgentRunStatusResponse]:
        """List recent runs from database.

        Returns runs from all workers, ordered by start time (most recent first).
        This provides a consistent view across the entire multi-worker deployment.
        """
        with get_session() as session:
            stmt = (
                select(RunHistory)
                .order_by(RunHistory.started_at.desc().nullslast())
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()

            return [
                AgentRunStatusResponse(
                    run_id=r.run_id,
                    status=r.status,
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                    message=r.message,
                )
                for r in results
            ]

    def list_run_history(self, limit: int = 100) -> list[AgentRunStatusResponse]:
        """List persisted run history from database (most recent first)."""
        with get_session() as session:
            stmt = select(RunHistory).order_by(RunHistory.started_at.desc()).limit(limit)
            results = session.execute(stmt).scalars().all()

            return [
                AgentRunStatusResponse(
                    run_id=r.run_id,
                    status=r.status,
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                    message=r.message,
                )
                for r in results
            ]

    async def stop_run(self, run_id: str) -> AgentRunStatusResponse:
        """Stop a running agent execution.

        If the task is running in this worker, cancels it. Otherwise, only
        updates the database status. Note: can only cancel tasks on the same
        worker that created them.
        """
        # Verify run exists in database
        with get_session() as session:
            db_run = session.get(RunHistory, run_id)
            if not db_run:
                raise AgentRunNotFound(f"Run '{run_id}' not found")

        # Try to cancel task if it's in this worker
        task = None
        async with self._lock:
            task = self._active_tasks.get(run_id)

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Successfully cancelled run %s", run_id)

        # Update database status
        finished_at = datetime.now(tz=timezone.utc)
        with get_session() as session:
            db_run = session.get(RunHistory, run_id)
            if db_run:
                db_run.status = AgentRunStatus.CANCELLED
                db_run.finished_at = finished_at
                db_run.message = "Run cancelled by user"
                session.commit()

                return AgentRunStatusResponse(
                    run_id=db_run.run_id,
                    status=db_run.status,
                    started_at=db_run.started_at,
                    finished_at=db_run.finished_at,
                    message=db_run.message,
                )

        raise AgentRunNotFound(f"Run '{run_id}' not found")

    async def _execute_run(
        self,
        run_id: str,
        project_id: str,
        agent_name: str,
        mode: ExecutionMode,
        execution_config: dict[str, Any],
    ) -> None:
        """Execute agent run and persist all state changes to database.

        All status updates are written to database immediately, ensuring
        visibility across all workers.
        """
        # Update to running status
        started_at = datetime.now(tz=timezone.utc)
        with get_session() as session:
            db_run = session.get(RunHistory, run_id)
            if db_run:
                db_run.status = AgentRunStatus.RUNNING
                db_run.started_at = started_at
                session.commit()

        project_path = self._resolve_project_path(project_id)

        final_status = AgentRunStatus.COMPLETED
        final_message = "Run completed successfully"

        try:
            launcher = AgentLauncher(project_path=str(project_path))
            await launcher.initialize()
            await launcher.launch(
                agent_config_name=agent_name,
                mode=mode,
                execution_config=execution_config,
            )
        except asyncio.CancelledError:
            final_status = AgentRunStatus.CANCELLED
            final_message = "Run cancelled"
            raise
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            # Known failure modes - log and mark as failed
            logger.exception("Agent run %s failed with known error", run_id)
            final_status = AgentRunStatus.FAILED
            final_message = str(exc)
        # Let all other unexpected exceptions propagate - fail fast!
        finally:
            # Update final status in database
            finished_at = datetime.now(tz=timezone.utc)
            with get_session() as session:
                db_run = session.get(RunHistory, run_id)
                if db_run:
                    db_run.status = final_status
                    db_run.finished_at = finished_at
                    db_run.message = final_message
                    session.commit()

            # Clean up task reference from this worker
            async with self._lock:
                self._active_tasks.pop(run_id, None)

    async def create_repl_session(self, project_id: str, agent_name: str) -> ReplSession:
        """Create interactive REPL session for an agent.

        MULTI-WORKER LIMITATION: REPL sessions are stored in-memory and tied
        to the worker that created them. In a multi-worker deployment:
        - Session creation must use sticky sessions or a load balancer
        - All requests for a session must route to the same worker
        - Session state is NOT shared across workers

        For true multi-worker REPL support, implement Redis-backed session
        storage or use sticky sessions at the load balancer level.
        """
        project_path = self._resolve_project_path(project_id)
        project = Project(str(project_path))
        project.initialize()
        project.load_configurations()

        config = build_agent_config(project, agent_name)
        session_id = str(uuid.uuid4())
        agent_id = f"repl-{session_id}"

        # Create manager and ensure cleanup on error
        manager = AgentThreadPoolManager()
        try:
            manager.create_agent(agent_id, config)
            router = manager.get_response_router(agent_id)
            if router:
                await router.start()

            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            queue.put_nowait({"type": "system", "message": f"Session started for '{agent_name}'."})

            session = ReplSession(
                session_id=session_id,
                project_id=project_id,
                agent_name=agent_name,
                agent_id=agent_id,
                manager=manager,
                event_queue=queue,
            )

            async with self._lock:
                self._repl_sessions[session_id] = session

            return session
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            # Cleanup manager on error
            logger.exception("Failed to create REPL session for agent '%s'", agent_name)
            manager.shutdown_agent(agent_id)
            raise RuntimeError(f"Failed to create REPL session: {exc}") from exc

    def get_repl_event_queue(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        session = self._repl_sessions.get(session_id)
        if not session:
            raise ReplSessionNotFound(f"Session '{session_id}' not found")
        return session.event_queue

    async def send_repl_input(self, session_id: str, message: str) -> None:
        session = self._repl_sessions.get(session_id)
        if not session or not session.active:
            raise ReplSessionNotFound(f"Session '{session_id}' not found")

        session.event_queue.put_nowait({"type": "user", "content": message})

        agent_message = AgentMessage(
            message_type=AgentMessageType.CONVERSATION,
            content=message,
            context={"mode": "chat"},
            response_queue_id=session.agent_id,
            priority=AgentMessagePriority.NORMAL,
            timeout=None,
        )

        try:
            message_id = await session.manager.send_message(session.agent_id, agent_message)
            response = await session.manager.wait_for_response(
                session.agent_id, message_id, agent_message.timeout
            )
            if not response.success:
                session.event_queue.put_nowait(
                    {"type": "error", "message": response.error or "Agent error"}
                )
                return

            response_data = response.response_data
            session.event_queue.put_nowait(
                {
                    "type": "agent",
                    "content": response_data.content,
                    "activity": getattr(response_data, "activity", None),
                }
            )
        except (asyncio.TimeoutError, ValueError, RuntimeError) as exc:
            # Known error conditions - send to event queue
            session.event_queue.put_nowait({"type": "error", "message": str(exc)})
        # Let unexpected exceptions propagate - fail fast!

    async def close_repl_session(self, session_id: str) -> None:
        session = self._repl_sessions.get(session_id)
        if not session:
            raise ReplSessionNotFound(f"Session '{session_id}' not found")
        if session.active:
            session.manager.shutdown_agent(session.agent_id)
            session.event_queue.put_nowait({"type": "system", "message": "Session closed."})
            session.active = False
        async with self._lock:
            self._repl_sessions.pop(session_id, None)

    def _resolve_project_path(self, project_id: str) -> Path:
        project_path = (self._root / project_id).resolve()
        if not project_path.exists() or not project_path.is_dir():
            raise FileNotFoundError(f"Project '{project_id}' not found under {self._root}")
        if project_path == self._root or self._root not in project_path.parents:
            raise ValueError(f"Project '{project_id}' resolved outside managed root {self._root}")
        return project_path

