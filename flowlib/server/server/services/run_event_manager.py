"""Run event manager for real-time agent monitoring.

This module provides a centralized manager for streaming agent execution
events to WebSocket clients. Each run has its own event queue that can
be subscribed to by multiple clients.

Architecture:
    Agent (ActivityStream) → RunEventManager → WebSocket clients

The manager handles:
- Creating/destroying event queues per run
- Publishing events from agent execution
- Broadcasting events to all subscribed clients
- Cleanup on run completion
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from server.models.events import RunEvent, RunEventType

logger = logging.getLogger(__name__)


class RunEventManager:
    """Manages event queues for agent run monitoring.

    Each run has a dedicated event queue. Clients subscribe via async
    iteration and receive events as they're published.

    Thread Safety:
        This manager uses asyncio.Lock for coordination. Agent execution
        happens in asyncio tasks, so all operations are async-safe.

    Usage:
        manager = RunEventManager()

        # In agent execution task:
        async with manager.create_run_stream(run_id):
            # Events can now be published
            await manager.publish(run_id, event)

        # In WebSocket handler:
        async for event in manager.subscribe(run_id):
            await websocket.send_json(event.model_dump())
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        """Initialize event manager.

        Args:
            max_queue_size: Maximum events buffered per run queue.
                When exceeded, oldest events are dropped.
        """
        self._max_queue_size = max_queue_size
        self._lock = asyncio.Lock()

        # run_id → list of subscriber queues
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | None]]] = {}

        # Track active runs for cleanup
        self._active_runs: set[str] = set()

    @asynccontextmanager
    async def create_run_stream(self, run_id: str) -> AsyncIterator[None]:
        """Context manager to create and manage a run's event stream.

        This should wrap the entire agent execution. Events can be published
        while inside this context. On exit, all subscribers are notified
        that the stream has ended.

        Args:
            run_id: Unique run identifier

        Yields:
            None - use publish() to send events

        Example:
            async with manager.create_run_stream(run_id):
                await manager.publish(run_id, RunEvent.run_started(run_id))
                # ... agent execution ...
                await manager.publish(run_id, RunEvent.run_completed(run_id))
        """
        async with self._lock:
            if run_id in self._active_runs:
                raise ValueError(f"Run stream already exists for run_id: {run_id}")
            self._active_runs.add(run_id)
            self._subscribers[run_id] = []

        logger.debug("Created event stream for run %s", run_id)

        try:
            yield
        finally:
            # Signal end of stream to all subscribers
            async with self._lock:
                subscriber_queues = self._subscribers.pop(run_id, [])
                self._active_runs.discard(run_id)

            # Send None sentinel to signal stream end
            for queue in subscriber_queues:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass  # Queue full, subscriber will timeout anyway

            logger.debug("Closed event stream for run %s (%d subscribers)", run_id, len(subscriber_queues))

    async def publish(self, run_id: str, event: RunEvent) -> None:
        """Publish an event to all subscribers of a run.

        Args:
            run_id: Run identifier
            event: Event to publish

        Raises:
            ValueError: If run stream doesn't exist (not inside create_run_stream context)
        """
        async with self._lock:
            if run_id not in self._active_runs:
                # Run stream not active - this can happen if publish is called
                # after cleanup or for a run that doesn't exist
                logger.warning("Attempted to publish to non-existent run stream: %s", run_id)
                return

            subscriber_queues = self._subscribers.get(run_id, [])
            subscriber_count = len(subscriber_queues)

        logger.info("Publishing event %s to run %s (%d subscribers)", event.event_type, run_id, subscriber_count)

        # Publish to all subscribers outside lock to avoid blocking
        published = 0
        for queue in subscriber_queues:
            try:
                queue.put_nowait(event)
                published += 1
            except asyncio.QueueFull:
                # Drop oldest event to make room
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                    published += 1
                except asyncio.QueueEmpty:
                    pass

        if subscriber_count > 0:
            logger.debug("Published event to %d/%d subscribers", published, subscriber_count)

    async def subscribe(self, run_id: str, timeout: float | None = None) -> AsyncIterator[RunEvent]:
        """Subscribe to events from a run.

        Yields events as they're published until the run completes
        or the connection is closed.

        Args:
            run_id: Run identifier to subscribe to
            timeout: Optional timeout in seconds for waiting on events.
                None means wait indefinitely.

        Yields:
            RunEvent instances as they're published

        Raises:
            ValueError: If run doesn't exist
            asyncio.TimeoutError: If timeout specified and no event received

        Example:
            async for event in manager.subscribe(run_id):
                print(f"Got event: {event.event_type}")
        """
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(maxsize=self._max_queue_size)

        # Register subscriber
        async with self._lock:
            if run_id not in self._active_runs:
                raise ValueError(f"No active run stream for run_id: {run_id}")
            self._subscribers[run_id].append(queue)

        total_subscribers = len(self._subscribers.get(run_id, []))
        logger.info("New subscriber connected for run %s (total: %d)", run_id, total_subscribers)

        # Send a welcome event to confirm connection works
        welcome_event = RunEvent(
            run_id=run_id,
            event_type=RunEventType.LOG,
            timestamp=datetime.now(tz=timezone.utc),
            data={
                "level": "INFO",
                "message": f"Connected to event stream for run {run_id[:8]}...",
                "logger_name": "run_event_manager",
            },
        )
        try:
            queue.put_nowait(welcome_event)
            logger.info("Sent welcome event to new subscriber for run %s", run_id)
        except asyncio.QueueFull:
            pass

        try:
            while True:
                try:
                    if timeout is not None:
                        event = await asyncio.wait_for(queue.get(), timeout=timeout)
                    else:
                        event = await queue.get()
                except asyncio.TimeoutError:
                    raise

                # None sentinel signals end of stream
                if event is None:
                    logger.debug("Subscriber received end-of-stream for run %s", run_id)
                    return

                yield event
        finally:
            # Unregister subscriber
            async with self._lock:
                subscriber_list = self._subscribers.get(run_id, [])
                if queue in subscriber_list:
                    subscriber_list.remove(queue)

            logger.debug("Subscriber disconnected from run %s", run_id)

    def is_run_active(self, run_id: str) -> bool:
        """Check if a run stream is currently active.

        Args:
            run_id: Run identifier

        Returns:
            True if run has an active event stream
        """
        return run_id in self._active_runs

    def get_active_runs(self) -> list[str]:
        """Get list of all active run IDs.

        Returns:
            List of run IDs with active event streams
        """
        return list(self._active_runs)

    def get_subscriber_count(self, run_id: str) -> int:
        """Get number of subscribers for a run.

        Args:
            run_id: Run identifier

        Returns:
            Number of active subscribers, 0 if run not active
        """
        return len(self._subscribers.get(run_id, []))


# Global singleton instance
_run_event_manager: RunEventManager | None = None


def get_run_event_manager() -> RunEventManager:
    """Get the global run event manager instance.

    Returns:
        RunEventManager singleton
    """
    global _run_event_manager
    if _run_event_manager is None:
        _run_event_manager = RunEventManager()
    return _run_event_manager


def create_activity_handler(run_id: str, manager: RunEventManager, loop: asyncio.AbstractEventLoop) -> callable:
    """Create an ActivityStream output handler that publishes to RunEventManager.

    This function creates a handler that can be set as the ActivityStream's
    output_handler. It converts activity output to RunEvents and publishes
    them to the event manager.

    IMPORTANT: The agent runs in a separate thread, so we must use
    run_coroutine_threadsafe() to schedule async publishes on the main event loop.

    Args:
        run_id: Run identifier for events
        manager: RunEventManager instance
        loop: The asyncio event loop to schedule publishes on (must be the main loop)

    Returns:
        Handler function compatible with ActivityStream.set_output_handler()

    Usage:
        loop = asyncio.get_running_loop()
        handler = create_activity_handler(run_id, manager, loop)
        agent.set_activity_stream_handler(handler)
    """

    def handler(formatted_output: str) -> None:
        """Handle ActivityStream output by publishing as event.

        Note: This is called synchronously from the agent's thread,
        so we use run_coroutine_threadsafe to schedule on the main loop.
        """
        try:
            logger.info("Activity handler called for run %s: %s", run_id, formatted_output[:100] if len(formatted_output) > 100 else formatted_output)

            # Create event with the formatted output
            event = RunEvent(
                run_id=run_id,
                event_type=RunEventType.ACTIVITY,
                timestamp=datetime.now(tz=timezone.utc),
                data={
                    "formatted": formatted_output,
                    "activity_type": "ACTIVITY",
                    "message": formatted_output,
                },
            )

            # Schedule async publish on the main event loop from this thread
            # This is thread-safe and works across thread boundaries
            future = asyncio.run_coroutine_threadsafe(manager.publish(run_id, event), loop)

            # Wait briefly to catch any immediate errors
            try:
                future.result(timeout=0.1)
            except TimeoutError:
                pass  # Expected - we don't want to block
            except Exception as pub_error:
                logger.error("Failed to publish event: %s", pub_error)

        except Exception as e:
            # Don't let handler errors crash the agent
            logger.error("Error in activity handler: %s", e, exc_info=True)

    return handler


def create_structured_activity_handler(run_id: str, manager: RunEventManager) -> callable:
    """Create a handler that receives structured activity data.

    This is an alternative handler that expects to receive structured
    activity data (type, message, details) rather than formatted strings.
    Use this when you have access to modify how activities are emitted.

    Args:
        run_id: Run identifier for events
        manager: RunEventManager instance

    Returns:
        Handler function that accepts (activity_type, message, details)
    """

    async def async_handler(activity_type: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Handle structured activity data."""
        event = RunEvent.activity(
            run_id=run_id,
            activity_type=activity_type,
            message=message,
            details=details,
        )
        await manager.publish(run_id, event)

    def sync_handler(activity_type: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Sync wrapper for async handler."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(async_handler(activity_type, message, details))
            else:
                loop.run_until_complete(async_handler(activity_type, message, details))
        except Exception as e:
            logger.error("Error in structured activity handler: %s", e)

    return sync_handler
