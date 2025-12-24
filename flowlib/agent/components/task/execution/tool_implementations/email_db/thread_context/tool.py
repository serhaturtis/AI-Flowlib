"""Thread context tool - retrieves full conversation thread with analysis."""

import logging

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import ToolExecutionContext, ToolStatus
from flowlib.agent.components.task.execution.tool_implementations.email_db.db_service import (
    EmailDBService,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    ThreadContextParameters,
    ThreadContextResult,
)
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


@tool(
    parameter_type=ThreadContextParameters,
    name="thread_context",
    tool_category="email_db",
    description="Get full conversation thread context including all emails and contact info",
    planning_description="Retrieve conversation thread history",
)
class ThreadContextTool:
    """Tool for retrieving full conversation thread context.

    Returns:
    - Thread metadata (subject, status, dates)
    - All emails in the thread (chronological order)
    - Contact profile for the thread owner
    - Thread summary and analysis
    """

    def get_name(self) -> str:
        """Return tool name."""
        return "thread_context"

    def get_description(self) -> str:
        """Return tool description."""
        return "Get full conversation thread context including all emails and contact info"

    async def execute(
        self,
        todo: TodoItem,
        params: ThreadContextParameters,
        context: ToolExecutionContext,
    ) -> ThreadContextResult:
        """Execute thread context retrieval.

        Args:
            todo: The task description
            params: Validated ThreadContextParameters
            context: Execution context

        Returns:
            ThreadContextResult with thread data and emails
        """
        try:
            # Get database provider
            db_provider = await provider_registry.get_by_config("default-database")
            db_service = EmailDBService(db_provider)

            # Get thread - try by ID first, then by message_id
            thread = None
            if params.thread_id:
                thread = await db_service.get_thread_by_id(params.thread_id)
            elif params.message_id:
                thread = await db_service.get_thread_by_message_id(params.message_id)

            if not thread:
                return ThreadContextResult(
                    status=ToolStatus.SUCCESS,
                    message=f"Thread not found: {params.thread_id or params.message_id}",
                    exists=False,
                )

            # Get all emails in the thread
            emails = await db_service.get_thread_emails(
                thread_id=thread.id,
                limit=params.max_emails,
            )

            # Get contact profile if requested
            contact = None
            if params.include_contact and thread.contact_id:
                contact = await db_service.get_contact_by_id(thread.contact_id)

            # Generate thread summary if requested
            summary = None
            if params.include_summary and emails:
                summary = self._generate_thread_summary(thread, emails)

            return ThreadContextResult(
                status=ToolStatus.SUCCESS,
                message=f"Retrieved thread with {len(emails)} emails",
                thread=thread,
                emails=emails,
                contact=contact,
                summary=summary,
                exists=True,
            )

        except Exception as e:
            logger.error(f"Error retrieving thread context: {e}", exc_info=True)
            return ThreadContextResult(
                status=ToolStatus.ERROR,
                message=f"Failed to retrieve thread context: {str(e)}",
                exists=False,
            )

    def _generate_thread_summary(self, thread, emails: list) -> str:
        """Generate a summary of the thread for context.

        Args:
            thread: Thread object
            emails: List of emails in the thread

        Returns:
            Human-readable thread summary
        """
        if not emails:
            return "Empty thread - no emails found."

        # Count inbound/outbound
        inbound_count = sum(1 for e in emails if e.direction == "inbound")
        outbound_count = sum(1 for e in emails if e.direction == "outbound")

        # Get date range
        first_email = emails[0]
        last_email = emails[-1]

        # Build summary
        lines = [
            f"Thread: {thread.subject}",
            f"Status: {thread.status}",
            f"Emails: {len(emails)} total ({inbound_count} received, {outbound_count} sent)",
            f"Started: {first_email.received_at.isoformat()}",
            f"Last activity: {last_email.received_at.isoformat()}",
        ]

        # Add key points from analysis if available
        topics_seen = set()
        for email in emails:
            if email.topics:
                topics_seen.update(email.topics)

        if topics_seen:
            lines.append(f"Topics discussed: {', '.join(sorted(topics_seen))}")

        # Add last email preview
        if last_email.body:
            preview = last_email.body[:200].replace("\n", " ")
            if len(last_email.body) > 200:
                preview += "..."
            lines.append(f"Last message preview: {preview}")

        return "\n".join(lines)
