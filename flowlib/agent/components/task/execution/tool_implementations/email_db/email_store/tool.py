"""Email store tool - saves emails to PostgreSQL with auto-analysis."""

import logging
from datetime import datetime

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import ToolExecutionContext, ToolStatus
from flowlib.agent.components.task.execution.tool_implementations.email_db.analyze_email.tool import (
    AnalyzeEmailTool,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.db_service import (
    EmailDBService,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    AnalyzeEmailParameters,
    EmailAnalysis,
    EmailStoreParameters,
    EmailStoreResult,
)
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


@tool(
    parameter_type=EmailStoreParameters,
    name="email_store",
    tool_category="email_db",
    description="Store an email in the database with auto-analysis, contact creation, and thread linking",
    planning_description="Save email to database with analysis",
)
class EmailStoreTool:
    """Tool for storing emails in PostgreSQL.

    Features:
    - Auto-creates contacts for new senders
    - Auto-links to existing threads (via in_reply_to)
    - Creates new threads for new conversations
    - Optional LLM analysis (sentiment, topics, intent)
    - Logs interaction for customer timeline
    """

    def get_name(self) -> str:
        """Return tool name."""
        return "email_store"

    def get_description(self) -> str:
        """Return tool description."""
        return "Store an email in the database with auto-analysis, contact creation, and thread linking"

    async def execute(
        self,
        todo: TodoItem,
        params: EmailStoreParameters,
        context: ToolExecutionContext,
    ) -> EmailStoreResult:
        """Execute email storage.

        Args:
            todo: The task description
            params: Validated EmailStoreParameters
            context: Execution context

        Returns:
            EmailStoreResult with storage details
        """
        try:
            # Get database provider
            db_provider = await provider_registry.get_by_config("default-database")
            db_service = EmailDBService(db_provider)

            # Parse received_at timestamp
            received_at = datetime.fromisoformat(params.received_at.replace("Z", "+00:00"))

            # 1. Get or create contact for sender
            contact, is_new_contact = await db_service.get_or_create_contact(
                email=params.from_address,
                display_name=None,  # Could extract from From header
            )
            contact_id = contact.id

            # 2. Find or create thread
            thread_id: str | None = None
            is_new_thread = False

            if params.in_reply_to:
                # Try to find thread for the email being replied to
                thread_id = await db_service.find_thread_for_reply(params.in_reply_to)

            if not thread_id:
                # Create new thread
                thread_id = await db_service.create_thread(
                    subject=params.subject,
                    contact_id=contact_id,
                    original_message_id=params.message_id,
                )
                is_new_thread = True

            # 3. Analyze email if requested
            analysis: EmailAnalysis | None = None
            if params.auto_analyze:
                analysis = await self._analyze_email(params.subject, params.body)

            # 4. Store the email
            email_id = await db_service.store_email(
                message_id=params.message_id,
                thread_id=thread_id,
                contact_id=contact_id,
                direction=params.direction,
                from_address=params.from_address,
                to_addresses=params.to_addresses,
                cc_addresses=params.cc_addresses,
                subject=params.subject,
                body=params.body,
                html_body=params.html_body,
                received_at=received_at,
                in_reply_to=params.in_reply_to,
                raw_headers=params.raw_headers,
                analysis=analysis,
            )

            # 5. Update contact stats
            if params.direction == "inbound":
                await db_service.update_contact_stats(
                    contact_id=contact_id,
                    increment_received=1,
                    sentiment_score=analysis.sentiment_score if analysis else None,
                )
            else:
                await db_service.update_contact_stats(
                    contact_id=contact_id,
                    increment_sent=1,
                )

            # 6. Update thread activity
            await db_service.update_thread_activity(thread_id)

            # 7. Log interaction
            interaction_type = "email_received" if params.direction == "inbound" else "email_sent"
            subject_preview = (params.subject or "(no subject)")[:100]
            await db_service.log_interaction(
                contact_id=contact_id,
                interaction_type=interaction_type,
                summary=f"{params.direction.capitalize()} email: {subject_preview}",
                thread_id=thread_id,
                email_id=email_id,
            )

            return EmailStoreResult(
                status=ToolStatus.SUCCESS,
                message=f"Email stored successfully (ID: {email_id})",
                email_id=email_id,
                thread_id=thread_id,
                contact_id=contact_id,
                is_new_contact=is_new_contact,
                is_new_thread=is_new_thread,
                analysis=analysis,
            )

        except Exception as e:
            logger.error(f"Error storing email: {e}", exc_info=True)
            return EmailStoreResult(
                status=ToolStatus.ERROR,
                message=f"Failed to store email: {str(e)}",
            )

    async def _analyze_email(self, subject: str, body: str) -> EmailAnalysis | None:
        """Analyze email using the analyze_email tool.

        Args:
            subject: Email subject
            body: Email body

        Returns:
            EmailAnalysis or None if analysis fails
        """
        try:
            analyze_tool = AnalyzeEmailTool()
            analyze_params = AnalyzeEmailParameters(
                subject=subject,
                body=body,
                include_suggestions=True,
            )

            # Create a minimal TodoItem for the tool call
            todo = TodoItem(
                id="analyze-for-store",
                description="Analyze email for storage",
                status="in_progress",
            )

            # Create minimal context
            context = ToolExecutionContext(
                original_user_message="",
                conversation_history=[],
                allowed_tool_categories=["email_db"],
            )

            result = await analyze_tool.execute(todo, analyze_params, context)
            return result.analysis

        except Exception as e:
            logger.warning(f"Email analysis failed, storing without analysis: {e}")
            return None
