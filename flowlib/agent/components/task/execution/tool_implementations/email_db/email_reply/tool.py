"""Email reply tool - sends email replies and stores them in the database."""

import logging
from datetime import datetime, timezone

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import ToolExecutionContext, ToolStatus
from flowlib.agent.components.task.execution.tool_implementations.email_db.db_service import (
    EmailDBService,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    EmailReplyParameters,
    EmailReplyResult,
)
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.email.base import EmailProvider, SendEmailResult

logger = logging.getLogger(__name__)


@tool(
    parameter_type=EmailReplyParameters,
    name="email_reply",
    tool_category="email_db",
    description="Send an email reply and store it in the database with thread linking",
    planning_description="Send email reply to customer",
)
class EmailReplyTool:
    """Tool for sending email replies.

    Features:
    - Sends email via configured SMTP provider
    - Stores sent email in database
    - Links to existing thread
    - Updates contact interaction history
    - Marks original email as replied
    """

    def get_name(self) -> str:
        """Return tool name."""
        return "email_reply"

    def get_description(self) -> str:
        """Return tool description."""
        return "Send an email reply and store it in the database with thread linking"

    async def execute(
        self,
        todo: TodoItem,
        params: EmailReplyParameters,
        context: ToolExecutionContext,
    ) -> EmailReplyResult:
        """Execute email reply.

        Args:
            todo: The task description
            params: Validated EmailReplyParameters
            context: Execution context

        Returns:
            EmailReplyResult with send status and storage details
        """
        try:
            # Get providers
            db_provider = await provider_registry.get_by_config("default-database")
            email_provider: EmailProvider = await provider_registry.get_by_config("default-email")

            db_service = EmailDBService(db_provider)

            # Get the original email to reply to
            original_email = await db_service.get_email_by_message_id(params.reply_to_message_id)
            if not original_email:
                return EmailReplyResult(
                    status=ToolStatus.ERROR,
                    message=f"Original email not found: {params.reply_to_message_id}",
                    sent=False,
                )

            # Validate thread and contact context exist (required for proper reply handling)
            if not original_email.thread_id:
                return EmailReplyResult(
                    status=ToolStatus.ERROR,
                    message=f"Original email has no thread context: {params.reply_to_message_id}",
                    sent=False,
                )

            if not original_email.contact_id:
                return EmailReplyResult(
                    status=ToolStatus.ERROR,
                    message=f"Original email has no contact context: {params.reply_to_message_id}",
                    sent=False,
                )

            # Build subject (add Re: if not present)
            subject = params.subject or original_email.subject or ""
            if subject and not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"

            # Determine recipient
            to_address = params.to_address or original_email.from_address

            # Send email via provider - returns SendEmailResult with message_id and from_address
            try:
                send_result: SendEmailResult = await email_provider.send_email(
                    to=[to_address],
                    subject=subject,
                    body=params.body,
                    html_body=params.html_body,
                    cc=params.cc_addresses if params.cc_addresses else None,
                    in_reply_to=params.reply_to_message_id,
                )
            except Exception as send_error:
                logger.error(f"Failed to send email: {send_error}")
                return EmailReplyResult(
                    status=ToolStatus.ERROR,
                    message=f"Failed to send email: {str(send_error)}",
                    sent=False,
                )

            if not send_result.success:
                return EmailReplyResult(
                    status=ToolStatus.ERROR,
                    message=send_result.error or "Email provider returned failure status",
                    sent=False,
                )

            # Store the sent email in database using message_id and from_address from provider
            email_id = await db_service.store_email(
                message_id=send_result.message_id,
                thread_id=original_email.thread_id,
                contact_id=original_email.contact_id,
                direction="outbound",
                from_address=send_result.from_address,
                to_addresses=[to_address],
                cc_addresses=params.cc_addresses or [],
                subject=subject,
                body=params.body,
                html_body=params.html_body,
                received_at=datetime.now(timezone.utc),
                in_reply_to=params.reply_to_message_id,
                raw_headers=None,
                analysis=None,
            )

            # Update contact stats
            await db_service.update_contact_stats(
                contact_id=original_email.contact_id,
                increment_sent=1,
            )

            # Update thread activity
            await db_service.update_thread_activity(original_email.thread_id)

            # Mark original email as replied
            await db_service.mark_email_replied(original_email.id)

            # Log interaction
            subject_preview = subject[:100] if subject else "(no subject)"
            await db_service.log_interaction(
                contact_id=original_email.contact_id,
                interaction_type="email_sent",
                summary=f"Sent reply: {subject_preview}",
                thread_id=original_email.thread_id,
                email_id=email_id,
            )

            return EmailReplyResult(
                status=ToolStatus.SUCCESS,
                message=f"Reply sent successfully to {to_address}",
                sent=True,
                email_id=email_id,
                thread_id=original_email.thread_id,
                message_id=send_result.message_id,
            )

        except Exception as e:
            logger.error(f"Error sending email reply: {e}", exc_info=True)
            return EmailReplyResult(
                status=ToolStatus.ERROR,
                message=f"Failed to send reply: {str(e)}",
                sent=False,
            )
