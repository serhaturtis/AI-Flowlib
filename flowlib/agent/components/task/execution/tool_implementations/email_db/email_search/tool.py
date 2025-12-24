"""Email search tool - searches emails by various criteria."""

import logging
from datetime import datetime

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import ToolExecutionContext, ToolStatus
from flowlib.agent.components.task.execution.tool_implementations.email_db.db_service import (
    EmailDBService,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    EmailSearchParameters,
    EmailSearchResult,
)
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


@tool(
    parameter_type=EmailSearchParameters,
    name="email_search",
    tool_category="email_db",
    description="Search emails by keyword, topic, sender, date range, or sentiment",
    planning_description="Search for emails matching criteria",
)
class EmailSearchTool:
    """Tool for searching emails in the database.

    Supports:
    - Full-text search on subject and body
    - Filter by topic
    - Filter by contact email or contact_id
    - Filter by direction (inbound/outbound)
    - Filter by date range
    - Filter by sentiment category or score range
    - Sort by date or sentiment
    """

    def get_name(self) -> str:
        """Return tool name."""
        return "email_search"

    def get_description(self) -> str:
        """Return tool description."""
        return "Search emails by keyword, topic, sender, date range, or sentiment"

    async def execute(
        self,
        todo: TodoItem,
        params: EmailSearchParameters,
        context: ToolExecutionContext,
    ) -> EmailSearchResult:
        """Execute email search.

        Args:
            todo: The task description
            params: Validated EmailSearchParameters
            context: Execution context

        Returns:
            EmailSearchResult with matching emails
        """
        try:
            # Get database provider
            db_provider = await provider_registry.get_by_config("default-database")
            db_service = EmailDBService(db_provider)

            # Parse date strings to datetime if provided
            date_from = self._parse_date(params.date_from) if params.date_from else None
            date_to = self._parse_date(params.date_to) if params.date_to else None

            # Execute search - returns tuple of (emails, total_count)
            emails, total_count = await db_service.search_emails(
                query_text=params.query,
                contact_id=params.contact_id,
                contact_email=params.contact_email,
                thread_id=params.thread_id,
                date_from=date_from,
                date_to=date_to,
                sentiment=params.sentiment,
                min_sentiment=params.min_sentiment,
                max_sentiment=params.max_sentiment,
                intent=params.intent,
                topic=params.topic,
                direction=params.direction,
                is_replied=params.is_replied,
                sort_by=params.sort_by,
                limit=params.limit,
                offset=params.offset,
            )

            return EmailSearchResult(
                status=ToolStatus.SUCCESS,
                message=f"Found {len(emails)} emails matching criteria (total: {total_count})",
                emails=emails,
                total_count=total_count,
                has_more=(params.offset + len(emails)) < total_count,
            )

        except Exception as e:
            logger.error(f"Error searching emails: {e}", exc_info=True)
            return EmailSearchResult(
                status=ToolStatus.ERROR,
                message=f"Failed to search emails: {str(e)}",
                emails=[],
                total_count=0,
                has_more=False,
            )

    def _parse_date(self, date_str: str) -> datetime:
        """Parse ISO date string to datetime.

        Args:
            date_str: ISO format date string

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date string is invalid
        """
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
