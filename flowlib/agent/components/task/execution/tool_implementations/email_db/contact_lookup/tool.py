"""Contact lookup tool - retrieves customer profiles and history."""

import logging

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import ToolExecutionContext, ToolStatus
from flowlib.agent.components.task.execution.tool_implementations.email_db.db_service import (
    EmailDBService,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    ContactLookupParameters,
    ContactLookupResult,
)
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


@tool(
    parameter_type=ContactLookupParameters,
    name="contact_lookup",
    tool_category="email_db",
    description="Look up a customer contact profile with interaction history and recent threads",
    planning_description="Get customer profile and history",
)
class ContactLookupTool:
    """Tool for looking up customer contact profiles.

    Returns:
    - Contact profile (email, name, company, sentiment trends)
    - Recent interaction history
    - Recent conversation threads
    """

    def get_name(self) -> str:
        """Return tool name."""
        return "contact_lookup"

    def get_description(self) -> str:
        """Return tool description."""
        return "Look up a customer contact profile with interaction history and recent threads"

    async def execute(
        self,
        todo: TodoItem,
        params: ContactLookupParameters,
        context: ToolExecutionContext,
    ) -> ContactLookupResult:
        """Execute contact lookup.

        Args:
            todo: The task description
            params: Validated ContactLookupParameters
            context: Execution context

        Returns:
            ContactLookupResult with profile and history
        """
        try:
            # Validate parameters
            if not params.email and not params.contact_id:
                return ContactLookupResult(
                    status=ToolStatus.ERROR,
                    message="Either email or contact_id must be provided",
                    exists=False,
                )

            # Get database provider
            db_provider = await provider_registry.get_by_config("default-database")
            db_service = EmailDBService(db_provider)

            # Look up contact
            contact = None
            if params.contact_id:
                contact = await db_service.get_contact_by_id(params.contact_id)
            elif params.email:
                contact = await db_service.get_contact_by_email(params.email)

            if not contact:
                return ContactLookupResult(
                    status=ToolStatus.SUCCESS,
                    message=f"Contact not found: {params.email or params.contact_id}",
                    exists=False,
                )

            # Get interaction history if requested
            recent_interactions = []
            if params.include_history:
                recent_interactions = await db_service.get_contact_interactions(
                    contact_id=contact.id,
                    limit=params.history_limit,
                )

            # Get recent threads if requested
            recent_threads = []
            if params.include_threads:
                recent_threads = await db_service.get_contact_threads(
                    contact_id=contact.id,
                    limit=params.threads_limit,
                )

            return ContactLookupResult(
                status=ToolStatus.SUCCESS,
                message=f"Found contact: {contact.email}",
                contact=contact,
                recent_interactions=recent_interactions,
                recent_threads=recent_threads,
                exists=True,
            )

        except Exception as e:
            logger.error(f"Error looking up contact: {e}", exc_info=True)
            return ContactLookupResult(
                status=ToolStatus.ERROR,
                message=f"Failed to look up contact: {str(e)}",
                exists=False,
            )
