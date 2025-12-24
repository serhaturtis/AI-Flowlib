"""Email database tools for smart email sales agent.

This module provides tools for:
- Email analysis (sentiment, topics, intent)
- Email storage and retrieval
- Contact management
- Thread context
- Email replies
"""

from flowlib.agent.components.task.execution.tool_implementations.email_db.analyze_email import (
    AnalyzeEmailTool,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.contact_lookup import (
    ContactLookupTool,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.email_reply import (
    EmailReplyTool,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.email_search import (
    EmailSearchTool,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.email_store import (
    EmailStoreTool,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    AnalyzeEmailParameters,
    AnalyzeEmailResult,
    ContactLookupParameters,
    ContactLookupResult,
    ContactProfile,
    EmailAnalysis,
    EmailReplyParameters,
    EmailReplyResult,
    EmailSearchParameters,
    EmailSearchResult,
    EmailStoreParameters,
    EmailStoreResult,
    EmailSummary,
    InteractionRecord,
    ThreadContextParameters,
    ThreadContextResult,
    ThreadSummary,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.thread_context import (
    ThreadContextTool,
)

__all__ = [
    # Tools
    "AnalyzeEmailTool",
    "EmailStoreTool",
    "EmailSearchTool",
    "ContactLookupTool",
    "ThreadContextTool",
    "EmailReplyTool",
    # Analysis models
    "AnalyzeEmailParameters",
    "AnalyzeEmailResult",
    "EmailAnalysis",
    # Store models
    "EmailStoreParameters",
    "EmailStoreResult",
    # Search models
    "EmailSearchParameters",
    "EmailSearchResult",
    "EmailSummary",
    # Contact models
    "ContactLookupParameters",
    "ContactLookupResult",
    "ContactProfile",
    "InteractionRecord",
    # Thread models
    "ThreadContextParameters",
    "ThreadContextResult",
    "ThreadSummary",
    # Reply models
    "EmailReplyParameters",
    "EmailReplyResult",
]
