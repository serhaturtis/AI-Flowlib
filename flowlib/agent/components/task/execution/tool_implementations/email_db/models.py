"""Pydantic models for email database tools.

This module defines all parameter and result models for:
- analyze_email: LLM-based email analysis
- email_store: Store email with analysis
- email_search: Search past emails
- contact_lookup: Get customer profile
- thread_context: Get conversation thread
- email_reply: Send reply with threading
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from flowlib.agent.components.task.execution.models import (
    ToolParameters,
    ToolResult,
    ToolStatus,
)
from flowlib.core.models import StrictBaseModel


# =============================================================================
# Common Models (shared across tools)
# =============================================================================


class EmailAnalysis(StrictBaseModel):
    """LLM-extracted email analysis data."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the email"
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0, description="Sentiment score from -1.0 (negative) to 1.0 (positive)"
    )
    topics: list[str] = Field(default_factory=list, description="Extracted topic tags")
    intent: Literal[
        "inquiry", "complaint", "purchase", "support", "feedback", "escalation", "other"
    ] = Field(description="Primary intent of the email")
    urgency: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal", description="Urgency level"
    )
    key_entities: list[str] = Field(
        default_factory=list, description="Key entities mentioned (products, features, etc.)"
    )
    suggested_action: str = Field(default="", description="Suggested response action")
    requires_human: bool = Field(
        default=False, description="Whether this requires human escalation"
    )


class EmailSummary(StrictBaseModel):
    """Summary of an email for search results and thread context."""

    id: str = Field(description="Email UUID")
    message_id: str | None = Field(default=None, description="Email Message-ID header")
    thread_id: str | None = Field(default=None, description="Thread UUID")
    contact_id: str | None = Field(default=None, description="Contact UUID")
    direction: Literal["inbound", "outbound"] = Field(description="Email direction")
    from_address: str = Field(description="Sender email address")
    to_addresses: list[str] = Field(description="Recipient email addresses")
    subject: str | None = Field(default=None, description="Email subject")
    body: str = Field(description="Full email body text")
    body_preview: str = Field(default="", description="First 500 chars of body")
    received_at: datetime = Field(description="When email was received/sent")
    sentiment: str | None = Field(default=None, description="Sentiment classification")
    sentiment_score: float | None = Field(default=None, description="Sentiment score")
    topics: list[str] = Field(default_factory=list, description="Extracted topic tags")
    intent: str | None = Field(default=None, description="Intent classification")
    urgency: str | None = Field(default=None, description="Urgency level")
    is_replied: bool = Field(default=False, description="Whether email has been replied to")


class ContactProfile(StrictBaseModel):
    """Customer contact profile."""

    id: str = Field(description="Contact UUID")
    email: str = Field(description="Email address")
    display_name: str | None = Field(default=None, description="Display name")
    company: str | None = Field(default=None, description="Company name")
    first_seen_at: datetime = Field(description="First contact timestamp")
    last_contact_at: datetime = Field(description="Last contact timestamp")
    total_emails_sent: int = Field(default=0, description="Emails sent to this contact")
    total_emails_received: int = Field(default=0, description="Emails received from this contact")
    sentiment_score: float | None = Field(default=None, description="Rolling average sentiment")
    tags: list[str] = Field(default_factory=list, description="Contact tags")
    profile_data: dict = Field(default_factory=dict, description="Additional profile data")


class InteractionRecord(StrictBaseModel):
    """Record of a customer interaction."""

    id: str = Field(description="Interaction UUID")
    interaction_type: str = Field(description="Type of interaction")
    summary: str | None = Field(default=None, description="Interaction summary")
    outcome: str | None = Field(default=None, description="Outcome of interaction")
    created_at: datetime = Field(description="When interaction occurred")
    email_id: str | None = Field(default=None, description="Related email UUID")
    thread_id: str | None = Field(default=None, description="Related thread UUID")


class ThreadSummary(StrictBaseModel):
    """Summary of an email thread/conversation."""

    id: str = Field(description="Thread UUID")
    subject: str | None = Field(default=None, description="Thread subject")
    contact_id: str | None = Field(default=None, description="Primary contact UUID")
    contact_email: str | None = Field(default=None, description="Primary contact email")
    status: str = Field(default="open", description="Thread status")
    priority: str = Field(default="normal", description="Thread priority")
    category: str | None = Field(default=None, description="Thread category")
    started_at: datetime = Field(description="When thread started")
    last_activity_at: datetime = Field(description="Last activity timestamp")
    message_count: int = Field(default=0, description="Number of messages in thread")


# =============================================================================
# analyze_email Tool Models
# =============================================================================


class AnalyzeEmailParameters(ToolParameters):
    """Parameters for analyze_email tool."""

    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text")
    sender_email: str | None = Field(default=None, description="Sender email for context")
    include_suggestions: bool = Field(
        default=True, description="Include suggested response actions"
    )


class AnalyzeEmailResult(ToolResult):
    """Result from analyze_email tool."""

    analysis: EmailAnalysis | None = Field(default=None, description="Email analysis data")


# =============================================================================
# email_store Tool Models
# =============================================================================


class EmailStoreParameters(ToolParameters):
    """Parameters for email_store tool."""

    message_id: str = Field(description="Email Message-ID header (unique identifier)")
    from_address: str = Field(description="Sender email address")
    to_addresses: list[str] = Field(description="Recipient email addresses")
    cc_addresses: list[str] = Field(default_factory=list, description="CC recipients")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body text")
    html_body: str | None = Field(default=None, description="HTML body if available")
    received_at: str = Field(description="ISO timestamp when email was received")
    in_reply_to: str | None = Field(default=None, description="Message-ID being replied to")
    direction: Literal["inbound", "outbound"] = Field(
        default="inbound", description="Email direction"
    )
    raw_headers: dict | None = Field(default=None, description="Raw email headers")
    auto_analyze: bool = Field(
        default=True, description="Automatically analyze email after storing"
    )


class EmailStoreResult(ToolResult):
    """Result from email_store tool."""

    email_id: str | None = Field(default=None, description="UUID of stored email")
    thread_id: str | None = Field(default=None, description="UUID of associated thread")
    contact_id: str | None = Field(default=None, description="UUID of sender contact")
    is_new_contact: bool = Field(default=False, description="Whether contact was newly created")
    is_new_thread: bool = Field(default=False, description="Whether thread was newly created")
    analysis: EmailAnalysis | None = Field(default=None, description="Analysis if auto_analyze=True")


# =============================================================================
# email_search Tool Models
# =============================================================================


class EmailSearchParameters(ToolParameters):
    """Parameters for email_search tool."""

    query: str | None = Field(default=None, description="Full-text search query")
    contact_id: str | None = Field(default=None, description="Filter by contact UUID")
    contact_email: str | None = Field(default=None, description="Filter by contact email")
    thread_id: str | None = Field(default=None, description="Filter by thread UUID")
    date_from: str | None = Field(default=None, description="Filter emails after this ISO date")
    date_to: str | None = Field(default=None, description="Filter emails before this ISO date")
    sentiment: Literal["positive", "negative", "neutral"] | None = Field(
        default=None, description="Filter by sentiment"
    )
    min_sentiment: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Minimum sentiment score"
    )
    max_sentiment: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Maximum sentiment score"
    )
    intent: str | None = Field(default=None, description="Filter by intent")
    topic: str | None = Field(default=None, description="Filter by topic")
    direction: Literal["inbound", "outbound"] | None = Field(
        default=None, description="Filter by direction"
    )
    is_replied: bool | None = Field(default=None, description="Filter by replied status")
    sort_by: Literal["date", "sentiment"] = Field(
        default="date", description="Sort order for results"
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class EmailSearchResult(ToolResult):
    """Result from email_search tool."""

    emails: list[EmailSummary] = Field(default_factory=list, description="Matching emails")
    total_count: int = Field(default=0, description="Total matching emails (before pagination)")
    has_more: bool = Field(default=False, description="Whether more results exist")


# =============================================================================
# contact_lookup Tool Models
# =============================================================================


class ContactLookupParameters(ToolParameters):
    """Parameters for contact_lookup tool."""

    email: str | None = Field(default=None, description="Contact email address")
    contact_id: str | None = Field(default=None, description="Contact UUID")
    include_history: bool = Field(
        default=True, description="Include recent interaction history"
    )
    history_limit: int = Field(
        default=10, ge=1, le=50, description="Max interactions to include"
    )
    include_threads: bool = Field(default=True, description="Include recent threads")
    threads_limit: int = Field(default=5, ge=1, le=20, description="Max threads to include")


class ContactLookupResult(ToolResult):
    """Result from contact_lookup tool."""

    contact: ContactProfile | None = Field(default=None, description="Contact profile")
    recent_interactions: list[InteractionRecord] = Field(
        default_factory=list, description="Recent interactions"
    )
    recent_threads: list[ThreadSummary] = Field(
        default_factory=list, description="Recent conversation threads"
    )
    exists: bool = Field(default=False, description="Whether contact exists in database")


# =============================================================================
# thread_context Tool Models
# =============================================================================


class ThreadContextParameters(ToolParameters):
    """Parameters for thread_context tool."""

    thread_id: str | None = Field(default=None, description="Thread UUID")
    message_id: str | None = Field(
        default=None, description="Message-ID to find thread for"
    )
    include_contact: bool = Field(default=True, description="Include contact profile")
    include_summary: bool = Field(default=True, description="Include thread summary text")
    max_emails: int = Field(
        default=50, ge=1, le=100, description="Max emails to include"
    )


class ThreadContextResult(ToolResult):
    """Result from thread_context tool."""

    thread: ThreadSummary | None = Field(default=None, description="Thread summary")
    emails: list[EmailSummary] = Field(
        default_factory=list, description="Emails in thread (chronological)"
    )
    contact: ContactProfile | None = Field(
        default=None, description="Primary contact profile"
    )
    summary: str | None = Field(default=None, description="Human-readable thread summary")
    exists: bool = Field(default=False, description="Whether thread exists")


# =============================================================================
# email_reply Tool Models
# =============================================================================


class EmailReplyParameters(ToolParameters):
    """Parameters for email_reply tool."""

    reply_to_message_id: str = Field(description="Message-ID of the email being replied to")
    body: str = Field(description="Reply body text")
    to_address: str | None = Field(
        default=None, description="Override recipient address (defaults to original sender)"
    )
    cc_addresses: list[str] = Field(default_factory=list, description="CC recipients")
    subject: str | None = Field(default=None, description="Override subject (defaults to Re: original)")
    html_body: str | None = Field(default=None, description="HTML body if different from text")


class EmailReplyResult(ToolResult):
    """Result from email_reply tool."""

    sent: bool = Field(default=False, description="Whether the email was sent successfully")
    email_id: str | None = Field(default=None, description="UUID of sent email")
    thread_id: str | None = Field(default=None, description="Thread UUID")
    message_id: str | None = Field(default=None, description="Message-ID of sent email")
