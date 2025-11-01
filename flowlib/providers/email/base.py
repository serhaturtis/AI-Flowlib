"""Email provider base class and related functionality.

This module provides the base class for implementing email providers
such as IMAP/SMTP, Gmail API, etc.
"""

import logging
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.base import Provider, ProviderSettings

logger = logging.getLogger(__name__)


class EmailMessage(StrictBaseModel):
    """Email message model.

    Represents a single email message with all relevant metadata.
    """

    id: str = Field(..., description="Unique message identifier")
    from_address: str = Field(..., description="Sender email address")
    to_addresses: list[str] = Field(default_factory=list, description="Recipient email addresses")
    cc_addresses: list[str] = Field(default_factory=list, description="CC email addresses")
    bcc_addresses: list[str] = Field(default_factory=list, description="BCC email addresses")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    timestamp: datetime = Field(..., description="Email timestamp")
    thread_id: str | None = Field(default=None, description="Thread/conversation ID")
    in_reply_to: str | None = Field(default=None, description="Message ID this is replying to")
    attachments: list[dict[str, Any]] = Field(
        default_factory=list, description="Email attachments metadata"
    )
    headers: dict[str, str] = Field(default_factory=dict, description="Additional email headers")
    is_read: bool = Field(default=False, description="Whether email has been read")
    folder: str = Field(default="INBOX", description="Folder/mailbox containing this email")


class EmailProviderSettings(ProviderSettings):
    """Base settings for email providers.

    Common settings that apply to all email provider implementations.
    Individual providers can extend this with provider-specific settings.
    """

    # Connection settings
    imap_host: str = Field(..., description="IMAP server hostname")
    imap_port: int = Field(default=993, description="IMAP server port")
    smtp_host: str = Field(..., description="SMTP server hostname")
    smtp_port: int = Field(default=587, description="SMTP server port")

    # Authentication
    username: str = Field(..., description="Email account username")
    password: str = Field(..., description="Email account password (use environment variables)")

    # Security
    use_ssl: bool = Field(default=True, description="Use SSL/TLS for IMAP connection")
    use_tls: bool = Field(default=True, description="Use STARTTLS for SMTP connection")

    # Behavior
    timeout: float = Field(default=30.0, description="Operation timeout in seconds")
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed operations"
    )


SettingsT = TypeVar("SettingsT", bound=EmailProviderSettings)


class EmailProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for email providers.

    This class provides the interface for:
    1. Fetching emails from mailboxes
    2. Sending emails via SMTP
    3. Managing email folders
    4. Email operations (read, move, delete)
    """

    def __init__(
        self, name: str, provider_type: str, settings: SettingsT | None = None, **kwargs: Any
    ):
        """Initialize email provider.

        Args:
            name: Unique provider name
            provider_type: Provider type (e.g., 'email')
            settings: Optional provider settings
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the email provider."""
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False

    async def list_folders(self) -> list[str]:
        """List available email folders/mailboxes.

        Returns:
            List of folder names (e.g., ['INBOX', 'Sent', 'Drafts'])

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement list_folders()")

    async def fetch_emails(
        self,
        folder: str = "INBOX",
        unread_only: bool = True,
        limit: int | None = None,
        since_date: datetime | None = None,
    ) -> list[EmailMessage]:
        """Fetch emails from a folder.

        Args:
            folder: Folder name to fetch from (default: INBOX)
            unread_only: Only fetch unread emails
            limit: Maximum number of emails to fetch
            since_date: Only fetch emails after this date

        Returns:
            List of EmailMessage objects

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement fetch_emails()")

    async def send_email(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to: str | None = None,
        in_reply_to: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send an email.

        Args:
            to: List of recipient email addresses
            subject: Email subject
            body: Email body content
            cc: Optional CC recipients
            bcc: Optional BCC recipients
            reply_to: Optional Reply-To header
            in_reply_to: Optional In-Reply-To header (for threading)
            attachments: Optional list of attachments

        Returns:
            True if email sent successfully

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement send_email()")

    async def mark_as_read(self, email_id: str, folder: str = "INBOX") -> bool:
        """Mark an email as read.

        Args:
            email_id: Email message ID
            folder: Folder containing the email

        Returns:
            True if operation successful

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement mark_as_read()")

    async def mark_as_unread(self, email_id: str, folder: str = "INBOX") -> bool:
        """Mark an email as unread.

        Args:
            email_id: Email message ID
            folder: Folder containing the email

        Returns:
            True if operation successful

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement mark_as_unread()")

    async def move_to_folder(self, email_id: str, from_folder: str, to_folder: str) -> bool:
        """Move an email to a different folder.

        Args:
            email_id: Email message ID
            from_folder: Source folder
            to_folder: Destination folder

        Returns:
            True if operation successful

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement move_to_folder()")

    async def delete_email(self, email_id: str, folder: str = "INBOX") -> bool:
        """Delete an email.

        Args:
            email_id: Email message ID
            folder: Folder containing the email

        Returns:
            True if operation successful

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement delete_email()")

    async def search_emails(
        self, query: str, folder: str = "INBOX", limit: int | None = None
    ) -> list[EmailMessage]:
        """Search for emails matching a query.

        Args:
            query: Search query (provider-specific syntax)
            folder: Folder to search in
            limit: Maximum number of results

        Returns:
            List of matching EmailMessage objects

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement search_emails()")

    async def check_connection(self) -> bool:
        """Check if the email provider connection is active.

        Returns:
            True if connection is active and working

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement check_connection()")
