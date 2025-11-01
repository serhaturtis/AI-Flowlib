"""IMAP/SMTP email provider implementation.

This module implements email operations using standard IMAP and SMTP protocols
with async support via aioimaplib and aiosmtplib.
"""

import email
import logging
from datetime import datetime
from email.message import EmailMessage as StdEmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.email.base import EmailMessage, EmailProvider, EmailProviderSettings

logger = logging.getLogger(__name__)

# Lazy import email libraries
try:
    import aioimaplib
    import aiosmtplib

    EMAIL_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("Email libraries not found. Install with: pip install aioimaplib aiosmtplib")
    EMAIL_LIBS_AVAILABLE = False
    # Create placeholders
    aioimaplib = None  # type: ignore
    aiosmtplib = None  # type: ignore


class IMAPSMTPSettings(EmailProviderSettings):
    """Settings for IMAP/SMTP email provider.

    Inherits all settings from EmailProviderSettings.
    This class exists for future IMAP/SMTP-specific extensions.
    """

    pass


@provider(provider_type="email", name="imap-smtp", settings_class=IMAPSMTPSettings)
class IMAPSMTPProvider(EmailProvider[IMAPSMTPSettings]):
    """IMAP/SMTP email provider implementation.

    Provides email operations using standard IMAP (for reading) and
    SMTP (for sending) protocols.
    """

    def __init__(
        self,
        name: str,
        provider_type: str,
        settings: IMAPSMTPSettings | None = None,
        **kwargs: Any,
    ):
        """Initialize IMAP/SMTP provider.

        Args:
            name: Provider name
            provider_type: Provider type
            settings: Provider settings
            **kwargs: Additional arguments
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._imap_client: Any | None = None
        self._current_folder: str | None = None

    async def _initialize(self) -> None:
        """Initialize IMAP connection."""
        if not EMAIL_LIBS_AVAILABLE:
            raise ProviderError(
                message="Email libraries not available. Install with: pip install aioimaplib aiosmtplib",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="DependencyError",
                    error_location="IMAPSMTPProvider._initialize",
                    component=self.name,
                    operation="initialize",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="initialize",
                    retry_count=0,
                ),
            )

        try:
            # Connect to IMAP server
            if self.settings.use_ssl:
                self._imap_client = aioimaplib.IMAP4_SSL(
                    host=self.settings.imap_host,
                    port=self.settings.imap_port,
                    timeout=self.settings.timeout,
                )
            else:
                self._imap_client = aioimaplib.IMAP4(
                    host=self.settings.imap_host,
                    port=self.settings.imap_port,
                    timeout=self.settings.timeout,
                )

            await self._imap_client.wait_hello_from_server()

            # Login
            response = await self._imap_client.login(self.settings.username, self.settings.password)

            if response.result != "OK":
                raise ProviderError(
                    message=f"IMAP login failed: {response.lines}",
                    context=ErrorContext.create(
                        flow_name="email_provider",
                        error_type="AuthenticationError",
                        error_location="IMAPSMTPProvider._initialize",
                        component=self.name,
                        operation="login",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="login",
                        retry_count=0,
                    ),
                )

            logger.info(f"IMAP connection established for {self.settings.username}")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize IMAP connection: {e}")
            raise ProviderError(
                message=f"Failed to initialize IMAP connection: {str(e)}",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="ConnectionError",
                    error_location="IMAPSMTPProvider._initialize",
                    component=self.name,
                    operation="connect",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="connect",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def _shutdown(self) -> None:
        """Shutdown IMAP connection."""
        if self._imap_client:
            try:
                await self._imap_client.logout()
                logger.info("IMAP connection closed")
            except Exception as e:
                logger.warning(f"Error closing IMAP connection: {e}")
            finally:
                self._imap_client = None
                self._current_folder = None

    async def list_folders(self) -> list[str]:
        """List available IMAP folders."""
        if not self._imap_client:
            raise ProviderError(
                message="IMAP client not initialized",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="StateError",
                    error_location="IMAPSMTPProvider.list_folders",
                    component=self.name,
                    operation="list_folders",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="list_folders",
                    retry_count=0,
                ),
            )

        try:
            response = await self._imap_client.list()
            if response.result != "OK":
                raise ProviderError(
                    message=f"Failed to list folders: {response.lines}",
                    context=ErrorContext.create(
                        flow_name="email_provider",
                        error_type="OperationError",
                        error_location="IMAPSMTPProvider.list_folders",
                        component=self.name,
                        operation="list_folders",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="list_folders",
                        retry_count=0,
                    ),
                )

            folders = []
            for line in response.lines:
                # Parse IMAP LIST response: (flags) delimiter "folder name"
                parts = line.decode().split('"')
                if len(parts) >= 2:
                    folder_name = parts[-2]
                    folders.append(folder_name)

            return folders

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            raise ProviderError(
                message=f"Error listing folders: {str(e)}",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="OperationError",
                    error_location="IMAPSMTPProvider.list_folders",
                    component=self.name,
                    operation="list_folders",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="list_folders",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def fetch_emails(
        self,
        folder: str = "INBOX",
        unread_only: bool = True,
        limit: int | None = None,
        since_date: datetime | None = None,
    ) -> list[EmailMessage]:
        """Fetch emails from IMAP folder."""
        if not self._imap_client:
            raise ProviderError(
                message="IMAP client not initialized",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="StateError",
                    error_location="IMAPSMTPProvider.fetch_emails",
                    component=self.name,
                    operation="fetch_emails",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="fetch_emails",
                    retry_count=0,
                ),
            )

        try:
            # Select folder if not already selected
            if self._current_folder != folder:
                response = await self._imap_client.select(folder)
                if response.result != "OK":
                    raise ProviderError(
                        message=f"Failed to select folder {folder}: {response.lines}",
                        context=ErrorContext.create(
                            flow_name="email_provider",
                            error_type="OperationError",
                            error_location="IMAPSMTPProvider.fetch_emails",
                            component=self.name,
                            operation="select_folder",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type=self.provider_type,
                            operation="select_folder",
                            retry_count=0,
                        ),
                    )
                self._current_folder = folder

            # Build search criteria
            search_criteria = "UNSEEN" if unread_only else "ALL"
            if since_date:
                date_str = since_date.strftime("%d-%b-%Y")
                search_criteria = f"{search_criteria} SINCE {date_str}"

            # Search for emails
            response = await self._imap_client.search(search_criteria)
            if response.result != "OK":
                raise ProviderError(
                    message=f"Search failed: {response.lines}",
                    context=ErrorContext.create(
                        flow_name="email_provider",
                        error_type="OperationError",
                        error_location="IMAPSMTPProvider.fetch_emails",
                        component=self.name,
                        operation="search",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="search",
                        retry_count=0,
                    ),
                )

            # Get email IDs
            email_ids_bytes = response.lines[0]
            if not email_ids_bytes:
                return []

            email_ids = email_ids_bytes.decode().split()
            if limit:
                email_ids = email_ids[-limit:]  # Get most recent N emails

            # Fetch emails
            emails = []
            for email_id in email_ids:
                try:
                    email_msg = await self._fetch_single_email(email_id, folder)
                    if email_msg:
                        emails.append(email_msg)
                except Exception as e:
                    logger.warning(f"Failed to fetch email {email_id}: {e}")
                    continue

            return emails

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            raise ProviderError(
                message=f"Error fetching emails: {str(e)}",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="OperationError",
                    error_location="IMAPSMTPProvider.fetch_emails",
                    component=self.name,
                    operation="fetch_emails",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="fetch_emails",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def _fetch_single_email(self, email_id: str, folder: str) -> EmailMessage | None:
        """Fetch a single email by ID."""
        response = await self._imap_client.fetch(email_id, "(RFC822 FLAGS)")
        if response.result != "OK":
            return None

        # Parse response
        raw_email = None
        is_read = False

        for line in response.lines:
            if isinstance(line, bytes):
                raw_email = line
            elif isinstance(line, str) and "\\Seen" in line:
                is_read = True

        if not raw_email:
            return None

        # Parse email
        msg = email.message_from_bytes(raw_email)

        # Extract addresses
        from_addr = parseaddr(msg.get("From", ""))[1]
        to_addrs = [parseaddr(addr)[1] for addr in msg.get_all("To", [])]
        cc_addrs = [parseaddr(addr)[1] for addr in msg.get_all("Cc", [])]

        # Extract body
        body = self._extract_body(msg)

        # Extract timestamp
        date_str = msg.get("Date")
        try:
            timestamp = parsedate_to_datetime(date_str) if date_str else datetime.now()
        except Exception:
            timestamp = datetime.now()

        return EmailMessage(
            id=email_id,
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs,
            subject=msg.get("Subject", ""),
            body=body,
            timestamp=timestamp,
            thread_id=msg.get("Message-ID"),
            in_reply_to=msg.get("In-Reply-To"),
            is_read=is_read,
            folder=folder,
            headers=dict(msg.items()),
        )

    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract email body from message."""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(errors="ignore")
            # Fallback to HTML if no plain text
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(errors="ignore")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode(errors="ignore")
        return ""

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
        """Send email via SMTP."""
        try:
            # Create email message
            msg = StdEmailMessage()
            msg["From"] = self.settings.username
            msg["To"] = ", ".join(to)
            if cc:
                msg["Cc"] = ", ".join(cc)
            if bcc:
                msg["Bcc"] = ", ".join(bcc)
            msg["Subject"] = subject
            msg.set_content(body)

            if reply_to:
                msg["Reply-To"] = reply_to
            if in_reply_to:
                msg["In-Reply-To"] = in_reply_to
                msg["References"] = in_reply_to

            # TODO: Handle attachments in future version

            # Send via SMTP
            smtp_kwargs = {
                "hostname": self.settings.smtp_host,
                "port": self.settings.smtp_port,
                "username": self.settings.username,
                "password": self.settings.password,
                "use_tls": self.settings.use_tls,
                "timeout": self.settings.timeout,
            }

            async with aiosmtplib.SMTP(**smtp_kwargs) as smtp:
                await smtp.send_message(msg)

            logger.info(f"Email sent successfully to {to}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise ProviderError(
                message=f"Failed to send email: {str(e)}",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="OperationError",
                    error_location="IMAPSMTPProvider.send_email",
                    component=self.name,
                    operation="send_email",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="send_email",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def mark_as_read(self, email_id: str, folder: str = "INBOX") -> bool:
        """Mark email as read by adding \\Seen flag."""
        return await self._set_flag(email_id, folder, "\\Seen", add=True)

    async def mark_as_unread(self, email_id: str, folder: str = "INBOX") -> bool:
        """Mark email as unread by removing \\Seen flag."""
        return await self._set_flag(email_id, folder, "\\Seen", add=False)

    async def _set_flag(self, email_id: str, folder: str, flag: str, add: bool) -> bool:
        """Set or remove a flag on an email."""
        if not self._imap_client:
            raise ProviderError(
                message="IMAP client not initialized",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="StateError",
                    error_location="IMAPSMTPProvider._set_flag",
                    component=self.name,
                    operation="set_flag",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="set_flag",
                    retry_count=0,
                ),
            )

        try:
            # Select folder if needed
            if self._current_folder != folder:
                await self._imap_client.select(folder)
                self._current_folder = folder

            # Set/remove flag
            command = "+FLAGS" if add else "-FLAGS"
            response = await self._imap_client.store(email_id, command, flag)

            return response.result == "OK"

        except Exception as e:
            logger.error(f"Error setting flag on email {email_id}: {e}")
            return False

    async def move_to_folder(self, email_id: str, from_folder: str, to_folder: str) -> bool:
        """Move email to different folder."""
        if not self._imap_client:
            raise ProviderError(
                message="IMAP client not initialized",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="StateError",
                    error_location="IMAPSMTPProvider.move_to_folder",
                    component=self.name,
                    operation="move_to_folder",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="move_to_folder",
                    retry_count=0,
                ),
            )

        try:
            # Select source folder
            if self._current_folder != from_folder:
                await self._imap_client.select(from_folder)
                self._current_folder = from_folder

            # Copy to destination
            response = await self._imap_client.copy(email_id, to_folder)
            if response.result != "OK":
                return False

            # Mark as deleted in source
            response = await self._imap_client.store(email_id, "+FLAGS", "\\Deleted")
            if response.result != "OK":
                return False

            # Expunge deleted messages
            await self._imap_client.expunge()

            return True

        except Exception as e:
            logger.error(f"Error moving email {email_id}: {e}")
            return False

    async def delete_email(self, email_id: str, folder: str = "INBOX") -> bool:
        """Delete email by marking as deleted and expunging."""
        if not self._imap_client:
            raise ProviderError(
                message="IMAP client not initialized",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="StateError",
                    error_location="IMAPSMTPProvider.delete_email",
                    component=self.name,
                    operation="delete_email",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="delete_email",
                    retry_count=0,
                ),
            )

        try:
            # Select folder
            if self._current_folder != folder:
                await self._imap_client.select(folder)
                self._current_folder = folder

            # Mark as deleted
            response = await self._imap_client.store(email_id, "+FLAGS", "\\Deleted")
            if response.result != "OK":
                return False

            # Expunge
            await self._imap_client.expunge()

            return True

        except Exception as e:
            logger.error(f"Error deleting email {email_id}: {e}")
            return False

    async def search_emails(
        self, query: str, folder: str = "INBOX", limit: int | None = None
    ) -> list[EmailMessage]:
        """Search emails using IMAP search criteria."""
        if not self._imap_client:
            raise ProviderError(
                message="IMAP client not initialized",
                context=ErrorContext.create(
                    flow_name="email_provider",
                    error_type="StateError",
                    error_location="IMAPSMTPProvider.search_emails",
                    component=self.name,
                    operation="search_emails",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="search_emails",
                    retry_count=0,
                ),
            )

        try:
            # Select folder
            if self._current_folder != folder:
                await self._imap_client.select(folder)
                self._current_folder = folder

            # Search
            response = await self._imap_client.search(query)
            if response.result != "OK":
                return []

            # Get email IDs
            email_ids_bytes = response.lines[0]
            if not email_ids_bytes:
                return []

            email_ids = email_ids_bytes.decode().split()
            if limit:
                email_ids = email_ids[-limit:]

            # Fetch emails
            emails = []
            for email_id in email_ids:
                try:
                    email_msg = await self._fetch_single_email(email_id, folder)
                    if email_msg:
                        emails.append(email_msg)
                except Exception as e:
                    logger.warning(f"Failed to fetch email {email_id}: {e}")
                    continue

            return emails

        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            return []

    async def check_connection(self) -> bool:
        """Check if IMAP connection is active."""
        if not self._imap_client:
            return False

        try:
            response = await self._imap_client.noop()
            return response.result == "OK"
        except Exception:
            return False
