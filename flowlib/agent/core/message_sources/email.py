"""Email-based message source."""

import asyncio
import logging
import queue
from datetime import datetime
from typing import Any

from flowlib.agent.core.message_source import MessageSource
from flowlib.agent.core.models.messages import AgentMessage, AgentMessageType
from flowlib.core.message_source_config import EmailMessageSourceConfig

logger = logging.getLogger(__name__)


class EmailMessageSource(MessageSource):
    """Produces messages when new emails arrive.

    This source monitors an email inbox and sends a message to the agent's
    input queue for each new email received.

    Example:
        >>> config = EmailMessageSourceConfig(
        ...     name="email_monitor",
        ...     email_provider_name="gmail-default",
        ...     check_interval_seconds=300,
        ...     folder="INBOX",
        ...     only_unread=True,
        ...     mark_as_read=True
        ... )
        >>> source = EmailMessageSource(config)
        >>> await source.start(agent.input_queue)
    """

    def __init__(self, config: EmailMessageSourceConfig):
        super().__init__(config)
        self.config: EmailMessageSourceConfig = config
        self._email_provider: Any | None = None

    async def start(self, input_queue: queue.Queue[AgentMessage]) -> None:
        """Start monitoring email.

        Args:
            input_queue: Agent's input queue to send messages to

        Raises:
            ValueError: If email provider cannot be loaded
        """
        await super().start(input_queue)

        # Get email provider
        from flowlib.providers.core.registry import provider_registry

        try:
            self._email_provider = await provider_registry.get_by_config(
                self.config.email_provider_name
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load email provider '{self.config.email_provider_name}': {e}"
) from e

        logger.info(
            f"Starting email source '{self.config.name}' "
            f"checking every {self.config.check_interval_seconds}s"
        )

        # Main check loop
        while not self._stopped:
            try:
                if self._enabled:
                    await self._check_and_send()

                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                logger.debug(f"Email source '{self.config.name}' cancelled")
                break
            except Exception as e:
                logger.error(f"Error in email source '{self.config.name}': {e}", exc_info=True)

        logger.info(f"Email source '{self.config.name}' stopped")

    async def _check_and_send(self) -> None:
        """Check for new emails and send messages."""
        if not self._input_queue or not self._email_provider:
            return

        try:
            # Fetch emails
            emails = await self._email_provider.fetch_emails(
                folder=self.config.folder, unread_only=self.config.only_unread
            )

            for email in emails:
                message = AgentMessage(
                    message_type=AgentMessageType.SYSTEM,
                    content=f"New email: {email.subject}",
                    context={
                        "source_name": self.config.name,
                        "source_type": "email",
                        "email_id": email.id,
                        "email_from": email.from_address,
                        "email_to": email.to_addresses,
                        "email_cc": email.cc_addresses,
                        "email_subject": email.subject,
                        "email_body": email.body,
                        "email_timestamp": email.timestamp.isoformat(),
                        "email_thread_id": email.thread_id,
                        "email_folder": email.folder,
                        "trigger_time": datetime.now().isoformat(),
                    },
                    response_queue_id="email_source",
                )

                self._input_queue.put(message)
                logger.info(f"Email source '{self.config.name}' sent message for: {email.subject}")

                if self.config.mark_as_read:
                    await self._email_provider.mark_as_read(email.id, folder=self.config.folder)

        except Exception as e:
            logger.error(
                f"Error checking emails in source '{self.config.name}': {e}",
                exc_info=True,
            )

    async def stop(self) -> None:
        """Stop email monitoring.

        Disconnects from email provider and releases resources.
        """
        logger.info(f"Stopping email source '{self.config.name}'")
        await super().stop()

        if self._email_provider:
            try:
                await self._email_provider.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down email provider: {e}", exc_info=True)
